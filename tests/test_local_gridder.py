import numpy as np
import pytest
from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsRectangle,
    QgsVectorLayer,
)

from faulted_surface_gridder.algorithms.fault_blocks import (
    assign_point_blocks,
    build_fault_blocks,
)
from faulted_surface_gridder.algorithms.helpers import (
    GridSpec,
    convex_hull_nodata_mask,
)
from faulted_surface_gridder.algorithms.local_gridder_impl import (
    build_point_samples,
    local_gridder_impl,
    weighted_mean,
)


def _point_layer(points: list[tuple[float, float, float]]) -> QgsVectorLayer:
    layer = QgsVectorLayer(
        "Point?crs=EPSG:3857&field=z:double", "points", "memory"
    )
    assert layer.isValid()
    features = []
    for x, y, z in points:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        feature["z"] = z
        features.append(feature)
    assert layer.dataProvider().addFeatures(features)
    return layer


def _line_layer(wkts: list[str]) -> QgsVectorLayer:
    layer = QgsVectorLayer("LineString?crs=EPSG:3857", "faults", "memory")
    assert layer.isValid()
    features = []
    for wkt in wkts:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromWkt(wkt))
        features.append(feature)
    assert layer.dataProvider().addFeatures(features)
    return layer


@pytest.fixture
def gridspec():
    return GridSpec(0.0, 0.0, 5, 5, 2.0)


def test_weighted_mean_includes_exact_node_point():
    value = weighted_mean(
        2.0,
        4.0,
        np.array([2.0, 3.0]),
        np.array([4.0, 4.0]),
        np.array([10.0, 20.0]),
    )
    expected = (10.0 + 20.0 * np.tanh(1.0)) / (1.0 + np.tanh(1.0))
    assert value == pytest.approx(expected)
    # The exact-node point must not simply override the other sample.
    assert value > 10.0


def test_vertical_fault_splits_blocks_and_marks_trace(gridspec):
    layer = _line_layer(["LINESTRING (4 0, 4 8)"])
    heave = np.zeros((5, 5), dtype=bool)
    blocks = build_fault_blocks(gridspec, layer, heave)

    # The fault column is NoData and separates left/right blocks.
    assert blocks.barrier[:, 2].all()
    assert blocks.block_ids[2, 0] != blocks.block_ids[2, 4]
    assert blocks.block_ids[2, 2] == -1


def test_partial_fault_reconnects_around_tip(gridspec):
    layer = _line_layer(["LINESTRING (4 4, 4 8)"])
    heave = np.zeros((5, 5), dtype=bool)
    blocks = build_fault_blocks(gridspec, layer, heave)

    # Nodes on the trace from y=8 down to y=4 are NoData.
    assert blocks.barrier[:3, 2].all()
    assert not blocks.barrier[3, 2]

    # Because the fault does not span the grid, both sides reconnect below
    # the tip and therefore belong to one connected block overall.
    assert blocks.block_ids[0, 0] == blocks.block_ids[0, 4]
    assert blocks.block_ids[4, 0] == blocks.block_ids[4, 4]


def test_off_node_vertical_fault_blocks_edges_without_barrier(gridspec):
    # x=3 falls between grid columns, so no node is exactly on the trace.
    layer = _line_layer(["LINESTRING (3 0, 3 8)"])
    blocks = build_fault_blocks(gridspec, layer, np.zeros((5, 5), dtype=bool))

    assert not blocks.barrier.any()
    assert blocks.blocked_h[:, 1].all()
    assert blocks.block_ids[2, 0] != blocks.block_ids[2, 3]


def test_off_node_fault_prevents_cross_fault_contribution(gridspec):
    # x=3 is between columns, so the fault is represented by blocked edges.
    layer = _line_layer(["LINESTRING (3 0, 3 8)"])
    points = _point_layer([(2.0, 2.0, 10.0), (4.0, 2.0, 20.0)])
    samples = build_point_samples(points, "z")
    blocks = build_fault_blocks(gridspec, layer, np.zeros((5, 5), dtype=bool))
    point_blocks = assign_point_blocks(
        samples.xs, samples.ys, samples.fids, gridspec, blocks
    )

    assert list(point_blocks) == [0, 1]

    result = local_gridder_impl(
        gridspec,
        samples,
        blocks.block_ids,
        point_blocks,
        blocks.barrier,
    )
    # Node (row=3, col=1) is at (2, 2).  The right-side sample is inside the
    # square window but must not contribute across the fault.
    assert result[3, 1] == pytest.approx(10.0)
    assert result[3, 2] == pytest.approx(20.0)


def test_point_on_fault_is_ambiguous(gridspec):
    layer = _line_layer(["LINESTRING (4 0, 4 8)"])
    blocks = build_fault_blocks(gridspec, layer, np.zeros((5, 5), dtype=bool))
    point_blocks = assign_point_blocks(
        np.array([2.0, 4.0, 6.0]),
        np.array([4.0, 4.0, 4.0]),
        np.array([1, 2, 3]),
        gridspec,
        blocks,
        heave_mask=np.zeros((5, 5), dtype=bool),
    )
    assert list(point_blocks) == [0, -1, 1]


def test_heave_cell_marks_its_corner_nodes(gridspec):
    heave = np.zeros((5, 5), dtype=bool)
    heave[2, 2] = True
    blocks = build_fault_blocks(gridspec, None, heave)
    assert blocks.barrier[2, 2]
    assert blocks.barrier[2, 3]
    assert blocks.barrier[3, 2]
    assert blocks.barrier[3, 3]
    assert not blocks.barrier[1, 1]


def test_convex_hull_mask_excludes_outside_nodes(gridspec):
    outside = convex_hull_nodata_mask(
        np.array([0.0, 8.0, 8.0]),
        np.array([0.0, 0.0, 8.0]),
        gridspec,
    )
    # Node (row=0, col=0) = (0, 8) is outside the triangle; (8, 0) is inside.
    assert outside[0, 0]
    assert not outside[4, 4]
    # Boundary node on the diagonal is retained.
    assert not outside[2, 2]


def test_degenerate_hull_does_not_clip(gridspec):
    outside = convex_hull_nodata_mask(
        np.array([0.0, 4.0]),
        np.array([0.0, 4.0]),
        gridspec,
    )
    assert not outside.any()


def test_local_gridder_uses_only_same_fault_block(gridspec):
    points = _point_layer([(2.0, 4.0, 10.0), (6.0, 4.0, 20.0)])
    fault = _line_layer(["LINESTRING (4 0, 4 8)"])
    samples = build_point_samples(points, "z")
    heave = np.zeros((5, 5), dtype=bool)
    blocks = build_fault_blocks(gridspec, fault, heave)
    point_blocks = assign_point_blocks(
        samples.xs, samples.ys, samples.fids, gridspec, blocks, heave
    )

    result = local_gridder_impl(
        gridspec,
        samples,
        blocks.block_ids,
        point_blocks,
        blocks.barrier,
    )

    # Row 2 is y=4: left node sees only z=10, right node only z=20.
    assert result[2, 1] == pytest.approx(10.0)
    assert result[2, 3] == pytest.approx(20.0)
    assert np.isnan(result[2, 2])
    assert np.isnan(result[0, 0])


def test_square_window_includes_diagonal_neighbour(gridspec):
    # Node (row=3, col=1) is at (2, 2).  This sample is within the square
    # window but outside a circular radius of one grid step.
    points = _point_layer([(3.9, 3.9, 42.0)])
    samples = build_point_samples(points, "z")
    heave = np.zeros((5, 5), dtype=bool)
    blocks = build_fault_blocks(gridspec, None, heave)
    point_blocks = assign_point_blocks(
        samples.xs, samples.ys, samples.fids, gridspec, blocks, heave
    )

    result = local_gridder_impl(
        gridspec,
        samples,
        blocks.block_ids,
        point_blocks,
        blocks.barrier,
    )

    assert result[3, 1] == pytest.approx(42.0)


def _polygon_layer(wkts: list[str]) -> QgsVectorLayer:
    layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "heave", "memory")
    assert layer.isValid()
    features = []
    for wkt in wkts:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromWkt(wkt))
        features.append(feature)
    assert layer.dataProvider().addFeatures(features)
    return layer


def _run_algorithm(tmp_path, points, **extra_parameters):
    from osgeo import gdal

    from faulted_surface_gridder.algorithms.local_gridder import LocalGridder

    algorithm = LocalGridder()
    algorithm.initAlgorithm()
    out_path = str(tmp_path / "grid.tif")
    parameters = {
        LocalGridder.INPUT_POINTS: points,
        LocalGridder.Z_FIELD: "z",
        LocalGridder.GRID_EXTENT: QgsRectangle(0, 0, 8, 8),
        LocalGridder.GRID_STEP: 2.0,
        LocalGridder.CLIP_TO_HULL: False,
        LocalGridder.OUTPUT: out_path,
    }
    parameters.update(extra_parameters)

    result = algorithm.processAlgorithm(
        parameters, QgsProcessingContext(), QgsProcessingFeedback()
    )
    assert result[LocalGridder.OUTPUT] == out_path

    dataset = gdal.Open(out_path)
    assert dataset is not None
    return dataset.GetRasterBand(1).ReadAsArray()


def test_process_algorithm_writes_fault_split(tmp_path):
    points = _point_layer([(2.0, 4.0, 10.0), (6.0, 4.0, 20.0)])
    fault = _line_layer(["LINESTRING (4 0, 4 8)"])
    result = _run_algorithm(tmp_path, points, FAULT_LINES=fault)

    assert result[1, 1] == pytest.approx(10.0)
    assert result[1, 3] == pytest.approx(20.0)
    assert np.isnan(result[1, 2])
    # The bottom row is more than one step from every sample.
    assert np.isnan(result[3, 0])
    assert np.isnan(result[3, 3])


def test_process_algorithm_writes_heave_nodata_and_hull_clip(tmp_path):
    points = _point_layer(
        [
            (0.0, 0.0, 10.0),
            (0.0, 6.0, 10.0),
            (6.0, 0.0, 10.0),
            (6.0, 6.0, 10.0),
            (3.0, 3.0, 10.0),
        ]
    )
    heave = _polygon_layer(["POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))"])
    result = _run_algorithm(
        tmp_path,
        points,
        FAULT_POLYGONS=heave,
        CLIP_TO_HULL=True,
    )

    # The heave polygon's corner nodes become NoData.
    assert np.isnan(result[1, 1])
    assert np.isnan(result[1, 2])
    assert np.isnan(result[2, 1])
    assert np.isnan(result[2, 2])
    # Nodes outside the heave polygon are still interpolated.
    assert result[0, 0] == pytest.approx(10.0)
    assert result[3, 3] == pytest.approx(10.0)


def test_no_valid_points_returns_all_nan(gridspec):
    points = _point_layer([])
    samples = build_point_samples(points, "z")
    heave = np.zeros((5, 5), dtype=bool)
    blocks = build_fault_blocks(gridspec, None, heave)
    point_blocks = assign_point_blocks(
        samples.xs, samples.ys, samples.fids, gridspec, blocks, heave
    )
    result = local_gridder_impl(
        gridspec,
        samples,
        blocks.block_ids,
        point_blocks,
        blocks.barrier,
    )
    assert np.isnan(result).all()
