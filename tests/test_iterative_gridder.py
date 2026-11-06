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

from faulted_surface_gridder.algorithms.helpers import GridSpec
from faulted_surface_gridder.algorithms.iterative_gridder_impl import (
    SCIPY_INSTALL_HINT,
    coarsen_gridspec,
    iterative_gridder_impl,
    require_scipy,
    resolve_scale_count,
    upsample_grid,
)


class _Feedback:
    """Minimal feedback object that records warnings."""

    def __init__(self):
        self.warnings = []
        self.messages = []

    def pushWarning(self, message):
        self.warnings.append(message)

    def pushInfo(self, message):
        self.messages.append(message)

    def setProgress(self, value):
        pass

    def isCanceled(self):
        return False


def _empty_masks(rows, cols):
    return (
        np.zeros((rows, cols), dtype=np.int32),
        np.zeros((rows, cols - 1), dtype=bool),
        np.zeros((rows - 1, cols), dtype=bool),
        np.zeros((rows, cols), dtype=bool),
    )


def test_require_scipy_available():
    require_scipy()


def test_require_scipy_reports_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="requires SciPy"):
        require_scipy()
    assert "pip install scipy" in SCIPY_INSTALL_HINT


def test_plugin_modules_import_without_scipy():
    """The local gridder and plugin must load even when SciPy is absent."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy blocked for test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        import faulted_surface_gridder.plugin
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_plane_is_reproduced_from_four_known_nodes():
    spec = GridSpec(0.0, 0.0, 7, 7, 1.0)
    x, y = spec.nodes()
    plane = 2.0 + 0.5 * x + 0.25 * y

    local = np.full(plane.shape, np.nan)
    for r, c in [(0, 0), (0, 6), (6, 0), (3, 3)]:
        local[r, c] = plane[r, c]

    block_ids, blocked_h, blocked_v, nodata = _empty_masks(7, 7)
    result = iterative_gridder_impl(
        spec,
        local,
        block_ids,
        blocked_h,
        blocked_v,
        nodata,
        iterations=500,
        tolerance=1e-12,
    )

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, plane, atol=1e-8)


def test_faults_isolate_blocks():
    spec = GridSpec(0.0, 0.0, 5, 5, 2.0)
    block_ids = np.zeros((5, 5), dtype=np.int32)
    block_ids[:, 2] = -1
    block_ids[:, 3:] = 1
    blocked_h = np.zeros((5, 4), dtype=bool)
    blocked_h[:, 1] = True
    blocked_h[:, 2] = True
    blocked_v = np.zeros((4, 5), dtype=bool)
    nodata = block_ids < 0

    local = np.full((5, 5), np.nan)
    for r, c in [(0, 0), (4, 0), (0, 1)]:
        local[r, c] = 10.0
    for r, c in [(0, 4), (4, 4), (0, 3)]:
        local[r, c] = 20.0

    result = iterative_gridder_impl(
        spec,
        local,
        block_ids,
        blocked_h,
        blocked_v,
        nodata,
        iterations=500,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(result[:, :2], 10.0, atol=1e-6)
    np.testing.assert_allclose(result[:, 3:], 20.0, atol=1e-6)
    assert np.isnan(result[:, 2]).all()


def test_mask_nodes_are_never_filled():
    spec = GridSpec(0.0, 0.0, 5, 5, 1.0)
    block_ids, blocked_h, blocked_v, _ = _empty_masks(5, 5)
    nodata = np.zeros((5, 5), dtype=bool)
    nodata[0, 0] = True

    local = np.full((5, 5), np.nan)
    for r, c in [(4, 0), (4, 4), (0, 4)]:
        local[r, c] = 1.0

    result = iterative_gridder_impl(
        spec,
        local,
        block_ids,
        blocked_h,
        blocked_v,
        nodata,
        iterations=100,
    )

    assert np.isnan(result[0, 0])
    assert np.isfinite(result[~nodata]).all()


def test_underconstrained_block_falls_back_to_mean():
    spec = GridSpec(0.0, 0.0, 4, 4, 1.0)
    block_ids, blocked_h, blocked_v, nodata = _empty_masks(4, 4)
    local = np.full((4, 4), np.nan)
    local[0, 0] = 7.0
    feedback = _Feedback()

    result = iterative_gridder_impl(
        spec,
        local,
        block_ids,
        blocked_h,
        blocked_v,
        nodata,
        iterations=100,
        feedback=feedback,
    )

    np.testing.assert_allclose(result, 7.0)
    assert feedback.warnings
    assert "fewer than 3" in feedback.warnings[0]


def test_block_without_data_stays_nodata():
    spec = GridSpec(0.0, 0.0, 4, 4, 1.0)
    block_ids = np.zeros((4, 4), dtype=np.int32)
    block_ids[:, 2:] = 1
    blocked_h = np.zeros((4, 3), dtype=bool)
    blocked_h[:, 1] = True
    blocked_v = np.zeros((3, 4), dtype=bool)
    nodata = np.zeros((4, 4), dtype=bool)

    local = np.full((4, 4), np.nan)
    for r, c in [(0, 0), (3, 0), (0, 1)]:
        local[r, c] = 5.0
    feedback = _Feedback()

    result = iterative_gridder_impl(
        spec,
        local,
        block_ids,
        blocked_h,
        blocked_v,
        nodata,
        iterations=100,
        feedback=feedback,
    )

    np.testing.assert_allclose(result[:, :2], 5.0)
    assert np.isnan(result[:, 2:]).all()
    assert any("contain no data" in message for message in feedback.warnings)


def test_coarsen_keeps_fine_nodes():
    fine = GridSpec(10.0, 20.0, 9, 7, 2.0)
    coarse = coarsen_gridspec(fine, 2)

    assert coarse.step == 4.0
    assert coarse.ncols == 5
    assert coarse.nrows == 4
    fine_x, fine_y = fine.nodes()
    coarse_x, coarse_y = coarse.nodes()
    np.testing.assert_allclose(coarse_x, fine_x[::2, ::2])
    np.testing.assert_allclose(coarse_y, fine_y[::2, ::2])


def test_upsample_reproduces_linear_surface():
    fine = GridSpec(0.0, 0.0, 9, 9, 1.0)
    coarse = coarsen_gridspec(fine, 2)
    x, y = coarse.nodes()

    upsampled = upsample_grid(1.0 + 2.0 * x - 3.0 * y, coarse, fine)

    fine_x, fine_y = fine.nodes()
    np.testing.assert_allclose(
        upsampled, 1.0 + 2.0 * fine_x - 3.0 * fine_y, atol=1e-10
    )


def test_automatic_scale_count_respects_grid_size():
    assert resolve_scale_count(GridSpec(0, 0, 2, 2, 1.0), 0) == 1
    assert resolve_scale_count(GridSpec(0, 0, 9, 9, 1.0), 0) == 4
    assert resolve_scale_count(GridSpec(0, 0, 65, 65, 1.0), 0) == 6


def _point_layer(points):
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


def _line_layer(wkts):
    layer = QgsVectorLayer("LineString?crs=EPSG:3857", "faults", "memory")
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

    from faulted_surface_gridder.algorithms.iterative_gridder import (
        IterativeGridder,
    )

    algorithm = IterativeGridder()
    algorithm.initAlgorithm()
    out_path = str(tmp_path / "grid.tif")
    parameters = {
        IterativeGridder.INPUT_POINTS: points,
        IterativeGridder.Z_FIELD: "z",
        IterativeGridder.GRID_EXTENT: QgsRectangle(0, 0, 8, 8),
        IterativeGridder.GRID_STEP: 2.0,
        IterativeGridder.CLIP_TO_HULL: False,
        IterativeGridder.ITERATIONS: 200,
        IterativeGridder.SCALES: 1,
        IterativeGridder.TOLERANCE: 1e-10,
        IterativeGridder.OUTPUT: out_path,
    }
    parameters.update(extra_parameters)

    result = algorithm.processAlgorithm(
        parameters, QgsProcessingContext(), QgsProcessingFeedback()
    )
    assert result[IterativeGridder.OUTPUT] == out_path

    dataset = gdal.Open(out_path)
    assert dataset is not None
    return dataset.GetRasterBand(1).ReadAsArray()


def test_process_algorithm_fills_full_grid(tmp_path):
    points = _point_layer(
        [(0.0, 0.0, 10.0), (0.0, 8.0, 10.0), (8.0, 0.0, 10.0), (8.0, 8.0, 10.0)]
    )
    result = _run_algorithm(tmp_path, points)

    assert result.shape == (5, 5)
    np.testing.assert_allclose(result, 10.0, atol=1e-5)


def test_process_algorithm_multiscale(tmp_path):
    points = _point_layer(
        [(0.0, 0.0, 10.0), (0.0, 8.0, 10.0), (8.0, 0.0, 10.0), (8.0, 8.0, 10.0)]
    )
    single = _run_algorithm(tmp_path, points, SCALES=1)

    multiscale_path = tmp_path / "multi"
    multiscale_path.mkdir()
    multi = _run_algorithm(multiscale_path, points, SCALES=3)

    np.testing.assert_allclose(multi, single, atol=1e-5)


def test_process_algorithm_respects_fault_split(tmp_path):
    points = _point_layer([(2.0, 4.0, 10.0), (6.0, 4.0, 20.0)])
    fault = _line_layer(["LINESTRING (4 0, 4 8)"])
    result = _run_algorithm(tmp_path, points, FAULT_LINES=fault)

    assert np.isnan(result[2, 2])
    np.testing.assert_allclose(result[:, :2], 10.0, atol=1e-4)
    np.testing.assert_allclose(result[:, 3:], 20.0, atol=1e-4)


def _polygon_layer(wkts):
    layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "heave", "memory")
    assert layer.isValid()
    features = []
    for wkt in wkts:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromWkt(wkt))
        features.append(feature)
    assert layer.dataProvider().addFeatures(features)
    return layer


def test_process_algorithm_respects_heave_and_hull(tmp_path):
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
        SCALES=3,
    )

    # Heave corner nodes are barriers.
    assert np.isnan(result[2, 1])
    assert np.isnan(result[2, 2])
    assert np.isnan(result[3, 1])
    assert np.isnan(result[3, 2])
    # Nodes above the convex hull are clipped.
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 4])
    # Interior nodes away from both masks are solved.
    assert np.isfinite(result[1, 1])
    assert np.isfinite(result[4, 1])
    np.testing.assert_allclose(result[1, 1], 10.0, atol=1e-4)
    np.testing.assert_allclose(result[4, 1], 10.0, atol=1e-4)
