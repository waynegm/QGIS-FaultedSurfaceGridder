from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

from faulted_surface_gridder.algorithms.fault_blocks import build_fault_blocks
from faulted_surface_gridder.algorithms.gdal_helpers import (
    rasterize_heave_interior,
)
from faulted_surface_gridder.algorithms.helpers import GridSpec


def _polygon_layer(wkts, crs="EPSG:3857"):
    layer = QgsVectorLayer(f"Polygon?crs={crs}", "heave", "memory")
    assert layer.isValid()
    features = []
    for wkt in wkts:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromWkt(wkt))
        features.append(feature)
    assert layer.dataProvider().addFeatures(features)
    return layer


def _line_layer(wkts, crs="EPSG:3857"):
    layer = QgsVectorLayer(f"LineString?crs={crs}", "faults", "memory")
    assert layer.isValid()
    features = []
    for wkt in wkts:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromWkt(wkt))
        features.append(feature)
    assert layer.dataProvider().addFeatures(features)
    return layer


def test_thin_heave_polygon_creates_continuous_barrier():
    # A 0.14-wide sliver extending beyond the grid.  Its interior misses every
    # cell centre (they lie at x = 4.5 and 5.5), so only the polygon-boundary
    # barrier keeps the two sides apart.
    spec = GridSpec(0.0, 0.0, 11, 11, 1.0)
    layer = _polygon_layer(
        ["POLYGON ((5.03 -1, 5.17 -1, 5.17 11, 5.03 11, 5.03 -1))"]
    )

    heave = rasterize_heave_interior(layer, spec)
    assert not heave.any()

    blocks = build_fault_blocks(spec, None, heave, heave_polygons_layer=layer)
    assert blocks.barrier[:, 5].all()
    assert blocks.block_ids[5, 2] != blocks.block_ids[5, 8]


def test_fault_line_between_nodes_separates_blocks():
    # Fault lines use exact segment/grid-edge intersections, so a trace that
    # runs between node columns still blocks the edges it crosses.
    spec = GridSpec(0.0, 0.0, 11, 11, 1.0)
    layer = _line_layer(["LINESTRING (5.03 -1, 5.03 11)"])

    blocks = build_fault_blocks(spec, layer, None)
    assert blocks.block_ids[5, 2] != blocks.block_ids[5, 8]


def test_diagonal_fault_line_separates_blocks():
    spec = GridSpec(0.0, 0.0, 11, 11, 1.0)
    layer = _line_layer(["LINESTRING (-1 -0.5, 11 11.5)"])

    blocks = build_fault_blocks(spec, layer, None)
    assert blocks.block_ids[5, 2] != blocks.block_ids[5, 8]
