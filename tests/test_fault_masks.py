from dataclasses import dataclass

import numpy as np
import pytest
from osgeo import gdal, ogr, osr

from faulted_surface_gridder.algorithms.gdal_helpers import generate_fault_masks

# Enable GDAL exceptions for clearer assertions
gdal.UseExceptions()


@dataclass
class GridSpec:
    origin_x: float = 0.0
    top_y: float = 10.0
    nrows: int = 5
    ncols: int = 5
    step: float = 2.0

@pytest.fixture
def gridspec():
    return GridSpec()

def create_ogr_memory_layer(geom_type: int, wkt_geometries: list[str]) -> ogr.Layer:
    """Helper to build in-memory OGR layers for unit tests."""
    driver = ogr.GetDriverByName("MEM")
    ds = driver.CreateDataSource(f"mem_ds_{id(wkt_geometries)}")
    assert ds is not None, "Failed to create in-memory OGR DataSource"

    layer = ds.CreateLayer("test_layer", geom_type=geom_type)
    for wkt in wkt_geometries:
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        layer.CreateFeature(feat)

    # Attach dataset reference to prevent premature C++ garbage collection
    layer._ds_ref = ds
    return layer


# =============================================================================
# TESTS
# =============================================================================


def test_none_sources_returns_empty_masks(gridspec):
    """Verify that passing None for both sources returns all-False boolean masks."""
    fault_left, fault_up, heave_int = generate_fault_masks(
        fault_line_source=None,
        fault_polygon_source=None,
        gridspec=gridspec,
    )

    assert fault_left.shape == (5, 5)
    assert fault_up.shape == (5, 5)
    assert heave_int.shape == (5, 5)

    assert fault_left.dtype == bool
    assert fault_up.dtype == bool
    assert heave_int.dtype == bool

    assert not np.any(fault_left)
    assert not np.any(fault_up)
    assert not np.any(heave_int)

def test_line_and_polygon_edge_alignment(gridspec):
    """
    Verify that a standalone line and a polygon boundary at the EXACT same
    physical coordinate map to identical array indices in fault_left and fault_up.
    """
    # Vertical coordinate X = 4.0 (Column 2)
    line_wkt = ["LINESTRING (4.0 10.0, 4.0 0.0)"]
    # Polygon edge along X = 4.0 from Y = 0.0 to 10.0
    poly_wkt = ["POLYGON ((0.0 0.0, 4.0 0.0, 4.0 10.0, 0.0 10.0, 0.0 0.0))"]

    line_layer = create_ogr_memory_layer(ogr.wkbLineString, line_wkt)
    poly_layer = create_ogr_memory_layer(ogr.wkbPolygon, poly_wkt)

    # Rasterize line source only
    left_line, _, _ = generate_fault_masks(
        fault_line_source=line_layer,
        fault_polygon_source=None,
        gridspec=gridspec,
    )

    # Rasterize polygon source only
    left_poly, _, _ = generate_fault_masks(
        fault_line_source=None,
        fault_polygon_source=poly_layer,
        gridspec=gridspec,
    )
    print(left_line)
    print(left_poly)
    # Both must burn onto Column 2
    assert np.any(left_line[:, 2])
    assert np.any(left_poly[:, 2])

    # Both outputs must match index-for-index across column 2
    np.testing.assert_array_equal(left_line[:, 2], left_poly[:, 2])

def test_vertical_and_horizontal_lines_rasterization(gridspec):
    """Verify that vertical lines mark fault_left and horizontal lines mark fault_up."""
    # Vertical line down X = 4.0 (grid col index 2)
    # Horizontal line across Y = 6.0 (grid row index 2)
    lines_wkt = [
        "LINESTRING (4.0 10.0, 4.0 0.0)",
        "LINESTRING (0.0 6.0, 10.0 6.0)",
    ]
    line_layer = create_ogr_memory_layer(ogr.wkbLineString, lines_wkt)

    fault_left, fault_up, nodata_mask = generate_fault_masks(
        fault_line_source=line_layer,
        fault_polygon_source=None,
        gridspec=gridspec,
    )

    # No heave polygons were supplied
    assert not np.any(nodata_mask)

    # Vertical line at X=4.0 corresponds to column index 2
    assert np.any(fault_left[:, 2])

    # Horizontal line at Y=6.0 corresponds to row index 2
    assert np.any(fault_up[2, :])


def test_polygon_heave_rasterization(gridspec):
    """Verify polygon interior generates heave_interior and perimeter edges burn to left/up."""
    # Square polygon covering interior cells (X: 2.0 to 6.0, Y: 4.0 to 8.0)
    poly_wkt = ["POLYGON ((2.0 4.0, 6.0 4.0, 6.0 8.0, 2.0 8.0, 2.0 4.0))"]
    poly_layer = create_ogr_memory_layer(ogr.wkbPolygon, poly_wkt)

    fault_left, fault_up, nodata_mask = generate_fault_masks(
        fault_line_source=None,
        fault_polygon_source=poly_layer,
        gridspec=gridspec,
    )

    # Central cell interior must be flagged True
    assert nodata_mask.sum() > 0

    # Outer perimeters should burn onto edge masks
    assert np.any(fault_left)
    assert np.any(fault_up)


def test_datasource_input_support(gridspec):
    """Verify function works when passing an ogr.DataSource directly instead of ogr.Layer."""
    driver = ogr.GetDriverByName("Memory")
    ds = driver.CreateDataSource("ds_test")
    assert ds is not None

    layer = ds.CreateLayer("lines", geom_type=ogr.wkbLineString)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt("LINESTRING (4.0 10.0, 4.0 0.0)"))
    layer.CreateFeature(feat)

    # Pass the DataSource object `ds`
    fault_left, _, _ = generate_fault_masks(
        fault_line_source=ds,
        fault_polygon_source=None,
        gridspec=gridspec,
    )
    print(fault_left)

    assert np.any(fault_left[:, 2])


def test_boundary_integrity_enforcement(gridspec):
    """Verify Step 3 strictly enforces fault_left[:, 0] = False and fault_up[0, :] = False."""
    # Place lines directly on outer left boundary (X = 0.0) and top boundary (Y = 10.0)
    edge_lines = [
        "LINESTRING (0.0 10.0, 0.0 0.0)",
        "LINESTRING (0.0 10.0, 10.0 10.0)",
    ]
    line_layer = create_ogr_memory_layer(ogr.wkbLineString, edge_lines)

    fault_left, fault_up, _ = generate_fault_masks(
        fault_line_source=line_layer,
        fault_polygon_source=None,
        gridspec=gridspec,
    )

    # Outer boundaries must be forced to False regardless of rasterization
    assert not np.any(fault_left[:, 0])
    assert not np.any(fault_up[0, :])
