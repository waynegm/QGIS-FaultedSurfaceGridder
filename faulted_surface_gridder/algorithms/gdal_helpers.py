import os
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr
from qgis.core import (
    QgsVectorLayer,
)

from .helpers import GridSpec


def _vector_memory_driver():
    """Return the in-memory OGR driver, supporting old and new GDAL names."""
    driver = ogr.GetDriverByName("MEM")
    if driver is None:
        driver = ogr.GetDriverByName("Memory")
    return driver


def qgs_layer_to_ogr_memory_layer(layer: QgsVectorLayer) -> ogr.Layer | None:
    """Converts a QgsVectorLayer (including QGIS memory scratch layers)
    into an in-memory GDAL/OGR layer.
    """
    if layer is None or not layer.isValid():
        return None

    # Map QGIS WkbType to standard OGR WKB geometry type
    ogr_geom_type = layer.wkbType()

    # Create an OGR in-memory DataSource and Layer
    driver = _vector_memory_driver()
    # Unique dataset name to avoid memory clashes across multiple calls
    ds = driver.CreateDataSource(f"qgis_mem_ds_{id(layer)}")
    ogr_layer = ds.CreateLayer(layer.name(), geom_type=ogr_geom_type)

    # Transfer features using WKB fast-serialization
    for feature in iter(layer.getFeatures()):
        geom = feature.geometry()
        if geom.isNull() or geom.isEmpty():
            continue

        # Convert QGIS geometry bytes directly to OGR geometry
        ogr_geom = ogr.CreateGeometryFromWkb(geom.asWkb())

        ogr_feature = ogr.Feature(ogr_layer.GetLayerDefn())
        ogr_feature.SetGeometry(ogr_geom)
        ogr_layer.CreateFeature(ogr_feature)

    # Retain DataSource reference on the layer so GDAL doesn't garbage collect it
    ogr_layer._ds_ref = ds
    return ogr_layer

def get_ogr_layer(layer: QgsVectorLayer) -> tuple[ogr.Layer | None, object | None]:
    """Returns an (ogr.Layer, ogr.DataSource) tuple from a QgsVectorLayer."""
    if layer is None or not layer.isValid():
        return None, None

    # Attempt direct read for standard disk sources
    source_path = layer.source()
    try:
        ds = ogr.Open(source_path) if source_path else None
    except (RuntimeError, OSError):
        # GDAL may raise instead of returning None for non-file sources
        # such as QGIS memory layers.
        ds = None

    if ds is not None:
        return ds.GetLayer(0), ds

    # Fallback for QGIS scratch/reprojected layers: convert directly in memory
    mem_layer = qgs_layer_to_ogr_memory_layer(layer)
    if mem_layer is not None:
        return mem_layer, getattr(mem_layer, "_ds_ref", None)

    return None, None

def _resolve_ogr_layer(source) -> tuple[ogr.Layer | None, object | None]:
    """Resolve an OGR layer from a QgsVectorLayer, OGR layer/datasource or path."""
    if source is None:
        return None, None
    if isinstance(source, QgsVectorLayer):
        return get_ogr_layer(source)
    if isinstance(source, ogr.Layer):
        return source, getattr(source, "_ds_ref", None)
    if hasattr(source, "GetLayer") and callable(source.GetLayer):
        return source.GetLayer(0), source
    if isinstance(source, (str, Path)):
        try:
            ds = ogr.Open(str(source))
        except (RuntimeError, OSError):
            return None, None
        if ds is not None:
            return ds.GetLayer(0), ds
    return None, None


def rasterize_heave_interior(
    fault_polygon_source: str | Path | ogr.Layer | object | None,
    gridspec: GridSpec,
) -> np.ndarray:
    """Rasterise the interiors of fault-heave polygons to a boolean cell mask.

    Cells whose centre lies inside a polygon are True. Polygon boundaries are
    not included; the local gridder treats boundary/interior nodes as NoData.
    """
    rows = gridspec.nrows
    cols = gridspec.ncols
    mask = np.zeros((rows, cols), dtype=bool)

    heave_ogr, heave_ds = _resolve_ogr_layer(fault_polygon_source)
    if heave_ogr is None:
        return mask

    mem_driver = gdal.GetDriverByName("MEM")
    ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(
        [gridspec.origin_x, gridspec.step, 0, gridspec.top_y, 0, -gridspec.step]
    )
    heave_ogr.ResetReading()
    gdal.RasterizeLayer(
        ds, [1], heave_ogr, burn_values=[1], options=["ALL_TOUCHED=FALSE"]
    )
    mask = ds.GetRasterBand(1).ReadAsArray().astype(bool)
    ds = heave_ds = None
    return mask


def write_geotiff(path, array, spec, crs):
    """Write a node-centred grid as a GeoTIFF.

    Grid values are samples at ``GridSpec.nodes()``.  A GeoTIFF cell is
    registered by its top-left corner, so the transform is shifted by half a
    cell: each output pixel centre then falls exactly on a grid node.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, spec.ncols, spec.nrows, 1, gdal.GDT_Float32)
    half_step = spec.step / 2.0
    ds.SetGeoTransform(
        [
            spec.xmin - half_step,
            spec.step,
            0.0,
            spec.top_y + half_step,
            0.0,
            -spec.step,
        ]
    )
    if crs.isValid():
        ds.SetProjection(crs.toWkt())
    band = ds.GetRasterBand(1)
    data = np.asarray(array, dtype=np.float32)
    band.WriteArray(data)
    band.SetNoDataValue(float("nan"))
    band.FlushCache()
    ds = None
