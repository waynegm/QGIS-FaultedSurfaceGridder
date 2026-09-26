"""Grid faulted horizon algorithm.

Combines XYZ points, horizon polylines and a background raster into a
single sample set, splits the target grid into fault blocks and
interpolates each block independently.
"""

import math
import os

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProject,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication

from ..core import idw as _idw  # noqa: F401 - registers "idw"
from ..core import tps as _tps  # noqa: F401 - registers "tps"
from ..core.base import get_interpolator
from ..core.faults import BARRIER, label_blocks, rasterize_line_segments, rasterize_polygons
from ..core.grid import GridSpec, interpolate_blocks
from ..core.sampling import SampleSet, densify_polyline, merge_samples, sample_raster_values

METHOD_IDW = 0
METHOD_TPS = 1


class FaultedGridAlgorithm(QgsProcessingAlgorithm):
    INPUT_POINTS = "INPUT_POINTS"
    VALUE_FIELD = "VALUE_FIELD"
    USE_Z = "USE_Z"
    FAULT_LINES = "FAULT_LINES"
    FAULT_POLYGONS = "FAULT_POLYGONS"
    METHOD = "METHOD"
    IDW_POWER = "IDW_POWER"
    IDW_RADIUS = "IDW_RADIUS"
    IDW_MAX_NEIGHBORS = "IDW_MAX_NEIGHBORS"
    TPS_SMOOTHING = "TPS_SMOOTHING"
    TPS_NEIGHBORS = "TPS_NEIGHBORS"
    EXTENT = "EXTENT"
    PIXEL_SIZE = "PIXEL_SIZE"
    MIN_POINTS = "MIN_POINTS"
    CLIP_TO_HULL = "CLIP_TO_HULL"
    HULL_BUFFER = "HULL_BUFFER"
    OUTPUT = "OUTPUT"

    def tr(self, string, context=""):  # QGIS 4: QgsProcessingAlgorithm has no tr()
        if context == "":
            context = self.__class__.__name__
        return QCoreApplication.translate(context, string)

    def initAlgorithm(self, config=None):  # noqa: N802 - QGIS API
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                self.tr("Horizon points (XYZ)"),
                [QgsProcessing.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.VALUE_FIELD,
                self.tr("Point value field (ignored when Use Z is set and Z exists)"),
                parentLayerParameterName=self.INPUT_POINTS,
                allowMultiple=False,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_Z, self.tr("Use Z coordinate for points/lines when present"), True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.FAULT_LINES,
                self.tr("Fault traces (polylines, vertical faults)"),
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.FAULT_POLYGONS,
                self.tr("Fault heave polygons (interiors become NoData)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                self.tr("Interpolation method"),
                ["Inverse distance weighted (IDW)", "Thin plate spline"],
                defaultValue=METHOD_IDW,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IDW_POWER,
                self.tr("IDW power"),
                QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.1,
                maxValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IDW_RADIUS,
                self.tr("IDW search radius (0 = global)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IDW_MAX_NEIGHBORS,
                self.tr("IDW max neighbours"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=12,
                minValue=1,
                maxValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TPS_SMOOTHING,
                self.tr("TPS smoothing (0 = exact)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TPS_NEIGHBORS,
                self.tr("TPS neighbours per point (0 = auto)"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=500,
            )
        )
        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr("Grid extent")))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PIXEL_SIZE,
                self.tr("Pixel size (layer units)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=1e-9,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_POINTS,
                self.tr("Minimum samples per fault block"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLIP_TO_HULL,
                self.tr("Clip output to convex hull of input samples"),
                False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.HULL_BUFFER,
                self.tr("Hull buffer (layer units, 0 = exact hull)"),
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, self.tr("Gridded horizon"))
        )

    def checkParameterValues(self, parameters, context):
        if (
            not parameters.get(self.INPUT_POINTS)
            and not parameters.get(self.INPUT_LINES)
            and not parameters.get(self.INPUT_RASTER)
        ):
            return (
                False,
                self.tr("Provide at least one of points, polylines or background raster."),
            )
        return super().checkParameterValues(parameters, context)

    # ------------------------------------------------------------------ #
    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        import numpy as np

        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        extent_crs = self.parameterAsExtentCrs(parameters, self.EXTENT, context)
        pixel = self.parameterAsDouble(parameters, self.PIXEL_SIZE, context)
        if extent.isEmpty() or pixel <= 0:
            raise QgsProcessingException(self.tr("Invalid extent or pixel size."))
        spec = GridSpec(extent.xMinimum(), extent.yMinimum(), extent.xMaximum(),
                        extent.yMaximum(), pixel)
        feedback.pushInfo(f"Grid: {spec.ncols} x {spec.nrows} @ {pixel}")

        spacing = self.parameterAsDouble(parameters, self.LINE_SAMPLE_SPACING, context)
        if not spacing or spacing <= 0:
            spacing = pixel / 2.0

        sample_sets: list[SampleSet] = []
        target_crs = extent_crs if extent_crs.isValid() else QgsCoordinateReferenceSystem()

        if parameters.get(self.INPUT_POINTS):
            sample_sets.append(
                self._sample_points(parameters, context, target_crs, feedback)
            )
        if parameters.get(self.INPUT_LINES):
            sample_sets.append(
                self._sample_lines(parameters, context, target_crs, spacing, feedback)
            )
        if parameters.get(self.INPUT_RASTER):
            sample_sets.append(
                self._sample_raster(parameters, context, spec, target_crs, feedback)
            )
        merged = merge_samples(*sample_sets, tolerance=pixel / 10.0)
        if merged.xy.shape[0] == 0:
            raise QgsProcessingException(self.tr("No valid horizon samples found."))
        feedback.pushInfo(f"Total samples: {merged.xy.shape[0]}")

        barrier, gap = self._fault_masks(parameters, context, target_crs, spec, feedback)
        block_ids = label_blocks(barrier, gap)

        sample_block = self._locate_samples(merged.xy, block_ids, spec)
        usable = int(np.sum(sample_block != BARRIER))
        feedback.pushInfo(
            f"Fault blocks: {len(set(block_ids.ravel().tolist())) - (1 if (block_ids == BARRIER).any() else 0)}, "
            f"samples inside blocks: {usable}"
        )
        self._warn_if_tps_auto_cap(parameters, context, sample_block, feedback)

        factory = self._make_factory(parameters, context)
        min_points = self.parameterAsInt(parameters, self.MIN_POINTS, context)
        is_tps = self.parameterAsEnum(parameters, self.METHOD, context) == METHOD_TPS
        # IDW has no matrix solve, so it survives the degenerate sample
        # layouts (e.g. all picks along one line) that can defeat TPS.
        fallback_factory = None
        if is_tps:
            def fallback_factory():
                return get_interpolator("idw", power=2.0, max_neighbors=12)

        def _progress(frac: float) -> bool:
            if feedback.isCanceled():
                return False
            feedback.setProgress(int(100 * frac * 0.9))
            return True

        result = interpolate_blocks(
            factory, merged.xy, merged.z, block_ids, sample_block, spec,
            min_points=min_points, progress=_progress,
            warn=feedback.pushWarning,
            fallback_factory=fallback_factory,
        )
        if self.parameterAsBoolean(parameters, self.CLIP_TO_HULL, context):
            result = self._clip_to_hull(result, merged, sample_block, spec,
                                        pixel, parameters, context, feedback)

        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        self._write_geotiff(out_path, result, spec, target_crs)
        feedback.setProgress(100)
        return {self.OUTPUT: out_path}

    # ------------------------- sample extraction ---------------------- #
    def _transform(self, source_crs, target_crs):
        if not target_crs.isValid() or source_crs == target_crs:
            return None
        return QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

    def _sample_points(self, parameters, context, target_crs, feedback):
        import numpy as np

        source = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT_POINTS))
        field = self.parameterAsString(parameters, self.VALUE_FIELD, context)
        use_z = self.parameterAsBoolean(parameters, self.USE_Z, context)
        has_z = QgsWkbTypes.hasZ(source.wkbType())
        fidx = source.fields().lookupField(field) if field else -1
        tr = self._transform(source.sourceCrs(), target_crs)
        xs, zs = [], []
        total = source.featureCount() or 0
        for i, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geom = f.geometry()
            if geom.isEmpty():
                continue
            try:
                pt = next(geom.vertices())
            except StopIteration:
                continue
            x, y = pt.x(), pt.y()
            if tr is not None:
                from qgis.core import QgsPointXY

                t = tr.transform(QgsPointXY(x, y))
                x, y = t.x(), t.y()
            z = None
            if use_z and has_z:
                try:
                    z = float(pt.z())
                except Exception:
                    z = None
            if z is None or (z is not None and math.isnan(z)):
                if fidx >= 0:
                    try:
                        z = float(f.attributes()[fidx])
                    except (TypeError, ValueError):
                        z = None
            if z is None or math.isnan(z):
                continue
            xs.append((x, y))
            zs.append(z)
            if total and i % 10000 == 0:
                feedback.setProgress(int(5 * i / max(1, total)))
        return SampleSet(
            xy=np.array(xs, dtype=float).reshape(-1, 2),
            z=np.array(zs, dtype=float).reshape(-1),
        )

    def _sample_lines(self, parameters, context, target_crs, spacing, feedback):
        import numpy as np

        source = self.parameterAsSource(parameters, self.INPUT_LINES, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT_LINES))
        field = self.parameterAsString(parameters, self.LINE_VALUE_FIELD, context)
        const = self.parameterAsDouble(parameters, self.LINE_VALUE_CONSTANT, context)
        use_z = self.parameterAsBoolean(parameters, self.USE_Z, context)
        has_z = QgsWkbTypes.hasZ(source.wkbType())
        fidx = source.fields().lookupField(field) if field else -1
        tr = self._transform(source.sourceCrs(), target_crs)
        all_xy, all_z = [], []
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            geom = f.geometry()
            if geom.isEmpty():
                continue
            z_attr = None
            if fidx >= 0:
                try:
                    z_attr = float(f.attributes()[fidx])
                except (TypeError, ValueError):
                    z_attr = None
            for part in self._line_parts(geom):
                pts = [(p.x(), p.y()) for p in part]
                if tr is not None:
                    from qgis.core import QgsPointXY

                    pts = [
                        (t2.x(), t2.y())
                        for t2 in (tr.transform(QgsPointXY(x, y)) for x, y in pts)
                    ]
                dense = densify_polyline(np.array(pts, dtype=float), spacing)
                n = dense.shape[0]
                if use_z and has_z and len(part) == n:
                    # exact vertices: keep per-vertex Z
                    zv = np.array([self._safe_z(p) for p in part], dtype=float)
                    zval = None if np.isnan(zv).all() else zv
                elif use_z and has_z:
                    zval = None  # densified: interpolate Z along original
                    orig = np.array(pts, dtype=float)
                    orig_z = np.array([self._safe_z(p) for p in part], dtype=float)
                    if np.isnan(orig_z).all():
                        zval = None
                    else:
                        # linear interpolation of Z along cumulative distance
                        cum_o = np.concatenate([[0.0], np.cumsum(
                            np.linalg.norm(np.diff(orig, axis=0), axis=1))])
                        cum_d = np.concatenate([[0.0], np.cumsum(
                            np.linalg.norm(np.diff(dense, axis=0), axis=1))])
                        valid = ~np.isnan(orig_z)
                        zval = np.interp(cum_d, cum_o[valid], orig_z[valid])
                else:
                    zval = None
                if zval is None:
                    fill = z_attr if z_attr is not None and not math.isnan(z_attr) else const
                    zval = np.full(n, fill)
                all_xy.append(dense)
                all_z.append(np.asarray(zval, dtype=float).reshape(-1))
        if not all_xy:
            import numpy as np

            return SampleSet(xy=np.empty((0, 2)), z=np.empty((0,)))
        import numpy as np

        return SampleSet(xy=np.vstack(all_xy), z=np.concatenate(all_z))

    @staticmethod
    def _safe_z(qgs_point) -> float:
        try:
            return float(qgs_point.z())
        except Exception:
            return float("nan")

    @staticmethod
    def _line_parts(geom):
        """Return polyline parts as lists of QgsPoint (Z preserved).

        Note: in QGIS 4 ``asPolyline()`` returns 2D QgsPointXY, so the
        abstract geometry interface is used to keep Z values.
        """
        abstract = geom.constGet()
        if abstract is None:
            return []
        children = []
        num = getattr(abstract, "numGeometries", None)
        if callable(num):
            try:
                n = int(num())
            except Exception:
                n = 0
            if n > 0 and hasattr(abstract, "geometryN"):
                try:
                    children = [abstract.geometryN(i) for i in range(n)]
                except Exception:
                    children = []
        if not children:
            children = [abstract]
        parts = []
        for child in children:
            points_fn = getattr(child, "points", None)
            if not callable(points_fn):
                continue
            try:
                pts = points_fn()
            except Exception:
                continue
            if pts:
                parts.append(list(pts))
        return parts

    def _sample_raster(self, parameters, context, spec, target_crs, feedback):
        import numpy as np

        from ..core.grid import grid_centers

        layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        if layer is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT_RASTER)
            )
        thinning = max(1, self.parameterAsInt(parameters, self.RASTER_THINNING, context))
        gx, gy = grid_centers(spec.origin_x, spec.top_y, spec.pixel, spec.nrows, spec.ncols)
        values, valid = self._read_raster_on_grid(layer, spec, target_crs, feedback)
        return sample_raster_values(gx, gy, values, valid, thinning=thinning)

    def _read_raster_on_grid(self, layer, spec, target_crs, feedback):
        """Read background raster resampled to the target grid via GDAL."""
        import numpy as np

        from osgeo import gdal

        src_path = layer.source().split("|")[0]
        wkt = target_crs.toWkt() if target_crs.isValid() else layer.crs().toWkt()
        feedback.pushInfo(f"Resampling background raster: {src_path}")
        warped = gdal.Translate(
            "/vsimem/fsg_bg.tif",
            src_path,
            format="GTiff",
            outputBounds=[spec.xmin, spec.ymin, spec.xmax, spec.ymax],
            width=spec.ncols,
            height=spec.nrows,
            outputSRS=wkt if wkt else None,
            resampleAlg="bilinear",
        )
        if warped is None:
            raise QgsProcessingException(self.tr("Could not resample background raster."))
        band = warped.GetRasterBand(1)
        values = band.ReadAsArray().astype(float)
        nodata = band.GetNoDataValue()
        warped = None
        gdal.Unlink("/vsimem/fsg_bg.tif")
        if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
            valid = values != nodata
        else:
            valid = np.isfinite(values)
        valid &= np.isfinite(values)
        return values, valid

    # ------------------------------ faults ---------------------------- #
    def _fault_masks(self, parameters, context, target_crs, spec, feedback):
        import numpy as np

        nrows, ncols = spec.nrows, spec.ncols
        barrier = np.zeros((nrows, ncols), dtype=bool)
        gap = np.zeros((nrows, ncols), dtype=bool)

        if parameters.get(self.FAULT_LINES):
            src = self.parameterAsSource(parameters, self.FAULT_LINES, context)
            if src is None:
                raise QgsProcessingException(
                    self.invalidSourceError(parameters, self.FAULT_LINES)
                )
            segs = self._extract_segments(src, target_crs, feedback)
            barrier |= rasterize_line_segments(
                segs, spec.origin_x, spec.top_y, spec.pixel, nrows, ncols
            )
        if parameters.get(self.FAULT_POLYGONS):
            src = self.parameterAsSource(parameters, self.FAULT_POLYGONS, context)
            if src is None:
                raise QgsProcessingException(
                    self.invalidSourceError(parameters, self.FAULT_POLYGONS)
                )
            rings = self._extract_rings(src, target_crs, feedback)
            if rings:
                interior, boundary = rasterize_polygons(
                    rings, spec.origin_x, spec.top_y, spec.pixel, nrows, ncols
                )
                gap |= interior
                barrier |= boundary
        return barrier, gap

    def _extract_segments(self, source, target_crs, feedback):
        tr = self._transform(source.sourceCrs(), target_crs)
        segs = []
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            geom = f.geometry()
            if geom.isEmpty():
                continue
            for part in self._line_parts(geom):
                pts = [(p.x(), p.y()) for p in part]
                if tr is not None:
                    from qgis.core import QgsPointXY

                    pts = [
                        (t2.x(), t2.y())
                        for t2 in (tr.transform(QgsPointXY(x, y)) for x, y in pts)
                    ]
                import numpy as np

                segs.append(np.array(pts, dtype=float))
        return segs

    def _extract_rings(self, source, target_crs, feedback):
        """Extract polygon outer rings as (N,2) arrays (planimetric only)."""
        import numpy as np

        tr = self._transform(source.sourceCrs(), target_crs)
        rings = []
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            geom = f.geometry()
            if geom.isEmpty():
                continue
            abstract = geom.constGet()
            if abstract is None:
                continue
            polygons = []
            num = getattr(abstract, "numGeometries", None)
            if callable(num):
                try:
                    n = int(num())
                except Exception:
                    n = 0
                if n > 0 and hasattr(abstract, "geometryN"):
                    try:
                        polygons = [abstract.geometryN(i) for i in range(n)]
                    except Exception:
                        polygons = []
            if not polygons:
                polygons = [abstract]
            for poly in polygons:
                ring_fn = getattr(poly, "exteriorRing", None)
                if not callable(ring_fn):
                    continue
                try:
                    ring = ring_fn()
                    pts = [(p.x(), p.y()) for p in ring.points()]
                except Exception:
                    continue
                if len(pts) < 3:
                    continue
                if tr is not None:
                    from qgis.core import QgsPointXY

                    pts = [
                        (t2.x(), t2.y())
                        for t2 in (tr.transform(QgsPointXY(x, y)) for x, y in pts)
                    ]
                rings.append(np.array(pts, dtype=float))
        return rings

    @staticmethod
    def _locate_samples(xy, block_ids, spec):
        import numpy as np

        cols = ((xy[:, 0] - spec.origin_x) / spec.pixel).astype(int)
        rows = ((spec.top_y - xy[:, 1]) / spec.pixel).astype(int)
        out = np.full(xy.shape[0], BARRIER, dtype=int)
        ok = (rows >= 0) & (rows < spec.nrows) & (cols >= 0) & (cols < spec.ncols)
        out[ok] = block_ids[rows[ok], cols[ok]]
        return out

    # --------------------------- interpolation ------------------------ #
    def _clip_to_hull(self, result, merged, sample_block, spec, pixel,
                      parameters, context, feedback):
        """Mask grid cells outside the convex hull of usable samples."""
        import numpy as np

        from ..core.grid import convex_hull_mask

        usable = merged.xy[sample_block != BARRIER]
        if usable.shape[0] < 3:
            feedback.pushWarning(
                self.tr("Hull clipping skipped: fewer than 3 usable samples.")
            )
            return result
        buffer_units = self.parameterAsDouble(parameters, self.HULL_BUFFER, context)
        buffer_cells = int(round(max(0.0, buffer_units) / pixel)) if pixel > 0 else 0
        try:
            mask = convex_hull_mask(usable, spec, buffer_cells=buffer_cells)
        except Exception as e:  # ValueError / scipy QhullError on degenerate layouts
            feedback.pushWarning(
                self.tr("Hull clipping skipped: cannot build convex hull ({err}).")
                .format(err=e)
            )
            return result
        clipped = np.asarray(result).copy()
        clipped[~mask] = np.nan
        feedback.pushInfo(
            f"Hull clip: {int(np.sum(~mask))} extrapolation cells masked"
            + (f" (buffer {buffer_cells} cells)" if buffer_cells else "")
        )
        return clipped.astype(np.float32)

    def _warn_if_tps_auto_cap(self, parameters, context, sample_block, feedback):
        """Warn when the TPS global solve would exceed memory limits.

        The per-block fit transparently falls back to a local solve in
        that case; this message makes the approximation explicit.
        """
        import numpy as np

        if self.parameterAsEnum(parameters, self.METHOD, context) != METHOD_TPS:
            return
        if self.parameterAsInt(parameters, self.TPS_NEIGHBORS, context):
            return  # explicit neighbour count: no auto behaviour
        defaults = _tps.TPSInterpolator()
        livestats = sample_block[sample_block != BARRIER]
        if livestats.size == 0:
            return
        biggest = int(np.max(np.bincount(livestats)))
        if biggest > defaults.max_global_points:
            feedback.pushWarning(
                self.tr(
                    "Largest fault block holds {n} samples, above the exact "
                    "global TPS limit ({limit}). A local solve with {k} "
                    "nearest neighbours is used instead: much faster and "
                    "memory-safe, but an approximation. Set 'TPS neighbours' "
                    "explicitly to control it, or increase background raster "
                    "thinning / IDW for very dense data."
                ).format(n=biggest, limit=defaults.max_global_points,
                         k=defaults.auto_neighbors)
            )

    def _make_factory(self, parameters, context):
        method = self.parameterAsEnum(parameters, self.METHOD, context)

        def factory():
            if method == METHOD_TPS:
                neighbors = self.parameterAsInt(parameters, self.TPS_NEIGHBORS, context)
                return get_interpolator(
                    "tps",
                    smoothing=self.parameterAsDouble(
                        parameters, self.TPS_SMOOTHING, context
                    ),
                    neighbors=neighbors if neighbors and neighbors > 0 else None,
                )
            return get_interpolator(
                "idw",
                power=self.parameterAsDouble(parameters, self.IDW_POWER, context),
                radius=self.parameterAsDouble(parameters, self.IDW_RADIUS, context),
                max_neighbors=self.parameterAsInt(
                    parameters, self.IDW_MAX_NEIGHBORS, context
                ),
            )

        return factory

    @staticmethod
    def _write_geotiff(path, array, spec, crs):
        import numpy as np

        from osgeo import gdal

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(path, spec.ncols, spec.nrows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([spec.xmin, spec.pixel, 0.0, spec.top_y, 0.0, -spec.pixel])
        if crs.isValid():
            ds.SetProjection(crs.toWkt())
        band = ds.GetRasterBand(1)
        data = np.asarray(array, dtype=np.float32)
        band.WriteArray(data)
        band.SetNoDataValue(float("nan"))
        band.FlushCache()
        ds = None

    # ------------------------------ metadata -------------------------- #
    def name(self):  # noqa: N802
        return "faultedgrid"

    def displayName(self):  # noqa: N802
        return self.tr("Grid faulted horizon")

    def group(self):  # noqa: N802
        return self.tr("Faulted surfaces")

    def groupId(self):  # noqa: N802
        return "faultedsurfaces"

    def shortHelpString(self):  # noqa: N802
        return self.tr(
            "Interpolates scattered horizon picks (points, polylines, background "
            "raster) onto a regular Float32 grid. Fault polylines act as vertical "
            "barriers; fault polygon interiors become NoData gaps. Each fault block "
            "is interpolated independently with IDW or thin plate spline. "
            "TPS solves exactly up to 3000 samples per block, then switches to "
            "a memory-safe local-neighbour approximation (see log warning). "
            "Near-collinear picks (e.g. along a single section line) "
            "automatically step down the TPS polynomial and, if still "
            "unsolvable, fall back to IDW for the affected block. "
            "Optionally clip the output to the convex hull of the input "
            "samples to suppress extrapolation."
        )

    def createInstance(self):  # noqa: N802
        return FaultedGridAlgorithm()
