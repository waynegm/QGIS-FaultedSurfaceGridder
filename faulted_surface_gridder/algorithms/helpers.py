import os
from dataclasses import dataclass

import numpy as np
from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProject,
    QgsSpatialIndex,
)


@dataclass
class GridSpec:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    step: float

    @property
    def ncols(self) -> int:
        return max(1, int(np.ceil((self.xmax - self.xmin) / self.step)))

    @property
    def nrows(self) -> int:
        return max(1, int(np.ceil((self.ymax - self.ymin) / self.step)))

    @property
    def origin_x(self) -> float:
        return self.xmin

    @property
    def top_y(self) -> float:
        return self.ymax  # row 0 is the top edge

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.origin_x + np.arange(self.ncols) * self.step
        ys = self.top_y - np.arange(self.nrows) * self.step
        return np.meshgrid(xs, ys)


def get_gridspec(algo: QgsProcessingAlgorithm, parameters, context, feedback):
    extent = algo.parameterAsExtent(parameters, algo.GRID_EXTENT, context)
    grid_crs = algo.parameterAsExtentCrs(parameters, algo.GRID_EXTENT, context)
    step = algo.parameterAsDouble(parameters, algo.GRID_SIZE, context)
    if extent.isEmpty() or step <= 0:
        raise QgsProcessingException("Invalid extent or grid step.")
    gridspec = GridSpec(
        extent.xMinimum(), extent.yMinimum(), extent.xMaximum(),
        extent.yMaximum(), step
    )
    feedback.pushInfo(f"Grid: {gridspec.ncols} x {gridspec.nrows} @ {step}")
    return gridspec, grid_crs


def get_transform(source_crs, target_crs):
    if not target_crs.isValid() or source_crs == target_crs:
        return None
    return QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())


def _reproject_request(source, context, target_crs):
    """Build a feature request, reprojecting to target_crs when needed."""
    req = QgsFeatureRequest()
    if (
        target_crs is not None
        and target_crs.isValid()
        and source.sourceCrs() != target_crs
    ):
        req.setDestinationCrs(target_crs, context.transformContext())
    return req


def _feature_point_xy(feature) -> tuple[float, float] | None:
    geom = feature.geometry()
    if geom is None or geom.isEmpty():
        return None
    try:
        pt = geom.asPoint()
        return pt.x(), pt.y()
    except Exception:
        try:
            c = geom.centroid().asPoint()
            return c.x(), c.y()
        except Exception:
            return None


def get_points(algo: QgsProcessingAlgorithm, parameters, context, target_crs, feedback):
    """Materialize (x, y, value) samples in the target CRS.

    Returns a list of (x, y, float) tuples. Replaces the old behaviour of
    returning a feature source *or* a one-shot feature iterator, which broke
    downstream ``fields()``/``getFeatures()`` calls after reprojection.
    """
    points: list[tuple[float, float, float]] = []
    if not parameters.get(algo.INPUT_POINTS):
        return points
    src = algo.parameterAsSource(parameters, algo.INPUT_POINTS, context)
    if src is None:
        raise QgsProcessingException(
            algo.invalidSourceError(parameters, algo.INPUT_POINTS)
        )
    field = algo.parameterAsString(parameters, algo.VALUE_FIELD, context)
    if not field:
        raise QgsProcessingException(
            algo.invalidSourceError(parameters, algo.VALUE_FIELD)
        )
    fidx = src.fields().lookupField(field)
    if fidx < 0:
        raise QgsProcessingException(f"Value field '{field}' not found.")
    req = _reproject_request(src, context, target_crs)
    skipped = 0
    for feat in src.getFeatures(req):
        xy = _feature_point_xy(feat)
        if xy is None:
            skipped += 1
            continue
        try:
            val = feat[fidx]
        except Exception:
            skipped += 1
            continue
        if val is None:
            skipped += 1
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if v != v:  # NaN
            skipped += 1
            continue
        points.append((xy[0], xy[1], v))
    feedback.pushInfo(f"Input points: {len(points)} (skipped {skipped})")
    return points


def _get_geometries(algo, param_key, parameters, context, target_crs, feedback, label):
    if not parameters.get(param_key):
        return []
    src = algo.parameterAsSource(parameters, param_key, context)
    if src is None:
        raise QgsProcessingException(algo.invalidSourceError(parameters, param_key))
    req = _reproject_request(src, context, target_crs)
    geoms = []
    for feat in src.getFeatures(req):
        g = feat.geometry()
        if g is not None and not g.isEmpty():
            geoms.append(QgsGeometry(g))
    feedback.pushInfo(f"Total input {label}: {len(geoms)}")
    return geoms


def get_fault_lines(algo: QgsProcessingAlgorithm, parameters, context, target_crs, feedback):
    """Return fault-trace geometries materialized in the target CRS."""
    return _get_geometries(
        algo, algo.FAULT_LINES, parameters, context, target_crs, feedback, "fault lines"
    )


def get_fault_polygons(algo: QgsProcessingAlgorithm, parameters, context, target_crs, feedback):
    """Return fault-polygon geometries materialized in the target CRS."""
    return _get_geometries(
        algo, algo.FAULT_POLYGONS, parameters, context, target_crs, feedback,
        "fault polygons",
    )


def build_point_spatial_index(points_xyv, feedback=None,
                              progress_base=0.0, progress_span=100.0):
    """Build a fresh point index; id of each feature is its row in points_xyv."""
    _set_progress(feedback, progress_base)
    index = QgsSpatialIndex()
    total = len(points_xyv)
    tick = max(1, total // 50)
    for idx, (x, y, _v) in enumerate(points_xyv):
        if idx % 256 == 0 and _is_canceled(feedback):
            raise PreparationCanceled()
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
        f.setId(idx)
        index.addFeature(f)
        if total and idx % tick == 0:
            _set_progress(feedback, progress_base + progress_span * (idx + 1) / total)
    _set_progress(feedback, progress_base + progress_span)
    return index


def _as_geom_list(obj) -> list:
    """Accept None / list[QgsGeometry] / legacy feature source."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return [g for g in obj if g is not None]
    if hasattr(obj, "getFeatures"):
        out = []
        for f in obj.getFeatures():
            g = f.geometry()
            if g is not None and not g.isEmpty():
                out.append(QgsGeometry(g))
        return out
    return []


def _as_point_xy(p) -> QgsPointXY:
    if isinstance(p, QgsPointXY):
        return p
    return QgsPointXY(float(p[0]), float(p[1]))


class PreparationCanceled(Exception):
    """Raised when feedback.isCanceled() aborts fault-grid preparation."""


def _is_canceled(feedback) -> bool:
    try:
        return bool(feedback is not None and feedback.isCanceled())
    except Exception:
        return False


def _set_progress(feedback, percent: float):
    if feedback is None:
        return
    try:
        p = min(100.0, max(0.0, float(percent)))
        feedback.setProgress(p)
    except Exception:
        pass


def fault_between(fault_line_geoms, fault_poly_geoms, point_a, point_b,
                  line_index=None, poly_index=None) -> bool:
    """Reference oracle: True if segment a-b crosses any fault.

    Accepts lists of QgsGeometry (or legacy feature sources / None).
    Any intersection with a fault line or fault-polygon boundary/interior
    counts as blocked. Polygon interiors are treated as barriers here because
    valid nodes/points are never inside them (interiors are NoData and interior
    points are excluded) — see ``build_fault_blocks``.
    """
    lines = _as_geom_list(fault_line_geoms)
    polys = _as_geom_list(fault_poly_geoms)
    if not lines and not polys:
        return False
    pa = _as_point_xy(point_a)
    pb = _as_point_xy(point_b)
    segment = QgsGeometry.fromPolylineXY([pa, pb])
    seg_bbox = segment.boundingBox()

    if line_index is not None and hasattr(line_index, "intersects"):
        try:
            if not line_index.intersects(seg_bbox) and not (
                poly_index is not None and poly_index.intersects(seg_bbox)
            ):
                return False
        except Exception:
            pass

    engine = QgsGeometry.createGeometryEngine(segment.constGet())
    try:
        engine.prepareGeometry()
    except Exception:
        pass
    for g in lines:
        try:
            if not seg_bbox.intersects(g.boundingBox()):
                continue
            if engine.intersects(g.constGet()):
                return True
        except Exception:
            continue
    for g in polys:
        try:
            if not seg_bbox.intersects(g.boundingBox()):
                continue
            if engine.intersects(g.constGet()):
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Raster block labelling (option 3a)
# ---------------------------------------------------------------------------

def _polygon_parts(geom: QgsGeometry):
    """Yield (outer_x, outer_y, holes) with numpy arrays per polygon part."""
    parts = []
    try:
        if geom.isMultipart():
            polys = geom.asMultiPolygon()
        else:
            polys = [geom.asPolygon()]
    except Exception:
        return parts
    for poly in polys:
        if not poly or not poly[0]:
            continue
        try:
            ox = np.array([p.x() for p in poly[0]], dtype=float)
            oy = np.array([p.y() for p in poly[0]], dtype=float)
        except Exception:
            continue
        holes = []
        for ring in poly[1:]:
            try:
                holes.append((
                    np.array([p.x() for p in ring], dtype=float),
                    np.array([p.y() for p in ring], dtype=float),
                ))
            except Exception:
                continue
        parts.append((ox, oy, holes))
    return parts


def _polyline_parts(geom: QgsGeometry):
    """Yield (lx, ly) numpy arrays per polyline part."""
    parts = []
    try:
        if geom.isMultipart():
            lines = geom.asMultiPolyline()
        else:
            lines = [geom.asPolyline()]
    except Exception:
        return parts
    for line in lines:
        if len(line) < 2:
            continue
        try:
            parts.append((
                np.array([p.x() for p in line], dtype=float),
                np.array([p.y() for p in line], dtype=float),
            ))
        except Exception:
            continue
    return parts


def _points_in_ring(px: np.ndarray, py: np.ndarray, rx: np.ndarray, ry: np.ndarray):
    """Even-odd ray casting over flattened point arrays."""
    inside = np.zeros(px.shape, dtype=bool)
    n = len(rx)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi = rx[i], ry[i]
        xj, yj = rx[j], ry[j]
        if yi != yj:
            cond = ((yi > py) != (yj > py)) & (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            )
            inside ^= cond
        j = i
    return inside


def _window_from_bbox(spec: GridSpec, nrows: int, ncols: int,
                      minx: float, maxx: float, miny: float, maxy: float,
                      pad: float = 0.0):
    c0 = int(np.floor((minx - pad - spec.xmin) / spec.step))
    c1 = int(np.floor((maxx + pad - spec.xmin) / spec.step))
    r0 = int(np.floor((spec.top_y - (maxy + pad)) / spec.step))
    r1 = int(np.floor((spec.top_y - (miny - pad)) / spec.step))
    c0 = max(0, c0)
    r0 = max(0, r0)
    c1 = min(ncols - 1, c1)
    r1 = min(nrows - 1, r1)
    if r1 < r0 or c1 < c0:
        return None
    return r0, r1, c0, c1


def build_fault_blocks(gridspec: GridSpec, fault_line_geoms, fault_poly_geoms,
                       feedback=None, progress_base=0.0, progress_span=100.0):
    """Rasterize faults and flood-fill fault blocks.

    Returns dict with:
      barrier: bool (nrows, ncols) — fault lines (buffered by step/2),
        polygon interiors/boundaries.
      nodata_mask: bool — cells forced to NoData (polygon interiors +
        cells centred within step/2 of a fault trace).
      block_ids: int — connected-component labels of ~barrier using
        4-connectivity (diagonal walls still block). Partial faults that do
        not span the grid leave a path around the tip, so both sides share
        one block id there — by design.

    Progress occupies [progress_base, progress_base + progress_span];
    raises PreparationCanceled if feedback is canceled.
    """
    from scipy.ndimage import label as nd_label

    lines = _as_geom_list(fault_line_geoms)
    polys = _as_geom_list(fault_poly_geoms)
    nrows, ncols = gridspec.nrows, gridspec.ncols
    nodes_x, nodes_y = gridspec.nodes()
    tol = gridspec.step * 0.5

    barrier = np.zeros((nrows, ncols), dtype=bool)
    nodata = np.zeros((nrows, ncols), dtype=bool)

    # Pre-expand so progress can be shared fairly across units of work.
    poly_parts = [(g, _polygon_parts(g)) for g in polys]
    line_segs = []
    for g in lines:
        for lx, ly in _polyline_parts(g):
            for k in range(len(lx) - 1):
                line_segs.append((lx[k], ly[k], lx[k + 1], ly[k + 1]))
    total = sum(len(parts) for _, parts in poly_parts) + len(line_segs) + 1
    done = 0

    def _tick():
        nonlocal done
        done += 1
        _set_progress(feedback, progress_base + progress_span * done / total)

    # Fault polygons: interiors (+boundaries) are barrier and NoData.
    for _g, parts in poly_parts:
        for ox, oy, holes in parts:
            if _is_canceled(feedback):
                raise PreparationCanceled()
            win = _window_from_bbox(
                gridspec, nrows, ncols,
                float(np.min(ox)), float(np.max(ox)),
                float(np.min(oy)), float(np.max(oy)),
            )
            if win is None:
                _tick()
                continue
            r0, r1, c0, c1 = win
            subx = nodes_x[r0:r1 + 1, c0:c1 + 1].ravel()
            suby = nodes_y[r0:r1 + 1, c0:c1 + 1].ravel()
            inside = _points_in_ring(subx, suby, ox, oy)
            for hx, hy in holes:
                inside &= ~_points_in_ring(subx, suby, hx, hy)
            if inside.any():
                sel = inside.reshape((r1 - r0 + 1, c1 - c0 + 1))
                barrier[r0:r1 + 1, c0:c1 + 1][sel] = True
                nodata[r0:r1 + 1, c0:c1 + 1][sel] = True
            _tick()

    # Fault lines: cells within tol of any segment are barrier + NoData.
    for x1, y1, x2, y2 in line_segs:
        if _is_canceled(feedback):
            raise PreparationCanceled()
        win = _window_from_bbox(
            gridspec, nrows, ncols,
            min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2),
            pad=tol,
        )
        if win is None:
            _tick()
            continue
        r0, r1, c0, c1 = win
        subx = nodes_x[r0:r1 + 1, c0:c1 + 1]
        suby = nodes_y[r0:r1 + 1, c0:c1 + 1]
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            dist = np.hypot(subx - x1, suby - y1)
        else:
            t = ((subx - x1) * dx + (suby - y1) * dy) / denom
            t = np.clip(t, 0.0, 1.0)
            dist = np.hypot(subx - (x1 + t * dx), suby - (y1 + t * dy))
        hit = dist <= tol
        if hit.any():
            barrier[r0:r1 + 1, c0:c1 + 1][hit] = True
            nodata[r0:r1 + 1, c0:c1 + 1][hit] = True
        _tick()

    if _is_canceled(feedback):
        raise PreparationCanceled()
    structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int
    )  # 4-connectivity: diagonal barrier cells still separate blocks
    block_ids, _n = nd_label(~barrier, structure=structure)
    _tick()
    return {"barrier": barrier, "nodata_mask": nodata, "block_ids": block_ids}


def assign_point_blocks(points_xyv, gridspec: GridSpec, block_ids,
                        fault_line_geoms, fault_poly_geoms,
                        feedback=None, progress_base=0.0, progress_span=100.0):
    """Assign each sample its fault-block id; exclude unusable samples.

    Excluded (valid=False): points inside a fault polygon, or within
    step/2 of a fault trace (ambiguous side), or falling on a barrier cell.
    Out-of-extent points are clamped to the nearest edge cell — harmless
    because the node search radius (step) means only near-edge points are
    ever queried.

    Progress occupies [progress_base, progress_base + progress_span];
    raises PreparationCanceled if feedback is canceled.
    """
    lines = _as_geom_list(fault_line_geoms)
    polys = _as_geom_list(fault_poly_geoms)
    n = len(points_xyv)
    nrows, ncols = block_ids.shape
    tol = gridspec.step * 0.5

    point_blocks = np.zeros(n, dtype=int)
    point_valid = np.zeros(n, dtype=bool)
    if n == 0:
        _set_progress(feedback, progress_base + progress_span)
        return point_blocks, point_valid

    poly_engines = []
    for g in polys:
        try:
            e = QgsGeometry.createGeometryEngine(g.constGet())
            e.prepareGeometry()
            poly_engines.append(e)
        except Exception:
            continue

    tick = max(1, n // 50)
    for i, (x, y, _v) in enumerate(points_xyv):
        if i % 256 == 0 and _is_canceled(feedback):
            raise PreparationCanceled()
        pt = QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y)))
        try:
            inside_poly = any(e.contains(pt.constGet()) for e in poly_engines)
        except Exception:
            inside_poly = False
        if not inside_poly:
            on_fault = False
            for g in lines:
                try:
                    if g.distance(pt) <= tol:
                        on_fault = True
                        break
                except Exception:
                    continue
            if not on_fault:
                col = int(np.floor((float(x) - gridspec.xmin) / gridspec.step))
                row = int(np.floor((gridspec.top_y - float(y)) / gridspec.step))
                rc = min(max(row, 0), nrows - 1)
                cc = min(max(col, 0), ncols - 1)
                b = int(block_ids[rc, cc])
                if b != 0:
                    point_blocks[i] = b
                    point_valid[i] = True
        if i % tick == 0:
            _set_progress(feedback, progress_base + progress_span * (i + 1) / n)
    _set_progress(feedback, progress_base + progress_span)
    return point_blocks, point_valid


def write_geotiff(path, array, spec, crs):
    from osgeo import gdal

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, spec.ncols, spec.nrows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([spec.xmin, spec.step, 0.0, spec.top_y, 0.0, -spec.step])
    if crs.isValid():
        ds.SetProjection(crs.toWkt())
    band = ds.GetRasterBand(1)
    data = np.asarray(array, dtype=np.float32)
    band.WriteArray(data)
    band.SetNoDataValue(float("nan"))
    band.FlushCache()
    ds = None
