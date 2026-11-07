from dataclasses import dataclass

import numpy as np
from qgis import processing
from qgis.core import (
    QgsGeometry,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsVectorLayer,
    QgsWkbTypes,
)


@dataclass
class GridSpec:
    """Node-centred regular grid.

    Node ``(row, col)`` is at ``xmin + col*step`` and
    ``top_y - row*step``; ``top_y`` is derived from ``nrows``, so the lattice
    spans the requested extent up to the last node.
    """

    xmin: float
    ymin: float
    ncols: int
    nrows: int
    step: float

    @property
    def origin_x(self) -> float:
        return self.xmin

    @property
    def top_y(self) -> float:
        return (self.nrows-1)*self.step+self.ymin

    @property
    def right_x(self) -> float:
        return (self.ncols-1)*self.step+self.xmin

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.origin_x + np.arange(self.ncols) * self.step
        ys = self.top_y - np.arange(self.nrows) * self.step
        return np.meshgrid(xs, ys)

def get_gridspec(algo: QgsProcessingAlgorithm, parameters, context, feedback):
    extent = algo.parameterAsExtent(parameters, algo.GRID_EXTENT, context)
    grid_crs = algo.parameterAsExtentCrs(parameters, algo.GRID_EXTENT, context)
    step = algo.parameterAsDouble(parameters, algo.GRID_STEP, context)
    if extent.isEmpty() or step <= 0:
        raise QgsProcessingException("Invalid extent or grid step.")
    # Grid values live at nodes: the first node is at the extent's lower-left
    # corner and nodes continue at `step` intervals until the far edge is
    # enclosed.  The ``- 1e-9`` tolerance avoids an extra node from floating
    # point error when the extent is an exact multiple of the step.
    width = extent.xMaximum() - extent.xMinimum()
    height = extent.yMaximum() - extent.yMinimum()
    ncols = max(2, int(np.ceil(width / step - 1e-9)) + 1)
    nrows = max(2, int(np.ceil(height / step - 1e-9)) + 1)
    gridspec = GridSpec(extent.xMinimum(), extent.yMinimum(), ncols, nrows, step)
    feedback.pushInfo(
        f"Grid: {gridspec.ncols} x {gridspec.nrows} nodes @ {step}"
    )
    return gridspec, grid_crs

def _get_vector_layer(algo, param_key, parameters, context, target_crs, feedback) -> QgsVectorLayer|None:
    if not parameters.get(param_key):
        return None
    src = algo.parameterAsVectorLayer(parameters, param_key, context)
    if src is None:
        raise QgsProcessingException(algo.invalidSourceError(parameters, param_key))
    if src.crs() != target_crs:
        reproject_params = {
            'INPUT': src,
            'TARGET_CRS': target_crs,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        result = processing.run("native:reprojectlayer", reproject_params, context=context, feedback=feedback)
        return context.getMapLayer(result['OUTPUT'])
    else:
        return src

def get_fault_lines_layer(algo: QgsProcessingAlgorithm, parameters, context, target_crs, feedback) -> QgsVectorLayer|None:
    """Return fault-line layer materialized in the target CRS."""
    layer = _get_vector_layer(algo, algo.FAULT_LINES, parameters, context, target_crs, feedback)
    if layer is not None:
        feedback.pushInfo(f'Loaded {layer.featureCount()} fault lines')
    return layer

def get_fault_polygons_layer(algo: QgsProcessingAlgorithm, parameters, context, target_crs, feedback) -> QgsVectorLayer|None:
    """Return fault-polygon layer materialized in the target CRS."""
    layer = _get_vector_layer(algo, algo.FAULT_POLYGONS, parameters, context, target_crs, feedback)
    if layer is not None:
        feedback.pushInfo(f'Loaded {layer.featureCount()} fault polygons')
    return layer

def get_points_layer(algo: QgsProcessingAlgorithm, parameters, context, target_crs, feedback) -> QgsVectorLayer|None:
    """Return points layer materialized in the target CRS."""
    layer = _get_vector_layer(algo, algo.INPUT_POINTS, parameters, context, target_crs, feedback)
    if layer is not None:
        feedback.pushInfo(f'Loaded {layer.featureCount()} input points')
    return layer


def convex_hull_nodata_mask(xs, ys, gridspec: GridSpec, feedback=None) -> np.ndarray:
    """Return a boolean mask of grid nodes outside the samples' convex hull.

    The hull is built with QGIS' ``QgsGeometry.convexHull()`` and then tested
    against the exact grid-node coordinates with a vectorised half-plane test.
    Boundary nodes are included in the hull.

    Degenerate hulls (fewer than three non-collinear points) cannot enclose an
    area; in that case no clipping is applied and an all-False mask is returned.
    """
    outside = np.zeros((gridspec.nrows, gridspec.ncols), dtype=bool)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        if feedback is not None:
            feedback.pushWarning(
                "Convex hull clipping skipped: fewer than 3 input points"
            )
        return outside

    geom = QgsGeometry.fromMultiPointXY(
        [QgsPointXY(float(x), float(y)) for x, y in zip(xs, ys)]
    )
    hull = geom.convexHull()
    if hull.isEmpty() or hull.type() != QgsWkbTypes.PolygonGeometry:
        if feedback is not None:
            feedback.pushWarning(
                "Convex hull clipping skipped: could not build a polygon hull"
            )
        return outside

    rings = hull.asPolygon()
    if not rings or len(rings[0]) < 4:
        return outside

    ring = rings[0]
    # Drop the repeated closing vertex if present.
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        return outside

    vx = np.asarray([p.x() for p in ring], dtype=float)
    vy = np.asarray([p.y() for p in ring], dtype=float)

    area2 = float(np.sum(vx * np.roll(vy, -1) - np.roll(vx, -1) * vy))
    if abs(area2) < 1e-12:
        if feedback is not None:
            feedback.pushWarning(
                "Convex hull clipping skipped: hull has zero area"
            )
        return outside
    orientation = 1.0 if area2 > 0 else -1.0

    node_x, node_y = gridspec.nodes()
    extent = max(
        float(np.ptp(vx)), float(np.ptp(vy)), gridspec.step, 1.0
    )
    tol = 1e-9 * extent * extent

    inside = np.ones(node_x.shape, dtype=bool)
    for i in range(len(vx)):
        x1, y1 = vx[i], vy[i]
        x2, y2 = vx[(i + 1) % len(vx)], vy[(i + 1) % len(vy)]
        cross = (x2 - x1) * (node_y - y1) - (y2 - y1) * (node_x - x1)
        inside &= orientation * cross >= -tol

    outside[:] = ~inside
    return outside
