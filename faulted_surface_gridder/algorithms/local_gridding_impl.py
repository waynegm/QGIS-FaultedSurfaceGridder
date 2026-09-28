import numpy as np
from qgis.core import QgsGeometry, QgsPointXY

from .helpers import (
    assign_point_blocks,
    build_fault_blocks,
    build_point_spatial_index,
)


def _cancelled(feedback) -> bool:
    try:
        return bool(feedback and feedback.isCanceled())
    except Exception:
        return False


def local_gridding_impl(gridspec, points_xyv, fault_line_geoms, fault_poly_geoms,
                        feedback=None):
    """IDW interpolation constrained to fault blocks.

    points_xyv: sequence of (x, y, value) in grid CRS (list or (N,3) array).
    fault_*: lists of QgsGeometry (or None/[]).
    Cells inside fault polygons or within step/2 of a fault trace stay NaN.
    Only samples sharing the node's fault-block id contribute.
    Search radius is unchanged: one grid step.
    """
    pts = np.asarray(list(points_xyv) if not isinstance(points_xyv, np.ndarray)
                     else points_xyv, dtype=float).reshape(-1, 3) \
        if len(points_xyv) else np.empty((0, 3))
    nodes_x, nodes_y = gridspec.nodes()
    results = np.full(nodes_x.shape, dtype=float, fill_value=np.nan)

    blocks = build_fault_blocks(gridspec, fault_line_geoms, fault_poly_geoms)
    block_ids = blocks["block_ids"]
    nodata_mask = blocks["nodata_mask"]

    if pts.shape[0] == 0:
        return results

    point_blocks, point_valid = assign_point_blocks(
        pts, gridspec, block_ids, fault_line_geoms, fault_poly_geoms
    )
    spatial_index = build_point_spatial_index(pts)
    nrows = nodes_x.shape[0]

    for i in range(nodes_x.shape[0]):
        if _cancelled(feedback):
            break
        for j in range(nodes_x.shape[1]):
            if nodata_mask[i, j]:
                continue
            node_x = nodes_x[i, j]
            node_y = nodes_y[i, j]
            bbox = QgsGeometry.fromPointXY(
                QgsPointXY(node_x, node_y)).boundingBox().buffered(gridspec.step)
            candidate_ids = spatial_index.intersects(bbox)
            if not candidate_ids:
                continue
            node_block = int(block_ids[i, j])
            same = [pid for pid in candidate_ids
                    if 0 <= pid < len(pts) and point_valid[pid]
                    and point_blocks[pid] == node_block]
            if not same:
                continue
            data = pts[same]
            local_coords = data[:, :2]
            local_V = data[:, 2]
            dx = local_coords[:, 0] - node_x
            dy = local_coords[:, 1] - node_y
            distances = np.sqrt(dx**2 + dy**2)
            weights = np.ones_like(distances)
            nz = distances != 0
            weights[nz] = np.tanh(distances[nz]) / distances[nz]
            results[i, j] = np.sum(local_V * weights) / np.sum(weights)
        if feedback is not None and nrows:
            try:
                feedback.setProgress(int(100 * (i + 1) / nrows))
            except Exception:
                pass
    return results
