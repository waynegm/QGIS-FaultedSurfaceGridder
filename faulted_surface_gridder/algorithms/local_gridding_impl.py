import numpy as np

from .gridding_context import (
    is_canceled,
    prepare_faulted_grid,
    same_block_candidates,
)
from .helpers import PreparationCanceled

# Setup (fault rasterization + point assignment + index) owns the first
# slice of the progress bar; the interpolation loop owns the rest.
SETUP_END = 20.0


def _set_progress(feedback, percent: float):
    if feedback is None:
        return
    try:
        feedback.setProgress(min(100.0, max(0.0, float(percent))))
    except Exception:
        pass


def local_gridding_impl(gridspec, points_xyv, fault_line_geoms, fault_poly_geoms,
                        feedback=None):
    """IDW interpolation constrained to fault blocks.

    points_xyv: sequence of (x, y, value) in grid CRS (list or (N,3) array).
    fault_*: lists of QgsGeometry (or None/[]).
    Cells inside fault polygons or within step/2 of a fault trace stay NaN.
    Only samples sharing the node's fault-block id contribute.
    Search radius is unchanged: one grid step.
    Returns an all-NaN grid if preparation is canceled.
    """
    if feedback is not None:
        feedback.pushInfo('Local interpolation')
    try:
        ctx = prepare_faulted_grid(
            gridspec, points_xyv, fault_line_geoms, fault_poly_geoms,
            feedback=feedback, progress_base=0.0, progress_end=SETUP_END,
        )
    except PreparationCanceled:
        return np.full((gridspec.nrows, gridspec.ncols), np.nan)
    results = np.full(ctx.nodes_x.shape, dtype=float, fill_value=np.nan)
    if len(ctx.points) == 0:
        _set_progress(feedback, 100.0)
        return results
    nrows = ctx.nodes_x.shape[0]

    for i in range(ctx.nodes_x.shape[0]):
        if is_canceled(feedback):
            break
        for j in range(ctx.nodes_x.shape[1]):
            node_x = ctx.nodes_x[i, j]
            node_y = ctx.nodes_y[i, j]
            same = same_block_candidates(ctx, i, j, node_x, node_y)
            if not same:
                continue
            data = ctx.points[same]
            local_coords = data[:, :2]
            local_V = data[:, 2]
            dx = local_coords[:, 0] - node_x
            dy = local_coords[:, 1] - node_y
            distances = np.sqrt(dx**2 + dy**2)
            weights = np.ones_like(distances)
            nz = distances != 0
            weights[nz] = np.tanh(distances[nz]) / distances[nz]
            results[i, j] = np.sum(local_V * weights) / np.sum(weights)
        _set_progress(feedback, SETUP_END + (100.0 - SETUP_END) * (i + 1) / nrows)
    return results
