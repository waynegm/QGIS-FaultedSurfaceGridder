from dataclasses import dataclass

import numpy as np
from qgis.core import QgsGeometry, QgsPointXY

from .helpers import (
    GridSpec,
    PreparationCanceled,
    _set_progress,
    assign_point_blocks,
    build_fault_blocks,
    build_point_spatial_index,
)


@dataclass
class FaultedGridContext:
    """Shared per-run setup for fault-constrained gridding algorithms.

    Owns the point spatial index (not thread-safe; sequential use only).
    Search radius is locked to one grid step (``gridspec.step``).
    """

    gridspec: GridSpec
    points: np.ndarray  # (N, 3) float array of (x, y, value)
    nodes_x: np.ndarray
    nodes_y: np.ndarray
    block_ids: np.ndarray
    nodata_mask: np.ndarray
    point_blocks: np.ndarray
    point_valid: np.ndarray
    spatial_index: object


def is_canceled(feedback) -> bool:
    try:
        return bool(feedback and feedback.isCanceled())
    except Exception:
        return False


def prepare_faulted_grid(gridspec: GridSpec, points_xyv,
                         fault_line_geoms, fault_poly_geoms,
                         feedback=None, progress_base=0.0,
                         progress_end=100.0) -> FaultedGridContext:
    """Normalize samples, rasterize fault blocks and index points (once per run).

    Progress occupies [progress_base, progress_end]; raises
    PreparationCanceled if feedback is canceled. Split 60/30/10 across
    block rasterization, point assignment and index build.
    """
    span = progress_end - progress_base
    pts = np.asarray(
        list(points_xyv) if not isinstance(points_xyv, np.ndarray) else points_xyv,
        dtype=float,
    ).reshape(-1, 3) if len(points_xyv) else np.empty((0, 3))
    nodes_x, nodes_y = gridspec.nodes()
    _set_progress(feedback, progress_base)
    blocks = build_fault_blocks(
        gridspec, fault_line_geoms, fault_poly_geoms,
        feedback=feedback,
        progress_base=progress_base, progress_span=span * 0.6,
    )
    block_ids = blocks["block_ids"]
    nodata_mask = blocks["nodata_mask"]
    point_blocks, point_valid = assign_point_blocks(
        pts, gridspec, block_ids, fault_line_geoms, fault_poly_geoms,
        feedback=feedback,
        progress_base=progress_base + span * 0.6, progress_span=span * 0.3,
    )
    spatial_index = build_point_spatial_index(
        pts, feedback=feedback,
        progress_base=progress_base + span * 0.9, progress_span=span * 0.1,
    )
    _set_progress(feedback, progress_end)
    return FaultedGridContext(
        gridspec=gridspec,
        points=pts,
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        block_ids=block_ids,
        nodata_mask=nodata_mask,
        point_blocks=point_blocks,
        point_valid=point_valid,
        spatial_index=spatial_index,
    )


def same_block_candidates(ctx: FaultedGridContext, row: int, col: int,
                          node_x: float, node_y: float) -> list:
    """Sample indices within one step sharing the node's fault block.

    Returns [] for NoData cells (fault-polygon interiors, on-trace nodes).
    """
    if ctx.nodata_mask[row, col]:
        return []
    bbox = QgsGeometry.fromPointXY(
        QgsPointXY(node_x, node_y)
    ).boundingBox().buffered(ctx.gridspec.step)
    candidate_ids = ctx.spatial_index.intersects(bbox)
    if not candidate_ids:
        return []
    node_block = int(ctx.block_ids[row, col])
    n = len(ctx.points)
    return [
        pid for pid in candidate_ids
        if 0 <= pid < n and ctx.point_valid[pid]
        and int(ctx.point_blocks[pid]) == node_block
    ]
