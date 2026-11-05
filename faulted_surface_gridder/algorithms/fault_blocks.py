"""Geometry-based fault-block construction for local gridding.

The grid nodes are the vertices of a regular lattice.  Fault traces block
movement across the grid edges they cross, and nodes lying on a trace are
removed from the interpolated grid.  The remaining nodes are labelled into
connected fault blocks; a sample is only allowed to contribute to nodes in its
own block.

This module deliberately works from the original QGIS line layer rather than
from rasterised ``fault_left``/``fault_up`` masks.  Rasterising a line with
``ALL_TOUCHED`` marks both the left and top edges of every touched cell, which
over-blocks and can shift a trace that does not pass exactly through a node.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from qgis.core import QgsVectorLayer

from .helpers import GridSpec


@dataclass
class FaultBlocks:
    """Node-level fault connectivity for one grid."""

    block_ids: np.ndarray  # (rows, cols) int32, -1 for NoData nodes
    barrier: np.ndarray  # (rows, cols) bool, NoData nodes
    blocked_h: np.ndarray  # (rows, cols-1) bool, edge (r,c)-(r,c+1)
    blocked_v: np.ndarray  # (rows-1, cols) bool, edge (r,c)-(r+1,c)


def _is_canceled(feedback) -> bool:
    try:
        return bool(feedback is not None and feedback.isCanceled())
    except Exception:
        return False


def _iter_line_segments(layer: QgsVectorLayer):
    """Yield ``(x1, y1, x2, y2)`` for every straight segment in a line layer."""
    if layer is None:
        return
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            if geom.isMultipart():
                parts = geom.asMultiPolyline()
            else:
                parts = [geom.asPolyline()]
        except Exception:
            continue
        for part in parts:
            if part is None or len(part) < 2:
                continue
            for i in range(len(part) - 1):
                p1 = part[i]
                p2 = part[i + 1]
                yield (
                    float(p1.x()),
                    float(p1.y()),
                    float(p2.x()),
                    float(p2.y()),
                )


def _point_segment_distance(px, py, x1, y1, x2, y2):
    """Vectorised distance from points to one line segment."""
    dx = x2 - x1
    dy = y2 - y1
    den = dx * dx + dy * dy
    if den == 0.0:
        return np.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / den
    np.clip(t, 0.0, 1.0, out=t)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _mark_fault_nodes(
    barrier: np.ndarray,
    gridspec: GridSpec,
    segments,
    feedback=None,
) -> int:
    """Mark nodes lying on a fault trace.  Returns the number of segments."""
    rows = gridspec.nrows
    cols = gridspec.ncols
    step = gridspec.step
    tol = max(step * 1e-6, 1e-9)
    node_x = gridspec.xmin + np.arange(cols) * step
    node_y = gridspec.top_y - np.arange(rows) * step

    count = 0
    for x1, y1, x2, y2 in segments:
        count += 1
        if count % 256 == 0 and _is_canceled(feedback):
            break

        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        c0 = max(0, int(np.floor((min_x - tol - gridspec.xmin) / step)))
        c1 = min(cols - 1, int(np.ceil((max_x + tol - gridspec.xmin) / step)))
        r0 = max(0, int(np.floor((gridspec.top_y - (max_y + tol)) / step)))
        r1 = min(rows - 1, int(np.ceil((gridspec.top_y - (min_y - tol)) / step)))
        if c1 < c0 or r1 < r0:
            continue

        sx, sy = np.meshgrid(
            node_x[c0 : c1 + 1], node_y[r0 : r1 + 1]
        )
        dist = _point_segment_distance(sx, sy, x1, y1, x2, y2)
        hit = dist <= tol
        if hit.any():
            barrier[r0 : r1 + 1, c0 : c1 + 1] |= hit

    return count


def _mark_edge_crossings(
    blocked_h: np.ndarray,
    blocked_v: np.ndarray,
    gridspec: GridSpec,
    segments,
    feedback=None,
) -> None:
    """Block grid edges crossed by fault segments in their interiors."""
    rows = gridspec.nrows
    cols = gridspec.ncols
    step = gridspec.step
    xmin = gridspec.xmin
    xmax = gridspec.right_x
    ymin = gridspec.ymin
    ymax = gridspec.top_y
    eps = 1e-12

    for seg_index, (x1, y1, x2, y2) in enumerate(segments):
        if seg_index % 256 == 0 and _is_canceled(feedback):
            break

        dx = x2 - x1
        dy = y2 - y1

        # Crossings of vertical grid lines x = xmin + c*step.  Each crossing
        # blocks the vertical edge between two rows.
        if abs(dx) > eps:
            c0 = max(0, int(np.ceil((min(x1, x2) - xmin) / step - eps)))
            c1 = min(cols - 1, int(np.floor((max(x1, x2) - xmin) / step + eps)))
            for c in range(c0, c1 + 1):
                x = xmin + c * step
                t = (x - x1) / dx
                if t < -eps or t > 1.0 + eps:
                    continue
                y = y1 + t * dy
                if y <= ymin + eps or y >= ymax - eps:
                    continue
                r_float = (ymax - y) / step
                r = int(np.floor(r_float))
                frac = r_float - r
                if r < 0 or r >= rows - 1:
                    continue
                # A crossing exactly at a node is handled by the node barrier.
                if frac < 1e-9 or frac > 1.0 - 1e-9:
                    continue
                blocked_v[r, c] = True

        # Crossings of horizontal grid lines y = ymax - r*step.  Each crossing
        # blocks the horizontal edge between two columns.
        if abs(dy) > eps:
            r0 = max(0, int(np.ceil((ymax - max(y1, y2)) / step - eps)))
            r1 = min(rows - 1, int(np.floor((ymax - min(y1, y2)) / step + eps)))
            for r in range(r0, r1 + 1):
                y = ymax - r * step
                t = (y - y1) / dy
                if t < -eps or t > 1.0 + eps:
                    continue
                x = x1 + t * dx
                if x <= xmin + eps or x >= xmax - eps:
                    continue
                c_float = (x - xmin) / step
                c = int(np.floor(c_float))
                frac = c_float - c
                if c < 0 or c >= cols - 1:
                    continue
                if frac < 1e-9 or frac > 1.0 - 1e-9:
                    continue
                blocked_h[r, c] = True


def _label_blocks(
    barrier: np.ndarray,
    blocked_h: np.ndarray,
    blocked_v: np.ndarray,
    feedback=None,
) -> np.ndarray:
    """Flood-fill 4-connected fault blocks over non-barrier nodes."""
    rows, cols = barrier.shape
    block_ids = np.full((rows, cols), -1, dtype=np.int32)

    if not barrier.any() and not blocked_h.any() and not blocked_v.any():
        block_ids[:] = 0
        return block_ids

    next_id = 0
    for start_r in range(rows):
        if start_r % 16 == 0 and _is_canceled(feedback):
            break
        for start_c in range(cols):
            if barrier[start_r, start_c] or block_ids[start_r, start_c] != -1:
                continue

            block_ids[start_r, start_c] = next_id
            queue = deque([(start_r, start_c)])
            while queue:
                r, c = queue.popleft()

                if (
                    c > 0
                    and not barrier[r, c - 1]
                    and not blocked_h[r, c - 1]
                    and block_ids[r, c - 1] == -1
                ):
                    block_ids[r, c - 1] = next_id
                    queue.append((r, c - 1))

                if (
                    c + 1 < cols
                    and not barrier[r, c + 1]
                    and not blocked_h[r, c]
                    and block_ids[r, c + 1] == -1
                ):
                    block_ids[r, c + 1] = next_id
                    queue.append((r, c + 1))

                if (
                    r > 0
                    and not barrier[r - 1, c]
                    and not blocked_v[r - 1, c]
                    and block_ids[r - 1, c] == -1
                ):
                    block_ids[r - 1, c] = next_id
                    queue.append((r - 1, c))

                if (
                    r + 1 < rows
                    and not barrier[r + 1, c]
                    and not blocked_v[r, c]
                    and block_ids[r + 1, c] == -1
                ):
                    block_ids[r + 1, c] = next_id
                    queue.append((r + 1, c))

            next_id += 1

    return block_ids


def build_fault_blocks(
    gridspec: GridSpec,
    fault_lines_layer: QgsVectorLayer | None,
    heave_mask: np.ndarray | None = None,
    feedback=None,
) -> FaultBlocks:
    """Build node barriers and fault-block labels from QGIS geometries.

    ``heave_mask`` is the rasterised polygon-interior cell mask.  Its cells are
    expanded to their corner nodes so the whole polygon interior, including its
    boundary nodes, is NoData.
    """
    rows = gridspec.nrows
    cols = gridspec.ncols

    barrier = np.zeros((rows, cols), dtype=bool)
    if heave_mask is not None:
        heave = np.asarray(heave_mask, dtype=bool)
        if heave.shape != (rows, cols):
            raise ValueError(
                f"heave_mask shape {heave.shape} does not match grid "
                f"({rows}, {cols})"
            )
        barrier |= heave
        if cols > 1:
            barrier[:, 1:] |= heave[:, :-1]
            if rows > 1:
                barrier[1:, 1:] |= heave[:-1, :-1]
        if rows > 1:
            barrier[1:, :] |= heave[:-1, :]

    blocked_h = np.zeros((rows, max(cols - 1, 0)), dtype=bool)
    blocked_v = np.zeros((max(rows - 1, 0), cols), dtype=bool)

    if fault_lines_layer is not None:
        # Materialise the segment list once; it is used by both passes.
        segments = list(_iter_line_segments(fault_lines_layer))
        if feedback is not None:
            feedback.pushInfo(f"Fault segments: {len(segments)}")

        _mark_fault_nodes(barrier, gridspec, segments, feedback=feedback)
        _mark_edge_crossings(
            blocked_h, blocked_v, gridspec, segments, feedback=feedback
        )

    block_ids = _label_blocks(barrier, blocked_h, blocked_v, feedback=feedback)
    return FaultBlocks(
        block_ids=block_ids,
        barrier=barrier,
        blocked_h=blocked_h,
        blocked_v=blocked_v,
    )


def assign_point_blocks(
    xs,
    ys,
    fids,
    gridspec: GridSpec,
    blocks: FaultBlocks,
    heave_mask: np.ndarray | None = None,
    feedback=None,
) -> np.ndarray:
    """Assign each sample to a fault block (``-1`` = unusable sample).

    A sample gets the block of its nearest non-barrier node.  If several
    equidistant valid nodes belong to different blocks the sample is ambiguous
    (it lies on a fault) and is rejected.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    fids = np.asarray(fids)
    n = len(fids)
    point_blocks = np.full(n, -1, dtype=np.int32)

    if n == 0:
        return point_blocks

    # Fast path when there are no fault or heave barriers at all.
    if (
        not blocks.barrier.any()
        and not blocks.blocked_h.any()
        and not blocks.blocked_v.any()
    ):
        point_blocks[:] = 0
        return point_blocks

    rows = gridspec.nrows
    cols = gridspec.ncols
    step = gridspec.step
    xmin = gridspec.xmin
    top_y = gridspec.top_y
    node_x = xmin + np.arange(cols) * step
    node_y = top_y - np.arange(rows) * step
    tol_d2 = max((step * 1e-6) ** 2, 1e-12)

    heave = None
    if heave_mask is not None:
        heave = np.asarray(heave_mask, dtype=bool)

    for i in range(n):
        if i % 1024 == 0 and _is_canceled(feedback):
            break

        x = xs[i]
        y = ys[i]
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if heave is not None:
            cc = int(np.floor((x - xmin) / step))
            rr = int(np.floor((top_y - y) / step))
            if 0 <= rr < rows and 0 <= cc < cols and heave[rr, cc]:
                continue

        c_center = int(round((x - xmin) / step))
        r_center = int(round((top_y - y) / step))
        if (
            c_center < -2
            or c_center > cols + 1
            or r_center < -2
            or r_center > rows + 1
        ):
            continue

        r0 = max(0, r_center - 2)
        r1 = min(rows - 1, r_center + 2)
        c0 = max(0, c_center - 2)
        c1 = min(cols - 1, c_center + 2)

        best_d2 = np.inf
        best_blocks = set()
        for r in range(r0, r1 + 1):
            dy = node_y[r] - y
            for c in range(c0, c1 + 1):
                block = int(blocks.block_ids[r, c])
                if block < 0:
                    continue
                dx = node_x[c] - x
                d2 = dx * dx + dy * dy
                if d2 < best_d2 - tol_d2:
                    best_d2 = d2
                    best_blocks = {block}
                elif abs(d2 - best_d2) <= tol_d2:
                    best_blocks.add(block)

        if len(best_blocks) == 1:
            point_blocks[i] = next(iter(best_blocks))

    return point_blocks
