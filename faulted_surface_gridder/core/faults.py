"""Fault barrier rasterization and fault-block labeling.

V1 model:
  * fault polylines  -> 1-cell-thick barriers (vertical faults)
  * fault polygons   -> interiors are NoData gaps (heave), boundaries are barriers
  * connected-component labeling of the remaining cells yields fault blocks
Interpolation is then run independently per block.
"""

import numpy as np

BARRIER = -1  # block id for barrier/gap cells


def rasterize_line_segments(
    segments: list[np.ndarray],
    origin_x: float,
    origin_y: float,
    pixel: float,
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """Rasterize 2D segments (each (K,2)) to a boolean barrier mask."""
    mask = np.zeros((nrows, ncols), dtype=bool)
    for seg in segments:
        seg = np.asarray(seg, dtype=float)
        for a, b in zip(seg[:-1], seg[1:]):
            length = float(np.linalg.norm(b - a))
            steps = max(1, int(np.ceil(length / (pixel * 0.5))))
            for t in np.linspace(0.0, 1.0, steps + 1):
                x, y = a + (b - a) * t
                col = int((x - origin_x) / pixel)
                row = int((origin_y - y) / pixel)  # origin_y = top edge
                if 0 <= row < nrows and 0 <= col < ncols:
                    mask[row, col] = True
    return mask


def rasterize_polygons(
    polygons: list[np.ndarray],
    origin_x: float,
    origin_y: float,
    pixel: float,
    nrows: int,
    ncols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize polygon rings to (interior_mask, boundary_mask).

    Each polygon is an (K,2) ring. Ray-casting is vectorized per polygon
    over grid cell centers; suitable for modest grids. Very large grids
    with many polygons should pre-simplify in QGIS.
    """
    from .grid import grid_centers  # local import to avoid cycle

    gx, gy = grid_centers(origin_x, origin_y, pixel, nrows, ncols)
    interior = np.zeros((nrows, ncols), dtype=bool)
    boundary = np.zeros((nrows, ncols), dtype=bool)
    for ring in polygons:
        ring = np.asarray(ring, dtype=float)
        if ring.shape[0] < 3:
            continue
        # boundary via segment rasterization
        boundary |= rasterize_line_segments(
            [np.vstack([ring, ring[:1]])], origin_x, origin_y, pixel, nrows, ncols
        )
        # interior via ray casting on flattened centers
        px, py = gx.reshape(-1), gy.reshape(-1)
        inside = _points_in_ring(px, py, ring[:, 0], ring[:, 1])
        interior |= inside.reshape(nrows, ncols)
    interior &= ~boundary  # boundary cells stay barriers, not gap
    return interior, boundary


def _points_in_ring(px, py, rx, ry) -> np.ndarray:
    x0, x1 = rx[:-1], rx[1:]
    y0, y1 = ry[:-1], ry[1:]
    cond = ((y0 > py[:, None]) != (y1 > py[:, None])) & (
        px[:, None]
        < (x1 - x0) * (py[:, None] - y0) / (y1 - y0 + 1e-300) + x0
    )
    return (np.sum(cond, axis=1) % 2) == 1


def label_blocks(
    barrier_mask: np.ndarray, gap_mask: np.ndarray | None = None
) -> np.ndarray:
    """Label 4-connected fault blocks. Barriers/gaps get id ``BARRIER``."""
    try:
        from scipy.ndimage import label
    except ImportError as e:
        raise ImportError(
            "Fault-block labeling requires scipy, which is not installed "
            "in the QGIS Python environment."
        ) from e
    blocked = np.asarray(barrier_mask, dtype=bool)
    if gap_mask is not None:
        blocked = blocked | np.asarray(gap_mask, dtype=bool)
    open_cells = ~blocked
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connectivity
    labeled, _ = label(open_cells, structure=structure)
    return np.where(blocked, BARRIER, labeled - 1)  # block ids 0..B-1
