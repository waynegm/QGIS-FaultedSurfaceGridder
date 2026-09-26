"""Target grid definition and block-wise interpolation driver."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .base import BaseInterpolator, ProgressCallback
from .faults import BARRIER

WarnCallback = Callable[[str], None] | None


@dataclass
class GridSpec:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    pixel: float

    @property
    def ncols(self) -> int:
        return max(1, int(np.ceil((self.xmax - self.xmin) / self.pixel)))

    @property
    def nrows(self) -> int:
        return max(1, int(np.ceil((self.ymax - self.ymin) / self.pixel)))

    @property
    def origin_x(self) -> float:
        return self.xmin

    @property
    def top_y(self) -> float:
        return self.ymax  # row 0 is the top edge


def grid_centers(
    origin_x: float, top_y: float, pixel: float, nrows: int, ncols: int
) -> tuple[np.ndarray, np.ndarray]:
    xs = origin_x + (np.arange(ncols) + 0.5) * pixel
    ys = top_y - (np.arange(nrows) + 0.5) * pixel
    return np.meshgrid(xs, ys)


def convex_hull_mask(
    samples_xy: np.ndarray, spec: GridSpec, buffer_cells: int = 0
) -> np.ndarray:
    """Boolean mask of grid cells inside the convex hull of sample points.

    Cells outside the hull are extrapolation; masking them keeps the grid
    honest about data support. ``buffer_cells`` dilates the mask outward
    by that many cells (e.g. to keep a rim around edge samples).

    Raises ``ValueError`` for fewer than 3 distinct points and propagates
    scipy's ``QhullError`` for degenerate (e.g. collinear) layouts —
    callers should catch these and leave the grid unclipped.
    """
    from scipy.ndimage import binary_dilation
    from scipy.spatial import ConvexHull, Delaunay

    pts = np.asarray(samples_xy, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = np.unique(np.round(pts, decimals=9), axis=0)
    if pts.shape[0] < 3:
        raise ValueError("At least 3 distinct samples are needed for a convex hull")
    hull = ConvexHull(pts)  # raises QhullError if degenerate
    tri = Delaunay(pts[hull.vertices])
    gx, gy = grid_centers(spec.origin_x, spec.top_y, spec.pixel, spec.nrows, spec.ncols)
    mask = (tri.find_simplex(np.column_stack([gx.ravel(), gy.ravel()])) >= 0)
    mask = mask.reshape(spec.nrows, spec.ncols)
    if buffer_cells and int(buffer_cells) > 0:
        mask = binary_dilation(mask, iterations=int(buffer_cells))
    return mask


def interpolate_blocks(
    make_interpolator,
    samples_xy: np.ndarray,
    samples_z: np.ndarray,
    block_ids: np.ndarray,
    sample_block: np.ndarray,
    spec: GridSpec,
    min_points: int = 1,
    progress: ProgressCallback = None,
    warn: WarnCallback = None,
    fallback_factory=None,
) -> np.ndarray:
    """Interpolate each fault block independently.

    :param make_interpolator: zero-arg factory returning a fresh
        ``BaseInterpolator`` (fresh instance per block is required).
    :param block_ids: (nrows, ncols) array with BARRIER for gaps.
    :param sample_block: block id per sample (-1 if on barrier/gap,
        in which case the sample is ignored).
    :param warn: optional callable receiving warning strings.
    :param fallback_factory: optional zero-arg factory for a robust
        interpolator (e.g. IDW) used for a block whose primary method
        fails numerically. Without it, the first such error propagates.
    :returns: (nrows, ncols) float32 array with NaN for barriers/gaps
        and under-constrained blocks.
    """
    gx, gy = grid_centers(spec.origin_x, spec.top_y, spec.pixel, spec.nrows, spec.ncols)
    out = np.full((spec.nrows, spec.ncols), np.nan, dtype=float)
    blocks = sorted(b for b in np.unique(block_ids).tolist() if b != BARRIER)
    total = max(1, len(blocks))
    for i, b in enumerate(blocks):
        rows, cols = np.nonzero(block_ids == b)
        if rows.size == 0:
            continue
        m = sample_block == b
        n_samples = int(np.sum(m))
        if n_samples < min_points:
            continue
        targets = np.column_stack([gx[rows, cols], gy[rows, cols]])
        frac_base = i / total

        def _cb(frac, _base=frac_base, _i=i, _total=total):
            if progress is None:
                return True
            return progress((_base + frac / _total))

        try:
            interp: BaseInterpolator = make_interpolator()
            interp.fit(samples_xy[m], samples_z[m])
            out[rows, cols] = interp.predict(targets, progress=_cb)
        except np.linalg.LinAlgError as e:
            if fallback_factory is None:
                raise
            if warn is not None:
                warn(
                    f"Fault block {b} ({n_samples} samples): primary method "
                    f"failed numerically ({e}); used robust fallback for "
                    "this block."
                )
            fallback: BaseInterpolator = fallback_factory()
            fallback.fit(samples_xy[m], samples_z[m])
            out[rows, cols] = fallback.predict(targets, progress=_cb)
        if progress is not None and not progress((i + 1) / total):
            break
    return out.astype(np.float32)
