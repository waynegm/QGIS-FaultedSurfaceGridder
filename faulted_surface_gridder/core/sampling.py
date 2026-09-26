"""Merge horizon sample sources into a single (xy, z) array.

Pure-numpy helpers so unit tests don't need QGIS. The QGIS wrapper
(processing/grid_algorithm.py) extracts raw coordinates/values from
layers and calls :func:`merge_samples`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleSet:
    xy: np.ndarray  # (N, 2) float
    z: np.ndarray  # (N,) float


def densify_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Densify a 2D polyline so no segment exceeds ``spacing``.

    ``points``: (M, 2) vertices. Returns densified (K, 2) coordinates.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2 or spacing <= 0:
        return points
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        seg_len = float(np.linalg.norm(b - a))
        if seg_len == 0 or not np.isfinite(seg_len):
            continue
        n = max(1, int(np.ceil(seg_len / spacing)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out)


def sample_raster_values(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
    thinning: int = 1,
) -> SampleSet:
    """Convert valid raster cells to point samples.

    ``grid_x``/``grid_y``/``values``/``valid_mask`` are 2D arrays of the
    background raster already resampled to the target grid. ``thinning``
    keeps every Nth row/col to bound sample counts.
    """
    thinning = max(1, int(thinning))
    xs = grid_x[::thinning, ::thinning].reshape(-1)
    ys = grid_y[::thinning, ::thinning].reshape(-1)
    vs = values[::thinning, ::thinning].reshape(-1)
    ok = valid_mask[::thinning, ::thinning].reshape(-1) & np.isfinite(vs)
    return SampleSet(
        xy=np.column_stack([xs, ys])[ok], z=np.asarray(vs[ok], dtype=float)
    )


def merge_samples(*sets: SampleSet, tolerance: float = 0.0) -> SampleSet:
    """Concatenate sample sets and drop exact duplicates.

    When duplicates coincide, the first occurrence wins. ``tolerance``
    rounds coordinates to ``-log10(tolerance)`` decimals before dedup.
    """
    xys = [np.asarray(s.xy, dtype=float).reshape(-1, 2) for s in sets if s.xy.size]
    zs = [np.asarray(s.z, dtype=float).reshape(-1) for s in sets if s.xy.size]
    if not xys:
        return SampleSet(xy=np.empty((0, 2)), z=np.empty((0,)))
    xy = np.vstack(xys)
    z = np.concatenate(zs)
    finite = np.isfinite(z) & np.all(np.isfinite(xy), axis=1)
    xy, z = xy[finite], z[finite]
    if tolerance and tolerance > 0:
        decimals = max(0, int(round(-np.log10(tolerance))))
        _, idx = np.unique(np.round(xy, decimals=decimals), axis=0, return_index=True)
        keep = np.sort(idx)
        xy, z = xy[keep], z[keep]
    return SampleSet(xy=xy, z=z)
