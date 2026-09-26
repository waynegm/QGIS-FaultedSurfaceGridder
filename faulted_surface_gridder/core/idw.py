"""Inverse distance weighted interpolation."""

import numpy as np

from .base import ProgressCallback, register
from .base import BaseInterpolator


@register("idw")
class IDWInterpolator(BaseInterpolator):
    """Standard IDW with optional search radius and neighbour cap.

    :param power: distance exponent (typically 1-3).
    :param radius: search radius in layer units (0 = global).
    :param max_neighbors: max nearest neighbours used per output point.
    :param min_neighbors: outputs with fewer neighbours become NaN.
    :param nodata: fill value for under-constrained outputs.
    """

    def __init__(
        self,
        power: float = 2.0,
        radius: float = 0.0,
        max_neighbors: int = 12,
        min_neighbors: int = 1,
        nodata: float = float("nan"),
    ):
        if power <= 0:
            raise ValueError("power must be > 0")
        self.power = float(power)
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        self.min_neighbors = int(min_neighbors)
        self.nodata = nodata
        self._tree: cKDTree | None = None
        self._z: np.ndarray | None = None

    def fit(self, xy: np.ndarray, z: np.ndarray) -> None:
        try:
            from scipy.spatial import cKDTree
        except ImportError as e:
            raise ImportError(
                "IDW interpolation requires scipy, which is not installed "
                "in the QGIS Python environment."
            ) from e
        xy = np.asarray(xy, dtype=float)
        z = np.asarray(z, dtype=float)
        if xy.shape[0] == 0:
            raise ValueError("No samples to fit")
        if xy.shape[0] != z.shape[0]:
            raise ValueError("xy and z length mismatch")
        mask = np.isfinite(z).reshape(-1) & np.all(np.isfinite(xy), axis=1)
        if not np.any(mask):
            raise ValueError("No finite samples to fit")
        self._tree = cKDTree(xy[mask])
        self._z = z[mask].reshape(-1)

    def predict(
        self, xy_out: np.ndarray, progress: ProgressCallback = None
    ) -> np.ndarray:
        if self._tree is None or self._z is None:
            raise RuntimeError("fit() must be called before predict()")
        xy_out = np.asarray(xy_out, dtype=float)
        n_out = xy_out.shape[0]
        out = np.full(n_out, self.nodata, dtype=float)

        k = min(self.max_neighbors, self._z.shape[0])
        chunk = 4096
        for start in range(0, n_out, chunk):
            stop = min(start + chunk, n_out)
            block = xy_out[start:stop]
            if self.radius > 0:
                # neighbours within radius, capped at k nearest
                idx_list = self._tree.query_ball_point(block, r=self.radius)
                for i, idx in enumerate(idx_list):
                    if len(idx) < self.min_neighbors:
                        continue
                    idx = np.asarray(idx)
                    if len(idx) > k:
                        d_all = np.linalg.norm(
                            self._tree.data[idx] - block[i], axis=1
                        )
                        idx = idx[np.argpartition(d_all, k - 1)[:k]]
                    out[start + i] = self._idw_point(
                        block[i], self._tree.data[idx], self._z[idx]
                    )
            else:
                dist, idx = self._tree.query(block, k=k)
                if k == 1:
                    dist = dist[:, None]
                    idx = idx[:, None]
                valid = np.sum(np.isfinite(dist), axis=1) >= self.min_neighbors
                for i in np.flatnonzero(valid):
                    # drop inf-distance columns (k > n edge case)
                    m = np.isfinite(dist[i])
                    out[start + i] = self._idw_point(
                        block[i],
                        self._tree.data[idx[i][m]],
                        self._z[idx[i][m]],
                        precomputed_dist=dist[i][m],
                    )
            if progress is not None and not progress(stop / n_out):
                break
        return out

    def _idw_point(
        self,
        point: np.ndarray,
        neighbors_xy: np.ndarray,
        neighbors_z: np.ndarray,
        precomputed_dist: np.ndarray | None = None,
    ) -> float:
        if precomputed_dist is None:
            precomputed_dist = np.linalg.norm(neighbors_xy - point, axis=1)
        # exact co-location -> return sample value directly
        zero = precomputed_dist == 0.0
        if np.any(zero):
            return float(neighbors_z[np.flatnonzero(zero)[0]])
        w = 1.0 / np.power(precomputed_dist, self.power)
        return float(np.sum(w * neighbors_z) / np.sum(w))
