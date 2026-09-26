"""Thin plate spline interpolation via scipy RBFInterpolator."""

import warnings

import numpy as np

from .base import ProgressCallback, register
from .base import BaseInterpolator


@register("tps")
class TPSInterpolator(BaseInterpolator):
    """Thin plate spline (kernel='thin_plate_spline').

    :param smoothing: smoothing parameter; 0 = exact interpolation.
    :param neighbors: number of nearest neighbours per evaluation point.
        None (default) means auto: an exact global solve up to
        ``max_global_points`` samples, otherwise a local solve with
        ``auto_neighbors`` neighbours. Pass an explicit int to override.
    :param degree: polynomial degree (default 1, minimum valid for TPS).
    :param max_global_points: sample count above which the global solve
        is refused (its memory grows with N squared).
    :param auto_neighbors: neighbour count used for the automatic
        fallback local solve.
    """

    def __init__(
        self,
        smoothing: float = 0.0,
        neighbors: int | None = None,
        degree: int = 1,
        max_global_points: int = 3000,
        auto_neighbors: int = 32,
    ):
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0")
        self.smoothing = float(smoothing)
        self.neighbors = int(neighbors) if neighbors else None
        self.degree = int(degree)
        self.max_global_points = int(max_global_points)
        self.auto_neighbors = int(auto_neighbors)
        self._interp = None
        # fit() populates these for reporting:
        self.n_points = 0
        self.used_neighbors: int | None = None
        self.auto_capped = False
        self.used_degree = int(degree)
        self.degraded = False  # True if degree had to be stepped down
        self._xy: np.ndarray | None = None
        self._z: np.ndarray | None = None

    def _degrees_to_try(self, start: int) -> list[int]:
        """Candidate degrees from ``start`` down to no polynomial."""
        out = [start]
        for d in (0, -1):
            if d < start and d not in out:
                out.append(d)
        return out

    def _build_one(self, xy: np.ndarray, z: np.ndarray,
                   neighbors: int | None, degree: int):
        """Construct a single-degree scipy interpolant (may raise)."""
        try:
            from scipy.interpolate import RBFInterpolator
        except ImportError as e:
            raise ImportError(
                "Thin plate spline interpolation requires scipy, which is not "
                "installed in the QGIS Python environment."
            ) from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scipy warns on degree < 1
            return RBFInterpolator(
                xy,
                z,
                kernel="thin_plate_spline",
                smoothing=self.smoothing,
                neighbors=neighbors,
                degree=degree,
            )

    def _build(self, xy: np.ndarray, z: np.ndarray, neighbors: int | None,
               start_degree: int):
        """Build the scipy interpolant, stepping the polynomial degree down
        on singular systems.

        Collinear/coplanar neighbourhoods (common with picks digitized along
        seismic lines) make the degree-1 monomial matrix rank-deficient. A
        constant (degree 0) or absent (degree -1) polynomial restores
        solvability; the RBF terms still carry the spatial structure.

        Note: with ``neighbors`` set, scipy solves lazily at evaluation
        time, so construction alone cannot validate solvability — callers
        must probe-evaluate (see predict()).
        """
        last: Exception | None = None
        for degree in self._degrees_to_try(start_degree):
            try:
                interp = self._build_one(xy, z, neighbors, degree)
            except np.linalg.LinAlgError as e:
                last = e
                continue
            self.used_degree = degree
            self.degraded = degree != self.degree
            return interp
        raise np.linalg.LinAlgError(
            "Thin plate spline solve failed: singular system even without "
            f"polynomial terms ({last}). The samples are likely (near-) "
            "duplicated; try IDW or increase smoothing."
        )

    def fit(self, xy: np.ndarray, z: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=float)
        z = np.asarray(z, dtype=float).reshape(-1)
        mask = np.isfinite(z) & np.all(np.isfinite(xy), axis=1)
        xy, z = xy[mask], z[mask]
        if xy.shape[0] == 0:
            raise ValueError("No finite samples to fit")
        # deduplicate coincident points (singular matrix otherwise)
        _, unique = np.unique(np.round(xy, decimals=9), axis=0, return_index=True)
        xy, z = xy[np.sort(unique)], z[np.sort(unique)]
        n = xy.shape[0]
        self.n_points = n
        self._xy, self._z = xy, z
        used = self.neighbors
        if used is not None and used >= n:
            used = None  # global solve is cheaper than degenerate local solve
        if used is None and n > self.max_global_points:
            # Global solve needs an N x N dense matrix; refuse and fall back
            # to a local solve instead of raising MemoryError deep in scipy.
            used = min(self.auto_neighbors, n - 1)
            self.auto_capped = True
        self.used_neighbors = used
        self._interp = self._build(xy, z, used, self.degree)

    def predict(
        self, xy_out: np.ndarray, progress: ProgressCallback = None
    ) -> np.ndarray:
        if self._interp is None:
            raise RuntimeError("fit() must be called before predict()")
        xy_arr = np.asarray(xy_out, dtype=float)
        try:
            return self._predict_with(self._interp, xy_arr, progress)
        except np.linalg.LinAlgError:
            # Local neighbourhoods are solved lazily inside scipy, so a
            # (near-)collinear neighbourhood only surfaces here — and
            # rebuilding the same degree would fail identically. Step down
            # to lower degrees, probe-evaluating each candidate to force
            # the lazy solves before committing to a full prediction.
            if self._xy is None or self._z is None:
                raise
            probe = xy_arr[: min(256, xy_arr.shape[0])]
            for degree in self._degrees_to_try(self.used_degree)[1:]:
                candidate = self._build_one(self._xy, self._z,
                                            self.used_neighbors, degree)
                try:
                    candidate(probe)
                except np.linalg.LinAlgError:
                    continue
                self._interp = candidate
                self.used_degree = degree
                self.degraded = True
                break
            else:
                raise
            return self._predict_with(self._interp, xy_arr, progress)

    def _predict_with(self, interp, xy_out: np.ndarray,
                      progress: ProgressCallback) -> np.ndarray:
        xy_out = np.asarray(xy_out, dtype=float)
        chunk = 8192
        parts = []
        n = xy_out.shape[0]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            parts.append(interp(xy_out[start:stop]))
            if progress is not None and not progress(stop / n):
                # pad remainder with NaN on abort
                rest = np.full(n - stop, np.nan)
                return np.concatenate(parts + [rest])
        return np.concatenate(parts)
