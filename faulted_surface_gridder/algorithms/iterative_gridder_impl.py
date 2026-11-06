"""Second-derivative least-squares interpolation for the iterative gridder.

This module is deliberately free of QGIS objects.  It takes a locally gridded
array (the stage-1 local grid), fault-block connectivity, and a NoData mask,
and fills the remaining active nodes by minimising the squared second
derivatives of the surface.

The minimisation is assembled as a sparse least-squares problem

    A x = b

where each row is one second-derivative stencil (``d_xx``, ``d_yy`` or
``d_xy``) with known local-grid values moved to the right-hand side.  It is
solved with :func:`scipy.sparse.linalg.lsmr`.  Faults are handled by simply
omitting stencils that cross a blocked grid edge, exactly as described by
Léger and Clochard (2020).

Multi-scale support is provided by coarse-grid construction and bilinear
upsampling; the orchestration lives in ``iterative_gridder.py``.

SciPy is imported lazily so that the plugin (and the local gridder) can load
without it; only this algorithm requires SciPy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .helpers import GridSpec

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

# Eq. (6) of the paper weights the cross-derivative term twice relative to
# the two axial terms, so its least-squares row is scaled by sqrt(2).
CROSS_WEIGHT = math.sqrt(2.0)

SCIPY_INSTALL_HINT = (
    "The Iterative Gridding algorithm requires SciPy, which could not be "
    "imported in QGIS's Python environment. Install SciPy for the Python "
    "interpreter used by QGIS, then restart QGIS. On Windows run "
    '"<QGIS install>\\bin\\python-qgis-ltr.bat" -m pip install scipy '
    "from an elevated Command Prompt (use python-qgis.bat for non-LTR "
    "builds). On Linux install the python3-scipy package or run "
    "'python3 -m pip install scipy' with QGIS's Python."
)


def require_scipy() -> None:
    """Raise ``ImportError`` with installation guidance if SciPy is missing."""
    try:
        import scipy  # noqa: F401
        from scipy.ndimage import map_coordinates  # noqa: F401
        from scipy.sparse import csr_matrix  # noqa: F401
        from scipy.sparse.linalg import lsmr  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{SCIPY_INSTALL_HINT}\n\nOriginal error: {exc}") from exc


def coarsen_gridspec(spec: GridSpec, factor: int) -> GridSpec:
    """Return a coarser grid whose nodes are a subset of ``spec``'s nodes.

    A factor of 2 keeps every second node in each direction.  The coarse
    lattice shares the lower-left origin with the fine lattice.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("Coarsening factor must be a positive integer")
    return GridSpec(
        spec.xmin,
        spec.ymin,
        (spec.ncols - 1) // factor + 1,
        (spec.nrows - 1) // factor + 1,
        spec.step * factor,
    )


def resolve_scale_count(spec: GridSpec, requested: int) -> int:
    """Resolve ``0 = automatic`` multi-scale level count.

    The coarsest level must still have at least one cell in each direction.
    Automatic mode aims for a handful of levels and is capped at six.
    """
    max_factor = max(1, min(spec.nrows - 1, spec.ncols - 1))
    max_levels = 1
    while max_levels < 32 and (2 ** max_levels) <= max_factor:
        max_levels += 1

    requested = int(requested)
    if requested <= 0:
        requested = min(max_levels, 6)
    return max(1, min(requested, max_levels))


def _nearest_fill(values: np.ndarray) -> np.ndarray | None:
    """Fill NaNs with the nearest finite value, or return None if none exist."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return values.copy()
    if not finite.any():
        return None

    from scipy.ndimage import distance_transform_edt

    indices = distance_transform_edt(
        ~finite, return_distances=False, return_indices=True
    )
    return values[tuple(indices)]


def upsample_grid(
    coarse_values: np.ndarray,
    coarse_spec: GridSpec,
    fine_spec: GridSpec,
) -> np.ndarray | None:
    """Bilinearly upsample a coarse grid to ``fine_spec`` node coordinates.

    Returns ``None`` when the coarse grid has no finite values.  The
    interpolation is only used as an initial guess; the fine-grid solve still
    enforces the exact fault and NoData masks.
    """
    coarse_values = np.asarray(coarse_values, dtype=float)
    if coarse_values.shape != (coarse_spec.nrows, coarse_spec.ncols):
        raise ValueError("coarse_values shape does not match coarse_spec")

    filled = _nearest_fill(coarse_values)
    if filled is None:
        return None

    from scipy.ndimage import map_coordinates

    fine_x, fine_y = fine_spec.nodes()
    col_coords = (fine_x - coarse_spec.xmin) / coarse_spec.step
    row_coords = (coarse_spec.top_y - fine_y) / coarse_spec.step
    return map_coordinates(
        filled,
        [row_coords, col_coords],
        order=1,
        mode="nearest",
        prefilter=False,
    )


def _append_stencil_rows(
    valid: np.ndarray,
    coefficients: list[float],
    unknowns: list[np.ndarray],
    known_values: list[np.ndarray],
    unknown_indices: list[np.ndarray],
    weight: float,
    row_offset: int,
    row_parts: list[np.ndarray],
    col_parts: list[np.ndarray],
    data_parts: list[np.ndarray],
    rhs_parts: list[np.ndarray],
) -> int:
    """Append sparse rows for one vectorised stencil family."""
    any_unknown = np.zeros(valid.shape, dtype=bool)
    for unknown in unknowns:
        any_unknown |= unknown
    row_mask = valid & any_unknown
    if not row_mask.any():
        return row_offset

    r_idx, c_idx = np.nonzero(row_mask)
    row_numbers = row_offset + np.arange(r_idx.size)
    rhs = np.zeros(r_idx.size, dtype=float)

    for coefficient, unknown, known, indices in zip(
        coefficients, unknowns, known_values, unknown_indices
    ):
        selected = unknown[r_idx, c_idx]
        if selected.any():
            row_parts.append(row_numbers[selected])
            col_parts.append(indices[r_idx[selected], c_idx[selected]])
            data_parts.append(
                np.full(int(selected.sum()), weight * coefficient)
            )
        rhs += coefficient * known[r_idx, c_idx]

    rhs_parts.append(weight * (-rhs))
    return row_offset + r_idx.size


def build_least_squares_system(
    gridspec: GridSpec,
    values: np.ndarray,
    active: np.ndarray,
    known: np.ndarray,
    unknown: np.ndarray,
    blocked_h: np.ndarray,
    blocked_v: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Assemble the sparse second-derivative least-squares system.

    ``values`` supplies the known node values.  ``active``, ``known`` and
    ``unknown`` are boolean node masks; known values are eliminated from the
    left-hand side and moved to the right-hand side.
    """
    rows = gridspec.nrows
    cols = gridspec.ncols
    if values.shape != (rows, cols):
        raise ValueError("values shape does not match gridspec")
    if blocked_h.shape != (rows, max(cols - 1, 0)):
        raise ValueError("blocked_h shape does not match gridspec")
    if blocked_v.shape != (max(rows - 1, 0), cols):
        raise ValueError("blocked_v shape does not match gridspec")

    from scipy.sparse import csr_matrix

    unknown_index = np.full((rows, cols), -1, dtype=np.int64)
    n_unknown = int(unknown.sum())
    if n_unknown:
        unknown_index[unknown] = np.arange(n_unknown)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    rhs_parts: list[np.ndarray] = []
    row_offset = 0

    # d_xx at (r, c): f[r, c-1] - 2 f[r, c] + f[r, c+1].
    valid = (
        active[:, :-2]
        & active[:, 1:-1]
        & active[:, 2:]
        & ~blocked_h[:, :-1]
        & ~blocked_h[:, 1:]
    )
    row_offset = _append_stencil_rows(
        valid,
        [1.0, -2.0, 1.0],
        [unknown[:, :-2], unknown[:, 1:-1], unknown[:, 2:]],
        [
            np.where(known[:, :-2], values[:, :-2], 0.0),
            np.where(known[:, 1:-1], values[:, 1:-1], 0.0),
            np.where(known[:, 2:], values[:, 2:], 0.0),
        ],
        [
            np.where(unknown[:, :-2], unknown_index[:, :-2], -1),
            np.where(unknown[:, 1:-1], unknown_index[:, 1:-1], -1),
            np.where(unknown[:, 2:], unknown_index[:, 2:], -1),
        ],
        1.0,
        row_offset,
        row_parts,
        col_parts,
        data_parts,
        rhs_parts,
    )

    # d_yy at (r, c): f[r-1, c] - 2 f[r, c] + f[r+1, c].
    valid = (
        active[:-2, :]
        & active[1:-1, :]
        & active[2:, :]
        & ~blocked_v[:-1, :]
        & ~blocked_v[1:, :]
    )
    row_offset = _append_stencil_rows(
        valid,
        [1.0, -2.0, 1.0],
        [unknown[:-2, :], unknown[1:-1, :], unknown[2:, :]],
        [
            np.where(known[:-2, :], values[:-2, :], 0.0),
            np.where(known[1:-1, :], values[1:-1, :], 0.0),
            np.where(known[2:, :], values[2:, :], 0.0),
        ],
        [
            np.where(unknown[:-2, :], unknown_index[:-2, :], -1),
            np.where(unknown[1:-1, :], unknown_index[1:-1, :], -1),
            np.where(unknown[2:, :], unknown_index[2:, :], -1),
        ],
        1.0,
        row_offset,
        row_parts,
        col_parts,
        data_parts,
        rhs_parts,
    )

    # d_xy on a 2x2 cell: f[r,c] + f[r+1,c+1] - f[r+1,c] - f[r,c+1].
    valid = (
        active[:-1, :-1]
        & active[:-1, 1:]
        & active[1:, :-1]
        & active[1:, 1:]
        & ~blocked_h[:-1, :]
        & ~blocked_h[1:, :]
        & ~blocked_v[:, :-1]
        & ~blocked_v[:, 1:]
    )
    row_offset = _append_stencil_rows(
        valid,
        [1.0, -1.0, -1.0, 1.0],
        [
            unknown[:-1, :-1],
            unknown[:-1, 1:],
            unknown[1:, :-1],
            unknown[1:, 1:],
        ],
        [
            np.where(known[:-1, :-1], values[:-1, :-1], 0.0),
            np.where(known[:-1, 1:], values[:-1, 1:], 0.0),
            np.where(known[1:, :-1], values[1:, :-1], 0.0),
            np.where(known[1:, 1:], values[1:, 1:], 0.0),
        ],
        [
            np.where(unknown[:-1, :-1], unknown_index[:-1, :-1], -1),
            np.where(unknown[:-1, 1:], unknown_index[:-1, 1:], -1),
            np.where(unknown[1:, :-1], unknown_index[1:, :-1], -1),
            np.where(unknown[1:, 1:], unknown_index[1:, 1:], -1),
        ],
        CROSS_WEIGHT,
        row_offset,
        row_parts,
        col_parts,
        data_parts,
        rhs_parts,
    )

    if not data_parts:
        return csr_matrix((0, n_unknown), dtype=float), np.empty(0)

    matrix = csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(row_offset, n_unknown),
    )
    rhs = np.concatenate(rhs_parts)
    return matrix, rhs


def _classify_blocks(
    block_ids: np.ndarray,
    active: np.ndarray,
    known: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split active nodes into solvable, fallback-mean and no-data blocks."""
    rows, cols = block_ids.shape
    solvable = np.zeros((rows, cols), dtype=bool)
    fallback = np.zeros((rows, cols), dtype=bool)
    no_data = np.zeros((rows, cols), dtype=bool)
    block_mean = np.full((rows, cols), np.nan, dtype=float)

    block_values = block_ids[active]
    if block_values.size == 0:
        return solvable, fallback, no_data, block_mean

    for block_id in np.unique(block_values):
        block_mask = (block_ids == block_id) & active
        known_mask = block_mask & known
        n_known = int(known_mask.sum())
        if n_known == 0:
            no_data |= block_mask
            continue

        block_mean[block_mask] = float(np.mean(values[known_mask]))

        if n_known >= 3:
            r_idx, c_idx = np.nonzero(known_mask)
            design = np.column_stack(
                (np.ones(n_known), c_idx, r_idx)
            )
            if np.linalg.matrix_rank(design) >= 3:
                solvable |= block_mask
                continue

        fallback |= block_mask

    return solvable, fallback, no_data, block_mean


def iterative_gridder_impl(
    gridspec: GridSpec,
    local_grid: np.ndarray,
    block_ids: np.ndarray,
    blocked_h: np.ndarray,
    blocked_v: np.ndarray,
    nodata_mask: np.ndarray,
    iterations: int = 100,
    tolerance: float = 1e-6,
    initial: np.ndarray | None = None,
    feedback=None,
) -> np.ndarray:
    """Fill non-local nodes by minimising second derivatives.

    ``local_grid`` is the stage-1 local gridding result; its finite values are
    hard constraints.  Fault blocks with fewer than three independent known
    nodes are filled with their mean, while blocks with no known nodes remain
    NaN.
    """
    rows = gridspec.nrows
    cols = gridspec.ncols
    local_grid = np.asarray(local_grid, dtype=float)
    if local_grid.shape != (rows, cols):
        raise ValueError("local_grid shape does not match gridspec")
    if nodata_mask.shape != (rows, cols):
        raise ValueError("nodata_mask shape does not match gridspec")

    active = (block_ids >= 0) & ~np.asarray(nodata_mask, dtype=bool)
    known = active & np.isfinite(local_grid)
    result = local_grid.copy()
    result[~active] = np.nan

    solvable, fallback, no_data, block_mean = _classify_blocks(
        block_ids, active, known, local_grid
    )

    # Underconstrained blocks are completed with their local mean.
    fallback_unknown = fallback & ~known
    if fallback_unknown.any():
        result[fallback_unknown] = block_mean[fallback_unknown]
        if feedback is not None:
            feedback.pushWarning(
                f"{int(np.unique(block_ids[fallback]).size)} fault block(s) "
                "have fewer than 3 independent data nodes; filled with their "
                "mean value"
            )

    if no_data.any() and feedback is not None:
        feedback.pushWarning(
            f"{int(np.unique(block_ids[no_data]).size)} fault block(s) "
            "contain no data and remain NoData"
        )

    solvable_known = solvable & known
    solvable_unknown = solvable & ~known
    n_unknown = int(solvable_unknown.sum())
    if n_unknown == 0:
        result[~active] = np.nan
        return result

    matrix, rhs = build_least_squares_system(
        gridspec,
        result,
        solvable,
        solvable_known,
        solvable_unknown,
        blocked_h,
        blocked_v,
    )

    if matrix.shape[0] == 0:
        if feedback is not None:
            feedback.pushWarning(
                "No second-derivative equations were available; "
                "unconstrained nodes were filled with their block mean"
            )
        result[solvable_unknown] = block_mean[solvable_unknown]
        result[~active] = np.nan
        return result

    base = block_mean[solvable_unknown]
    base = np.where(np.isfinite(base), base, 0.0)
    if initial is None:
        x0 = base
    else:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (rows, cols):
            raise ValueError("initial shape does not match gridspec")
        initial_values = initial[solvable_unknown]
        x0 = np.where(np.isfinite(initial_values), initial_values, base)

    max_iterations = max(1, int(iterations))
    atol = max(0.0, float(tolerance))

    from scipy.sparse.linalg import lsmr

    x, istop, itn, normr, normar, norma, conda, normx = lsmr(
        matrix,
        rhs,
        x0=x0,
        atol=atol,
        btol=atol,
        maxiter=max_iterations,
    )

    result[solvable_unknown] = x
    result[~active] = np.nan

    if feedback is not None:
        feedback.pushInfo(
            f"Iterative solve: {n_unknown} unknowns, {matrix.shape[0]} "
            f"equations, {itn} lsmr iterations (istop={istop})"
        )
        if istop == 5:
            feedback.pushWarning(
                "lsmr reached the iteration limit before converging"
            )

    return result
