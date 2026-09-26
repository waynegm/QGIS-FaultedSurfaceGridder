import numpy as np

from faulted_surface_gridder.core import get_interpolator
from faulted_surface_gridder.core.faults import BARRIER, label_blocks
from faulted_surface_gridder.core.grid import GridSpec, grid_centers, interpolate_blocks
from faulted_surface_gridder.core.sampling import (
    SampleSet,
    densify_polyline,
    merge_samples,
    sample_raster_values,
)


def test_idw_exact_at_samples():
    interp = get_interpolator("idw", power=2.0)
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z = np.array([10.0, 20.0, 30.0])
    interp.fit(xy, z)
    np.testing.assert_allclose(interp.predict(xy), z, rtol=1e-9)


def test_idw_plane_reasonable():
    rng = np.random.default_rng(0)
    xy = rng.random((200, 2)) * 10.0
    z = 2 * xy[:, 0] + 3 * xy[:, 1] + 5
    interp = get_interpolator("idw", power=2.0, max_neighbors=12)
    interp.fit(xy, z)
    pred = interp.predict(np.array([[5.0, 5.0]]))
    assert abs(pred[0] - 30.0) < 1.5


def test_tps_exact_plane():
    interp = get_interpolator("tps")
    rng = np.random.default_rng(1)
    xy = rng.random((30, 2)) * 10.0
    z = 2 * xy[:, 0] - xy[:, 1] + 4
    interp.fit(xy, z)
    pred = interp.predict(np.array([[5.0, 5.0], [1.0, 2.0]]))
    np.testing.assert_allclose(pred, [9.0, 4.0], atol=0.5)


def test_tps_auto_cap_large_sample_count():
    rng = np.random.default_rng(4)
    xy = rng.random((120, 2)) * 10.0
    z = np.sin(xy[:, 0]) + np.cos(xy[:, 1])
    interp = get_interpolator("tps", max_global_points=50, auto_neighbors=8)
    interp.fit(xy, z)
    assert interp.auto_capped is True
    assert interp.used_neighbors == 8
    pred = interp.predict(np.array([[5.0, 5.0]]))
    assert np.isfinite(pred).all()


def test_tps_global_small_sample_count():
    rng = np.random.default_rng(5)
    xy = rng.random((40, 2)) * 10.0
    z = xy[:, 0] + xy[:, 1]
    interp = get_interpolator("tps", max_global_points=3000)
    interp.fit(xy, z)
    assert interp.auto_capped is False
    assert interp.used_neighbors is None


def test_tps_collinear_global_steps_down_degree():
    # Picks along a single straight line: degree-1 monomials rank-deficient.
    t = np.linspace(0, 10, 60)
    xy = np.column_stack([t, t])
    z = 2 * t + 1
    interp = get_interpolator("tps")
    interp.fit(xy, z)  # must not raise
    assert interp.degraded is True
    assert interp.used_degree < 1
    pred = interp.predict(np.array([[5.0, 5.0], [0.0, 0.0]]))
    assert np.isfinite(pred).all()
    assert abs(pred[1] - 1.0) < 1.0


def test_tps_collinear_local_neighbors_recovers_in_predict():
    t = np.linspace(0, 10, 120)
    xy = np.column_stack([t, t])
    z = 2 * t + 1
    interp = get_interpolator("tps", neighbors=8)
    interp.fit(xy, z)  # local solves are lazy: no failure yet
    pred = interp.predict(np.array([[5.0, 5.0]]))  # must not raise
    assert np.isfinite(pred).all()


def test_block_fallback_to_idw_on_singular():
    warnings_list = []

    def boom():
        raise np.linalg.LinAlgError("singular test double")

    spec = GridSpec(0, 0, 4, 4, 1.0)
    block_ids = np.zeros((4, 4), dtype=int)
    sxy = np.array([[0.5, 0.5], [3.5, 3.5], [0.5, 3.5]])
    sz = np.array([1.0, 2.0, 3.0])
    sblock = np.zeros(3, dtype=int)
    out = interpolate_blocks(boom, sxy, sz, block_ids, sblock, spec,
                             min_points=1, warn=warnings_list.append,
                             fallback_factory=lambda: get_interpolator("idw"))
    assert np.isfinite(out).all()
    assert len(warnings_list) == 1 and "fallback" in warnings_list[0]


def test_convex_hull_mask_clips_corners():
    from faulted_surface_gridder.core.grid import convex_hull_mask

    spec = GridSpec(0, 0, 10, 10, 1.0)
    # diamond of samples: corners of the grid are outside the hull
    sxy = np.array([[5.0, 1.0], [9.0, 5.0], [5.0, 9.0], [1.0, 5.0]])
    mask = convex_hull_mask(sxy, spec)
    assert mask.shape == (10, 10)
    assert mask[5, 5]  # center cell inside
    assert not mask[0, 0] and not mask[0, 9]  # corners outside
    assert int(mask.sum()) < 100


def test_convex_hull_mask_buffer_expands():
    from faulted_surface_gridder.core.grid import convex_hull_mask

    spec = GridSpec(0, 0, 10, 10, 1.0)
    sxy = np.array([[5.0, 1.0], [9.0, 5.0], [5.0, 9.0], [1.0, 5.0]])
    assert convex_hull_mask(sxy, spec, buffer_cells=2).sum() > \
        convex_hull_mask(sxy, spec).sum()


def test_convex_hull_mask_degenerate_raises():
    from faulted_surface_gridder.core.grid import convex_hull_mask

    spec = GridSpec(0, 0, 10, 10, 1.0)
    try:
        convex_hull_mask(np.array([[1.0, 1.0], [5.0, 5.0]]), spec)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <3 points")
    import pytest
    with pytest.raises(Exception):  # QhullError: collinear points
        t = np.linspace(0, 9, 10)
        convex_hull_mask(np.column_stack([t, t]), spec)


def test_fault_blocks_split_by_line():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    segs = [np.array([[5.0, 0.0], [5.0, 10.0]])]
    from faulted_surface_gridder.core.faults import rasterize_line_segments

    barrier = rasterize_line_segments(segs, spec.origin_x, spec.top_y,
                                      spec.pixel, spec.nrows, spec.ncols)
    blocks = label_blocks(barrier)
    left = blocks[5, 2]
    right = blocks[5, 7]
    assert left != BARRIER and right != BARRIER and left != right


def test_block_interpolation_discontinuity():
    # step function across x=5 fault; per-block IDW must preserve the step
    spec = GridSpec(0, 0, 10, 4, 1.0)
    from faulted_surface_gridder.core.faults import rasterize_line_segments

    barrier = rasterize_line_segments(
        [np.array([[5.0, 0.0], [5.0, 4.0]])],
        spec.origin_x, spec.top_y, spec.pixel, spec.nrows, spec.ncols)
    blocks = label_blocks(barrier)
    gx, gy = grid_centers(spec.origin_x, spec.top_y, spec.pixel, spec.nrows, spec.ncols)
    left_m = gx < 5.0
    right_m = gx > 5.0
    rng = np.random.default_rng(2)
    lx = gx[left_m].reshape(-1)[::3][:20]
    ly = gy[left_m].reshape(-1)[::3][:20]
    rx = gx[right_m].reshape(-1)[::3][:20]
    ry = gy[right_m].reshape(-1)[::3][:20]
    sxy = np.vstack([np.column_stack([lx, ly]), np.column_stack([rx, ry])])
    # add slight noise so geometry is not degenerate
    sxy = sxy + rng.normal(0, 0.05, sxy.shape)
    sz = np.where(sxy[:, 0] < 5.0, 100.0, 200.0)

    # locate samples in blocks without importing QGIS code
    cols = ((sxy[:, 0] - spec.origin_x) / spec.pixel).astype(int)
    rows = ((spec.top_y - sxy[:, 1]) / spec.pixel).astype(int)
    sblock = np.full(sxy.shape[0], BARRIER, dtype=int)
    ok = (rows >= 0) & (rows < spec.nrows) & (cols >= 0) & (cols < spec.ncols)
    sblock[ok] = blocks[rows[ok], cols[ok]]
    out = interpolate_blocks(lambda: get_interpolator("idw", max_neighbors=6),
                             sxy, sz, blocks, sblock, spec, min_points=1)
    assert abs(float(np.nanmean(out[:, 1:3])) - 100.0) < 5.0
    assert abs(float(np.nanmean(out[:, 6:8])) - 200.0) < 5.0


def test_merge_and_densify():
    d = densify_polyline(np.array([[0.0, 0.0], [2.0, 0.0]]), spacing=0.5)
    assert d.shape[0] == 5
    m = merge_samples(SampleSet(np.array([[0.0, 0.0]]), np.array([1.0])),
                      SampleSet(np.array([[1.0, 1.0]]), np.array([2.0])))
    assert m.xy.shape == (2, 2)
    gx, gy = np.meshgrid(np.arange(4.0), np.arange(4.0))
    vals = gx + gy
    s = sample_raster_values(gx, gy, vals, np.ones_like(vals, bool), thinning=2)
    assert s.xy.shape[0] == 4
