import numpy as np
from qgis.core import QgsGeometry, QgsPointXY

from faulted_surface_gridder.algorithms.gridding_context import (
    is_canceled,
    prepare_faulted_grid,
    same_block_candidates,
)
from faulted_surface_gridder.algorithms.helpers import (
    GridSpec,
    PreparationCanceled,
    assign_point_blocks,
    build_fault_blocks,
    fault_between,
)
from faulted_surface_gridder.algorithms.local_gridding_impl import (
    local_gridding_impl,
)


def _vline(x, ymin=-1.0, ymax=11.0):
    return QgsGeometry.fromPolylineXY(
        [QgsPointXY(x, ymin), QgsPointXY(x, ymax)]
    )


def test_fault_between_crossing_and_same_side():
    fl = [_vline(5.0)]
    assert fault_between(fl, [], QgsPointXY(1, 5), QgsPointXY(9, 5)) is True
    assert fault_between(fl, [], QgsPointXY(1, 5), QgsPointXY(2, 5)) is False


def test_fault_between_empty_and_none():
    a, b = QgsPointXY(1, 5), QgsPointXY(9, 5)
    assert fault_between([], [], a, b) is False
    assert fault_between(None, None, a, b) is False


def test_fault_between_polygon():
    poly = [
        QgsGeometry.fromPolygonXY(
            [[QgsPointXY(2, 2), QgsPointXY(4, 2), QgsPointXY(4, 4),
              QgsPointXY(2, 4), QgsPointXY(2, 2)]]
        )
    ]
    assert fault_between([], poly, QgsPointXY(0, 3), QgsPointXY(9, 3)) is True
    assert fault_between([], poly, QgsPointXY(0, 0), QgsPointXY(9, 0)) is False


def test_full_fault_splits_blocks():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    blocks = build_fault_blocks(spec, [_vline(5.0)], [])
    ids = blocks["block_ids"]
    assert ids[0, 0] != ids[0, -1]  # left/right separated
    assert bool(blocks["barrier"][0, 5])  # fault column is barrier
    assert bool(blocks["nodata_mask"][0, 5])  # on-trace cells are NoData


def test_partial_fault_reconnects_around_tip():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    fl = [QgsGeometry.fromPolylineXY([QgsPointXY(5, 5), QgsPointXY(5, 11)])]
    blocks = build_fault_blocks(spec, fl, [])
    ids = blocks["block_ids"]
    # Beyond the tip (bottom rows) both sides share one block.
    assert ids[-1, 0] == ids[-1, -1]


def test_polygon_interior_is_nodata():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    poly = [
        QgsGeometry.fromPolygonXY(
            [[QgsPointXY(2, 2), QgsPointXY(4, 2), QgsPointXY(4, 4),
              QgsPointXY(2, 4), QgsPointXY(2, 2)]]
        )
    ]
    blocks = build_fault_blocks(spec, [], poly)
    assert blocks["nodata_mask"].sum() > 0
    # Interior node must come out as NaN in the grid.
    res = local_gridding_impl(spec, [(0.5, 9.5, 7.0)], [], poly)
    assert np.isnan(res).any()


def test_gridding_does_not_mix_across_full_fault():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    pts = [(2.0, 5.0, 10.0), (2.0, 6.0, 10.0),
           (8.0, 5.0, 20.0), (8.0, 6.0, 20.0)]
    res = local_gridding_impl(spec, pts, [_vline(5.0)], [])
    assert res[5, 2] == 10.0
    assert res[5, 8] == 20.0
    assert np.isnan(res[5, 5])  # node on the trace


def test_points_inside_polygon_or_on_trace_excluded():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    poly = [
        QgsGeometry.fromPolygonXY(
            [[QgsPointXY(2, 2), QgsPointXY(4, 2), QgsPointXY(4, 4),
              QgsPointXY(2, 4), QgsPointXY(2, 2)]]
        )
    ]
    pts = [(3.0, 3.0, 99.0), (0.5, 9.5, 7.0)]
    _, valid = assign_point_blocks(
        np.array(pts), spec, build_fault_blocks(spec, [], poly)["block_ids"],
        [], poly,
    )
    assert list(valid) == [False, True]


def test_prepare_faulted_grid_empty_points():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    ctx = prepare_faulted_grid(spec, [], [_vline(5.0)], [])
    assert ctx.points.shape == (0, 3)
    assert ctx.block_ids.shape == (spec.nrows, spec.ncols)
    assert same_block_candidates(ctx, 0, 0, 0.0, 10.0) == []


def test_same_block_candidates_only_same_side():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    pts = [(2.0, 5.0, 10.0), (2.0, 6.0, 10.0),
           (8.0, 5.0, 20.0), (8.0, 6.0, 20.0)]
    ctx = prepare_faulted_grid(spec, pts, [_vline(5.0)], [])
    left = same_block_candidates(ctx, 5, 2, 2.0, 5.0)
    assert sorted(left) == [0, 1]
    assert same_block_candidates(ctx, 5, 5, 5.0, 5.0) == []  # on-trace NoData


class FakeFeedback:
    def __init__(self, cancel_after=None):
        self.progresses = []
        self.cancel_after = cancel_after
        self.calls = 0

    def setProgress(self, percent):
        self.progresses.append(float(percent))

    def isCanceled(self):
        self.calls += 1
        return self.cancel_after is not None and self.calls > self.cancel_after


def test_prepare_reports_progress_in_range():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    pts = [(2.0, 5.0, 10.0), (8.0, 5.0, 20.0)]
    fb = FakeFeedback()
    prepare_faulted_grid(spec, pts, [_vline(5.0)], [], feedback=fb)
    assert fb.progresses, "expected progress updates during preparation"
    assert all(0.0 <= p <= 100.0 for p in fb.progresses)
    assert fb.progresses == sorted(fb.progresses), "progress must not go backwards"
    assert fb.progresses[-1] == 100.0


def test_prepare_canceled_aborts():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    pts = [(float(x), 5.0, 1.0) for x in range(10)]
    fb = FakeFeedback(cancel_after=0)  # canceled from the start
    try:
        prepare_faulted_grid(spec, pts, [_vline(5.0)], [], feedback=fb)
    except PreparationCanceled:
        pass
    else:
        raise AssertionError("expected PreparationCanceled")
    assert is_canceled(fb) is True


def test_impl_returns_nan_grid_when_canceled():
    spec = GridSpec(0, 0, 10, 10, 1.0)
    pts = [(2.0, 5.0, 10.0), (8.0, 5.0, 20.0)]
    res = local_gridding_impl(
        spec, pts, [_vline(5.0)], [], feedback=FakeFeedback(cancel_after=0)
    )
    assert np.isnan(res).all()
