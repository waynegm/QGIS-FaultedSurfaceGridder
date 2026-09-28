import numpy as np
from qgis.core import QgsGeometry, QgsPointXY

from faulted_surface_gridder.algorithms.helpers import (
    GridSpec,
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
