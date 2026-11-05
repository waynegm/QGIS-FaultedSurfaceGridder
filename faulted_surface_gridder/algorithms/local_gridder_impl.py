"""Local tanh-distance interpolation implementation.

The heavy lifting is deliberately split into:
  * a QGIS/QgsSpatialIndexKDBush gather phase, and
  * a pure NumPy weighted-mean calculation.

Keeping the weighted mean free of QGIS objects makes it possible to replace
the node loop with a numba kernel later without changing the data preparation.
"""

from dataclasses import dataclass, field

import numpy as np
from qgis.core import (
    QgsFeatureRequest,
    QgsProcessingException,
    QgsRectangle,
    QgsSpatialIndexKDBush,
    QgsVectorLayer,
)

from .helpers import GridSpec


@dataclass
class PointSamples:
    """Valid point samples and their QGIS KD-tree index."""

    index: QgsSpatialIndexKDBush | None
    z_by_fid: dict[int, float] = field(default_factory=dict)
    xs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    fids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def count(self) -> int:
        return len(self.fids)


def _empty_samples() -> PointSamples:
    return PointSamples(index=None)


def build_point_samples(
    points_layer: QgsVectorLayer | None,
    z_field: str,
    feedback=None,
) -> PointSamples:
    """Extract finite ``(x, y, z)`` samples and build a QGIS KDBush index."""
    if points_layer is None or not points_layer.isValid():
        return _empty_samples()

    z_idx = points_layer.fields().lookupField(z_field)
    if z_idx < 0:
        raise QgsProcessingException(
            f"Point value field '{z_field}' was not found in the input layer"
        )

    xs: list[float] = []
    ys: list[float] = []
    fids: list[int] = []
    z_by_fid: dict[int, float] = {}

    for i, feature in enumerate(points_layer.getFeatures()):
        if i % 1024 == 0 and feedback is not None and feedback.isCanceled():
            break

        geom = feature.geometry()
        if geom is None or geom.isEmpty() or geom.isMultipart():
            continue
        try:
            point = geom.asPoint()
        except Exception:
            continue

        try:
            value = feature[z_idx]
            if value is None:
                continue
            z = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(z):
            continue

        fid = int(feature.id())
        z_by_fid[fid] = z
        xs.append(float(point.x()))
        ys.append(float(point.y()))
        fids.append(fid)

    if not fids:
        if feedback is not None:
            feedback.pushWarning("No valid input points with a finite value field")
        return _empty_samples()

    request = QgsFeatureRequest().setSubsetOfAttributes([z_idx])
    iterator = points_layer.getFeatures(request)
    if feedback is not None:
        index = QgsSpatialIndexKDBush(iterator, feedback)
    else:
        index = QgsSpatialIndexKDBush(iterator)

    if feedback is not None:
        feedback.pushInfo(f"Indexed {len(fids)} point samples")

    return PointSamples(
        index=index,
        z_by_fid=z_by_fid,
        xs=np.asarray(xs, dtype=float),
        ys=np.asarray(ys, dtype=float),
        fids=np.asarray(fids, dtype=np.int64),
    )


def weighted_mean(
    node_x: float,
    node_y: float,
    cand_x: np.ndarray,
    cand_y: np.ndarray,
    cand_z: np.ndarray,
) -> float:
    """Tanh-distance weighted mean for one node.

    ``w = tanh(d) / d`` with the removable singularity at ``d == 0`` set to 1.
    All candidates, including points exactly on the node, contribute.
    """
    dx = cand_x - node_x
    dy = cand_y - node_y
    dist = np.hypot(dx, dy)

    weights = np.ones_like(dist, dtype=float)
    np.divide(np.tanh(dist), dist, out=weights, where=dist > 0)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.dot(weights, cand_z) / weight_sum)


def local_gridder_impl(
    gridspec: GridSpec,
    samples: PointSamples,
    block_ids: np.ndarray,
    point_blocks: np.ndarray,
    nodata_mask: np.ndarray,
    feedback=None,
) -> np.ndarray:
    """Interpolate samples onto adjacent grid nodes within one grid step."""
    if feedback is not None:
        feedback.pushInfo("Local tanh-distance interpolation")
    return tanh_idw(
        gridspec,
        samples,
        block_ids,
        point_blocks,
        nodata_mask,
        feedback=feedback,
    )


def tanh_idw(
    gridspec: GridSpec,
    samples: PointSamples,
    block_ids: np.ndarray,
    point_blocks: np.ndarray,
    nodata_mask: np.ndarray,
    feedback=None,
) -> np.ndarray:
    """Fill nodes from samples within a one-step square window."""
    rows = gridspec.nrows
    cols = gridspec.ncols
    step = gridspec.step
    xmin = gridspec.xmin
    top_y = gridspec.top_y

    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    if samples.index is None or samples.count == 0:
        grid[nodata_mask] = np.nan
        return grid

    block_by_fid = {
        int(fid): int(block)
        for fid, block in zip(samples.fids, point_blocks)
    }

    # Only visit rows/columns that can possibly have a sample within one step.
    c0 = max(0, int(np.floor((samples.xs.min() - step - xmin) / step)))
    c1 = min(cols - 1, int(np.ceil((samples.xs.max() + step - xmin) / step)))
    r0 = max(0, int(np.floor((top_y - (samples.ys.max() + step)) / step)))
    r1 = min(rows - 1, int(np.ceil((top_y - (samples.ys.min() - step)) / step)))
    if c1 < c0 or r1 < r0:
        grid[nodata_mask] = np.nan
        return grid

    total_rows = r1 - r0 + 1

    for row_index, r in enumerate(range(r0, r1 + 1)):
        if feedback is not None:
            if feedback.isCanceled():
                break
            feedback.setProgress(5.0 + 90.0 * (row_index + 1) / total_rows)

        y = top_y - r * step
        for c in range(c0, c1 + 1):
            if nodata_mask[r, c]:
                continue
            node_block = int(block_ids[r, c])
            if node_block < 0:
                continue

            x = xmin + c * step
            candidates = samples.index.intersects(
                QgsRectangle(x - step, y - step, x + step, y + step)
            )
            if not candidates:
                continue

            cand_x: list[float] = []
            cand_y: list[float] = []
            cand_z: list[float] = []
            for data in candidates:
                fid = int(data.id)
                z = samples.z_by_fid.get(fid)
                if z is None:
                    continue
                if block_by_fid.get(fid, -1) != node_block:
                    continue
                point = data.point()
                cand_x.append(float(point.x()))
                cand_y.append(float(point.y()))
                cand_z.append(z)

            if not cand_x:
                continue

            grid[r, c] = weighted_mean(
                x,
                y,
                np.asarray(cand_x, dtype=float),
                np.asarray(cand_y, dtype=float),
                np.asarray(cand_z, dtype=float),
            )

    grid[nodata_mask] = np.nan
    return grid
