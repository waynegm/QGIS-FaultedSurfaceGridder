"""Shared setup for the local and iterative gridding algorithms.

Both algorithms need the same input handling: resolve the target CRS, load
and reproject the point/fault layers, index the samples, rasterise heave
polygons, build fault blocks and fill nodes adjacent to the data.  Keeping
that in one place means the two algorithms only differ in how the remaining
nodes are interpolated.
"""

from dataclasses import dataclass

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProject,
    QgsVectorLayer,
)

from .fault_blocks import FaultBlocks, assign_point_blocks, build_fault_blocks
from .gdal_helpers import rasterize_heave_interior
from .helpers import (
    GridSpec,
    convex_hull_nodata_mask,
    get_fault_lines_layer,
    get_fault_polygons_layer,
    get_gridspec,
    get_points_layer,
)
from .local_gridder_impl import (
    PointSamples,
    build_point_samples,
    local_gridder_impl,
)


@dataclass
class GriddingInputs:
    """Inputs shared by every grid level."""

    gridspec: GridSpec
    target_crs: QgsCoordinateReferenceSystem
    samples: PointSamples
    fault_lines: QgsVectorLayer | None = None
    fault_polygons: QgsVectorLayer | None = None


@dataclass
class GridLevel:
    """Fault topology and locally gridded values for one grid resolution."""

    gridspec: GridSpec
    blocks: FaultBlocks
    point_blocks: np.ndarray
    heave_mask: np.ndarray
    nodata_mask: np.ndarray
    local_grid: np.ndarray


def add_common_parameters(algo: QgsProcessingAlgorithm) -> None:
    """Add the parameters shared by the local and iterative gridder."""
    algo.addParameter(
        QgsProcessingParameterFeatureSource(
            algo.INPUT_POINTS,
            "Horizon points (XYZ)",
            [QgsProcessing.TypeVectorPoint],
            optional=False,
        )
    )
    algo.addParameter(
        QgsProcessingParameterField(
            algo.Z_FIELD,
            "Point value field",
            parentLayerParameterName=algo.INPUT_POINTS,
            allowMultiple=False,
            optional=False,
        )
    )
    algo.addParameter(
        QgsProcessingParameterFeatureSource(
            algo.FAULT_LINES,
            "Fault traces (polylines, vertical faults)",
            [QgsProcessing.TypeVectorLine],
            optional=True,
        )
    )
    algo.addParameter(
        QgsProcessingParameterFeatureSource(
            algo.FAULT_POLYGONS,
            "Fault heave polygons (interiors become NoData)",
            [QgsProcessing.TypeVectorPolygon],
            optional=True,
        )
    )
    algo.addParameter(
        QgsProcessingParameterExtent(algo.GRID_EXTENT, "Grid extent")
    )
    algo.addParameter(
        QgsProcessingParameterNumber(
            algo.GRID_STEP,
            "Grid size (layer units)",
            QgsProcessingParameterNumber.Double,
            defaultValue=100.0,
            minValue=0.0,
        )
    )
    algo.addParameter(
        QgsProcessingParameterBoolean(
            algo.CLIP_TO_HULL,
            "Clip output to convex hull of input samples",
            False,
        )
    )
    algo.addParameter(
        QgsProcessingParameterRasterDestination(algo.OUTPUT, "Output Grid")
    )


def resolve_target_crs(
    algo: QgsProcessingAlgorithm,
    parameters,
    context,
    feedback,
    grid_crs: QgsCoordinateReferenceSystem,
) -> QgsCoordinateReferenceSystem:
    """Use the grid extent CRS, falling back to the points or project CRS."""
    if grid_crs.isValid():
        return grid_crs

    points_source = algo.parameterAsSource(
        parameters, algo.INPUT_POINTS, context
    )
    target_crs = (
        points_source.sourceCrs()
        if points_source is not None
        else QgsProject.instance().crs()
    )
    feedback.pushWarning(
        "Grid extent has no CRS; using the input points/project CRS"
    )
    return target_crs


def load_gridding_inputs(
    algo: QgsProcessingAlgorithm,
    parameters,
    context,
    feedback,
) -> GriddingInputs:
    """Resolve the target CRS and load/index all gridding inputs."""
    gridspec, grid_crs = get_gridspec(algo, parameters, context, feedback)
    target_crs = resolve_target_crs(
        algo, parameters, context, feedback, grid_crs
    )

    z_field = algo.parameterAsString(parameters, algo.Z_FIELD, context)
    if not z_field:
        raise QgsProcessingException("No point value field selected")

    points_layer = get_points_layer(
        algo, parameters, context, target_crs, feedback
    )
    if points_layer is None:
        raise QgsProcessingException("No input points layer")

    samples = build_point_samples(points_layer, z_field, feedback)
    fault_lines = get_fault_lines_layer(
        algo, parameters, context, target_crs, feedback
    )
    fault_polygons = get_fault_polygons_layer(
        algo, parameters, context, target_crs, feedback
    )

    return GriddingInputs(
        gridspec=gridspec,
        target_crs=target_crs,
        samples=samples,
        fault_lines=fault_lines,
        fault_polygons=fault_polygons,
    )


def prepare_grid_level(
    inputs: GriddingInputs,
    gridspec: GridSpec,
    clip_to_hull: bool,
    feedback=None,
) -> GridLevel:
    """Build fault topology and the local (near-data) grid for one level."""
    heave_mask = rasterize_heave_interior(inputs.fault_polygons, gridspec)
    blocks = build_fault_blocks(
        gridspec,
        inputs.fault_lines,
        heave_mask,
        feedback=feedback,
    )
    point_blocks = assign_point_blocks(
        inputs.samples.xs,
        inputs.samples.ys,
        inputs.samples.fids,
        gridspec,
        blocks,
        heave_mask=heave_mask,
        feedback=feedback,
    )

    nodata_mask = blocks.barrier.copy()
    if clip_to_hull:
        nodata_mask |= convex_hull_nodata_mask(
            inputs.samples.xs,
            inputs.samples.ys,
            gridspec,
            feedback,
        )

    local_grid = local_gridder_impl(
        gridspec,
        inputs.samples,
        blocks.block_ids,
        point_blocks,
        nodata_mask,
        feedback=feedback,
    )

    return GridLevel(
        gridspec=gridspec,
        blocks=blocks,
        point_blocks=point_blocks,
        heave_mask=heave_mask,
        nodata_mask=nodata_mask,
        local_grid=local_grid,
    )
