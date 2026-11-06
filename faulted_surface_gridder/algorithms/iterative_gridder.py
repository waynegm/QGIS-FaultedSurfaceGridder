"""Multi-scale iterative gridding algorithm.

Stage 1 runs the local gridder at every scale to pin the surface near the
input data.  Stage 2 fills the remaining nodes by minimising the squared
second derivatives of the surface (Léger & Clochard, 2020).  A coarse-to-fine
sequence of grids supplies the initial guess for each refinement, so the
final fine-grid solve only has to add local detail.
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterNumber,
)

from .gdal_helpers import write_geotiff
from .gridding_context import (
    add_common_parameters,
    load_gridding_inputs,
    prepare_grid_level,
)
from .iterative_gridder_impl import (
    coarsen_gridspec,
    iterative_gridder_impl,
    require_scipy,
    resolve_scale_count,
    upsample_grid,
)


class IterativeGridder(QgsProcessingAlgorithm):
    INPUT_POINTS = "INPUT_POINTS"
    Z_FIELD = "Z_FIELD"
    FAULT_LINES = "FAULT_LINES"
    FAULT_POLYGONS = "FAULT_POLYGONS"
    GRID_EXTENT = "GRID_EXTENT"
    GRID_STEP = "GRID_STEP"
    CLIP_TO_HULL = "CLIP_TO_HULL"
    ITERATIONS = "ITERATIONS"
    SCALES = "SCALES"
    TOLERANCE = "TOLERANCE"
    OUTPUT = 'OUTPUT'

    def name(self):
        return 'iterative_gridding'

    def displayName(self):
        return 'Iterative Gridding'

    def group(self):
        return 'Gridding Methods'

    def groupId(self):
        return 'gridding_methods'

    def shortHelpString(self):
        return (
            "Interpolate horizon points across a faulted surface.\n\n"
            "The algorithm first runs the local gridder to fill nodes near "
            "the input data, then iteratively fills the remaining nodes by "
            "minimising the surface's second derivatives.  Fault traces, "
            "heave polygons and (optionally) the input convex hull act as "
            "barriers.  Multi-scale refinement solves a coarse grid first "
            "and uses it to initialise progressively finer grids.\n\n"
            "Fault blocks with fewer than three independent data nodes are "
            "filled with their mean value; blocks with no data remain "
            "NoData.\n\n"
            "Requires SciPy in QGIS's Python environment.  On Windows run "
            '\"<QGIS install>\\bin\\python-qgis-ltr.bat\" -m pip install scipy'
            " from an elevated Command Prompt, then restart QGIS.  On Linux "
            "install the python3-scipy package or pip-install SciPy with "
            "QGIS's Python."
        )

    def createInstance(self):
        return IterativeGridder()

    def initAlgorithm(self, config=None):
        add_common_parameters(self)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ITERATIONS,
                "Solver iterations at the finest scale",
                QgsProcessingParameterNumber.Integer,
                defaultValue=100,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SCALES,
                "Multi-scale levels (0 = automatic)",
                QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TOLERANCE,
                "Solver tolerance",
                QgsProcessingParameterNumber.Double,
                defaultValue=1e-6,
                minValue=0.0,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        try:
            require_scipy()
        except ImportError as exc:
            raise QgsProcessingException(str(exc)) from exc

        inputs = load_gridding_inputs(self, parameters, context, feedback)
        clip_to_hull = self.parameterAsBoolean(
            parameters, self.CLIP_TO_HULL, context
        )
        iterations = self.parameterAsInt(parameters, self.ITERATIONS, context)
        requested_scales = self.parameterAsInt(
            parameters, self.SCALES, context
        )
        tolerance = self.parameterAsDouble(
            parameters, self.TOLERANCE, context
        )

        fine_spec = inputs.gridspec
        levels = resolve_scale_count(fine_spec, requested_scales)
        feedback.pushInfo(
            f"Multi-scale: {levels} level(s), {iterations} iteration(s) "
            "at the finest scale"
        )

        previous = None
        previous_spec = None
        for level_index in range(levels - 1, -1, -1):
            if feedback.isCanceled():
                raise QgsProcessingException("Iterative gridding cancelled")

            factor = 2**level_index
            level_spec = (
                fine_spec
                if factor == 1
                else coarsen_gridspec(fine_spec, factor)
            )
            feedback.pushInfo(
                f"Level {levels - level_index}/{levels}: "
                f"{level_spec.ncols} x {level_spec.nrows} nodes "
                f"@ {level_spec.step}"
            )

            level = prepare_grid_level(
                inputs, level_spec, clip_to_hull, feedback
            )

            initial = None
            if previous is not None:
                initial = upsample_grid(previous, previous_spec, level_spec)

            # Coarse levels are small but carry the large-scale structure,
            # so give them proportionally more iterations to converge.
            level_iterations = max(1, iterations * factor)
            previous = iterative_gridder_impl(
                level_spec,
                level.local_grid,
                level.blocks.block_ids,
                level.blocks.blocked_h,
                level.blocks.blocked_v,
                level.nodata_mask,
                iterations=level_iterations,
                tolerance=tolerance,
                initial=initial,
                feedback=feedback,
            )
            previous_spec = level_spec

            feedback.setProgress(100.0 * (levels - level_index) / levels)

        out_path = self.parameterAsOutputLayer(
            parameters, self.OUTPUT, context
        )
        write_geotiff(out_path, previous, fine_spec, inputs.target_crs)
        feedback.setProgress(100)
        feedback.pushInfo("Finished iterative gridding")
        return {self.OUTPUT: out_path}
