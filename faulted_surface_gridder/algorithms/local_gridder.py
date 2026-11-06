from qgis.core import (
    QgsProcessingAlgorithm,
)

from .gdal_helpers import write_geotiff
from .gridding_context import (
    add_common_parameters,
    load_gridding_inputs,
    prepare_grid_level,
)


class LocalGridder(QgsProcessingAlgorithm):
    INPUT_POINTS = "INPUT_POINTS"
    Z_FIELD = "Z_FIELD"
    FAULT_LINES = "FAULT_LINES"
    FAULT_POLYGONS = "FAULT_POLYGONS"
    GRID_EXTENT = "GRID_EXTENT"
    GRID_STEP = "GRID_STEP"
    CLIP_TO_HULL = "CLIP_TO_HULL"
    OUTPUT = 'OUTPUT'

    def name(self):
        return 'local_gridding'

    def displayName(self):
        return 'Local Gridding'

    def group(self):
        return 'Gridding Methods'

    def groupId(self):
        return 'gridding_methods'

    def shortHelpString(self):
        return "Grid input data to adjacent grid nodes."

    def createInstance(self):
        return LocalGridder()

    def initAlgorithm(self, config=None):
        add_common_parameters(self)

    def processAlgorithm(self, parameters, context, feedback):
        inputs = load_gridding_inputs(self, parameters, context, feedback)
        clip_to_hull = self.parameterAsBoolean(
            parameters, self.CLIP_TO_HULL, context
        )
        level = prepare_grid_level(
            inputs, inputs.gridspec, clip_to_hull, feedback
        )

        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        write_geotiff(
            out_path, level.local_grid, inputs.gridspec, inputs.target_crs
        )
        feedback.setProgress(100)
        feedback.pushInfo("Finished local gridding")
        return {self.OUTPUT: out_path}
