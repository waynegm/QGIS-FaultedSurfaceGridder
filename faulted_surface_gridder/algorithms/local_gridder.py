from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
)

from .helpers import (
    get_fault_lines,
    get_fault_polygons,
    get_gridspec,
    get_points,
    write_geotiff,
)
from .local_gridding_impl import local_gridding_impl


class LocalGridder(QgsProcessingAlgorithm):
    # Parameter naming keys
    INPUT_POINTS = "INPUT_POINTS"
    VALUE_FIELD = "VALUE_FIELD"
    USE_Z = "USE_Z"
    FAULT_LINES = "FAULT_LINES"
    FAULT_POLYGONS = "FAULT_POLYGONS"

    GRID_EXTENT = "GRID_EXTENT"
    GRID_SIZE = "GRID_SIZE"
    CLIP_TO_HULL = "CLIP_TO_HULL"
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                "Horizon points (XYZ)",
                [QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.VALUE_FIELD,
                "Point value field",
                parentLayerParameterName=self.INPUT_POINTS,
                allowMultiple=False,
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.FAULT_LINES,
                "Fault traces (polylines, vertical faults)",
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.FAULT_POLYGONS,
                "Fault heave polygons (interiors become NoData)",
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(QgsProcessingParameterExtent(self.GRID_EXTENT, "Grid extent"))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GRID_SIZE,
                "Grid size (layer units)",
                QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLIP_TO_HULL,
                "Clip output to convex hull of input samples",
                False,
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, 'Output Grid'
        ))

    def name(self):
        return 'local_interpolation'

    def displayName(self):
        return 'Local Interpolation'

    def group(self):
        return 'Deterministic Methods'

    def groupId(self):
        return 'deterministic'

    def shortHelpString(self):
        return  "Interpolate to local grid nodes."

    def createInstance(self):
        return LocalGridder()

    def processAlgorithm(self, parameters, context, feedback):
        gridspec, grid_crs = get_gridspec(self, parameters, context, feedback)
        target_crs = grid_crs if grid_crs.isValid() else QgsCoordinateReferenceSystem()
        points_xyv = get_points(self, parameters, context, target_crs, feedback)
        fault_lines = get_fault_lines(self, parameters, context, target_crs, feedback)
        fault_polys = get_fault_polygons(self, parameters, context, target_crs, feedback)
        result = local_gridding_impl(
            gridspec, points_xyv, fault_lines, fault_polys, feedback
        )
        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        write_geotiff(out_path, result, gridspec, target_crs)
        feedback.setProgress(100)
        return {self.OUTPUT: out_path}
