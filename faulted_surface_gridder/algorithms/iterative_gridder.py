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

class IterativeGridder(QgsProcessingAlgorithm):
    INPUT_POINTS = "INPUT_POINTS"
    Z_FIELD = "Z_FIELD"
    FAULT_LINES = "FAULT_LINES"
    FAULT_POLYGONS = "FAULT_POLYGONS"
    GRID_EXTENT = "GRID_EXTENT"
    GRID_STEP = "GRID_STEP"
    CLIP_TO_HULL = "CLIP_TO_HULL"
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
        return  "Grid input data using an iterative method."

    def createInstance(self):
        return IterativeGridder()

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
                self.Z_FIELD,
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
                self.GRID_STEP,
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

    def processAlgorithm(self, parameters, context, feedback):

        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
#        write_geotiff(out_path, result, gridspec, target_crs)
        feedback.setProgress(100)
        feedback.pushInfo("Finished iterative gridding")
        return {self.OUTPUT: out_path}
