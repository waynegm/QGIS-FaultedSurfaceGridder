from qgis.core import (
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
)

from .fault_blocks import assign_point_blocks, build_fault_blocks
from .gdal_helpers import rasterize_heave_interior, write_geotiff
from .helpers import (
    convex_hull_nodata_mask,
    get_fault_lines_layer,
    get_fault_polygons_layer,
    get_gridspec,
    get_points_layer,
)
from .local_gridder_impl import build_point_samples, local_gridder_impl


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
                minValue=0.0,
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
        gridspec, grid_crs = get_gridspec(self, parameters, context, feedback)

        if grid_crs.isValid():
            target_crs = grid_crs
        else:
            points_source = self.parameterAsSource(
                parameters, self.INPUT_POINTS, context
            )
            target_crs = (
                points_source.sourceCrs()
                if points_source is not None
                else QgsProject.instance().crs()
            )
            feedback.pushWarning(
                "Grid extent has no CRS; using the input points/project CRS"
            )

        z_field = self.parameterAsString(parameters, self.Z_FIELD, context)
        if not z_field:
            raise QgsProcessingException("No point value field selected")

        clip_to_hull = self.parameterAsBoolean(
            parameters, self.CLIP_TO_HULL, context
        )

        points_layer = get_points_layer(
            self, parameters, context, target_crs, feedback
        )
        if points_layer is None:
            raise QgsProcessingException("No input points layer")

        samples = build_point_samples(points_layer, z_field, feedback)

        fault_lines_layer = get_fault_lines_layer(
            self, parameters, context, target_crs, feedback
        )
        fault_polys_layer = get_fault_polygons_layer(
            self, parameters, context, target_crs, feedback
        )

        heave_mask = rasterize_heave_interior(fault_polys_layer, gridspec)
        blocks = build_fault_blocks(
            gridspec,
            fault_lines_layer,
            heave_mask,
            feedback=feedback,
        )
        point_blocks = assign_point_blocks(
            samples.xs,
            samples.ys,
            samples.fids,
            gridspec,
            blocks,
            heave_mask=heave_mask,
            feedback=feedback,
        )

        nodata_mask = blocks.barrier.copy()
        if clip_to_hull:
            nodata_mask |= convex_hull_nodata_mask(
                samples.xs, samples.ys, gridspec, feedback
            )

        result = local_gridder_impl(
            gridspec,
            samples,
            blocks.block_ids,
            point_blocks,
            nodata_mask,
            feedback,
        )

        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        write_geotiff(out_path, result, gridspec, target_crs)
        feedback.setProgress(100)
        feedback.pushInfo("Finished local gridding")
        return {self.OUTPUT: out_path}
