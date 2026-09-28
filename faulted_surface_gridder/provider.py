from qgis.core import QgsProcessingProvider

from .algorithms.local_gridder import LocalGridder


class FaultedSurfaceGridderProvider(QgsProcessingProvider):
    """Processing provider exposed in the Processing Toolbox."""

    def loadAlgorithms(self):
        """Register all algorithms for this provider."""
        self.addAlgorithm(LocalGridder())
        # Add additional algorithms here, e.g.:
        # self.addAlgorithm(MyOtherAlgorithm())

    def id(self):
        # Stable, lowercase, no spaces. Forms part of every algorithm id,
        # e.g. "faultedsurfacegridder:local_interpolation".
        return "faultedsurfacegridder"

    def name(self):
        return "Faulted Surface Gridding"

    def longName(self):
        return "Faulted Surface Gridding"

    def icon(self):
        # Default provider icon; replace with custom QIcon if desired.
        return QgsProcessingProvider.icon(self)
