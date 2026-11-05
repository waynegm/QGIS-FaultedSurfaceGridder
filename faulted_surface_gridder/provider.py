from qgis.core import QgsProcessingProvider

from .algorithms.iterative_gridder import IterativeGridder
from .algorithms.local_gridder import LocalGridder


class FaultedSurfaceGridderProvider(QgsProcessingProvider):
    """Processing provider exposed in the Processing Toolbox."""

    def loadAlgorithms(self):
        """Register all algorithms for this provider."""
        self.addAlgorithm(LocalGridder())
        self.addAlgorithm(IterativeGridder())

    def id(self):
        return "faultedsurfacegridder"

    def name(self):
        return "Faulted Surface Gridding"

    def icon(self):
        return QgsProcessingProvider.icon(self)
