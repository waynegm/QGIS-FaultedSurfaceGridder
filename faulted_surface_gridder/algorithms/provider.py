"""Processing provider exposing the Faulted Surface Gridder algorithms."""

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .grid_algorithm import FaultedGridAlgorithm


class FaultedSurfaceGridderProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(FaultedGridAlgorithm())

    def id(self) -> str:
        return "faultedsurfacegridder"

    def name(self) -> str:
        return self.tr("Faulted Surface Gridder")

    def icon(self) -> QIcon:
        return QgsProcessingProvider.icon(self)
