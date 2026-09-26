"""Main plugin class — registers the Processing provider."""

from qgis.core import QgsApplication

from .processing.provider import FaultedSurfaceGridderProvider


class FaultedSurfaceGridderPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = FaultedSurfaceGridderProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
