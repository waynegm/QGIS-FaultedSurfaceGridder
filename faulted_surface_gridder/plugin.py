from qgis.core import QgsApplication

from .provider import FaultedSurfaceGridderProvider


class FaultedSurfaceGriddingPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Create and register the provider with the Processing registry."""
        self.provider = FaultedSurfaceGridderProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        # Symmetrical teardown - required to avoid duplicate providers on reload.
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
