def classFactory(iface):
    from .plugin import FaultedSurfaceGriddingPlugin

    return FaultedSurfaceGriddingPlugin(iface)
