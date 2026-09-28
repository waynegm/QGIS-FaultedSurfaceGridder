"""Faulted Surface Gridding plugin entry point."""


def classFactory(iface):
    from .plugin import FaultedSurfaceGriddingPlugin

    return FaultedSurfaceGriddingPlugin(iface)
