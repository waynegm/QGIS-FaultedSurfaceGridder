"""Faulted Surface Gridder plugin entry point."""


def classFactory(iface):  # noqa: N802 - QGIS-required name
    from .plugin import FaultedSurfaceGridderPlugin

    return FaultedSurfaceGridderPlugin(iface)
