import os

# QGIS/Qt must be able to start without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session", autouse=True)
def qgis_application():
    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    yield app
    app.exitQgis()
