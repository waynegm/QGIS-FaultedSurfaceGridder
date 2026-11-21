import os

# QGIS/Qt must be able to start without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session", autouse=True)
def qgis_application():
    from processing.core.Processing import Processing
    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    # Algorithms that call processing.run() need the processing plugin.
    Processing.initialize()
    yield app
    app.exitQgis()
