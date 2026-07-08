######################################
# FDS SLCF Visualizer
# FZ-Juelich - Max Boehler, Lukas Arnold
# Layered UI: resizable layout, theming, accessibility, lazy data loading
######################################

import os
import sys
import logging

from PyQt5 import QtCore, QtGui, QtWidgets

from data_provider import load_simulation_data, DataLoadError
from main_window import MainWindow
from config import DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

LOGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo")


def _icon_or_none(filename: str) -> QtGui.QIcon:
    path = os.path.join(LOGO_DIR, filename)
    return QtGui.QIcon(path) if os.path.exists(path) else QtGui.QIcon()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FDS SLCF Visualizer")

    splash_path = os.path.join(LOGO_DIR, "fds_vis.png")
    if os.path.exists(splash_path):
        splash_pixmap = QtGui.QPixmap(splash_path)
        splash = QtWidgets.QSplashScreen(splash_pixmap, QtCore.Qt.WindowStaysOnTopHint)
        progress = QtWidgets.QProgressBar(splash)
        progress.setGeometry(0, splash_pixmap.height() - 22, splash_pixmap.width(), 20)
        progress.setMaximum(1)
        splash.setWindowIcon(_icon_or_none("fds_vis_small.png"))
        splash.show()
        app.processEvents()
    else:
        splash = None
        progress = None

    try:
        sim_data = load_simulation_data()
        # Warm the cache for the scenario the UI opens on, so the first frame
        # is ready the instant the window appears (only ~1-2s for one
        # scenario, vs. the ~37s the old eager loader took for all of them).
        initial_case = sim_data.data_matrix[DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC]
        sim_data.store.get(initial_case)
        if progress is not None:
            progress.setValue(1)
            app.processEvents()
    except DataLoadError as e:
        logger.error("data load failed: %s (%s)", e.message, e.technical_detail)
        if splash is not None:
            splash.close()
        QtWidgets.QMessageBox.critical(
            None,
            "Could not load simulation data",
            f"{e.message}\n\nDetails:\n{e.technical_detail}",
        )
        sys.exit(1)

    window = MainWindow(sim_data)
    window.setWindowIcon(_icon_or_none("fds_vis_small.png"))

    if splash is not None:
        splash.close()

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
