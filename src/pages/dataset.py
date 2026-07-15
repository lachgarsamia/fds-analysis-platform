"""Dataset page (FireLab roadmap Phase 4): re-hosts the experiment
browser's existing content (browser.py's ExperimentBrowserDock, unchanged)
as page content instead of a QDockWidget. Built eagerly (mirrors LivePage,
not the lazy PlaceholderPage pattern) since the content already exists by
the time this page is constructed -- see main_window.py's __init__
ordering comment."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from pages.base import Page


class DatasetPage(Page):
    title = "Dataset"

    def __init__(self, content: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if content is not None:
            layout.addWidget(content)
        else:
            # Demo mode: no manifest, nothing to browse.
            label = QtWidgets.QLabel("No experiment data available (demo mode).")
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
