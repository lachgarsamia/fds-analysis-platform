"""Analysis page (FireLab roadmap Phase 4; extended into the app's
combined "analysis workspace" by the scientific-visualization completion
pass, item 7): re-hosts the analytics panel's existing content
(analytics_panel.py's AnalyticsPanelDock, unchanged) as page content
instead of a QDockWidget/tab, with the new static/playback-independent
ForecastingPanel (forecasting_panel.py) stacked below it.

The one-shot background feature-index load used to be triggered by the
dock's own visibilityChanged signal (tab raised) -- a plain page has no
such signal, so on_enter() calls the supplied `on_shown` callback instead
(main_window.py wires this to the same guarded, one-shot handler,
unchanged in every other way).
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets

from pages.base import Page


class AnalysisPage(Page):
    title = "Analysis"

    def __init__(self, content: QtWidgets.QWidget = None,
                 on_shown: Optional[Callable[[], None]] = None,
                 forecasting_content: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self._on_shown = on_shown
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QLabel("Analysis")
        header.setProperty("role", "title")
        layout.addWidget(header)

        if content is None and forecasting_content is None:
            # Demo mode: no manifest, nothing to analyze.
            label = QtWidgets.QLabel("No experiment data available (demo mode).")
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
        elif content is not None and forecasting_content is not None:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            splitter.addWidget(content)
            splitter.addWidget(forecasting_content)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            layout.addWidget(splitter, 1)
        else:
            layout.addWidget(content or forecasting_content, 1)

    def on_enter(self) -> None:
        if self._on_shown is not None:
            self._on_shown()
