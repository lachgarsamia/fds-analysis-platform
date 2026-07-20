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
                 forecasting_content: QtWidgets.QWidget = None,
                 timeseries_content: QtWidgets.QWidget = None,
                 energy_content: QtWidgets.QWidget = None,
                 factor_effects_content: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self._on_shown = on_shown
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QLabel("Analysis")
        header.setProperty("role", "title")
        layout.addWidget(header)

        # V2 roadmap M1.1: the page grew from a two-way splitter to a tab
        # per analysis surface (PCA/clustering, time-series workspace,
        # forecasting) -- a 3+-way vertical splitter would starve every
        # pane at once. Only supplied (non-None) surfaces get a tab; demo
        # mode supplies none.
        sections = [
            ("Ensemble analytics", content),
            ("Factor effects", factor_effects_content),
            ("Time series", timeseries_content),
            ("Energy budget", energy_content),
            ("Forecasting", forecasting_content),
        ]
        available = [(label, w) for label, w in sections if w is not None]
        if not available:
            # Demo mode: no manifest, nothing to analyze.
            label = QtWidgets.QLabel("No experiment data available (demo mode).")
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
        elif len(available) == 1:
            layout.addWidget(available[0][1], 1)
        else:
            self.tabs = QtWidgets.QTabWidget()
            for label, w in available:
                self.tabs.addTab(w, label)
            layout.addWidget(self.tabs, 1)

    def on_enter(self) -> None:
        if self._on_shown is not None:
            self._on_shown()
