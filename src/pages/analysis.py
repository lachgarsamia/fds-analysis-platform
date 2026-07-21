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
                 factor_effects_content: QtWidgets.QWidget = None,
                 tenability_content: QtWidgets.QWidget = None,
                 fire_mri_content: QtWidgets.QWidget = None,
                 semantic_diff_content: QtWidgets.QWidget = None,
                 query_content: QtWidgets.QWidget = None,
                 state_space_content: QtWidgets.QWidget = None,
                 attention_content: QtWidgets.QWidget = None,
                 cause_content: QtWidgets.QWidget = None,
                 height_content: QtWidgets.QWidget = None,
                 linked_content: QtWidgets.QWidget = None,
                 zone_content: QtWidgets.QWidget = None,
                 interval_content: QtWidgets.QWidget = None,
                 measurement_content: QtWidgets.QWidget = None,
                 advanced_compare_content: QtWidgets.QWidget = None,
                 study_content: QtWidgets.QWidget = None,
                 sensitivity_content: QtWidgets.QWidget = None,
                 hazard_content: QtWidgets.QWidget = None,
                 dashboard_content: QtWidgets.QWidget = None,
                 experiments_content: QtWidgets.QWidget = None,
                 quantities_content: QtWidgets.QWidget = None,
                 assistant_content: QtWidgets.QWidget = None,
                 sessions_content: QtWidgets.QWidget = None, parent=None):
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
            ("Dashboard", dashboard_content),
            ("Hazard", hazard_content),
            ("Ensemble analytics", content),
            ("Height", height_content),
            ("Zones", zone_content),
            ("Intervals", interval_content),
            ("Measure", measurement_content),
            ("Experiments", experiments_content),
            ("Quantities", quantities_content),
            ("Assistant", assistant_content),
            ("Sessions", sessions_content),
            ("Inspect moment", linked_content),
            ("Ask", query_content),
            ("Fire MRI", fire_mri_content),
            ("Attention", attention_content),
            ("Why is it hot?", cause_content),
            ("State space", state_space_content),
            ("Semantic diff", semantic_diff_content),
            ("Compare axes", advanced_compare_content),
            ("Study", study_content),
            ("Sensitivity", sensitivity_content),
            ("Factor effects", factor_effects_content),
            ("Tenability", tenability_content),
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

    def show_tab(self, widget: QtWidgets.QWidget) -> None:
        """Raise the tab hosting `widget` (V4-M9 comparison hand-off)."""
        tabs = getattr(self, "tabs", None)
        if tabs is not None and widget is not None:
            idx = tabs.indexOf(widget)
            if idx >= 0:
                tabs.setCurrentIndex(idx)

    def on_enter(self) -> None:
        if self._on_shown is not None:
            self._on_shown()
