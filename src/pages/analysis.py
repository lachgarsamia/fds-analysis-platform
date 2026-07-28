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

# Analysis-section consolidation (Phase 1 of the approved consolidation
# plan, docs: the "Analysis Section Consolidation" audit): the Phase D
# grouping below organized tabs by investigation *stage*, but left several
# tools that answer the same research question scattered across different
# groups (e.g. Compare axes/Ensemble/Ensemble analytics/Study's parallel
# coordinates were split across "Comparison"/"Study-Level" despite all four
# answering "how do scenarios compare"). This re-groups by research
# question instead, purely a navigation change -- no panel is merged,
# split, or otherwise touched here (that's Phases 2-6). Membership:
# - Overview & Interpretation: "what is happening in this simulation?"
# - Compare & Discover: "how are scenarios similar or different?" (State
#   space's genome is ensemble-normalized, i.e. inherently a
#   this-scenario-vs-the-study comparison, so it lives here too)
# - Probe & Measure: "what happens at this location/region?"
# - Factors & Sensitivity: "what drives the observed response?" (Study
#   still also hosts its parallel-coordinates tab today -- that tab
#   conceptually belongs in Compare & Discover and is planned to move
#   there in Phase 3, once it's extracted into its own widget)
# - Spatiotemporal Analysis: "how does a quantity evolve across time
#   and/or space?"
# - Reference & Communication: authoring/browsing/reporting tools that
#   aren't themselves an investigation of the simulation
# The lower-confidence/exploratory tools (Experimental, collapsed by
# default) are unchanged from Phase D.
_GROUPS = [
    ("Overview & Interpretation", ["Dashboard", "Hazard & Tenability", "Narrative"]),
    ("Compare & Discover", ["Compare axes", "Ensemble analytics", "Ensemble", "State space"]),
    ("Probe & Measure", ["Devices", "Zones", "Velocity", "Measure"]),
    ("Factors & Sensitivity", ["Study", "Sensitivity"]),
    ("Spatiotemporal Analysis", ["Height", "Time series", "Intervals", "Space-time"]),
    ("Reference & Communication", ["Calculator", "Quantities", "Graph", "Assistant", "Sessions"]),
]
# Fire MRI, Attention, Why is it hot?, and Forecasting are each individually
# gated/heuristic/exploratory (per-panel disclaimers already say so) --
# grouped together and collapsed by default rather than competing for
# attention with the core workflow.
_EXPERIMENTAL = ["Fire MRI", "Attention", "Why is it hot?", "Forecasting"]


class _CollapsibleGroup(QtWidgets.QWidget):
    """A tab-group that starts collapsed (Phase D: the Experimental group)
    -- a checkable toggle button shows/hides the inner QTabWidget, instead
    of it competing for attention with the four always-open groups."""

    def __init__(self, inner_tabs: QtWidgets.QTabWidget, parent=None):
        super().__init__(parent)
        self.tabs = inner_tabs
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = QtWidgets.QPushButton("▶ Show experimental panels")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self.toggle)
        inner_tabs.setVisible(False)
        layout.addWidget(inner_tabs, 1)

    def _on_toggled(self, checked: bool) -> None:
        self.tabs.setVisible(checked)
        self.toggle.setText("▼ Hide experimental panels" if checked
                            else "▶ Show experimental panels")

    def expand(self) -> None:
        self.toggle.setChecked(True)


class AnalysisPage(Page):
    title = "Analysis"
    # Phase D: a tab switch at ANY level (outer group, or a group's own
    # inner tab, or expanding the collapsed Experimental group) must still
    # trigger the RC-polish "resend the current selection" catch-up
    # (main_window.py's freeze-while-hidden mechanism) -- previously that
    # was one flat QTabWidget's currentChanged; grouping adds two more
    # places a panel can go from hidden to visible.
    tab_shown = QtCore.pyqtSignal()

    def __init__(self, content: QtWidgets.QWidget = None,
                 on_shown: Optional[Callable[[], None]] = None,
                 playback_bar: QtWidgets.QWidget = None,
                 forecasting_content: QtWidgets.QWidget = None,
                 timeseries_content: QtWidgets.QWidget = None,
                 fire_mri_content: QtWidgets.QWidget = None,
                 state_space_content: QtWidgets.QWidget = None,
                 attention_content: QtWidgets.QWidget = None,
                 cause_content: QtWidgets.QWidget = None,
                 height_content: QtWidgets.QWidget = None,
                 zone_content: QtWidgets.QWidget = None,
                 interval_content: QtWidgets.QWidget = None,
                 measurement_content: QtWidgets.QWidget = None,
                 advanced_compare_content: QtWidgets.QWidget = None,
                 study_content: QtWidgets.QWidget = None,
                 sensitivity_content: QtWidgets.QWidget = None,
                 hazard_tenability_content: QtWidgets.QWidget = None,
                 dashboard_content: QtWidgets.QWidget = None,
                 spacetime_content: QtWidgets.QWidget = None,
                 narrative_content: QtWidgets.QWidget = None,
                 ensemble_content: QtWidgets.QWidget = None,
                 graph_content: QtWidgets.QWidget = None,
                 calculator_content: QtWidgets.QWidget = None,
                 devices_content: QtWidgets.QWidget = None,
                 velocity_content: QtWidgets.QWidget = None,
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

        # RC polish: a shared playback transport so the temporal analysis panels
        # play/pause/step in lockstep with the Live Viewer (same clock).
        if playback_bar is not None:
            layout.addWidget(playback_bar)

        # V2 roadmap M1.1: the page grew from a two-way splitter to a tab
        # per analysis surface (PCA/clustering, time-series workspace,
        # forecasting) -- a 3+-way vertical splitter would starve every
        # pane at once. Only supplied (non-None) surfaces get a tab; demo
        # mode supplies none.
        sections = [
            ("Dashboard", dashboard_content),
            ("Hazard & Tenability", hazard_tenability_content),
            ("Narrative", narrative_content),
            ("Space-time", spacetime_content),
            ("Ensemble analytics", content),
            ("Height", height_content),
            ("Zones", zone_content),
            ("Intervals", interval_content),
            ("Measure", measurement_content),
            ("Graph", graph_content),
            ("Quantities", quantities_content),
            ("Calculator", calculator_content),
            ("Devices", devices_content),
            ("Velocity", velocity_content),
            ("Assistant", assistant_content),
            ("Sessions", sessions_content),
            ("Fire MRI", fire_mri_content),
            ("Attention", attention_content),
            ("Why is it hot?", cause_content),
            ("State space", state_space_content),
            ("Compare axes", advanced_compare_content),
            ("Study", study_content),
            ("Sensitivity", sensitivity_content),
            ("Ensemble", ensemble_content),
            ("Time series", timeseries_content),
            ("Forecasting", forecasting_content),
        ]
        by_label = dict(sections)
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
            self.tabs.currentChanged.connect(lambda _i: self.tab_shown.emit())

            def add_group(group_label: str, member_labels: list) -> None:
                members = [(lbl, by_label[lbl]) for lbl in member_labels
                          if by_label.get(lbl) is not None]
                if not members:
                    return
                inner = QtWidgets.QTabWidget()
                inner.currentChanged.connect(lambda _i: self.tab_shown.emit())
                for lbl, w in members:
                    inner.addTab(w, lbl)
                self.tabs.addTab(inner, group_label)

            for group_label, member_labels in _GROUPS:
                add_group(group_label, member_labels)

            experimental = [(lbl, by_label[lbl]) for lbl in _EXPERIMENTAL
                            if by_label.get(lbl) is not None]
            if experimental:
                inner = QtWidgets.QTabWidget()
                inner.currentChanged.connect(lambda _i: self.tab_shown.emit())
                for lbl, w in experimental:
                    inner.addTab(w, lbl)
                collapsible = _CollapsibleGroup(inner)
                collapsible.toggle.toggled.connect(
                    lambda checked: self.tab_shown.emit() if checked else None)
                self.tabs.addTab(collapsible, "Experimental")

            layout.addWidget(self.tabs, 1)

    def show_tab(self, widget: QtWidgets.QWidget) -> None:
        """Raise the tab hosting `widget` (V4-M9 comparison hand-off).
        Phase D: tabs are grouped into an outer QTabWidget of QTabWidgets
        (or, for Experimental, a collapsible wrapper around one) -- search
        one level deeper when `widget` isn't a direct child, expanding a
        collapsed group so the revealed tab is actually visible."""
        tabs = getattr(self, "tabs", None)
        if tabs is None or widget is None:
            return
        idx = tabs.indexOf(widget)
        if idx >= 0:
            tabs.setCurrentIndex(idx)
            return
        for i in range(tabs.count()):
            group = tabs.widget(i)
            inner = group if isinstance(group, QtWidgets.QTabWidget) else getattr(group, "tabs", None)
            if inner is None:
                continue
            j = inner.indexOf(widget)
            if j >= 0:
                tabs.setCurrentIndex(i)
                inner.setCurrentIndex(j)
                expand = getattr(group, "expand", None)
                if callable(expand):
                    expand()
                return

    def on_enter(self) -> None:
        if self._on_shown is not None:
            self._on_shown()
