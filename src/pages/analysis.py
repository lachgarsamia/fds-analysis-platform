"""Analysis page (FireLab roadmap Phase 4; extended into the app's
combined "analysis workspace" by the scientific-visualization completion
pass, item 7): the "Cross-Scenario Comparison" tab hosts
compare_discover_panel.py's CompareDiscoverPanel, which re-hosts the
analytics panel's existing content (analytics_panel.py's
AnalyticsPanelDock, unchanged) as one of its own modes -- see that
module's docstring for how it and the other three "how do scenarios
compare?" tools got there. The new static/playback-independent
ForecastingPanel (forecasting_panel.py) is its own tab, Experimental
group.

The one-shot background feature-index load used to be triggered by the
dock's own visibilityChanged signal (tab raised) -- a plain page has no
such signal, so on_enter() calls the supplied `on_shown` callback instead
(main_window.py wires this to the same guarded, one-shot handler,
unchanged in every other way, still triggered by the Analysis *page*
being shown -- unaffected by which tab/mode is active, so nesting the
dock's widget one level deeper inside CompareDiscoverPanel doesn't change
when this fires).

Phase 6 similarly folded Height/Time series/Time Window into
spatiotemporal_panel.py's SpatiotemporalPanel (three modes of one
"Field & Time Explorer" tab) -- see that module's docstring for why
Space-time stays a separate top-level tab instead.

Analysis UX + reliability pass removed Calculator and Sessions from
Reference & Communication entirely (not hidden -- their backends,
field_calculator.py and session_store.py respectively, are called
directly by main_window.py/quantity_provider.py independent of either
panel, and stay).
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets

from pages.base import Page

# Analysis section consolidation (docs: the "Analysis Section
# Consolidation" audit + phased plan). Phase 1 re-grouped tabs by research
# question instead of investigation stage (several tools answering the
# same question were scattered across different Phase-D groups). Phase 3
# went further for the "how are scenarios similar or different?" question:
# Compare axes/Ensemble/Ensemble analytics/Study's former parallel-
# coordinates tab are now one "Cross-Scenario Comparison" tab (four modes
# of compare_discover_panel.py's CompareDiscoverPanel) instead of four
# separate tabs; Phase 4 did the same for Devices/Zones/Velocity/Measure,
# now one "Spatial Probes" tab (probe_measure_panel.py's
# ProbeMeasurePanel). Phase 5 folded Sensitivity into Study itself as a
# sub-tab (study_panel.py), the same "thin slot, not a rewrite" pattern
# already used there for Factor effects -- so "Factors & Sensitivity" now
# has a single "Study" tab rather than two separate ones. Phase 6 did the
# same for Height/Time series/Time Window, now one "Field & Time Explorer"
# tab (spatiotemporal_panel.py's SpatiotemporalPanel); Space-time stays a
# separate top-level tab in the same group (deferred, see that module's
# docstring). Membership:
# - Overview & Interpretation: "what is happening in this simulation?"
# - Compare & Discover: "how are scenarios similar or different?" (State
#   space's genome is ensemble-normalized, i.e. inherently a
#   this-scenario-vs-the-study comparison, so it lives here too, as its
#   own standalone tab alongside Cross-Scenario Comparison)
# - Probe & Measure: "what happens at this location/region?"
# - Factors & Sensitivity: "what drives the observed response?"
# - Spatiotemporal Analysis: "how does a quantity evolve across time
#   and/or space?"
# - Reference & Communication: authoring/browsing/reporting tools that
#   aren't themselves an investigation of the simulation
# The lower-confidence/exploratory tools (Experimental, collapsed by
# default) are unchanged from Phase D.
_GROUPS = [
    ("Overview & Interpretation", ["Dashboard", "Hazard & Tenability", "Narrative"]),
    ("Compare & Discover", ["Cross-Scenario Comparison", "State space"]),
    ("Probe & Measure", ["Spatial Probes"]),
    ("Factors & Sensitivity", ["Study"]),
    ("Spatiotemporal Analysis", ["Field & Time Explorer", "Space-time"]),
    ("Reference & Communication", ["Quantities", "Graph", "Assistant"]),
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

    def __init__(self, on_shown: Optional[Callable[[], None]] = None,
                 playback_bar: QtWidgets.QWidget = None,
                 forecasting_content: QtWidgets.QWidget = None,
                 fire_mri_content: QtWidgets.QWidget = None,
                 state_space_content: QtWidgets.QWidget = None,
                 attention_content: QtWidgets.QWidget = None,
                 cause_content: QtWidgets.QWidget = None,
                 spatiotemporal_content: QtWidgets.QWidget = None,
                 probe_measure_content: QtWidgets.QWidget = None,
                 compare_discover_content: QtWidgets.QWidget = None,
                 study_content: QtWidgets.QWidget = None,
                 hazard_tenability_content: QtWidgets.QWidget = None,
                 dashboard_content: QtWidgets.QWidget = None,
                 spacetime_content: QtWidgets.QWidget = None,
                 narrative_content: QtWidgets.QWidget = None,
                 graph_content: QtWidgets.QWidget = None,
                 quantities_content: QtWidgets.QWidget = None,
                 assistant_content: QtWidgets.QWidget = None, parent=None):
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
            ("Field & Time Explorer", spatiotemporal_content),
            ("Spatial Probes", probe_measure_content),
            ("Graph", graph_content),
            ("Quantities", quantities_content),
            ("Assistant", assistant_content),
            ("Fire MRI", fire_mri_content),
            ("Attention", attention_content),
            ("Why is it hot?", cause_content),
            ("State space", state_space_content),
            ("Cross-Scenario Comparison", compare_discover_content),
            ("Study", study_content),
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
        Tabs are grouped into an outer QTabWidget of QTabWidgets, some of
        which are themselves thin workspace wrappers over further nested
        QTabWidgets (Compare & Discover/Probe & Measure/... consolidation
        phases), or a collapsible wrapper (Experimental) -- searches to
        whatever depth `widget` is actually nested at, selecting every tab
        along the path and expanding any collapsible wrapper found there,
        so the revealed tab is actually visible."""
        tabs = getattr(self, "tabs", None)
        if tabs is None or widget is None:
            return
        self._reveal_in(tabs, widget)

    @staticmethod
    def _reveal_in(container: QtWidgets.QTabWidget, widget: QtWidgets.QWidget) -> bool:
        """Recursively find `widget` inside `container` or any QTabWidget
        nested inside its tabs (directly, or via a wrapper's own `.tabs`
        attribute -- see _CollapsibleGroup and every Phase 3+ consolidation
        wrapper). Selects every tab along the path and calls each
        container's own `expand()` if it has one. Returns True if found."""
        idx = container.indexOf(widget)
        if idx >= 0:
            container.setCurrentIndex(idx)
            return True
        for i in range(container.count()):
            child = container.widget(i)
            inner = child if isinstance(child, QtWidgets.QTabWidget) else getattr(child, "tabs", None)
            if inner is None:
                continue
            if AnalysisPage._reveal_in(inner, widget):
                container.setCurrentIndex(i)
                expand = getattr(child, "expand", None)
                if callable(expand):
                    expand()
                return True
        return False

    def on_enter(self) -> None:
        if self._on_shown is not None:
            self._on_shown()
