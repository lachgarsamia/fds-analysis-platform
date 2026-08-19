"""Analysis page (FireLab roadmap Phase 4; extended into the app's
combined "analysis workspace" by the scientific-visualization completion
pass, item 7).

The one-shot background feature-index load used to be triggered by the
analytics dock's own visibilityChanged signal (tab raised) -- a plain page
has no such signal, so on_enter() calls the supplied `on_shown` callback
instead (main_window.py wires this to the same guarded, one-shot handler,
unchanged in every other way, still triggered by the Analysis *page*
being shown -- unaffected by which tab is active).

Phase 6 folded Height/Time series/Time Window into
spatiotemporal_panel.py's SpatiotemporalPanel (three modes of one
"Field & Time Explorer" tab) -- see that module's docstring for why
Space-time stays a separate top-level tab instead.

Analysis UX + reliability pass removed Calculator and Sessions from
Reference & Communication entirely (not hidden -- their backends,
field_calculator.py and session_store.py respectively, are called
directly by main_window.py/quantity_provider.py independent of either
panel, and stay).

Analysis final-polish pass (pre-supervisor-meeting review): every
remaining tab was re-checked against "what scientific question does this
help answer?" and several didn't clear that bar relative to their
complexity/upkeep -- State space, Ensemble spread, Parallel coordinates,
the disposable "Quick probe" measurement tool, and the Assistant
template-summary layer were removed outright rather than kept because
they already existed. Compare & Discover's former 4-mode
CompareDiscoverPanel wrapper is unwrapped into two direct tabs (Pairwise
Comparison, PCA/Clustering) now that only two of its four modes remain --
a 2-child wrapper wasn't earning its own indirection layer. (Analysis
page pruning, item 8: Pairwise Comparison/advanced_compare_panel was
later removed outright; the top-level Compare page's story-preset
buttons moved in as its replacement -- see compare_presets_panel.py.)
Reference &
Communication's former Assistant tab is replaced by "Ask"
(query_panel.py's QueryPanel, a distinct deterministic physics-query
grammar previously reachable only as a secondary mode inside the removed
Assistant wrapper) restored to its own tab.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from pages.base import Page
from tour import ANALYSIS_STEPS, ANALYSIS_SETTINGS_KEY, TourOverlay, mark_tour_completed, should_show_tour

# Analysis section consolidation (docs: the "Analysis Section
# Consolidation" audit + phased plan, later re-checked by the Analysis
# final-polish pass -- see module docstring). Phase 4 folded Devices/
# Zones/Velocity into one "Spatial Probes" tab (probe_measure_panel.py's
# ProbeMeasurePanel). Phase 5 folded Sensitivity into Study itself as a
# sub-tab (study_panel.py), the same "thin slot, not a rewrite" pattern
# already used there for Factor effects -- so "Factors & Sensitivity" now
# has a single "Study" tab rather than two separate ones (that tab's own
# sub-tabs include Correlation & outliers -- already there, not moved).
# Phase 6 did the same for Height/Time series/Time Window, now one "Field
# & Time Explorer" tab (spatiotemporal_panel.py's SpatiotemporalPanel);
# Space-time stays a separate top-level tab in the same group (deferred,
# see that module's docstring). Membership:
# - Overview & Interpretation: "what is happening in this simulation?"
# - Compare & Discover: "how are scenarios similar or different?"
# - Probe & Measure: "what happens at this location/region?"
# - Factors & Sensitivity: "what drives the observed response?"
# - Spatiotemporal Analysis: "how does a quantity evolve across time
#   and/or space?"
# Analysis page pruning, item 9 (last of Phase C): the Experimental group
# (Fire MRI, Attention, Why is it hot?, Forecasting -- each individually
# gated/heuristic/exploratory) was removed outright, panels and engines
# both (fire_mri_panel.py/signatures.py, attention_panel.py/attention.py,
# cause_panel.py/cause_explorer.py, forecasting_panel.py) -- along with
# _EXPERIMENTAL and the _CollapsibleGroup wrapper that gave it its
# collapsed-by-default toggle, since nothing else used that mechanism.
# prediction_store.py stays: it's also used independently by the
# Experiment Browser's "View model prediction" button (main_window.py's
# _open_browser_model_eval), not exclusive to the removed Forecasting tab.
#
# Reference & Communication (Quantities/Graph/Ask) was removed entirely by
# direct product decision. Quantities/Graph/Ask are still constructed in
# main_window.py and still passed into this page's own
# quantities_content/graph_content/ask_content parameters (harmless --
# add_group() below already no-ops when a group's members are all None or,
# as now, when the group itself is absent from _GROUPS) -- they're just
# never added to any tab, the same "constructed but not exposed as a UI
# surface" pattern already used for velocity_panel (see main_window.py's
# ProbeMeasurePanel construction) and the removed Study response-arrival-
# time options (study_analytics.py's RESPONSE_FIELDS).
_GROUPS = [
    ("Overview & Interpretation", ["Dashboard"]),
    ("Compare & Discover", ["Compare Presets", "PCA / Clustering"]),
    ("Probe & Measure", ["Spatial Probes"]),
    ("Factors & Sensitivity", ["Study"]),
    ("Spatiotemporal Analysis", ["Field & Time Explorer", "Space-time", "Smoke-Layer Motion"]),
]


class _PanelJumpDialog(QtWidgets.QDialog):
    """Ctrl+K-style "type to jump" (roadmap A2): a filterable list of every
    leaf panel AnalysisPage.show_tab() can already reveal, so finding a
    tool doesn't require remembering which of the 5 groups it lives under.
    Pure UI over that existing navigation primitive -- selecting an entry
    just calls show_tab(widget), the same call every other cross-
    navigation hand-off in this app already uses; nothing new is taught
    to it about how to reveal a tab."""

    def __init__(self, entries: list, parent=None):
        """entries: [(display_name, widget), ...], as built by
        AnalysisPage._collect_leaves()."""
        super().__init__(parent)
        self.setWindowTitle("Jump to panel")
        self._entries = entries
        self._selected_widget = None

        layout = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Type a panel name…")
        self.search.setAccessibleName("Filter panels")
        layout.addWidget(self.search)
        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Matching panels")
        for name, _widget in entries:
            self.list.addItem(name)
        layout.addWidget(self.list, 1)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.resize(380, 340)

        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._accept_current)
        self.list.itemActivated.connect(self._accept_current)
        self.search.setFocus()

    def _filter(self, text: str) -> None:
        needle = text.lower()
        first_visible = None
        for i in range(self.list.count()):
            item = self.list.item(i)
            visible = needle in item.text().lower()
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = i
        if first_visible is not None:
            self.list.setCurrentRow(first_visible)

    def _accept_current(self) -> None:
        item = self.list.currentItem()
        if item is None or item.isHidden():
            return
        self._selected_widget = self._entries[self.list.row(item)][1]
        self.accept()

    def selected_widget(self) -> Optional[QtWidgets.QWidget]:
        return self._selected_widget


class AnalysisPage(Page):
    title = "Analysis"
    # Phase D: a tab switch at ANY level (outer group, or a group's own
    # inner tab) must still trigger the RC-polish "resend the current
    # selection" catch-up (main_window.py's freeze-while-hidden mechanism)
    # -- previously that was one flat QTabWidget's currentChanged; grouping
    # adds one more place a panel can go from hidden to visible.
    tab_shown = QtCore.pyqtSignal()

    def __init__(self, on_shown: Optional[Callable[[], None]] = None,
                 history_bar: QtWidgets.QWidget = None,
                 settings: QtCore.QSettings = None,
                 spatiotemporal_content: QtWidgets.QWidget = None,
                 probe_measure_content: QtWidgets.QWidget = None,
                 compare_presets_content: QtWidgets.QWidget = None,
                 clustering_content: QtWidgets.QWidget = None,
                 study_content: QtWidgets.QWidget = None,
                 dashboard_content: QtWidgets.QWidget = None,
                 spacetime_content: QtWidgets.QWidget = None,
                 smoke_layer_motion_content: QtWidgets.QWidget = None,
                 graph_content: QtWidgets.QWidget = None,
                 quantities_content: QtWidgets.QWidget = None,
                 ask_content: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self._on_shown = on_shown
        self._settings = settings
        self._tour_shown = False  # session guard, same convention as LivePage's own
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QLabel("Analysis")
        header.setProperty("role", "title")
        layout.addWidget(header)

        # Breadcrumb (live-testing roadmap A1): navigation is 2-3 levels
        # deep (group tab -> panel tab, sometimes a further mode-tab owned
        # by the panel itself, e.g. Study's Factor influence/Correlation/
        # Factor effects/Sensitivity) with nothing showing where you are.
        # Only meaningful once there's more than one tab to be "in" --
        # added to the layout below, alongside self.tabs, not here.
        self._breadcrumb = QtWidgets.QLabel("")
        self._breadcrumb.setProperty("role", "caption")

        # V2 roadmap M1.1: the page grew from a two-way splitter to a tab
        # per analysis surface (PCA/clustering, time-series workspace,
        # forecasting) -- a 3+-way vertical splitter would starve every
        # pane at once. Only supplied (non-None) surfaces get a tab; demo
        # mode supplies none.
        sections = [
            ("Dashboard", dashboard_content),
            ("Space-time", spacetime_content),
            ("Field & Time Explorer", spatiotemporal_content),
            ("Smoke-Layer Motion", smoke_layer_motion_content),
            ("Spatial Probes", probe_measure_content),
            ("Graph", graph_content),
            ("Quantities", quantities_content),
            ("Ask", ask_content),
            ("Compare Presets", compare_presets_content),
            ("PCA / Clustering", clustering_content),
            ("Study", study_content),
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
            # History back/forward (roadmap A3) sits directly left of the
            # breadcrumb (A1) -- "how do I get back" and "where am I" are
            # the same question, browser-style, and this is where deep
            # navigation (this page's own 2-3 tab levels) makes them worth
            # the screen space. history_bar is None only if MainWindow
            # ever constructs this page without it (no other caller does
            # today, but this page shouldn't hard-require it).
            nav_row = QtWidgets.QHBoxLayout()
            nav_row.setContentsMargins(0, 0, 0, 0)
            nav_row.setSpacing(8)
            if history_bar is not None:
                nav_row.addWidget(history_bar)
            nav_row.addWidget(self._breadcrumb, 1)
            layout.addLayout(nav_row)
            self.tab_shown.connect(self._update_breadcrumb)

            self.tabs = QtWidgets.QTabWidget()
            self.tabs.currentChanged.connect(lambda _i: self.tab_shown.emit())

            def add_group(group_label: str, member_labels: list) -> None:
                members = [(lbl, by_label[lbl]) for lbl in member_labels
                          if by_label.get(lbl) is not None]
                if not members:
                    return
                if len(members) == 1:
                    # Same "nothing to pick between" shortcut this page's
                    # own top-level `available` list already applies for a
                    # lone group (see `elif len(available) == 1` above) --
                    # a single-member group goes straight to that member,
                    # no inner tab bar with exactly one, unpickable tab.
                    self.tabs.addTab(members[0][1], group_label)
                    return
                inner = QtWidgets.QTabWidget()
                inner.currentChanged.connect(lambda _i: self.tab_shown.emit())
                for lbl, w in members:
                    inner.addTab(w, lbl)
                self.tabs.addTab(inner, group_label)

            for group_label, member_labels in _GROUPS:
                add_group(group_label, member_labels)

            layout.addWidget(self.tabs, 1)
            self._update_breadcrumb()  # tab_shown only fires on a *change*; set the initial text now

            # Searchable panel jump (roadmap A2). WidgetWithChildrenShortcut,
            # not the default WindowShortcut context: this page shares one
            # QMainWindow with every other nav-rail page (Home, Live, ...),
            # so an unscoped shortcut would also fire while looking at a
            # completely different page.
            self._jump_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+K"), self)
            self._jump_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            self._jump_shortcut.activated.connect(self._open_panel_jump)

    def _open_panel_jump(self) -> None:
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return
        entries = self._collect_leaves(tabs)
        if not entries:
            return
        dialog = _PanelJumpDialog(entries, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            widget = dialog.selected_widget()
            if widget is not None:
                self.show_tab(widget)

    @staticmethod
    def _collect_leaves(container: QtWidgets.QTabWidget, prefix: str = "") -> list:
        """[(display_name, widget), ...] for every leaf show_tab() can
        reveal -- the enumerate-everything counterpart to _reveal_in's
        find-one-thing, walking the identical structure: a group's own
        inner QTabWidget, or a further sub-tab QTabWidget a panel exposes as
        its own `.tabs` (Study, Spatial Probes, Field & Time Explorer all
        do -- picked up here for free, not hardcoded per panel).
        display_name is "Group › Panel" (and one level deeper for a panel's
        own sub-tabs),
        so two identically-named leaves in different places -- none exist
        today -- would still read as distinct entries."""
        out = []
        for i in range(container.count()):
            label = container.tabText(i)
            child = container.widget(i)
            full = f"{prefix} › {label}" if prefix else label
            inner = child if isinstance(child, QtWidgets.QTabWidget) else getattr(child, "tabs", None)
            if isinstance(inner, QtWidgets.QTabWidget):
                out.extend(AnalysisPage._collect_leaves(inner, full))
            else:
                out.append((full, child))
        return out

    def _update_breadcrumb(self) -> None:
        """"Group -> Panel" from the outer/inner QTabWidgets' own current
        state -- tab_shown carries no payload, so this reads whichever tab
        is actually current rather than tracking it independently (can't
        drift out of sync with what's really showing). Deeper levels a
        panel owns internally (e.g. Study's own Factor influence/
        Correlation/Factor effects/Sensitivity sub-tabs) aren't reflected
        here -- doing that would mean reaching into each panel's own
        widget tree, which tab_shown's existing emit sites don't cover."""
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return
        group_idx = tabs.currentIndex()
        if group_idx < 0:
            return
        group_label = tabs.tabText(group_idx)
        content = tabs.currentWidget()
        panel_label = None
        if isinstance(content, QtWidgets.QTabWidget):
            panel_label = content.tabText(content.currentIndex())
        self._breadcrumb.setText(f"{group_label} › {panel_label}" if panel_label else group_label)

    def show_tab(self, widget: QtWidgets.QWidget) -> None:
        """Raise the tab hosting `widget` (V4-M9 comparison hand-off).
        Tabs are grouped into an outer QTabWidget of QTabWidgets, some of
        which are themselves thin workspace wrappers over further nested
        QTabWidgets (Compare & Discover/Probe & Measure/... consolidation
        phases) -- searches to whatever depth `widget` is actually nested
        at, selecting every tab along the path, so the revealed tab is
        actually visible."""
        tabs = getattr(self, "tabs", None)
        if tabs is None or widget is None:
            return
        self._reveal_in(tabs, widget)

    @staticmethod
    def _reveal_in(container: QtWidgets.QTabWidget, widget: QtWidgets.QWidget) -> bool:
        """Recursively find `widget` inside `container` or any QTabWidget
        nested inside its tabs (directly, or via a wrapper's own `.tabs`
        attribute -- see every Phase 3+ consolidation wrapper). Selects
        every tab along the path and calls each container's own `expand()`
        if it has one (no wrapper defines one today; kept generic rather
        than removed since it costs nothing unused). Returns True if
        found."""
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
        self._maybe_show_tour()

    def _maybe_show_tour(self) -> None:
        """First-run coach-mark (roadmap A4) -- same should_show_tour/
        mark_tour_completed/TourOverlay mechanism as the Live page's own
        tour (tour.py), pointed at this page's own ANALYSIS_STEPS and
        recorded under its own ANALYSIS_SETTINGS_KEY, so dismissing this
        one never marks the Live tour seen (or vice versa)."""
        if self._tour_shown or self._settings is None:
            return
        if not should_show_tour(self._settings, ANALYSIS_SETTINGS_KEY):
            return
        self._tour_shown = True
        overlay = TourOverlay(self, steps=ANALYSIS_STEPS)
        overlay.finished.connect(lambda: mark_tour_completed(self._settings, ANALYSIS_SETTINGS_KEY))
