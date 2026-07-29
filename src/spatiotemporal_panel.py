"""Spatiotemporal Analysis workspace (Analysis section consolidation Phase
6), an Analysis-page tab.

A thin QTabWidget wrapper, not a rewrite: hosts the existing Height
(vertical temperature profile + smoke layer/plume/ceiling over time),
Time series (point/line/region probe -> XY plot over time or distance),
and Time Window (whole-field spatial-mean/max over time, interval/phase
selection, before-after split) panels as three modes of one "how does a
quantity evolve across time and/or space?" workspace, instead of three
same-level tabs previously scattered in the same group with no shared
framing. Every child's own construction, store access, lazy-load
convention, and SelectionBus wiring (including the Phase 2 point/region/
interval bus publishing already added directly on each child) is
completely unchanged -- only the tab-level presentation is consolidated,
the same pattern already proven for Compare & Discover and Probe & Measure.

Space-time (SpaceTimePanel) is deliberately NOT folded in here: it answers
a related question but via a structurally different mechanism (a full 2D
heatmap cross-section with plane/offset controls and its own Temperature/
FED toggle, no frame_slider at all) -- see the Analysis Section
Consolidation audit for why it's deferred to a later increment rather than
forced into this workspace's shape. StateSpacePanel is excluded for the
same reason it was relocated to Compare & Discover in Phase 1: its genome
is ensemble-normalized, not a single-scenario spatial view.

Any of the three children may be absent -- only supplied children get a
tab, same "only supplied surfaces get a tab" rule the outer AnalysisPage
already follows.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class SpatiotemporalPanel(QtWidgets.QWidget):
    def __init__(self, height: QtWidgets.QWidget = None,
                 timeseries: QtWidgets.QWidget = None,
                 time_window: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self.height_widget = height
        self.timeseries_widget = timeseries
        self.time_window_widget = time_window

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        for widget, label in ((height, "Vertical profile"),
                             (timeseries, "Point/Region/Line probe"),
                             (time_window, "Whole-field & interval")):
            if widget is not None:
                self.tabs.addTab(widget, label)
        layout.addWidget(self.tabs, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """All children load on first show, not just the one currently in
        view -- switching modes later must never reveal a blank panel."""
        for widget in (self.height_widget, self.timeseries_widget,
                      self.time_window_widget):
            if widget is not None and hasattr(widget, "ensure_loaded"):
                widget.ensure_loaded()
