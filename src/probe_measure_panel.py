"""Probe & Measure workspace (Analysis section consolidation Phase 4), an
Analysis-page tab.

A thin QTabWidget wrapper, not a rewrite: hosts the existing Devices
(instrument models: thermocouple/heat-detector/sprinkler), Zones (named,
persistent region statistics + cross-scenario compare), Velocity (true
vector-field probes), and Measure (disposable, un-named rectangle/point
quick reads) panels as four modes of one "what happens at this
location/region?" workspace, instead of four same-level tabs previously
scattered in the same group with no shared framing. Every child's own
construction, store access, lazy-load convention, and SelectionBus wiring
is completely unchanged -- only the tab-level presentation is
consolidated, the same pattern already proven for Hazard & Tenability,
Assistant + Ask, and Compare & Discover.

The four share a "click the map, place/define a spatial entity, read a
number" interaction and an identical session-persistence shape
(get_X()/set_X(), cached results, never recomputed on restore), but their
scientific semantics differ meaningfully enough (literal instrument
models vs. named persistent regions vs. vector reconstruction vs.
disposable single reads) that they stay four distinct child panels rather
than one shared canvas -- see the Analysis Section Consolidation audit for
why a true shared-canvas merge was explicitly left out of scope (real
regression risk to Live-Viewer-linked device markers, for a UX gain this
thin-wrapper version already mostly captures).

Any of the four children may be absent (Velocity needs a manifest;
Devices/Zones/Measure follow the same convention) -- only supplied
children get a tab, same "only supplied surfaces get a tab" rule the
outer AnalysisPage already follows.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class ProbeMeasurePanel(QtWidgets.QWidget):
    def __init__(self, devices: QtWidgets.QWidget = None,
                 zones: QtWidgets.QWidget = None,
                 velocity: QtWidgets.QWidget = None,
                 measure: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self.devices_widget = devices
        self.zones_widget = zones
        self.velocity_widget = velocity
        self.measure_widget = measure

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        for widget, label in ((devices, "Devices"), (zones, "Zones"),
                             (velocity, "Velocity"), (measure, "Quick probe")):
            if widget is not None:
                self.tabs.addTab(widget, label)
        layout.addWidget(self.tabs, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """All children load on first show, not just the one currently in
        view -- switching modes later must never reveal a blank panel."""
        for widget in (self.devices_widget, self.zones_widget,
                      self.velocity_widget, self.measure_widget):
            if widget is not None and hasattr(widget, "ensure_loaded"):
                widget.ensure_loaded()
