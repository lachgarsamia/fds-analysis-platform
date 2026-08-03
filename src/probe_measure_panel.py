"""Probe & Measure workspace (Analysis section consolidation Phase 4), an
Analysis-page tab.

A thin QTabWidget wrapper, not a rewrite: hosts the existing Devices
(instrument models: thermocouple/heat-detector/sprinkler), Zones (named,
persistent region statistics + cross-scenario compare), and Velocity (true
vector-field probes) panels as three modes of one "what happens at this
location/region?" workspace, instead of same-level tabs previously
scattered in the same group with no shared framing. Every child's own
construction, store access, lazy-load convention, and SelectionBus wiring
is completely unchanged -- only the tab-level presentation is
consolidated, the same pattern already proven for Hazard & Tenability
and Compare & Discover.

(Analysis final-polish pass: the fourth former mode, "Quick probe" --
disposable, un-named rectangle/point reads via measurement_panel.py's
MeasurementPanel -- was removed. It was a second, less deliberate way to
read the same field Devices/Zones/Velocity already cover more
purposefully; measure.py, the underlying probe/rect-stats engine, stays
-- velocity.py's streamline reconstruction depends on it directly.)

Any of the three children may be absent (Velocity needs a manifest;
Devices/Zones follow the same convention) -- only supplied children get a
tab, same "only supplied surfaces get a tab" rule the outer AnalysisPage
already follows.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class ProbeMeasurePanel(QtWidgets.QWidget):
    def __init__(self, devices: QtWidgets.QWidget = None,
                 zones: QtWidgets.QWidget = None,
                 velocity: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self.devices_widget = devices
        self.zones_widget = zones
        self.velocity_widget = velocity

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        for widget, label in ((devices, "Devices"), (zones, "Zones"),
                             (velocity, "Velocity")):
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
                      self.velocity_widget):
            if widget is not None and hasattr(widget, "ensure_loaded"):
                widget.ensure_loaded()
