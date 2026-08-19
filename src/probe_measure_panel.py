"""Probe & Measure workspace (Analysis section consolidation Phase 4), an
Analysis-page tab.

A thin QTabWidget wrapper, not a rewrite: hosts the existing Devices
(instrument models: thermocouple/heat-detector/sprinkler) and Velocity
(true vector-field probes) panels as modes of one "what happens at this
location/region?" workspace, instead of same-level tabs previously
scattered in the same group with no shared framing. Every child's own
construction, store access, lazy-load convention, and SelectionBus wiring
is completely unchanged -- only the tab-level presentation is
consolidated, the same pattern already proven for Hazard & Tenability
and Compare & Discover.

(Analysis final-polish pass: the fourth former mode, "Quick probe" --
disposable, un-named rectangle/point reads via measurement_panel.py's
MeasurementPanel -- was removed. It was a second, less deliberate way to
read the same field Devices/Velocity already cover more purposefully;
measure.py, the underlying probe/rect-stats engine, stays --
velocity.py's streamline reconstruction depends on it directly.

Analysis page pruning: the Zones mode/tab was removed too, panel and
all -- zone_stats.py's computation engine and Zone data stay (now held
headlessly by main_window.zone_panel, now a ZoneStore not a widget,
instead of a UI panel), since
graph_panel.py's Knowledge Graph and context.py's Context-Panel data
layer still read saved zones; there is no UI left to create one.)

Any child may be absent (Velocity needs a manifest; Devices follows the
same convention) -- only supplied children get a tab, same "only
supplied surfaces get a tab" rule the outer AnalysisPage already follows.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class ProbeMeasurePanel(QtWidgets.QWidget):
    def __init__(self, devices: QtWidgets.QWidget = None,
                 velocity: QtWidgets.QWidget = None,
                 streamlines: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self.devices_widget = devices
        self.velocity_widget = velocity
        # Velocity streamlines (matplotlib streamplot, a second, independent
        # visualization of the same U/W-VELOCITY field the Velocity tab's
        # quiver already draws) -- a sibling tab, not a change to Velocity
        # itself, so the two are directly next to each other for visual
        # comparison (see streamline_panel.py's module docstring).
        self.streamlines_widget = streamlines

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        for widget, label in ((devices, "Devices"),
                             (velocity, "Velocity"),
                             (streamlines, "Velocity (Streamlines)")):
            if widget is not None:
                self.tabs.addTab(widget, label)
        layout.addWidget(self.tabs, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """All children load on first show, not just the one currently in
        view -- switching modes later must never reveal a blank panel."""
        for widget in (self.devices_widget,
                      self.velocity_widget, self.streamlines_widget):
            if widget is not None and hasattr(widget, "ensure_loaded"):
                widget.ensure_loaded()
