"""Hazard & Tenability panel (Analysis-improvement roadmap Phase B), an
Analysis-page tab.

A thin mode-toggle wrapper around the pre-existing HazardPanel (map view)
and TenabilityPanel (time-to-untenable view) -- the audit found both
classify the same temperature/CO field into hazard bands via overlapping
engines, and were two separate top-level tabs for what a researcher
experiences as one investigation ("how dangerous, and when"). Neither
panel's own code changes: this only merges their presentation into one
tab with a mode switch, so both keep their full existing functionality,
SelectionBus wiring (unchanged, still generic bind_to_bus on each panel's
own scenario_combo), and honesty disclaimers exactly as before.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class HazardTenabilityPanel(QtWidgets.QWidget):
    def __init__(self, hazard_widget: QtWidgets.QWidget,
                 tenability_widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.hazard_widget = hazard_widget
        self.tenability_widget = tenability_widget

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Hazard & Tenability")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setAccessibleName("Hazard & Tenability mode")
        self.mode_combo.addItem("Map view (Safe/Warning/Critical/Untenable)")
        self.mode_combo.addItem("Time-to-untenable view")
        header.addWidget(self.mode_combo)
        layout.addLayout(header)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(hazard_widget)
        self.stack.addWidget(tenability_widget)
        layout.addWidget(self.stack, 1)

        self.mode_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """Both panels load on first show, not just the one currently in
        view -- switching modes later must never reveal a blank panel."""
        self.hazard_widget.ensure_loaded()
        self.tenability_widget.ensure_loaded()
