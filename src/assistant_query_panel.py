"""Assistant & Ask panel (Analysis-improvement roadmap Phase B), an
Analysis-page tab.

A thin mode-toggle wrapper around the pre-existing AssistantPanel (guided
actions + free-text chat over the bounded assistant engine) and QueryPanel
(deterministic physics-query grammar over query_engine) -- the audit found
both were separate "ask something, get a computed answer" surfaces, which a
researcher experiences as one Q&A workflow. Neither panel's own code
changes: this only merges their presentation into one tab with a mode
switch, so both keep their full existing functionality and SelectionBus
wiring (QueryPanel's own scenario_combo stays generically bus-bound;
AssistantPanel has no bus wiring of its own, unchanged either way).
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class AssistantQueryPanel(QtWidgets.QWidget):
    def __init__(self, assistant_widget: QtWidgets.QWidget,
                 query_widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.assistant_widget = assistant_widget
        self.query_widget = query_widget

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Assistant")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setAccessibleName("Assistant mode")
        self.mode_combo.addItem("Guided (summaries, findings, captions)")
        self.mode_combo.addItem("Ask (physics query grammar)")
        header.addWidget(self.mode_combo)
        layout.addLayout(header)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(assistant_widget)
        self.stack.addWidget(query_widget)
        layout.addWidget(self.stack, 1)

        self.mode_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """The query panel loads on first show too, not just when the mode
        is switched to it -- switching modes later must never reveal a
        blank panel."""
        if hasattr(self.query_widget, "ensure_loaded"):
            self.query_widget.ensure_loaded()
