"""Generic lazy-built placeholder page (FireLab roadmap Phase 1): reserves
a nav-rail slot for pages whose real content is later roadmap work
(Compare/Dataset/Analysis/Export re-hosting browser.py/analytics_panel.py
in Phase 4) without pulling that work into this task. Content is built on
first on_enter(), not at startup, since there's nothing to lose by
deferring a label nobody has looked at yet.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from pages.base import Page


class PlaceholderPage(Page):
    message = "Coming soon."

    def __init__(self, parent=None):
        super().__init__(parent)
        self._built = False

    def on_enter(self) -> None:
        if self._built:
            return
        self._built = True
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(self.message)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        label.setProperty("role", "title")
        layout.addWidget(label)
