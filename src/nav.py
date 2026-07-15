"""Left navigation rail (FireLab roadmap Phase 1): a vertical list of
page buttons, mutually exclusive (exactly one page active at a time), with
a collapse toggle for icon-only mode. Keyboard shortcuts (1-6) are wired
by MainWindow (same place the app's other shortcuts already live), not
here -- a rail widget shouldn't assume it owns global key handling.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

EXPANDED_WIDTH = 168
COLLAPSED_WIDTH = 48


class NavRail(QtWidgets.QWidget):
    """entries: [(key, label), ...] in display order, e.g.
    [("home", "Home"), ("live", "Live Viewer"), ...]."""

    page_selected = QtCore.pyqtSignal(str)  # page key

    def __init__(self, entries: list, parent=None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self._buttons: dict = {}
        self._labels: dict = {}
        self._collapsed = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(2)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        for i, (key, label) in enumerate(entries, start=1):
            button = QtWidgets.QPushButton()
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setAccessibleName(f"Go to {label}")
            button.setToolTip(f"{label} (key: {i})")
            button.clicked.connect(lambda _checked, k=key: self.page_selected.emit(k))
            self._group.addButton(button)
            self._buttons[key] = button
            self._labels[key] = f"{i}  {label}"
            layout.addWidget(button)

        layout.addStretch(1)

        self._collapse_button = QtWidgets.QPushButton()
        self._collapse_button.setObjectName("navCollapseButton")
        self._collapse_button.setAccessibleName("Collapse or expand the navigation rail")
        self._collapse_button.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self._collapse_button)

        self._relabel()
        if entries:
            self._buttons[entries[0][0]].setChecked(True)
        self.setFixedWidth(EXPANDED_WIDTH)

    def set_active(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.setFixedWidth(COLLAPSED_WIDTH if collapsed else EXPANDED_WIDTH)
        self._relabel()

    def _relabel(self) -> None:
        for key, button in self._buttons.items():
            full = self._labels[key]
            button.setText(full.split(None, 1)[0] if self._collapsed else full)
        self._collapse_button.setText(">>" if self._collapsed else "<<  Collapse")
