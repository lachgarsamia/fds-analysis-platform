"""Left navigation rail (FireLab roadmap Phase 1): a vertical list of
page buttons, mutually exclusive (exactly one page active at a time), with
a collapse toggle for icon-only mode. Keyboard shortcuts (1-6) are wired
by MainWindow (same place the app's other shortcuts already live), not
here -- a rail widget shouldn't assume it owns global key handling.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from branding import build_logo_widget

EXPANDED_WIDTH = 340
COLLAPSED_WIDTH = 48
LOGO_HEIGHT = 150


class NavRail(QtWidgets.QWidget):
    """entries: [(key, label), ...] in display order, e.g.
    [("home", "Home"), ("live", "Live Viewer"), ...]."""

    page_selected = QtCore.pyqtSignal(str)  # page key
    theme_toggle_requested = QtCore.pyqtSignal()

    def __init__(self, entries: list, parent=None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self._buttons: dict = {}
        self._labels: dict = {}
        self._collapsed = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(2)

        self._logo_row = QtWidgets.QHBoxLayout()
        self._logo_row.addStretch(1)
        self._logo = build_logo_widget(LOGO_HEIGHT)
        self._logo_row.addWidget(self._logo)
        self._logo_row.addStretch(1)
        layout.addLayout(self._logo_row)
        layout.addSpacing(8)

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

        # One-click light/dark toggle, always visible regardless of which
        # page is showing -- the full View > Theme menu (light/dark/system/
        # theatre) still exists in main_window.py for the less-common
        # choices; this is just a fast path for the two everyday ones.
        self._theme_button = QtWidgets.QPushButton()
        self._theme_button.setObjectName("navThemeButton")
        self._theme_button.setAccessibleName("Toggle light or dark mode")
        self._theme_button.clicked.connect(self.theme_toggle_requested.emit)
        # Placeholder state until the real theme is known -- main_window.py
        # calls set_dark() right after construction (its own _apply_theme()
        # already runs at startup), same as it does for set_active().
        self._is_dark = True
        self._theme_icon = "☀"
        self._theme_full_label = "Light mode"
        self._theme_button.setToolTip("Switch to light mode")
        layout.addWidget(self._theme_button)

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

    def set_dark(self, is_dark: bool) -> None:
        """Reflects the *currently applied* theme (main_window.py calls this
        after every _apply_theme(), including on startup and when the View
        menu changes it) -- so the button's own icon/label always describes
        what clicking it will switch *to*, not a state it owns itself."""
        self._is_dark = is_dark
        self._theme_icon = "☀" if is_dark else "◑"  # sun : half-moon
        self._theme_full_label = "Light mode" if is_dark else "Dark mode"
        self._theme_button.setToolTip(
            "Switch to light mode" if is_dark else "Switch to dark mode")
        self._relabel()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.setFixedWidth(COLLAPSED_WIDTH if collapsed else EXPANDED_WIDTH)
        self._logo.setVisible(not collapsed)
        self._relabel()

    def _relabel(self) -> None:
        for key, button in self._buttons.items():
            full = self._labels[key]
            button.setText(full.split(None, 1)[0] if self._collapsed else full)
        self._theme_button.setText(
            self._theme_icon if self._collapsed else f"{self._theme_icon}  {self._theme_full_label}")
        self._collapse_button.setText(">>" if self._collapsed else "<<  Collapse")
