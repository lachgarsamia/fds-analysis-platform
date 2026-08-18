"""Left navigation rail (FireLab roadmap Phase 1): a vertical list of
page buttons, mutually exclusive (exactly one page active at a time).

UI overhaul (global chrome pass): hover-to-reveal, not a manual collapse
button. The rail defaults to a slim icon/number-only strip and expands to
full width for as long as the mouse is over it (enterEvent/leaveEvent
below), collapsing again the instant the mouse leaves -- there's no
persisted "collapsed" preference to restore because there's nothing to
remember, the rail is always slim except while actively hovered.

This widget only owns its own width (via setFixedWidth in _set_expanded);
it does NOT position itself on screen. main_window.py parents it directly
onto page_stack *without* adding it to a layout, so it floats above page
content rather than sharing space with it, and repositions/raises it via
its own _layout_nav_rail() on resize and on this widget's expanded_changed
signal. Keyboard shortcuts (1-6) are wired by MainWindow (same place the
app's other shortcuts already live), not here -- a rail widget shouldn't
assume it owns global key handling.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from branding import build_partner_logos_widget

EXPANDED_WIDTH = 340
COLLAPSED_WIDTH = 48
LOGO_HEIGHT = 150


class NavRail(QtWidgets.QWidget):
    """entries: [(key, label), ...] in display order, e.g.
    [("home", "Home"), ("live", "Live Viewer"), ...]."""

    page_selected = QtCore.pyqtSignal(str)  # page key
    theme_toggle_requested = QtCore.pyqtSignal()
    expanded_changed = QtCore.pyqtSignal(bool)  # hover state, not a persisted preference

    def __init__(self, entries: list, parent=None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self._buttons: dict = {}
        self._labels: dict = {}
        self._expanded = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(2)

        self._logo_row = QtWidgets.QHBoxLayout()
        self._logo_row.addStretch(1)
        self._logo = build_partner_logos_widget(LOGO_HEIGHT)
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

        self._relabel()
        if entries:
            self._buttons[entries[0][0]].setChecked(True)
        # Starts slim; enterEvent expands it for as long as the mouse stays
        # over it. setFixedWidth (not min/max) because there's no drag-resize
        # anymore -- a transient hover flyout doesn't need a user-tunable
        # width the way a permanently-allocated pane did.
        self.setFixedWidth(COLLAPSED_WIDTH)

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
        self._logo.set_dark(is_dark)
        self._relabel()

    def enterEvent(self, event) -> None:
        self._set_expanded(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_expanded(False)
        super().leaveEvent(event)

    def _set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        width = EXPANDED_WIDTH if expanded else COLLAPSED_WIDTH
        self.setFixedWidth(width)
        self.resize(width, self.height())  # not under a layout, so the fixed-width change alone won't move pixels
        self._logo.setVisible(expanded)
        self._relabel()
        self.expanded_changed.emit(expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def _relabel(self) -> None:
        for key, button in self._buttons.items():
            full = self._labels[key]
            # The number prefix (not an icon) is what keeps the active page
            # identifiable in the slim rail -- combined with the :checked
            # QSS accent background (theme.py), which applies regardless of
            # width, so the active page stays visible without hovering.
            button.setText(full if self._expanded else full.split(None, 1)[0])
        self._theme_button.setText(
            f"{self._theme_icon}  {self._theme_full_label}" if self._expanded else self._theme_icon)
