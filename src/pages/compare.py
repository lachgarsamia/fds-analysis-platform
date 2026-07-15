"""Compare page (FireLab roadmap Phase 4): curated "story presets" that
jump into the Live Viewer's grid pre-configured to show one of the
dataset's honest, already-verified findings (M2.3) side by side. Reuses
the Live page's own grid/rendering machinery via the same combo-driven
signal chain a user's own clicks would trigger -- no separate rendering
path to build or maintain.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtWidgets

from pages.base import Page

# (key, button label, tooltip). Kept here (not main_window.py) since it's
# purely presentational; main_window._COMPARE_PRESETS carries the actual
# factor/quantity mapping each key resolves to.
PRESETS = [
    ("door", "Door open vs. closed",
     "Air speed reveals the doorway's effect on airflow -- M2.3 found "
     "velocity, not temperature, shows this most clearly."),
    ("candles", "One candle vs. two",
     "Temperature side by side for a single candle versus two."),
    ("ventilation", "Ventilation strong vs. weak",
     "Temperature side by side for an open versus closed vent."),
]


class ComparePage(Page):
    title = "Compare"

    def __init__(self, on_preset: Optional[Callable[[str], None]] = None, parent=None):
        super().__init__(parent)
        self._on_preset = on_preset

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Compare")
        title.setProperty("role", "title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Pick a story preset to see two scenarios side by side in the Live Viewer.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self._buttons = []
        for key, label, description in PRESETS:
            button = QtWidgets.QPushButton(label)
            button.setObjectName("primaryButton")
            button.setToolTip(description)
            button.clicked.connect(lambda _checked, k=key: self._apply(k))
            layout.addWidget(button)
            self._buttons.append(button)

        self._empty_label = QtWidgets.QLabel(
            "No experiment data available (demo mode) -- story presets need the real dataset.")
        self._empty_label.setWordWrap(True)
        self._empty_label.hide()
        layout.addWidget(self._empty_label)

        layout.addStretch(1)

    def set_available(self, available: bool) -> None:
        """No manifest (demo mode) means no scenarios to compare."""
        for b in self._buttons:
            b.setVisible(available)
        self._empty_label.setVisible(not available)

    def _apply(self, key: str) -> None:
        if self._on_preset is not None:
            self._on_preset(key)
