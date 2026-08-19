"""Compare & Discover's preset shortcuts (Analysis page pruning, item 8):
jump the Live Viewer's grid into one of the dataset's curated,
already-verified side-by-side comparisons (M2.3). Used to be the
top-level "Compare" nav page (pages/compare.py, removed); folded in here
as Compare & Discover's second sub-view alongside PCA/Clustering, same
buttons/tooltips/preset mapping unchanged -- only the container changed.
Reuses the Live page's own grid/rendering machinery via the same
combo-driven signal chain a user's own clicks would trigger, through
MainWindow._apply_compare_preset -- no separate rendering path here.

Factorial-only (candle/door/vent factor axes), like the sensitivity/
factor-effects panels: MainWindow only constructs this for a factorial
study, so there's no "no manifest" empty state to render here -- absent
entirely (not a tab) is this page's existing convention for that case.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtWidgets

PRESETS = [
    ("door", "Door open vs. closed",
     "Air speed reveals the doorway's effect on airflow -- M2.3 found "
     "velocity, not temperature, shows this most clearly."),
    ("candles", "One candle vs. two",
     "Temperature side by side for a single candle versus two."),
    ("ventilation", "Ventilation strong vs. weak",
     "Temperature side by side for an open versus closed vent."),
]


class ComparePresetsPanel(QtWidgets.QWidget):
    def __init__(self, on_preset: Optional[Callable[[str], None]] = None, parent=None):
        super().__init__(parent)
        self._on_preset = on_preset

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

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

        layout.addStretch(1)

    def _apply(self, key: str) -> None:
        if self._on_preset is not None:
            self._on_preset(key)
