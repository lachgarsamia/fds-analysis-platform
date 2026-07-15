"""Candle cards (FireLab roadmap Phase 3): a drop-in replacement for the
"Number of candles" ToggleGroup -- same value_changed/set_value contract,
so main_window.py's wiring and every existing test that calls
candle_toggle.set_value()/.toolTip() keeps working unchanged, but each
option now shows big, gently flickering flame icons instead of plain
text, mirroring the physical candles next to this app at the demo.
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from schematic import _flame_path, _icon_from_painter

FLICKER_FPS = 8
ICON_SIZE = 40


def _jittered_flame_icon(color: str, n_flames: int, size: int, rng: random.Random) -> QtGui.QIcon:
    def draw(painter, s):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(color))
        positions = [s / 2] if n_flames == 1 else [s * 0.34, s * 0.66]
        for cx in positions:
            jitter = 1.0 + rng.uniform(-0.08, 0.08)
            r = s * 0.16 * jitter
            painter.drawPath(_flame_path(cx, s * 0.62, r))
    return _icon_from_painter(draw, size)


class CandleCard(QtWidgets.QWidget):
    """options: [(label, value), ...] -- same shape as ToggleGroup. Truthy
    `value` means 2 flames, falsy means 1, matching this app's existing
    "Number of candles" convention ([("1 candle", 0), ("2 candles", 1)])."""

    value_changed = QtCore.pyqtSignal(object)

    def __init__(self, options: Sequence[Tuple[str, object]], default_index: int = 0,
                 accessible_name: str = "", parent=None):
        super().__init__(parent)
        self._buttons: list = []
        self._values = [v for _, v in options]
        self._n_flames = [2 if v else 1 for v in self._values]
        self._rng = random.Random(0)
        self._color = "#B33A3A"

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        for i, (label, _value) in enumerate(options):
            btn = QtWidgets.QPushButton(f"  {label}")
            btn.setCheckable(True)
            btn.setProperty("toggle", "true")
            btn.setIconSize(QtCore.QSize(ICON_SIZE, ICON_SIZE))
            btn.setFocusPolicy(QtCore.Qt.StrongFocus)
            if accessible_name:
                btn.setAccessibleName(f"{accessible_name}: {label}")
            self._group.addButton(btn, i)
            layout.addWidget(btn)
            self._buttons.append(btn)

        if self._buttons:
            self._buttons[default_index].setChecked(True)

        self._group.idClicked.connect(self._on_clicked)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._flicker)
        self._timer.start(round(1000 / FLICKER_FPS))
        self._flicker()

    def _flicker(self) -> None:
        for btn, n_flames in zip(self._buttons, self._n_flames):
            btn.setIcon(_jittered_flame_icon(self._color, n_flames, ICON_SIZE, self._rng))

    def set_palette(self, palette) -> None:
        """Recolor for the active theme (ROADMAP §4 M1.6.4's per-theme
        icon legibility pass) -- flame color follows palette.danger, the
        same role schematic.py's own room-diagram candles already use."""
        self._color = palette.danger
        self._flicker()

    def _on_clicked(self, index: int) -> None:
        self.value_changed.emit(self._values[index])

    @property
    def value(self):
        checked_id = self._group.checkedId()
        return self._values[checked_id] if checked_id >= 0 else None

    def set_value(self, value) -> None:
        """Programmatic selection without re-emitting the signal (mirrors
        ToggleGroup.set_value)."""
        if value in self._values:
            idx = self._values.index(value)
            self._buttons[idx].setChecked(True)

    def set_enabled_all(self, enabled: bool) -> None:
        for b in self._buttons:
            b.setEnabled(enabled)
