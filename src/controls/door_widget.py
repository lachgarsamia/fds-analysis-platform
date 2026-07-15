"""Door control (FireLab roadmap Phase 3): a drop-in replacement for the
"Door opening width" ToggleGroup -- same value_changed/set_value contract.
Swings an animated door-arc icon between states (QVariantAnimation-driven
angle, ease-out cubic, ~300ms) instead of a plain checked highlight,
mirroring the physical door next to this app at the demo.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from schematic import _icon_from_painter

ICON_SIZE = 40
ANIMATION_MS = 300

# Swing angle (degrees) the door arc sweeps through per state -- wider
# opening = wider arc. Falls back to a fully-open sweep for any value not
# listed here (only meant to cover config.py's existing door values).
_STATE_ANGLES = {1: 90.0, 0: 35.0}  # 1="Wide open", 0="Narrow"


def _door_icon(color: str, angle: float, size: int) -> QtGui.QIcon:
    def draw(painter, s):
        pen = QtGui.QPen(QtGui.QColor(color), max(1.5, s * 0.06))
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        m = s * 0.2
        painter.drawLine(QtCore.QPointF(m, m), QtCore.QPointF(m, s - m))
        painter.drawArc(QtCore.QRectF(m, m, s - 2 * m, s - 2 * m), 90 * 16, -round(angle * 16))
    return _icon_from_painter(draw, size)


class DoorWidget(QtWidgets.QWidget):
    """options: [(label, value), ...] -- value is looked up in
    _STATE_ANGLES for the icon's swing angle; same value_changed/set_value
    contract as ToggleGroup."""

    value_changed = QtCore.pyqtSignal(object)

    def __init__(self, options, default_index: int = 0, accessible_name: str = "", parent=None):
        super().__init__(parent)
        self._buttons: list = []
        self._values = [v for _, v in options]
        self._color = "#9CA3AF"
        self._accent = "#4C8DFF"
        self._last_angle = _STATE_ANGLES.get(self._values[default_index] if self._values else None, 90.0)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        for i, (label, value) in enumerate(options):
            btn = QtWidgets.QPushButton(f"  {label}")
            btn.setCheckable(True)
            btn.setProperty("toggle", "true")
            btn.setIconSize(QtCore.QSize(ICON_SIZE, ICON_SIZE))
            btn.setFocusPolicy(QtCore.Qt.StrongFocus)
            if accessible_name:
                btn.setAccessibleName(f"{accessible_name}: {label}")
            btn.setIcon(_door_icon(self._color, _STATE_ANGLES.get(value, 90.0), ICON_SIZE))
            self._group.addButton(btn, i)
            layout.addWidget(btn)
            self._buttons.append(btn)

        if self._buttons:
            self._buttons[default_index].setChecked(True)

        self._group.idClicked.connect(self._on_clicked)

        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._animate_tick)
        self._anim_from = self._anim_to = self._last_angle
        self._anim_elapsed = QtCore.QElapsedTimer()

    def set_palette(self, palette) -> None:
        self._color = palette.text_secondary
        self._accent = palette.accent
        for btn, value in zip(self._buttons, self._values):
            btn.setIcon(_door_icon(self._color, _STATE_ANGLES.get(value, 90.0), ICON_SIZE))

    def _on_clicked(self, index: int) -> None:
        new_value = self._values[index]
        self._start_swing(_STATE_ANGLES.get(new_value, 90.0))
        self.value_changed.emit(new_value)

    def _start_swing(self, target_angle: float) -> None:
        self._anim_from = self._last_angle
        self._anim_to = target_angle
        self._anim_elapsed.start()
        if not self._anim_timer.isActive():
            self._anim_timer.start(16)

    def _animate_tick(self) -> None:
        t = min(1.0, self._anim_elapsed.elapsed() / ANIMATION_MS)
        eased = 1 - (1 - t) ** 3  # ease-out cubic
        angle = self._anim_from + (self._anim_to - self._anim_from) * eased
        self._last_angle = angle
        checked_btn = self._group.checkedButton()
        if checked_btn is not None:
            color = self._accent if t < 1.0 else self._color
            checked_btn.setIcon(_door_icon(color, angle, ICON_SIZE))
        if t >= 1.0:
            self._anim_timer.stop()

    @property
    def value(self):
        checked_id = self._group.checkedId()
        return self._values[checked_id] if checked_id >= 0 else None

    def set_value(self, value) -> None:
        if value in self._values:
            idx = self._values.index(value)
            self._buttons[idx].setChecked(True)
            self._start_swing(_STATE_ANGLES.get(value, 90.0))

    def set_enabled_all(self, enabled: bool) -> None:
        for b in self._buttons:
            b.setEnabled(enabled)
