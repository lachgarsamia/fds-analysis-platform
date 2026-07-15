"""Vent control (FireLab roadmap Phase 3): a drop-in replacement for the
"Air vent" ToggleGroups (both VOD's 3-state and VOC's 2-state variants) --
same value_changed/set_value contract. Animated dash-offset flow arrows
when open/HVAC, a static vent otherwise, mirroring the physical vents
next to this app at the demo.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from schematic import _icon_from_painter

ICON_SIZE = 40
FLOW_FPS = 12
N_SLATS = 4

# Per-state flow speed (dash-offset step per tick) -- 0 means static
# (closed); HVAC flows faster than a plain open vent, a small but real
# visual distinction between "passively open" and "actively fan-driven",
# matching the plain-language explainer text already used for these
# controls elsewhere in the app.
_FLOW_SPEED = {"open": 0.4, "closed": 0.0, "HVAC": 0.9}


def _vent_icon(color: str, flowing: bool, phase: float, size: int) -> QtGui.QIcon:
    def draw(painter, s):
        m = s * 0.2
        pen = QtGui.QPen(QtGui.QColor(color), max(1.2, s * 0.05))
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(QtCore.QRectF(m, m, s - 2 * m, s - 2 * m))
        step = (s - 2 * m) / (N_SLATS - 1)
        for i in range(N_SLATS):
            y = m + i * step
            painter.drawLine(QtCore.QPointF(m, y), QtCore.QPointF(s - m, y))
        if flowing:
            arrow_pen = QtGui.QPen(QtGui.QColor(color), max(1.5, s * 0.06))
            arrow_pen.setStyle(QtCore.Qt.CustomDashLine)
            arrow_pen.setDashPattern([3, 3])
            arrow_pen.setDashOffset(phase * 6)
            painter.setPen(arrow_pen)
            painter.drawLine(QtCore.QPointF(s * 0.5, m), QtCore.QPointF(s * 0.5, s - m))
    return _icon_from_painter(draw, size)


class VentWidget(QtWidgets.QWidget):
    """options: [(label, value), ...]. `state_labels`: value -> one of
    "open"/"closed"/"HVAC" (schematic.py's own _VOD_STATES/_VOC_STATES
    convention), used to pick flow speed/color. Falls back to "open"
    behavior for any value not in state_labels."""

    value_changed = QtCore.pyqtSignal(object)

    def __init__(self, options, state_labels: dict, default_index: int = 0,
                 accessible_name: str = "", parent=None):
        super().__init__(parent)
        self._buttons: list = []
        self._values = [v for _, v in options]
        self._state_labels = dict(state_labels)
        self._colors = {"open": "#3DD68C", "closed": "#5B6270", "HVAC": "#F5A623"}
        self._phase = 0.0

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
        self._timer.timeout.connect(self._tick)
        self._timer.start(round(1000 / FLOW_FPS))
        self._tick()

    def _tick(self) -> None:
        self._phase += 1.0
        for btn, value in zip(self._buttons, self._values):
            state = self._state_labels.get(value, "open")
            speed = _FLOW_SPEED.get(state, 0.4)
            color = self._colors.get(state, "#3DD68C")
            btn.setIcon(_vent_icon(color, speed > 0, self._phase * speed, ICON_SIZE))

    def set_palette(self, palette) -> None:
        """Recolor per theme, reusing the exact same role mapping
        schematic.py's own room-diagram vents already use."""
        self._colors = {"open": palette.success, "closed": palette.text_disabled, "HVAC": palette.warning}
        self._tick()

    def _on_clicked(self, index: int) -> None:
        self.value_changed.emit(self._values[index])

    @property
    def value(self):
        checked_id = self._group.checkedId()
        return self._values[checked_id] if checked_id >= 0 else None

    def set_value(self, value) -> None:
        if value in self._values:
            idx = self._values.index(value)
            self._buttons[idx].setChecked(True)

    def set_enabled_all(self, enabled: bool) -> None:
        for b in self._buttons:
            b.setEnabled(enabled)
