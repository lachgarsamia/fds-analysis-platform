"""
schematic.py
------------
A live, proportionally-accurate top-down room diagram plus small category
icons, built for the M1.6 non-specialist usability pass (ROADMAP §0's
2026-07-09c entry, §4 M1.6, trade-off §7.10).

Design decisions this module encodes:

- Room *proportions* (the outline's aspect ratio) come from the mesh
  extents already parsed from the scenario's .smv file via
  `fds.slice.slice.readMeshes` -- the same extent data M2.6's probe feature
  will use, and the same data path M1.3s's fdsreader cross-validation
  checks. No new data path or measurement input is required to start.
- Door/vent/candle *positions inside* the outline are fixed proportional
  placements, not extent-derived -- per-object coordinates aren't
  recoverable from mesh bounding-box extents alone (that level of fidelity
  is M2.6's on-heatmap overlay work, explicitly out of scope here per the
  M1.6 scope boundary). This is a schematic, not a precise floor plan.
- A manual override hook exists for when the physical mockup yields exact
  measured dimensions later; it is a config constant, not a GUI feature.
"""

import os
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

import fds.slice.slice as fds_slice

# Fallback room footprint (meters), used only when no real .smv is available
# (e.g. demo-data mode). Matches the actual dataset's measured footprint
# (x: 0.0-1.0 m, y: -0.15-0.15 m, confirmed identical across every sampled
# scenario folder) rather than an arbitrary guess.
DEFAULT_ROOM_EXTENT = {"x": (0.0, 1.0), "y": (-0.15, 0.15)}

# Config-level manual override hook (2026-07-09c, ROADMAP §7.10 / task 1.6.1):
# once the physical mockup yields exact measured dimensions, set this to
# {"x": (min, max), "y": (min, max)} to override the parsed .smv extent.
# Intentionally not exposed anywhere in the GUI -- a future refinement
# point, not a feature to ship yet.
MANUAL_ROOM_EXTENT_OVERRIDE: Optional[dict] = None


def read_room_extent(root_dir: str) -> Optional[dict]:
    """Combined (x, y) bounding box across all mesh blocks in root_dir's .smv.

    Returns None if no .smv file is found rather than raising -- a missing
    geometry input should degrade the schematic to a fallback footprint,
    not crash the GUI.
    """
    smv_name = fds_slice.scanDirectory(root_dir)
    if smv_name is None:
        return None

    mesh_collection = fds_slice.readMeshes(os.path.join(root_dir, smv_name))
    if not mesh_collection.meshes:
        return None

    x_min = min(m.ranges[0][0] for m in mesh_collection.meshes)
    x_max = max(m.ranges[0][1] for m in mesh_collection.meshes)
    y_min = min(m.ranges[1][0] for m in mesh_collection.meshes)
    y_max = max(m.ranges[1][1] for m in mesh_collection.meshes)
    return {"x": (x_min, x_max), "y": (y_min, y_max)}


def resolve_room_extent(store, case_index: int) -> dict:
    """Best-available room extent: manual override > parsed .smv > fallback.

    `store` is whatever SimulationController is wired to -- a real
    ScenarioStore (has `.folders`) or DemoScenarioStore (doesn't). The mesh
    domain is identical across every scenario in this dataset (verified by
    spot-checking c1_d0_vod0_voc0 / c1_d1_vod0_voc0 / c2_d0_vod2_voc1), so
    callers only need to resolve this once at startup, not per toggle change.
    """
    if MANUAL_ROOM_EXTENT_OVERRIDE is not None:
        return MANUAL_ROOM_EXTENT_OVERRIDE

    folders = getattr(store, "folders", None)
    if folders and 0 <= case_index < len(folders):
        extent = read_room_extent(folders[case_index])
        if extent is not None:
            return extent

    return DEFAULT_ROOM_EXTENT


# --------------------------------------------------------------------- icons

def _icon_from_painter(draw_fn, size: int = 32) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    draw_fn(painter, size)
    painter.end()
    return QtGui.QIcon(pixmap)


def flame_icon(color: str, size: int = 32) -> QtGui.QIcon:
    def draw(painter, s):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(color))
        painter.drawPath(_flame_path(s / 2, s * 0.58, s * 0.32))
    return _icon_from_painter(draw, size)


def door_icon(color: str, size: int = 32) -> QtGui.QIcon:
    def draw(painter, s):
        pen = QtGui.QPen(QtGui.QColor(color), max(1.5, s * 0.06))
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        m = s * 0.2
        painter.drawLine(QtCore.QPointF(m, m), QtCore.QPointF(m, s - m))
        painter.drawArc(QtCore.QRectF(m, m, s - 2 * m, s - 2 * m), 0, 90 * 16)
    return _icon_from_painter(draw, size)


def vent_icon(color: str, size: int = 32) -> QtGui.QIcon:
    def draw(painter, s):
        pen = QtGui.QPen(QtGui.QColor(color), max(1.2, s * 0.05))
        painter.setPen(pen)
        m = s * 0.2
        n_slats = 4
        step = (s - 2 * m) / (n_slats - 1)
        for i in range(n_slats):
            y = m + i * step
            painter.drawLine(QtCore.QPointF(m, y), QtCore.QPointF(s - m, y))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(QtCore.QRectF(m, m, s - 2 * m, s - 2 * m))
    return _icon_from_painter(draw, size)


def _flame_path(cx: float, cy: float, r: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.moveTo(cx, cy - r * 1.4)
    path.cubicTo(cx + r, cy - r * 0.4, cx + r * 0.6, cy + r * 0.8, cx, cy + r)
    path.cubicTo(cx - r * 0.6, cy + r * 0.8, cx - r, cy - r * 0.4, cx, cy - r * 1.4)
    return path


# ---------------------------------------------------------------- the widget

_VOD_STATES = {0: "open", 1: "closed", 2: "HVAC"}
_VOC_STATES = {0: "open", 1: "closed"}


class SchematicWidget(QtWidgets.QWidget):
    """Top-down room diagram: outline, door, vents, candle(s).

    Pure presentation. State arrives via update_state()/set_room_extent()/
    apply_palette() -- nothing here duplicates SimulationController's
    SimulationParameters, it only mirrors the last values MainWindow pushed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("Scenario schematic")
        self._extent = DEFAULT_ROOM_EXTENT
        self._candles = 0   # 0 -> 1 candle, 1 -> 2 candles (ToggleGroup values)
        self._door = 1      # 1 -> wide open, 0 -> narrow
        self._vod = 0       # 0 open, 1 closed, 2 HVAC
        self._voc = 0       # 0 open, 1 closed
        self._palette = None
        self.setMinimumHeight(90)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._update_accessible_description()

    # -- state in -----------------------------------------------------------
    def set_room_extent(self, extent: dict):
        self._extent = extent
        self.updateGeometry()
        self.update()

    def apply_palette(self, palette):
        self._palette = palette
        self.update()

    def update_state(self, candles: int, door: int, vod: int, voc: int):
        self._candles = candles
        self._door = door
        self._vod = vod
        self._voc = voc
        self._update_accessible_description()
        self.update()

    def _update_accessible_description(self):
        n_candles = 2 if self._candles == 1 else 1
        self.setAccessibleDescription(
            "Room diagram: door {}, air vent 1 {}, air vent 2 {}, {} candle{}.".format(
                "wide open" if self._door == 1 else "narrow",
                _VOD_STATES.get(self._vod, "?"),
                _VOC_STATES.get(self._voc, "?"),
                n_candles, "s" if n_candles > 1 else "",
            )
        )

    # -- sizing ---------------------------------------------------------------
    def _aspect(self) -> float:
        x0, x1 = self._extent["x"]
        y0, y1 = self._extent["y"]
        return (y1 - y0) / (x1 - x0) if x1 > x0 else 0.3

    def sizeHint(self):
        width = max(self.width(), 260)
        return QtCore.QSize(width, max(80, int(width * self._aspect()) + 28))

    def heightForWidth(self, width: int) -> int:
        return max(70, int(width * self._aspect()) + 28)

    # -- painting -------------------------------------------------------------
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QtGui.QPainter):
        palette = self._palette
        if palette is None:
            return

        margin = 14
        rect = self.rect().adjusted(margin, margin, -margin, -margin)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        aspect = self._aspect()
        if rect.width() * aspect <= rect.height():
            draw_w = rect.width()
            draw_h = rect.width() * aspect
        else:
            draw_h = rect.height()
            draw_w = rect.height() / aspect if aspect > 0 else rect.width()
        draw_x = rect.x() + (rect.width() - draw_w) / 2
        draw_y = rect.y() + (rect.height() - draw_h) / 2
        room_rect = QtCore.QRectF(draw_x, draw_y, draw_w, draw_h)

        # --- room outline, sized to the extent-derived aspect ratio ---------
        painter.setPen(QtGui.QPen(QtGui.QColor(palette.border_strong), 2))
        painter.setBrush(QtGui.QColor(palette.bg_sunken))
        painter.drawRect(room_rect)

        # --- door: gap in the bottom wall, width by door state --------------
        door_frac = 0.34 if self._door == 1 else 0.16
        door_w = draw_w * door_frac
        door_cx = room_rect.left() + draw_w * 0.22
        door_y = room_rect.bottom()
        painter.setPen(QtGui.QPen(QtGui.QColor(palette.bg_base), 3))
        painter.drawLine(QtCore.QPointF(door_cx - door_w / 2, door_y),
                          QtCore.QPointF(door_cx + door_w / 2, door_y))
        accent = QtGui.QColor(palette.accent)
        painter.setPen(QtGui.QPen(accent, 1.5, QtCore.Qt.DashLine))
        painter.setBrush(QtCore.Qt.NoBrush)
        swing_rect = QtCore.QRectF(door_cx - door_w / 2, door_y - door_w, door_w, door_w)
        painter.drawArc(swing_rect, 0, 90 * 16)

        # --- vents: air vent 1 (VOD) + air vent 2 (VOC) on the right wall ----
        vent_colors = {"open": palette.success, "closed": palette.text_disabled, "HVAC": palette.warning}
        vod_label = _VOD_STATES[self._vod]
        voc_label = _VOC_STATES[self._voc]

        vent_w, vent_h = draw_w * 0.05, draw_h * 0.22
        painter.setPen(QtGui.QPen(QtGui.QColor(palette.border_strong), 1))

        vod_rect = QtCore.QRectF(room_rect.right() - vent_w / 2,
                                  room_rect.top() + draw_h * 0.15, vent_w, vent_h)
        painter.setBrush(QtGui.QColor(vent_colors[vod_label]))
        painter.drawRect(vod_rect)

        voc_rect = QtCore.QRectF(room_rect.right() - vent_w / 2,
                                  room_rect.bottom() - draw_h * 0.15 - vent_h, vent_w, vent_h)
        painter.setBrush(QtGui.QColor(vent_colors[voc_label]))
        painter.drawRect(voc_rect)

        # --- candle(s) ---------------------------------------------------------
        n_candles = 2 if self._candles == 1 else 1
        flame_color = QtGui.QColor(palette.danger)
        positions = [0.5] if n_candles == 1 else [0.38, 0.62]
        r = min(draw_w, draw_h) * 0.09
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(flame_color)
        for frac in positions:
            cx = room_rect.left() + draw_w * frac
            cy = room_rect.top() + draw_h * 0.5
            painter.drawPath(_flame_path(cx, cy, r))

        # --- caption -------------------------------------------------------------
        painter.setPen(QtGui.QColor(palette.text_secondary))
        font = painter.font()
        font.setPointSizeF(max(font.pointSizeF() * 0.8, 7))
        painter.setFont(font)
        caption = "Door: {}  ·  Vent 1: {}  ·  Vent 2: {}  ·  {} candle{}".format(
            "wide" if self._door == 1 else "narrow", vod_label, voc_label,
            n_candles, "s" if n_candles > 1 else "",
        )
        painter.drawText(self.rect().adjusted(0, 0, 0, -2),
                          QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, caption)
