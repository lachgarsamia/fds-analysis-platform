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

import numpy as np
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


def _flame_shape(cx: float, base_y: float, h: float) -> QtGui.QPainterPath:
    """A flame teardrop rooted at (cx, base_y) rising to a pointed tip
    `h` above it -- rounded, slightly wavy sides, tip drawn straight up."""
    w = h * 0.40
    path = QtGui.QPainterPath()
    path.moveTo(cx, base_y - h)                                   # tip
    path.cubicTo(cx + w, base_y - h * 0.5, cx + w * 0.65, base_y, cx, base_y)      # down the right
    path.cubicTo(cx - w * 0.65, base_y, cx - w, base_y - h * 0.5, cx, base_y - h)  # up the left
    return path


def draw_realistic_flame(painter: QtGui.QPainter, cx: float, base_y: float, height: float):
    """Layered candle flame sitting on the floor at (cx, base_y): a soft
    radial glow, a deep-red outer body, an orange mid, and a bright yellow
    core -- a warmer, more flame-like look than a single flat teardrop."""
    painter.setPen(QtCore.Qt.NoPen)
    # glow
    glow_c = QtCore.QPointF(cx, base_y - height * 0.5)
    glow_r = height * 1.0
    grad = QtGui.QRadialGradient(glow_c, glow_r)
    warm = QtGui.QColor("#FF7A18")
    warm.setAlpha(80); grad.setColorAt(0.0, warm)
    edge = QtGui.QColor("#FF7A18"); edge.setAlpha(0); grad.setColorAt(1.0, edge)
    painter.setBrush(QtGui.QBrush(grad))
    painter.drawEllipse(glow_c, glow_r, glow_r)
    # body layers, outer -> core
    for h, color, lift in ((height, "#D93415", 0.0),
                           (height * 0.70, "#FF8A1E", 0.06),
                           (height * 0.40, "#FFDD57", 0.12)):
        painter.setBrush(QtGui.QColor(color))
        painter.drawPath(_flame_shape(cx, base_y - height * lift, h))


# ---------------------------------------------------------------- the widget

_VOD_STATES = {0: "open", 1: "closed", 2: "HVAC"}
_VOC_STATES = {0: "open", 1: "closed"}

# Side-view (x-z plane) geometry, fixed across every scenario in this
# study (template.fds: MULT DX=0.25,I_UPPER=3 -> x 0..1.0 m;
# DZ=0.16,K_UPPER=2 -> z 0..0.48 m; the enclosed room is bounded by the
# vertical wall at x~0.27 and the ceiling at z~0.22, so it occupies the
# bottom-right of the domain). The schematic mirrors the heatmap's own
# orientation instead of an abstract top-down view.
# ROOM_X/ROOM_Z are public (not module-private) -- views.py reuses them
# to draw the same room outline directly on the Live Viewer's heatmap
# (physical mesh geometry, so it's identical for every quantity/scenario
# on a y-normal slice, not something to duplicate/hardcode a second time).
_DOMAIN_X = (0.0, 1.0)
_DOMAIN_Z = (0.0, 0.48)
ROOM_X = (0.27, 1.0)
ROOM_Z = (0.0, 0.22)
# Ceiling &OBST top face (XB=...,0.22,0.24 / Ceiling) -- the slab is
# ROOM_Z[1]=0.22 (bottom, the real wall) to _OBST_Z_TOP=0.24 (top, where
# real exterior gas resumes). TEMPERATURE/VELOCITY read exactly ambient/
# exactly 0 (flat, zero variation) at the one grid row strictly inside
# that slab (z=0.23) wherever it's solid -- a dead solid-interior grid
# point the gas-phase solver never touches, not real gas data or an
# overlay bug. Under an open VOD/VOC opening the same row instead varies
# smoothly with its real neighbors, since there's genuinely no solid
# there (confirmed directly, colormap expressiveness follow-up).
#
# Drawn as its true thickness on the heatmap: room_overlay_geometry's
# `walls` list has a dashed line at both ROOM_Z[1]=0.22 and _OBST_Z_TOP=
# 0.24, so the dead z=0.23 row sits visibly between two boundary lines
# instead of behind one unexplained offset line. That geometry alone
# isn't enough to make the row self-explanatory, though -- checked
# directly by rendering VELOCITY/DYNAMIC PRESSURE with the mask disabled:
# the dead cell's exact 0 sits in the same low-colormap range as the real,
# genuinely-near-zero calm air immediately above/below it (e.g. real
# VELOCITY of 0.007-0.13 m/s right next to the dead exact-0.0 row), so
# bracketing lines or not, it reads as plausible real data, not obviously
# "inside the wall." So it's still masked to background -- see
# ceiling_obstruction_mask() -- with the two lines left in place as the
# honest boundary of what's being masked.
_OBST_Z_TOP = 0.24
_CANDLE_X = (0.84, 0.96)      # candle burner x-band
# Vent x-ranges, shared by room_overlay_geometry (the drawn tick marks)
# and ceiling_obstruction_mask (which of these openings to leave unmasked)
# -- one source, so the two can't drift apart. From the real &HOLE lines,
# see room_overlay_geometry's docstring.
_VOD_X = (0.32, 0.40)
_VOC_X = (0.86, 0.94)
_DOMAIN_ASPECT = (_DOMAIN_Z[1] - _DOMAIN_Z[0]) / (_DOMAIN_X[1] - _DOMAIN_X[0])


def room_overlay_geometry(door: int, vod: int, voc: int) -> dict:
    """Physical-coordinate (meters) room geometry for the heatmap's room
    overlay (views.py) -- the actual &HOLE/&VENT opening positions from
    the FDS input deck (template.fds / the per-scenario .fds files),
    not a proportional guess. Confirmed by reading a real scenario file:
      &HOLE XB=0.25,0.29,...,-0.01,0.05  / Door (narrow, door=0)
      &HOLE XB=0.25,0.29,...,-0.01,0.15  / Door (wide, door=1)
      &HOLE XB=0.32,0.40,...,0.21,0.25   / vertical opening door  (VOD)
      &HOLE XB=0.86,0.94,...,0.21,0.25   / vertical opening candle (VOC)
    An earlier version used placeholder fractions (34%/64% across the
    ceiling) that didn't match these real positions -- most visibly wrong
    for VOC, which sits right next to the candle (x=0.92-0.96) at the
    room's far edge, not the ceiling's middle, so a real fire plume
    visibly passed the drawn vent without seeming to go through it.

    Returns {"walls": [(x0,z0,x1,z1), ...], "door": (x0,z0,x1,z1),
    "vents": [((x0,z0,x1,z1), state_name), ...]}."""
    x0, x1 = ROOM_X
    z0, z1 = ROOM_Z

    door_bottom = z0  # HOLE's z0 is -0.01, i.e. right at the floor
    door_top = door_bottom + (0.16 if door == 1 else 0.06)

    walls = [
        (x0, z0, x1, z0),             # floor
        (x1, z0, x1, z1),             # right wall
        (x0, z1, x1, z1),             # ceiling (slab underside, z=ROOM_Z[1] -- the real wall)
        # Ceiling &OBST slab top face (z=_OBST_Z_TOP), parallel to the line
        # above -- draws the slab's true ~2cm thickness instead of a single
        # line, so the dead solid-interior grid row between the two (see
        # _OBST_Z_TOP's docstring) reads as "inside the concrete" rather
        # than an unexplained second, offset boundary line. Static geometry,
        # same casing/dash style as every other room_walls segment -- no
        # vent-state coupling (real venting gas exists at both x-vent bands
        # regardless of z, so this line intentionally doesn't gap there,
        # matching the ceiling line's own convention of overlaying vent
        # color on top rather than cutting a hole in the wall segment).
        (x0, _OBST_Z_TOP, x1, _OBST_Z_TOP),
        (x0, z0, x0, door_bottom),    # left wall, below the door
        (x0, door_top, x0, z1),       # left wall, above the door
    ]
    door_seg = (x0, door_bottom, x0, door_top)

    vents = [
        ((_VOD_X[0], z1, _VOD_X[1], z1), _VOD_STATES.get(vod, "?")),   # vertical opening door
        ((_VOC_X[0], z1, _VOC_X[1], z1), _VOC_STATES.get(voc, "?")),   # vertical opening candle
    ]

    return {"walls": walls, "door": door_seg, "vents": vents}


def ceiling_obstruction_mask(vod: int, voc: int, extent, shape) -> np.ndarray:
    """Boolean (n_z, n_x) mask, True where a TEMPERATURE/VELOCITY-sourced
    cell sits inside the ceiling &OBST's solid slab (ROOM_Z[1]=0.22 to
    _OBST_Z_TOP=0.24) and is NOT under a currently-open VOD/VOC vent --
    i.e. a dead solid-interior grid point (see the comment above ROOM_Z)
    that should render as background, not a colored (mis)reading. Cells
    at exactly z=ROOM_Z[1] (the real wall, real gas) or under an open vent
    (real venting gas) are left unmasked -- False everywhere outside the
    slab's open interval too. `extent`/`shape` are the same (x0,x1,z0,z1)/
    (n_z,n_x) convention as SliceView's own extent (row 0 = z1/top)."""
    x0, x1, z0, z1 = extent
    n_z, n_x = shape
    xs = np.linspace(x0, x1, n_x)
    zs = np.linspace(z1, z0, n_z)
    # A small tolerance, well under the ~0.01 m grid spacing, so a grid
    # point that's really *on* a boundary (e.g. linspace(0,1,101)[94] ==
    # 0.9400000000000001, not exactly 0.94) doesn't fall on the wrong side
    # of it -- confirmed directly: without this, the real last column of
    # an open VOC vent got flagged as "outside the vent" and would have
    # been masked despite carrying real venting data.
    eps = 1e-6
    in_slab = (zs > ROOM_Z[1] + eps) & (zs < _OBST_Z_TOP - eps)
    # The ceiling &OBST only exists over the room itself (x>=ROOM_X[0]) --
    # the exterior corridor (x<ROOM_X[0]) has no ceiling at all, so it's
    # real, smoothly-varying gas at every z there, not a dead solid cell.
    # Confirmed directly: without this, the corridor's real data at z=0.23
    # (e.g. ~21-22.4 degC, clearly varying) was being masked too.
    in_room_x = xs >= ROOM_X[0] - eps
    open_vent_x = np.zeros(n_x, dtype=bool)
    if _VOD_STATES.get(vod) == "open":
        open_vent_x |= (xs >= _VOD_X[0] - eps) & (xs <= _VOD_X[1] + eps)
    if _VOC_STATES.get(voc) == "open":
        open_vent_x |= (xs >= _VOC_X[0] - eps) & (xs <= _VOC_X[1] + eps)
    return in_slab[:, None] & in_room_x[None, :] & ~open_vent_x[None, :]


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
        self._ui_scale = 1.0
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

    def set_ui_scale(self, scale: float) -> None:
        """View -> UI Scale (accessibility zoom): this widget draws itself
        entirely in paintEvent (no QSS-driven sizing/fonts reach it), so
        without this it stayed a fixed size and caption font regardless of
        the chosen scale while everything else around it grew/shrank."""
        self._ui_scale = scale
        self.setMinimumHeight(int(90 * scale))
        self.updateGeometry()
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
        desc = (
            "Room diagram (side view): the big rectangle is the full simulation "
            "domain; the smaller one at the bottom-right is the enclosed room. "
            "Door {} on the left wall, Vent 1 {} and Vent 2 {} on the ceiling, "
            "{} candle{} burning on the floor.".format(
                "wide open" if self._door == 1 else "narrow",
                _VOD_STATES.get(self._vod, "?"),
                _VOC_STATES.get(self._voc, "?"),
                n_candles, "s" if n_candles > 1 else "",
            )
        )
        self.setAccessibleDescription(desc)
        self.setToolTip(desc)   # hover info (Live Viewer polish)

    # -- sizing ---------------------------------------------------------------
    def _aspect(self) -> float:
        # Side view: the domain's z/x ratio, fixed for this study's geometry.
        return _DOMAIN_ASPECT

    def sizeHint(self):
        width = max(self.width(), int(260 * self._ui_scale))
        return QtCore.QSize(width, max(int(90 * self._ui_scale), int(width * self._aspect()) + 30))

    def heightForWidth(self, width: int) -> int:
        return max(int(80 * self._ui_scale), int(width * self._aspect()) + 30)

    def hasHeightForWidth(self) -> bool:
        # Without this, Qt's layout system never calls heightForWidth() at
        # all (its default is False) -- the widget's height stayed frozen
        # at whatever sizeHint() computed once at initial layout, so
        # resizing the sidebar (or changing ui_scale, which resizes it too)
        # left the room outline's proportions wrong instead of rescaling.
        return True

    # -- painting -------------------------------------------------------------
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _phys(self, dr: QtCore.QRectF, x: float, z: float) -> QtCore.QPointF:
        """Physical (x, z) meters -> a point inside the domain draw-rect
        `dr` (z points up, matching the heatmap)."""
        fx = (x - _DOMAIN_X[0]) / (_DOMAIN_X[1] - _DOMAIN_X[0])
        fz = (z - _DOMAIN_Z[0]) / (_DOMAIN_Z[1] - _DOMAIN_Z[0])
        return QtCore.QPointF(dr.left() + fx * dr.width(), dr.bottom() - fz * dr.height())

    def _paint(self, painter: QtGui.QPainter):
        palette = self._palette
        if palette is None:
            return

        margin = 12
        rect = self.rect().adjusted(margin, margin, -margin, -margin - 14)  # leave room for caption
        if rect.width() <= 0 or rect.height() <= 0:
            return

        aspect = self._aspect()
        if rect.width() * aspect <= rect.height():
            draw_w, draw_h = rect.width(), rect.width() * aspect
        else:
            draw_h = rect.height()
            draw_w = rect.height() / aspect if aspect > 0 else rect.width()
        draw_x = rect.x() + (rect.width() - draw_w) / 2
        draw_y = rect.y() + (rect.height() - draw_h) / 2
        domain_rect = QtCore.QRectF(draw_x, draw_y, draw_w, draw_h)

        # --- simulation domain (big outer rectangle) ------------------------
        painter.setPen(QtGui.QPen(QtGui.QColor(palette.border), 1, QtCore.Qt.DashLine))
        painter.setBrush(QtGui.QColor(palette.bg_base))
        painter.drawRect(domain_rect)

        # --- the room (smaller rectangle, bottom-right) ---------------------
        room_tl = self._phys(domain_rect, ROOM_X[0], ROOM_Z[1])   # top-left (x wall, ceiling)
        room_br = self._phys(domain_rect, ROOM_X[1], ROOM_Z[0])   # bottom-right (floor, right wall)
        room_rect = QtCore.QRectF(room_tl, room_br)
        painter.setPen(QtCore.Qt.NoPen)
        # bg_elevated (not bg_sunken) -- bg_sunken is *darker* than the
        # domain's own bg_base in every theme, so the room (the part with
        # the door/vents "openings") visually receded into the background
        # instead of standing out as the diagram's main subject.
        painter.setBrush(QtGui.QColor(palette.bg_elevated))
        painter.drawRect(room_rect)
        # text_primary (not border_strong) -- the room's own walls are the
        # diagram's main subject, so they get the same high-contrast tone
        # as body text rather than a subtle chrome-border gray that reads
        # too faint against bg_sunken in both themes.
        wall = QtGui.QPen(QtGui.QColor(palette.text_primary), 2.6)

        # left wall (with the door gap) and ceiling drawn as explicit walls;
        # floor and right wall are the plain room edges.
        painter.setPen(wall)
        painter.drawLine(room_rect.bottomLeft(), room_rect.bottomRight())   # floor
        painter.drawLine(room_rect.topRight(), room_rect.bottomRight())     # right wall

        # --- door on the LEFT wall, opening size by door state --------------
        # Real &HOLE z-range (see room_overlay_geometry's docstring): the
        # opening reaches the floor (not 6% above it) and is 0.06m (narrow)
        # or 0.16m (wide) tall, as a fraction of ROOM_Z's 0.22m room height.
        door_h_frac = (0.16 if self._door == 1 else 0.06) / 0.22
        wall_x = room_rect.left()
        door_h = room_rect.height() * door_h_frac
        door_bottom = room_rect.bottom()
        door_top = door_bottom - door_h
        # wall segments above and below the opening
        painter.setPen(wall)
        painter.drawLine(QtCore.QPointF(wall_x, room_rect.top()), QtCore.QPointF(wall_x, door_top))
        painter.drawLine(QtCore.QPointF(wall_x, door_bottom), QtCore.QPointF(wall_x, room_rect.bottom()))
        # opening marked with the accent, with a small swing arc
        accent = QtGui.QColor(palette.accent)
        painter.setPen(QtGui.QPen(accent, 2))
        painter.drawLine(QtCore.QPointF(wall_x, door_top), QtCore.QPointF(wall_x, door_bottom))
        painter.setPen(QtGui.QPen(accent, 1.2, QtCore.Qt.DashLine))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawArc(QtCore.QRectF(wall_x, door_top, door_h, door_h), 90 * 16, 90 * 16)

        # --- ventilation on the CEILING (Vent 1 = VOD, Vent 2 = VOC) --------
        vent_colors = {"open": palette.success, "closed": palette.text_disabled, "HVAC": palette.warning}
        vod_label, voc_label = _VOD_STATES[self._vod], _VOC_STATES[self._voc]
        ceiling_y = room_rect.top()
        painter.setPen(wall)
        painter.drawLine(room_rect.topLeft(), room_rect.topRight())  # ceiling
        # Real &HOLE x-ranges (see room_overlay_geometry's docstring): VOD
        # sits near the door at x=0.32-0.40, VOC sits right next to the
        # candle at x=0.86-0.94 -- as fractions of ROOM_X's 0.27-1.00 span,
        # not the evenly-spaced 34%/64% placeholder this used to draw.
        vent_w = room_rect.width() * (0.08 / 0.73)
        for frac, label in (((0.36 - 0.27) / 0.73, vod_label), ((0.90 - 0.27) / 0.73, voc_label)):
            cx = room_rect.left() + room_rect.width() * frac
            vent_rect = QtCore.QRectF(cx - vent_w / 2, ceiling_y - 3, vent_w, 6)
            painter.setPen(QtGui.QPen(QtGui.QColor(palette.border_strong), 1))
            painter.setBrush(QtGui.QColor(vent_colors[label]))
            painter.drawRect(vent_rect)
            # a couple of grille slats
            painter.setPen(QtGui.QPen(QtGui.QColor(palette.bg_base), 0.8))
            for k in (0.33, 0.66):
                sx = vent_rect.left() + vent_rect.width() * k
                painter.drawLine(QtCore.QPointF(sx, vent_rect.top()), QtCore.QPointF(sx, vent_rect.bottom()))

        # --- candle flame(s) on the floor -----------------------------------
        n_candles = 2 if self._candles == 1 else 1
        cx_mid = sum(_CANDLE_X) / 2
        span = (_CANDLE_X[1] - _CANDLE_X[0])
        xs = [cx_mid] if n_candles == 1 else [cx_mid - span * 0.35, cx_mid + span * 0.35]
        flame_h = room_rect.height() * 0.42
        for x in xs:
            base = self._phys(domain_rect, x, ROOM_Z[0])
            draw_realistic_flame(painter, base.x(), base.y(), flame_h)

        # --- caption -------------------------------------------------------------
        painter.setPen(QtGui.QColor(palette.text_secondary))
        font = painter.font()
        font.setPointSizeF(max(font.pointSizeF() * 0.8, 7) * self._ui_scale)
        painter.setFont(font)
        caption = "Domain + room (side view)  ·  Door: {}  ·  Vent 1: {}  ·  Vent 2: {}  ·  {} candle{}".format(
            "wide" if self._door == 1 else "narrow", vod_label, voc_label,
            n_candles, "s" if n_candles > 1 else "",
        )
        painter.drawText(self.rect().adjusted(0, 0, 0, -2),
                          QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, caption)
