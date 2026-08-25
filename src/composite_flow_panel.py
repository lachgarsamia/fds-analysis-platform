"""Composite flow panel: filled vertical-velocity (W) background + a
speed-scaled velocity quiver + translucent temperature zones, all on one
frame -- "where the air is going, colored by whether it's rising or
sinking, with the heat structure filled in on top." A third, independent
visualization of the validated U/W-VELOCITY field, alongside
VelocityPanel's quiver/streamlines and StreamlinePanel's matplotlib
streamplot -- this one composites velocity with TEMPERATURE instead of
showing velocity alone. Does not touch velocity_panel.py or
streamline_panel.py in any way; reuses velocity.py's VectorField/
quiver_grid exactly as they do (read-only), plus TEMPERATURE via the same
QuantityProvider.get() every other slice view uses.

Grid/cadence (confirmed directly, not assumed -- see the diagnostic pass
before this panel was built): TEMPERATURE and U/W-VELOCITY are both
481-frame .sf slices sharing the same (49, 101) grid and [0, 1, 0, 0.48]
extent at y=0 for all 24 scenarios, so `frame_index` indexes every layer
identically -- no regridding, no frame-remap (unlike the .s3d volume
quantities' own cadence, which is unrelated to this panel).

Row-0-is-ceiling convention: exactly the same as streamline_panel.py --
U/W/TEMPERATURE all have row 0 at the physical ceiling; each frame is
flipped vertically (row reindexing only, no sign changes) so `z` is
increasing for pcolormesh/contour alike, keeping the fill layers in the
same coordinate frame as ax.set_ylim(z0, z1).

Quiver is the one layer that must NOT receive the flipped arrays.
velocity.quiver_grid computes each sample's physical z itself from the
*raw* (row 0 = ceiling) array (see its own docstring/implementation --
the same convention VectorField.quiver_at() feeds it in velocity_panel.py
today); passing the already-flipped u/w in here as well (an earlier
version of this panel did exactly that) silently double-flips the
positions, so each arrow gets drawn at its physically mirrored z and
paired with a different row's velocity -- confirmed directly against
measure.probe_value ground truth before this fix landed. Fixed by
sampling quiver_grid from field.u[frame_index]/field.w[frame_index]
(unflipped) directly, not from the flipped u_frame/w_frame the pcolormesh/
contour layers use.

W sign: confirmed empirically (not assumed from FDS's nominal z-up
convention) against real data -- at the open VOC vent directly above a
candle, W averages +0.27 to +0.60 m/s while temperature there is well
above ambient (hot gas leaving through the ceiling opening); directly
above the candle, W stays positive while temperature falls with height
(classic buoyant-plume dilution). W > 0 is rising. Never flipped here.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
from schematic import room_overlay_geometry
from slice_key import SliceKey
from registry import get_quantity
from config import CONTOUR_OVERLAY_LEVELS
import velocity as vel

# Layer 2 (quiver): a fixed stride over the (49, 101) U/W grid --
# velocity.quiver_grid's positions are deterministic from stride alone
# (see its own docstring), so this is temporally stable by construction,
# same guarantee as StreamlinePanel's fixed seeds, with none of the
# caching those needed (nothing here depends on frame data to compute
# the *positions*, only the *directions*/*lengths* redraw per frame).
# Chosen via the same visual-comparison approach as the streamline
# panel's density passes: stride=6 (~17x9 arrows) reads as a direction
# texture over the W background rather than clutter competing with it.
QUIVER_STRIDE = 6
# Length encodes speed, sub-linearly (sqrt), not uniform and not linear:
# real speeds span ~20x here (room circulation ~0.03-0.15 m/s vs. plume
# ~0.6-1.2 m/s -- the same range streamline_panel.py measured and
# documented for this dataset), and a linear length scale would shrink
# circulation arrows to sub-pixel invisibility to leave headroom for the
# plume. sqrt(speed) compresses that to a ~4.5x range (sqrt(20)~=4.47)
# before QUIVER_LENGTH_FLOOR narrows it further so slow-but-real
# circulation stays legible instead of vanishing. QUIVER_SPEED_REF is the
# same empirical plume-peak reference streamline_panel.py's SPEED_REF_MS
# uses (duplicated, not imported -- this module's zero-coupling
# precedent): speeds at/above it render at QUIVER_LENGTH_MAX, everything
# else scales down to QUIVER_LENGTH_FLOOR. Color (Layer 1) still encodes
# magnitude too -- redundant encoding is deliberate, not a leftover.
QUIVER_SPEED_REF = 1.2
QUIVER_LENGTH_FLOOR = 0.2
QUIVER_LENGTH_MAX = 1.0
QUIVER_SCALE = 22.0
QUIVER_WIDTH = 0.0035
QUIVER_COLOR = "#1A1A1A"   # neutral dark -- reads over both coolwarm ends

# Layer 3 (temperature zones): TEMPERATURE's own registry-driven contour
# levels (config.CONTOUR_OVERLAY_LEVELS, the same list the View-menu
# Contour overlay and Temperature (Isolines) use), not a new hand-picked
# set -- translucent filled bands (ax.contourf), not lines, so Layer 1's
# W background and Layer 2's quiver both still read through them. A warm
# sequential colormap (not coolwarm/viridis) so "this is thermal" is
# visually unambiguous against the blue/red velocity field underneath.
# No `extend` (matplotlib default): cells below the lowest level (most of
# the room, ambient) get no fill at all rather than a wash across the
# whole domain -- these are meant to read as discrete hot *zones*, not a
# second full-domain background competing with Layer 1's.
TEMPERATURE_FILL_CMAP = "YlOrRd"
TEMPERATURE_FILL_ALPHA = 0.4

# Layer 4 (room geometry): same thinned/faded/receded-behind-the-flow
# values the streamline panel settled on (visual clarity passes 2-3) --
# duplicated rather than imported (this module's own "zero coupling to
# other velocity views" precedent, matching streamline_panel.py's).
_WALL_COLOR = "#FFFFFF"
_DOOR_COLOR = "#38BDF8"
_VENT_STATE_COLORS = {"open": "#22C55E", "closed": "#94A3B8", "HVAC": "#F59E0B"}
_WALL_LINEWIDTH = 1.4
_DOOR_LINEWIDTH = 1.6
_VENT_LINEWIDTH = 2.2
_GEOMETRY_ALPHA = 0.7
_GEOMETRY_ZORDER = 1   # above the pcolormesh background (default 0), below quiver/isotherms

_TEMPERATURE_KEY = SliceKey("TEMPERATURE", 1, 0)


class CompositeFlowPanel(QtWidgets.QWidget):
    """Analysis-page tab: filled W background + speed-scaled quiver +
    translucent temperature zones on one frame. Independent of
    VelocityPanel and StreamlinePanel -- no shared state, no probes,
    nothing here writes to either."""

    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._by_index = {e.case_index: e for e in self._manifest}
        self._fps = max(1, fps)
        self._loaded = False
        self._fields: dict = {}          # case_index -> vel.VectorField
        self._temperatures: dict = {}    # case_index -> full (t, z, x) TEMPERATURE array
        self._gate_reasons: dict = {}    # case_index -> str
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Velocity (composite)")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        header = QtWidgets.QHBoxLayout()
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Composite flow scenario")
        self.scenario_combo.setToolTip(
            "Which scenario's W-velocity/quiver/temperature composite to display")
        header.addWidget(self.scenario_combo)
        header.addStretch(1)
        layout.addLayout(header)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Composite flow canvas")
        layout.addWidget(self.canvas, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        was_loaded = self._loaded
        self.ensure_loaded()
        if was_loaded:
            self._render()

    def set_bus(self, bus) -> None:
        """Follow the shared playback frame (Selection.time_s), same
        set_bus precedent as streamline_panel.py -- no frame_slider for
        the generic bind_to_bus sync to hook. One-way: never publishes a
        selection, only reacts."""
        self._bus = bus
        bus.changed.connect(self._on_selection)
        self._on_selection(bus.current, None)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.time_s is None:
            return
        self._current_index = max(0, int(round(sel.time_s * self._fps)))
        if self._loaded and self.isVisible():
            self._render()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _reload(self) -> None:
        if not self._loaded:
            return
        self._render()

    # -------------------------------------------------- field/data access
    def _ensure_field(self, case_index: int):
        """Lazily compute (and cache) the VectorField for `case_index` --
        identical convention to StreamlinePanel._ensure_field, duplicated
        rather than shared (this module's own zero-coupling precedent)."""
        if case_index in self._fields:
            return self._fields[case_index]
        if case_index in self._gate_reasons:
            return None
        f = vel.VectorField(self._provider, case_index)
        try:
            f.compute()
        except GatedQuantityError as e:
            self._gate_reasons[case_index] = str(e)
            return None
        self._fields[case_index] = f
        return f

    def _ensure_temperature(self, case_index: int):
        """Lazily fetch (and cache) the full TEMPERATURE array for
        `case_index` at the same y=0 plane U/W use -- same lazy-cache
        shape as _ensure_field. TEMPERATURE is never gated in the
        registry, but get() can still raise GatedQuantityError for a
        scenario missing this specific plane (V6-M5's per-scenario
        inventory check), so this guards the same way."""
        if case_index in self._temperatures:
            return self._temperatures[case_index]
        if case_index in self._gate_reasons:
            return None
        try:
            temp = self._provider.get(case_index, _TEMPERATURE_KEY)
        except GatedQuantityError as e:
            self._gate_reasons[case_index] = str(e)
            return None
        temp = np.asarray(temp, dtype=float)
        self._temperatures[case_index] = temp
        return temp

    # --------------------------------------------------------------- render
    def _render(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        field = self._ensure_field(case_index)
        temp = self._ensure_temperature(case_index) if field is not None else None
        if field is None or temp is None:
            reason = self._gate_reasons.get(
                case_index, "U/W-VELOCITY or TEMPERATURE not available for this scenario.")
            ax.set_title("Composite unavailable -- needs U/W velocity and TEMPERATURE data",
                         fontsize=9, color="#B00020")
            ax.set_xticks([]); ax.set_yticks([])
            self.status.setText(reason)
            self.canvas.draw_idle()
            return
        self.status.setText("")

        entry = self._by_index.get(case_index)
        geometry = (room_overlay_geometry(entry.door, entry.vod, entry.voc)
                    if entry is not None else None)

        frame_index = min(max(self._current_index, 0),
                           field.n_frames - 1, temp.shape[0] - 1)
        x0, x1, z0, z1 = field.extent

        # Row 0 of u/w/temp is the ceiling; flip vertically (reindex only,
        # no sign change) so z is increasing for every layer alike -- same
        # convention as streamline_panel.py, kept identical so the three
        # layers stay in registration with each other.
        u_frame = np.flip(field.u[frame_index], axis=0)
        w_frame = np.flip(field.w[frame_index], axis=0)
        temp_frame = np.flip(temp[frame_index], axis=0)

        n_z, n_x = u_frame.shape
        x = np.linspace(x0, x1, n_x)
        z = np.linspace(z0, z1, n_z)

        # Layer 1: filled W background -- W-VELOCITY's own registry values,
        # unmodified (already calibrated against all 24 scenarios, see
        # registry.py's own comment). Zero-centered so rising (positive W,
        # confirmed empirically -- see module docstring) and sinking read
        # as symmetric red/blue.
        w_quantity = get_quantity("W-VELOCITY")
        w_norm = TwoSlopeNorm(vcenter=0.0, vmin=w_quantity.vmin, vmax=w_quantity.slider_default)
        mesh = ax.pcolormesh(x, z, w_frame, cmap=w_quantity.cmap, norm=w_norm,
                              shading="auto", zorder=0)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="W velocity (m/s)")

        # Layer 2: speed-scaled quiver -- fixed-stride grid positions (see
        # QUIVER_STRIDE), sampled from the RAW (unflipped) field.u/w, not
        # u_frame/w_frame (see the module docstring's "quiver_grid" note --
        # quiver_grid computes its own z from the raw row-0-is-ceiling
        # convention). Direction is unit-normalized, then scaled by
        # sqrt(speed) clamped between QUIVER_LENGTH_FLOOR and
        # QUIVER_LENGTH_MAX so both the plume and slow circulation stay
        # legible (see QUIVER_SPEED_REF's comment) -- a true stagnation
        # point (speed exactly 0) still renders as a zero-length (i.e. no)
        # arrow, only genuinely slow-but-moving cells get floored.
        xs, zs, us, ws = vel.quiver_grid(field.u[frame_index], field.w[frame_index],
                                         field.extent, QUIVER_STRIDE)
        speed = np.hypot(us, ws)
        eps = 1e-9
        us_dir = np.divide(us, speed, out=np.zeros_like(us), where=speed > eps)
        ws_dir = np.divide(ws, speed, out=np.zeros_like(ws), where=speed > eps)
        length_frac = np.clip(np.sqrt(speed / QUIVER_SPEED_REF), 0.0, 1.0)
        length = QUIVER_LENGTH_FLOOR + (QUIVER_LENGTH_MAX - QUIVER_LENGTH_FLOOR) * length_frac
        length = np.where(speed > eps, length, 0.0)
        ax.quiver(xs, zs, us_dir * length, ws_dir * length, color=QUIVER_COLOR, angles="xy",
                  scale_units="xy", scale=QUIVER_SCALE, width=QUIVER_WIDTH,
                  pivot="mid", zorder=2)

        # Layer 3: translucent temperature zones -- TEMPERATURE's own
        # registry-driven contour levels (same list the View-menu Contour
        # overlay and Temperature (Isolines) use), filled not lines, with
        # alpha so Layers 1-2 still read through them (see
        # TEMPERATURE_FILL_ALPHA's comment).
        levels = CONTOUR_OVERLAY_LEVELS.get("TEMPERATURE", [])
        if levels:
            temp_fill = ax.contourf(x, z, temp_frame, levels=levels, cmap=TEMPERATURE_FILL_CMAP,
                                    alpha=TEMPERATURE_FILL_ALPHA, zorder=3)
            fig.colorbar(temp_fill, ax=ax, fraction=0.046, pad=0.12, label="Temperature (°C)")

        # Layer 4: room outline -- thinned/faded/pushed just above the
        # background (below quiver/isotherms), same lesson as the
        # streamline panel's own visual-clarity passes: context, not focus.
        if entry is not None:
            for wx0, wz0, wx1, wz1 in geometry["walls"]:
                ax.plot([wx0, wx1], [wz0, wz1], color=_WALL_COLOR,
                        linestyle="--", linewidth=_WALL_LINEWIDTH,
                        alpha=_GEOMETRY_ALPHA, zorder=_GEOMETRY_ZORDER)
            dx0, dz0, dx1, dz1 = geometry["door"]
            ax.plot([dx0, dx1], [dz0, dz1], color=_DOOR_COLOR, linewidth=_DOOR_LINEWIDTH,
                    alpha=_GEOMETRY_ALPHA, zorder=_GEOMETRY_ZORDER)
            for (vx0, vz0, vx1, vz1), state in geometry["vents"]:
                ax.plot([vx0, vx1], [vz0, vz1],
                        color=_VENT_STATE_COLORS.get(state, "#94A3B8"),
                        linewidth=_VENT_LINEWIDTH, alpha=_GEOMETRY_ALPHA,
                        zorder=_GEOMETRY_ZORDER)

        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("auto")
        ax.set_xticks([]); ax.set_yticks([])
        t_s = frame_index / self._fps
        ax.set_title(f"W + quiver + temperature zones · t={t_s:.1f}s", fontsize=8)

        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()
