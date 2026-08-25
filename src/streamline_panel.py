"""Velocity streamlines panel: a second, independent visualization of the
same validated U/W-VELOCITY vector field VelocityPanel's quiver mode
already draws -- this one via matplotlib's own `ax.streamplot()` (a dense,
auto-seeded field of continuous flow lines) rather than VelocityPanel's
per-probe RK4 integrator (velocity.py's `integrate_streamline`/
`streamline_at`, seeded only at user-placed points). The two are
deliberately different tools: probe-seeded streamlines answer "where does
the flow starting *here* go", matplotlib's streamplot answers "what does
the whole flow field look like at a glance".

Does not touch velocity.py or velocity_panel.py -- reuses velocity.py's
VectorField exactly as VelocityPanel does (same QuantityProvider.get_vector
accessor underneath, zero re-derivation), read-only.

Architecture note (why this is a new panel, not a registry entry /
SliceView mode): views.py's PlotView Protocol contracts `show_frame(frame:
np.ndarray)` to a single 2D array -- the whole SliceView/ViewGrid/main
quantity-dropdown pipeline is built for one-scalar-quantity heatmaps.
Streamlines need two arrays (U, W) at once, so they cannot be "a flag on
SliceView" without breaking that contract. Nor is this a registry.py
QuantityInfo: the registry models real FDS output quantities (U-VELOCITY
and W-VELOCITY are legitimately registered as individual scalar
components); "streamlines" is a *rendering mode* of their combination,
exactly like velocity.py's own MODES tuple already treats "quiver" /
"streamlines" / "both" as plain constants, never registry entries. This
panel follows that existing precedent instead of inventing a new one.

Row-0-is-ceiling convention: VectorField.u/w (from QuantityProvider, same
as every other slice array in this app) have row 0 at the physical
*ceiling* (z1), with z decreasing as the row index increases -- see
load_data.py's np.flip(axis=1) and views.py's SliceView docstring.
ax.streamplot() requires a strictly *increasing* y-coordinate array, so
each frame is flipped vertically (row reindexing only -- W's sign, and
therefore its physical up/down meaning, is untouched) before the call.

Live playback (Analysis dynamic-visualizations pass): the whole streamplot
follows Selection.time_s via set_bus() -- same pattern as
device_panel.py/velocity_panel.py (state_at()-equivalent frame indexing,
isVisible()-gated redraw). Unlike those two, there's no separate
locator-background-vs-overlay split here (no placed probes, nothing to
click) -- the entire rendered content is the live frame.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import PowerNorm
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
from schematic import room_overlay_geometry, fire_positions
import velocity as vel

# Visual clarity pass, phase 2 (feature-based seeding): now that seeds
# cluster at the fire/door/vents instead of a blind 12x6 grid, the old
# 1.8 default -- tuned for that grid's much denser coverage -- reads as
# visual clutter competing with the feature clusters and the room outline
# for attention. Lowered to 0.9 (matplotlib's mask granularity, not seed
# count -- see the _seed_points/feature-seeding comment; seeds are
# unaffected) so surviving lines are fewer and less tangled, closer to
# ~15-30 visible strokes rather than filling the whole mask grid.
DEFAULT_DENSITY = 0.9
DEFAULT_CMAP = "viridis"  # perceptually uniform sequential -- never jet
# Linewidth-scales-with-speed formula (see _render): base + scale * ratio,
# ratio = local speed / SPEED_REF_MS (a fixed reference, not this frame's
# own peak -- see SPEED_REF_MS). Same comparison pass -- bumped from
# 0.5-2.5px to 1.0-3.5px so the peak-speed jet reads clearly bolder than
# ambient recirculation without the whole plot feeling heavy.
LINEWIDTH_BASE = 1.0
LINEWIDTH_SCALE = 2.5

# Fixed reference speed (m/s) for both color and linewidth, replacing two
# per-frame-relative references that crushed real signal: color used to
# normalize against VELOCITY.slider_default=2 (a generic UI slider ceiling,
# not calibrated to this dataset), and linewidth used to scale against each
# frame's own speed_frame.max(). Measured directly against real
# sim_stage1_prep data across all candle levels: room circulation runs
# ~0.03-0.15 m/s, the strongest candle case's plume peaks at ~1.17 m/s
# across its full run -- 1.2 gives that peak headroom without being so high
# that circulation still crushes toward zero. Shared by both fixes because
# they're the same bug: a reference that lets real signal get buried.
SPEED_REF_MS = 1.2

# Visual-density follow-up: matplotlib's streamplot places exactly one
# arrow per traced line, at that line's own midpoint -- with a single
# call per seed (integration_direction='both', the default), each of the
# 72 fixed seeds produces one long line and one arrow, so direction is
# only marked once per loop even when a line wanders across much of the
# room. Splitting each seed into two separate traces -- one
# 'forward'-only, one 'backward'-only, both from the *same* fixed
# start_points array (see _seed_points) -- doubles the arrow count (two
# midpoints instead of one) and roughly halves each visible line's
# length, without adding, moving, or removing a single seed. Confirmed
# against real data (scenario 12, t=72s): 34 arrows before -> 74 after.
# MAXLENGTH tuned down from streamplot's own default (4.0, halved to 2.0
# per direction internally when using a single 'both' call) via the same
# visual-comparison pass as DEFAULT_DENSITY.
#
# Visual clarity pass, phase 3: 1.2 was still long enough for a handful
# of trajectories to loop across most of the room (measured directly,
# scenario 0 t=72s: 2 of 39 traced lines spanned >75% of the room's
# diagonal) -- one line tracing "the whole room" reads as generic
# turbulence again, working against the feature-seeding goal of showing
# one meaningful local structure per seed (a plume rising into a vent, a
# door inflow) rather than everywhere-to-everywhere paths. Cut to 0.5 via
# the same measurement: 0 trajectories exceed 50% of the room diagonal at
# that value, while a fire-seeded plume trajectory (case 0, t=72s) still
# traces a real 0.27 m arc (down from 0.53 m at 1.2, not chopped to a
# stub) -- confirmed directly, not assumed. MINLENGTH is streamplot's own
# default (0.1); named explicitly rather than left implicit so both ends
# of "one meaningful structure, not the whole room or a speck" are
# visible together and independently tunable.
MAXLENGTH = 0.5
MINLENGTH = 0.1

# Visual clarity pass, phase 2: doubling arrows (above) was the right
# call for legibility, but at full size (arrowsize=1.0, matplotlib's own
# default) on *both* passes they read as clutter competing with the
# feature clusters for attention. Kept the forward/backward split (still
# wanted for its line-shortening effect), but only the forward pass gets
# a visible arrow now -- moderately shrunk, not full size. The backward
# pass's arrows are suppressed rather than removed outright:
# ARROWSIZE_HIDDEN isn't literally 0 because matplotlib's own arrow-head
# geometry divides by the arrow's length and misbehaves (RuntimeWarning)
# at exactly 0 -- confirmed directly; a tiny nonzero value renders as
# invisible without tripping that.
ARROWSIZE = 0.8            # moderate shrink from streamplot's own default (1.0)
ARROWSIZE_HIDDEN = 0.001   # effectively invisible -- backward pass only, see above

# Room outline colors (views.py's own _VENT_STATE_COLORS/wall/door
# convention, duplicated rather than imported -- same "zero coupling to
# the other velocity views" precedent this module already follows for
# _ensure_field). Drawn from schematic.room_overlay_geometry(), the same
# real &HOLE-derived geometry the Live Viewer overlays on its heatmap.
#
# Visual clarity pass, phase 2: at the original linewidths and zorder=6,
# the door/vent bars sat on top of and visually dominated the flow itself
# -- the room read as a decorated diagram with a streamplot in it, not a
# flow diagram with room context. Thinned, given _GEOMETRY_ALPHA<1, and
# moved to _GEOMETRY_ZORDER -- below streamplot's own default artist
# zorders (its LineCollection defaults to 2, its arrow FancyArrowPatches
# to 1; confirmed directly against matplotlib.collections/patches
# defaults) -- so flow lines and arrows now draw over the room outline,
# not under it.
_WALL_COLOR = "#FFFFFF"
_DOOR_COLOR = "#38BDF8"
_VENT_STATE_COLORS = {"open": "#22C55E", "closed": "#94A3B8", "HVAC": "#F59E0B"}
_WALL_LINEWIDTH = 1.4
_DOOR_LINEWIDTH = 1.6     # was 2.6
_VENT_LINEWIDTH = 2.2     # was 4.0
_GEOMETRY_ALPHA = 0.7
_GEOMETRY_ZORDER = 0

# Frame-to-frame stability fix: streamplot()'s automatic seeding
# (density=...) picks new seed locations independently on every call, so
# the rendered pattern reshuffled discontinuously between frames even
# though the underlying U/W field itself evolves smoothly -- the seeds
# moved, not the flow. Fixed explicit start_points (see _seed_points)
# sidesteps this: the same seed set drives every frame, so the pattern
# now shifts continuously with the field instead of jumping.
#
# Uniform-grid fallback (used only when a scenario has no manifest entry
# to read geometry from -- see _seed_points): NX:NZ roughly matches the
# room's ~2:1 x:z extent ratio; 12x6=72 seeds is comparable in on-screen
# density to the old density=1.8 auto-seeding, not excessive.
_SEED_GRID_NX = 12
_SEED_GRID_NZ = 6

# Feature-based seeding (replaces the uniform grid as the normal path):
# clusters at the fire(s), the door, and each open/HVAC vent, so the plot
# reads as "air enters at the openings, circulates, driven by the fire"
# instead of generic turbulence. Geometry-only inputs (entry.door/vod/
# voc/candles + schematic's static positions) -- never field.u/w/speed --
# so the result stays identical for every frame of a scenario (see
# _seed_points' cache), preserving the flicker fix. A closed vent gets no
# seeds at all: nothing flows through it, so nothing should emanate from
# it. Cluster size scales with each opening's physical extent (door
# height, vent width) via _FEATURE_SEED_SPACING rather than a fixed count
# per feature, so e.g. a wide door reads with more seeds than a narrow
# one. A light background grid (_BG_GRID_NX/NZ, well below the old
# uniform grid's 12x6) is layered underneath so the room's return
# circulation still draws -- feature seeding concentrates attention, it
# doesn't empty out the rest of the room.
_FEATURE_SEED_SPACING = 0.03      # m: target seed-to-seed spacing within a cluster
_FIRE_SEED_HEIGHTS = (0.0, 0.03, 0.06)   # m above the floor, per candle position
_BG_GRID_NX = 6
_BG_GRID_NZ = 4


class StreamlinePanel(QtWidgets.QWidget):
    """Analysis-page tab: a whole-plane matplotlib streamplot of the
    validated U/W-VELOCITY field, colored by local speed. Independent of
    VelocityPanel -- no shared state, no probes, nothing here writes to
    velocity.py or velocity_panel.py."""

    def __init__(self, provider, manifest: list, fps: int,
                 density: float = DEFAULT_DENSITY, cmap: str = DEFAULT_CMAP, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._by_index = {e.case_index: e for e in self._manifest}
        self._fps = max(1, fps)
        self._cmap = cmap
        self._loaded = False
        self._fields: dict = {}         # case_index -> vel.VectorField (successfully computed)
        self._gate_reasons: dict = {}   # case_index -> str
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()
        self._seed_cache: dict = {}   # (case_index, extent) -> fixed start_points array, see _seed_points

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Velocity (streamlines)")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        header = QtWidgets.QHBoxLayout()
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Streamline scenario")
        self.scenario_combo.setToolTip("Which scenario's vector field to compute and display")
        header.addWidget(self.scenario_combo)
        header.addStretch(1)
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Density"))
        # Constructor/config parameter (requirement 4): starts at
        # matplotlib's own default (1.0) and is exposed as a live control
        # so it's tunable without touching any rendering logic below.
        self.density_spin = QtWidgets.QDoubleSpinBox()
        self.density_spin.setAccessibleName("Streamline density")
        self.density_spin.setToolTip(
            "matplotlib streamplot density -- higher packs more streamlines closer together")
        self.density_spin.setRange(0.2, 5.0)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setValue(density)
        controls.addWidget(self.density_spin)
        self.linewidth_check = QtWidgets.QCheckBox("Line width scales with speed")
        self.linewidth_check.setAccessibleName("Scale line width by speed")
        self.linewidth_check.setChecked(True)
        controls.addWidget(self.linewidth_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Velocity streamlines canvas")
        layout.addWidget(self.canvas, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.density_spin.valueChanged.connect(lambda _v: self._render())
        self.linewidth_check.stateChanged.connect(lambda _s: self._render())

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        was_loaded = self._loaded
        self.ensure_loaded()
        if was_loaded:
            # Not the first show (ensure_loaded already rendered that case):
            # catch up on whatever frame playback moved to while this tab
            # was hidden and set_bus()'s isVisible() gate was skipping it.
            self._render()

    def set_bus(self, bus) -> None:
        """Follow the shared playback frame (Selection.time_s), same
        set_bus precedent as device_panel.py/velocity_panel.py -- this
        panel has no frame_slider for the generic bind_to_bus sync to
        hook. One-way: this panel never publishes a selection, only
        reacts."""
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

    # -------------------------------------------------- vector field access
    def _ensure_field(self, case_index: int):
        """Lazily compute (and cache) the VectorField for `case_index` --
        identical convention to VelocityPanel._ensure_field, duplicated
        rather than shared so this panel has zero coupling to
        velocity_panel.py's internals (module docstring)."""
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

    # ---------------------------------------------------------- seed points
    def _seed_points(self, case_index, entry, geometry, x0: float, x1: float,
                      z0: float, z1: float) -> np.ndarray:
        """Fixed start_points for streamplot(), generated once per
        (scenario, extent) and cached -- reused for every frame of that
        scenario/extent rather than regenerated, which is the actual
        flicker fix (see the class docstring's stability note). Feature-
        based when `entry` (and its `geometry`, already computed by
        _render()) is available -- see _feature_seed_points and the
        _FEATURE_SEED_SPACING/_BG_GRID_NX/NZ comment. Falls back to the
        old uniform grid only if `entry` is None (case_index absent from
        the manifest -- shouldn't normally happen, but there's no
        geometry to seed from in that case)."""
        key = (case_index, x0, x1, z0, z1)
        seeds = self._seed_cache.get(key)
        if seeds is None:
            if entry is not None:
                seeds = self._feature_seed_points(entry, geometry, x0, x1, z0, z1)
            else:
                xs = np.linspace(x0, x1, _SEED_GRID_NX)
                zs = np.linspace(z0, z1, _SEED_GRID_NZ)
                xx, zz = np.meshgrid(xs, zs)
                seeds = np.column_stack([xx.ravel(), zz.ravel()])
            self._seed_cache[key] = seeds
        return seeds

    def _feature_seed_points(self, entry, geometry: dict, x0: float, x1: float,
                              z0: float, z1: float) -> np.ndarray:
        """Geometry-only seed clusters: fire position(s), the door, each
        open/HVAC vent, plus a light background grid. Reads only
        entry.candles and `geometry` (itself derived only from
        entry.door/vod/voc) -- never field.u/w/speed -- see the
        _FEATURE_SEED_SPACING comment for why that matters."""
        points = []

        for fx, fz in fire_positions(entry.candles):
            for dz in _FIRE_SEED_HEIGHTS:
                points.append((fx, fz + dz))

        dx0, dz0, dx1, dz1 = geometry["door"]
        n_door = max(2, round(abs(dz1 - dz0) / _FEATURE_SEED_SPACING) + 1)
        for z in np.linspace(dz0, dz1, n_door):
            points.append((dx0, z))

        # geometry["vents"] is [VOD-lower, VOC-lower, VOD-topface,
        # VOC-topface] (see room_overlay_geometry's docstring) -- only
        # the first two are distinct physical openings; the top-face pair
        # redraws the same two vents on the ceiling slab's other face.
        for (vx0, vz0, vx1, vz1), state in geometry["vents"][:2]:
            if state == "closed":
                continue
            n_vent = max(2, round(abs(vx1 - vx0) / _FEATURE_SEED_SPACING) + 1)
            for x in np.linspace(vx0, vx1, n_vent):
                points.append((x, vz0))

        feature = np.array(points, dtype=float) if points else np.empty((0, 2))

        xs = np.linspace(x0, x1, _BG_GRID_NX)
        zs = np.linspace(z0, z1, _BG_GRID_NZ)
        xx, zz = np.meshgrid(xs, zs)
        background = np.column_stack([xx.ravel(), zz.ravel()])

        return np.vstack([feature, background])

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
        if field is None:
            reason = self._gate_reasons.get(case_index, "U/W-VELOCITY not available for this scenario.")
            ax.set_title("Streamlines unavailable -- needs U/W velocity data", fontsize=9, color="#B00020")
            ax.set_xticks([]); ax.set_yticks([])
            self.status.setText(reason)
            self.canvas.draw_idle()
            return
        self.status.setText("")

        # Scenario geometry (moved up from the room-outline draw below --
        # feature seeding needs it too, and this way it's computed once
        # per frame instead of twice). entry is None only if case_index is
        # somehow absent from the manifest.
        entry = self._by_index.get(case_index)
        geometry = (room_overlay_geometry(entry.door, entry.vod, entry.voc)
                    if entry is not None else None)

        # Live frame (Analysis dynamic-visualizations pass): field.u/w/speed
        # are indexed directly below (unlike VelocityPanel's quiver_at()/
        # streamline_at(), which clamp internally), so clamp here.
        frame_index = min(max(self._current_index, 0), field.n_frames - 1)
        x0, x1, z0, z1 = field.extent

        # Row 0 of u/w is the ceiling (z1); streamplot needs an increasing
        # y array -- flip the frame vertically (reindex only, W's sign is
        # untouched) rather than pass a decreasing y, which streamplot
        # does not support correctly. See module docstring.
        u_frame = np.flip(field.u[frame_index], axis=0)
        w_frame = np.flip(field.w[frame_index], axis=0)
        speed_frame = np.flip(field.speed[frame_index], axis=0)

        n_z, n_x = u_frame.shape
        x = np.linspace(x0, x1, n_x)
        z = np.linspace(z0, z1, n_z)  # increasing, matches the flipped frame

        linewidth = 1.0
        if self.linewidth_check.isChecked():
            linewidth = LINEWIDTH_BASE + LINEWIDTH_SCALE * (speed_frame / SPEED_REF_MS)

        # Fixed color scale against SPEED_REF_MS, not this frame's own
        # min/max speed -- same fixed-reference reasoning as linewidth
        # above. PowerNorm (gamma=0.45), not linear: real speeds span a
        # ~20x dynamic range (room circulation ~0.05 m/s vs. plume ~1.0
        # m/s), and a linear ramp buries circulation near-black to leave
        # headroom for the plume; gamma<1 stretches the low end so both
        # bands render legibly.
        norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=SPEED_REF_MS)
        # Two traces per seed (see MAXLENGTH) -- same fixed seeds, same
        # color/linewidth/density for both, only integration_direction
        # (and arrowsize, see ARROWSIZE/ARROWSIZE_HIDDEN) differs, so this
        # doesn't change what seeds exist or where.
        seeds = self._seed_points(case_index, entry, geometry, x0, x1, z0, z1)
        streamplot_kwargs = dict(
            color=speed_frame, cmap=self._cmap, norm=norm,
            density=self.density_spin.value(),
            linewidth=linewidth,
            start_points=seeds,
            maxlength=MAXLENGTH,
            minlength=MINLENGTH,
        )
        strm = ax.streamplot(x, z, u_frame, w_frame,
                              integration_direction="forward", arrowsize=ARROWSIZE,
                              **streamplot_kwargs)
        ax.streamplot(x, z, u_frame, w_frame,
                       integration_direction="backward", arrowsize=ARROWSIZE_HIDDEN,
                       **streamplot_kwargs)
        self.canvas.fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04, label="Speed (m/s)")

        # Room outline (visual-clarity follow-up): walls/door/vents from the
        # real &HOLE-derived geometry (schematic.room_overlay_geometry), the
        # same source and colors the Live Viewer overlays on its own
        # heatmap -- this plot had no spatial reference at all before, just
        # streamlines floating with no visible walls/door/vents to relate
        # the flow to. Drawn as plain ax.plot() calls (not views.py's
        # LineCollection/blit-cache machinery, which this from-scratch-
        # redraw-every-frame canvas doesn't use or need -- same "thin,
        # zero-coupling" precedent as this module's other duplicated bits).
        # Thinned/faded/pushed behind the flow -- see _GEOMETRY_ALPHA/
        # _GEOMETRY_ZORDER's comment above _WALL_COLOR.
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

        # Same axis scale/aspect convention as VelocityPanel's imshow
        # background (extent + aspect='auto') so the two views are
        # directly visually comparable (requirement 6).
        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("auto")
        ax.set_xticks([]); ax.set_yticks([])
        t_s = frame_index / self._fps
        ax.set_title(f"Streamlines · density {self.density_spin.value():.1f} · t={t_s:.1f}s", fontsize=8)

        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()
