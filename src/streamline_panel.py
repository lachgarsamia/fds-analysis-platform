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
from matplotlib.colors import Normalize
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, plot_fg_color
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
from registry import AMBIENT_C
from schematic import room_overlay_geometry, ROOM_X, ROOM_Z
from slice_key import SliceKey
import measure as mz
import velocity as vel

DEFAULT_DENSITY = 1.0  # matplotlib's own streamplot default
# Fallback colormap for the speed channel only (used if a scenario has no
# readable TEMPERATURE slice). Streamlines are normally coloured by
# TEMPERATURE (see _TEMP_CMAP) -- never jet.
DEFAULT_CMAP = "viridis"

# Streamlines are coloured by the local gas TEMPERATURE (same y=0 slice
# lic_flow_panel.py's isotherm contours pull from). "turbo" -- a
# perceptually-ordered rainbow whose bands stay distinct across the whole
# range and whose endpoints are dark (never white, so a line is always
# visible on the fixed-white canvas). Chosen over "coolwarm" (measured
# against real data: ~86% of this dataset's cells sit in 22-45 C, which
# coolwarm + a diverging norm crushed into indistinguishable near-white
# mid-tones) and over "jet"/"nipy_spectral" (jet's banding artefacts;
# nipy_spectral tops out white/grey and loses hot lines). See _render.
_TEMPERATURE_KEY = SliceKey("TEMPERATURE", 1, 0)
_TEMP_CMAP = "turbo"

# Opening-targeted seeds, in addition to the room-coverage grid. Placed by
# the *real prevailing flow direction* through each opening (mean normal
# velocity over the last third of the run, per opening location), so that
# inflow and outflow physically separate: an inflow line is seeded on the
# far side of the opening (corridor, or the plenum above a vent) and
# visibly crosses in; an outflow line is seeded on the near side and
# visibly crosses out.
#
# The doorway is *not* assumed one-way. Measured across configs
# (docs note / commit msg): with both ceiling vents open the door is pure
# supply air (inflow at every height) because the vents do all the
# exhaust; but with the ceiling vents closed or partly closed it develops
# the classic compartment two-layer pattern -- cool inflow low, hot
# outflow high, a neutral plane near mid-opening (e.g. z~0.075 m in
# c2_d1_vod1_voc1, outflow reaching -0.14 m/s and ~41 C at z~0.13). The
# per-z sign read below captures whichever pattern the scenario actually
# has.
#
# The doorway is probed over its FULL physical height (floor to lintel) at
# a FIXED spacing (_DOOR_PROBE_DZ), not a fixed sample count -- the
# thinnest real flow layer measured across the scenarios is the
# still-developing outflow in c2_d1_vod1_voc0, ~0.009 m at the run mean
# (~0.037 m at t=115 s). A fixed 15-point count on the 0.16 m wide door
# was 0.011 m spacing -- 0-1 samples across that layer, a knife-edge
# catch, and it would only get worse on a taller door. 0.004 m spacing
# puts >=2 samples across any layer >=0.008 m regardless of door height
# (~40 bilinear probes for the wide door, once, cached -- negligible).
# Detection (fine) is then decoupled from seed count: consecutive
# same-sign z-runs are grouped into bands and each band gets at most
# _DOOR_SEEDS_PER_BAND evenly spread seeds. Spans come from
# schematic.room_overlay_geometry -- never hardcoded.
_FLOW_EPS = 0.02            # m/s: |normal component| below this = no net flow, skip
_DOOR_PROBE_DZ = 0.004     # m: z spacing at which the doorway U sign is read (not a count)
_DOOR_SEEDS_PER_BAND = 5   # max seeds placed per detected inflow / outflow z-band
_DOOR_PROBE_DX = 0.014     # x (room side of the wall line) where U is sampled
_DOOR_IN_DX = -0.020       # inflow seeds sit this far corridor-side of the wall line
_DOOR_OUT_DX = 0.022       # outflow seeds this far room-side
_VENT_PROBE_NX = 6        # x-samples across a vent span (fixed 0.08 m span -> spacing OK)
_VENT_PROBE_DZ = 0.010    # depth below the ceiling underside where W is sampled
_VENT_OUT_DEPTHS = (0.006, 0.024, 0.044)   # outflow seeds: z below the ceiling underside
_VENT_IN_ABOVE = 0.016                      # inflow seed: z above the slab top (plenum)

# Room outline colors. Door/vent colors match views.py's own door/vent-
# state convention (duplicated rather than imported -- same "zero
# coupling to the other velocity views" precedent this module already
# follows for _ensure_field). Walls deliberately do NOT reuse views.py's
# own wall color (#FFFFFF): that reads fine there against a colored
# heatmap with a dark path-effect casing around the white line, but this
# panel's canvas background is plain, fixed white (MplCanvas.PLOT_BG) --
# a bare white line on white would be invisible, so this uses
# plot_fg_color() (widgets.py's own "readable against this canvas"
# helper) instead, resolved at render time. Geometry itself comes from
# schematic.room_overlay_geometry, the same real &HOLE-derived source
# the Live Viewer overlays.
_DOOR_COLOR = "#38BDF8"
_VENT_STATE_COLORS = {"open": "#22C55E", "closed": "#94A3B8", "HVAC": "#F59E0B"}

# Frame-to-frame stability fix: streamplot()'s automatic seeding
# (density=...) picks new seed locations independently on every call, so
# the rendered pattern reshuffled discontinuously between frames even
# though the underlying U/W field itself evolves smoothly -- the seeds
# moved, not the flow. Fixed explicit start_points (see _seed_points)
# sidesteps this: the same seed grid drives every frame, so the pattern
# now shifts continuously with the field instead of jumping.
#
# Seeded across the room's own bounds (schematic.ROOM_X/ROOM_Z), not the
# full raw mesh domain -- most of the domain outside the room is the FDS
# door-corridor/ambient-air buffer this app otherwise doesn't visualize
# (same room-vs-domain distinction views.py's room-outline crop and
# device_panel.py's locator canvas already make), so seeding it produced
# streamlines with no real fire/smoke relevance floating outside the
# drawn room outline below. A traced line can still leave the room
# through the door -- only the *starting* points are room-bounded, the
# vector field itself is unchanged.
_SEED_GRID_NX = 12
_SEED_GRID_NZ = 6


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
        self._temperatures: dict = {}   # case_index -> (t, z, x) TEMPERATURE array, or None
        self._gate_reasons: dict = {}   # case_index -> str
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()
        self._seed_cache: dict = {}   # (door, vod, voc, x0, x1, z0, z1) -> start_points array, see _seed_points

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

    def _ensure_temperature(self, case_index: int):
        """Lazily fetch (and cache) the full (t, z, x) TEMPERATURE array
        for `case_index` at the same y=0 plane U/W use -- for colouring the
        streamlines (CHANGE 3). Row 0 is the ceiling, same as u/w. Returns
        None (and the caller falls back to speed colouring) if TEMPERATURE
        can't be read for this scenario -- never fabricated."""
        if case_index in self._temperatures:
            return self._temperatures[case_index]
        try:
            temp = np.asarray(self._provider.get(case_index, _TEMPERATURE_KEY), dtype=float)
        except Exception:  # noqa: BLE001 - gated / missing plane -> fall back to speed colour
            self._temperatures[case_index] = None
            return None
        self._temperatures[case_index] = temp
        return temp

    # ---------------------------------------------------------- seed points
    def _opening_seed_points(self, geometry: dict, extent: tuple,
                             u_mean: np.ndarray, w_mean: np.ndarray) -> np.ndarray:
        """Extra start points at the doorway and each *open* ceiling vent,
        placed by the real prevailing flow direction there. `u_mean`/
        `w_mean` are the run-averaged (late-phase) velocity components,
        (n_z, n_x) with row 0 = ceiling (the raw VectorField convention);
        `geometry` is schematic.room_overlay_geometry (never hardcoded
        spans); `extent` is the plot grid (x0, x1, z0, z1) and every point
        is clamped inside it. Returns (N, 2), possibly empty.

        Direction is made legible by *where* the seed sits, not just by an
        arrowhead: an inflow seed is on the far side of the opening
        (corridor for a door, plenum above a vent) so its line visibly
        crosses in; an outflow seed is on the near side so its line
        visibly crosses out. Openings that the data shows are one-way get
        seeds on one side only."""
        x0, x1, z0, z1 = extent
        clamp = lambda a, lo, hi: min(max(a, lo), hi)
        probe = lambda arr, x, z: mz.probe_value(arr, extent, x, z)
        pts: list = []

        # --- Doorway (normal component = U; U > 0 is into the room). Read
        #     the U sign up the FULL opening height at _DOOR_PROBE_DZ
        #     spacing, group consecutive same-sign runs into bands, and
        #     seed each band (inflow -> corridor side, outflow -> room
        #     side) with up to _DOOR_SEEDS_PER_BAND evenly spread points.
        dx0, dz0, dx1, dz1 = geometry.get("door", (None,) * 4)
        if dx0 is not None:
            door_x = min(dx0, dx1)                       # the wall line (dx0 == dx1)
            lo_z, hi_z = clamp(min(dz0, dz1) + 0.004, z0, z1), clamp(max(dz0, dz1), z0, z1)
            if hi_z > lo_z:
                probe_x = clamp(door_x + _DOOR_PROBE_DX, x0, x1)
                zs = np.arange(lo_z, hi_z + _DOOR_PROBE_DZ * 0.5, _DOOR_PROBE_DZ)
                us = np.array([probe(u_mean, probe_x, z) for z in zs])
                sign = np.where(us > _FLOW_EPS, 1, np.where(us < -_FLOW_EPS, -1, 0))
                i = 0
                while i < len(sign):
                    if sign[i] == 0:
                        i += 1
                        continue
                    j = i
                    while j < len(sign) and sign[j] == sign[i]:
                        j += 1
                    seed_x = clamp(door_x + (_DOOR_IN_DX if sign[i] > 0 else _DOOR_OUT_DX), x0, x1)
                    k = min(_DOOR_SEEDS_PER_BAND, j - i)
                    pts.extend((seed_x, sz) for sz in np.linspace(zs[i], zs[j - 1], k))
                    i = j

        # --- Open ceiling vents (normal component = W; W > 0 is up/out).
        #     geometry["vents"] lists each vent twice (a segment at the slab
        #     underside and one at its top face) -- collapse to one span per
        #     open vent, then probe the W sign across it and seed outflow
        #     columns below the vent / inflow columns above it.
        spans: dict = {}
        for (vx0, vz0, vx1, vz1), state in geometry.get("vents", []):
            if state != "open":
                continue
            span = (round(min(vx0, vx1), 6), round(max(vx0, vx1), 6))
            lo, hi = spans.get(span, (vz0, vz1))
            spans[span] = (min(lo, vz0, vz1), max(hi, vz0, vz1))
        for (sx0, sx1), (underside, slab_top) in spans.items():
            for sx in np.linspace(sx0, sx1, _VENT_PROBE_NX):
                sx = clamp(sx, x0, x1)
                w = probe(w_mean, sx, underside - _VENT_PROBE_DZ)
                if w > _FLOW_EPS:                        # outflow -> below the vent
                    pts.extend((sx, clamp(underside - d, z0, z1)) for d in _VENT_OUT_DEPTHS)
                elif w < -_FLOW_EPS:                     # inflow -> above the slab
                    pts.append((sx, clamp(slab_top + _VENT_IN_ABOVE, z0, z1)))

        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def _seed_points(self, entry, field) -> np.ndarray:
        """start_points for streamplot(): the fixed 12x6 room-coverage grid
        (unchanged) concatenated with opening-targeted seeds
        (_opening_seed_points), placed by `field`'s late-run mean flow
        direction through each opening. Cached per (door, vod, voc,
        extent) -- the config is part of the key so a scenario switch
        regenerates the seeds; the mean is deterministic per scenario, so
        the seed set is still fixed across playback frames.

        `entry` is a manifest ScenarioEntry (its .door/.vod/.voc drive the
        geometry); `field` is the scenario's velocity.VectorField."""
        extent = field.extent
        x0, x1, z0, z1 = extent
        door = getattr(entry, "door", 0)
        vod = getattr(entry, "vod", 0)
        voc = getattr(entry, "voc", 0)
        key = (door, vod, voc, x0, x1, z0, z1)
        seeds = self._seed_cache.get(key)
        if seeds is None:
            # Room-coverage grid, clamped to the room's own bounds (see the
            # _SEED_GRID_NX/NZ comment) -- unchanged 12x6.
            gx0, gx1 = max(x0, min(ROOM_X)), min(x1, max(ROOM_X))
            gz0, gz1 = max(z0, min(ROOM_Z)), min(z1, max(ROOM_Z))
            xx, zz = np.meshgrid(np.linspace(gx0, gx1, _SEED_GRID_NX),
                                 np.linspace(gz0, gz1, _SEED_GRID_NZ))
            room_seeds = np.column_stack([xx.ravel(), zz.ravel()])
            # Prevailing flow direction: mean of the last third of the run
            # (the quasi-steady phase), per opening location.
            n = field.u.shape[0]
            lo = max(0, int(n * 2 / 3))
            u_mean = np.asarray(field.u[lo:]).mean(axis=0)
            w_mean = np.asarray(field.w[lo:]).mean(axis=0)
            opening_seeds = self._opening_seed_points(
                room_overlay_geometry(door, vod, voc), extent, u_mean, w_mean)
            seeds = (np.vstack([room_seeds, opening_seeds])
                     if opening_seeds.size else room_seeds)
            self._seed_cache[key] = seeds
        return seeds

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

        # Line WIDTH still scales with local speed (a separate channel from
        # colour -- CHANGE 3 keeps this as-is).
        linewidth = 1.0
        if self.linewidth_check.isChecked():
            peak = float(speed_frame.max())
            if peak > 1e-9:
                linewidth = 0.5 + 2.0 * (speed_frame / peak)

        # Line COLOUR = local gas TEMPERATURE (turbo). Same y=0 slice, same
        # row-0-is-ceiling flip as u/w. Falls back to speed (viridis) only
        # if this scenario has no readable TEMPERATURE -- never fabricated.
        temp = self._ensure_temperature(case_index)
        if temp is not None and temp.shape[0]:
            temp_frame = np.flip(temp[min(frame_index, temp.shape[0] - 1)], axis=0)
            color_arr, color_cmap = temp_frame, _TEMP_CMAP
            # Plain linear norm from ~ambient to the 95th percentile.
            # Measured on the real data (c2_d0_vod0_voc0, t=115s): p2=21,
            # p50=29, p90=40, p95=54, p99=126, max=378 -- a right-skewed
            # spread from ambient, NOT diverging around a centre, so a
            # TwoSlopeNorm(vcenter=ambient) just wasted half its range on
            # temperatures that don't occur and crushed the 22-45 C band
            # (86% of cells) into near-white. p95 as the ceiling keeps the
            # plume core (a <1%-of-cells outlier) from re-flattening the
            # scale; a [+20, +80] clamp keeps a cool or a very hot scenario
            # legible. With turbo this gives distinct bands: cold inflow
            # (blue) / mixing (green-yellow) / warm (orange) / hot outflow
            # (red).
            p95 = float(np.nanpercentile(temp_frame, 95))
            vmin = min(float(np.nanpercentile(temp_frame, 2)), AMBIENT_C)
            vmax = float(np.clip(p95, AMBIENT_C + 20.0, AMBIENT_C + 80.0))
            color_norm = Normalize(vmin=vmin, vmax=vmax)
            cbar_label = "Temperature (°C)"
        else:
            color_arr, color_cmap, color_norm = speed_frame, self._cmap, None
            cbar_label = "Speed (m/s)"

        # start_points = the 12x6 room grid + opening-targeted seeds
        # (doorway + open vents), placed by the scenario's real prevailing
        # flow direction through each opening and keyed on the door/vent
        # config so a scenario switch regenerates them. No
        # integration_direction override anywhere -- streamplot's default
        # 'both' still traces each seed forward AND backward.
        entry = self._by_index.get(case_index)
        strm = ax.streamplot(
            x, z, u_frame, w_frame,
            color=color_arr, cmap=color_cmap, norm=color_norm,
            density=self.density_spin.value(),
            linewidth=linewidth,
            start_points=self._seed_points(entry, field),
        )
        self.canvas.fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

        # Room outline (walls/door/vents) from the real &HOLE-derived
        # geometry (schematic.room_overlay_geometry), the same source and
        # colors the Live Viewer overlays on its own heatmap -- gives the
        # streamlines spatial context instead of floating with no visible
        # walls/door/vents to relate the flow to. Plain ax.plot() calls,
        # not views.py's LineCollection/blit-cache machinery, which this
        # from-scratch-redraw-every-frame canvas doesn't use or need.
        if entry is not None:
            geometry = room_overlay_geometry(entry.door, entry.vod, entry.voc)
            wall_color = plot_fg_color()
            for wx0, wz0, wx1, wz1 in geometry["walls"]:
                ax.plot([wx0, wx1], [wz0, wz1], color=wall_color,
                        linestyle="--", linewidth=1.4, zorder=6)
            dx0, dz0, dx1, dz1 = geometry["door"]
            ax.plot([dx0, dx1], [dz0, dz1], color=_DOOR_COLOR, linewidth=2.6, zorder=6)
            for (vx0, vz0, vx1, vz1), state in geometry["vents"]:
                ax.plot([vx0, vx1], [vz0, vz1],
                        color=_VENT_STATE_COLORS.get(state, "#94A3B8"),
                        linewidth=4.0, zorder=6)

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
