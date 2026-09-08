"""Tracer particles flow panel: a fourth, independent visualization of the
same validated U/W-VELOCITY field, alongside VelocityPanel's quiver/probe-
streamlines, StreamlinePanel's matplotlib streamplot, and LICFlowPanel's
Line Integral Convolution. Does not touch any of those three modules or
their panels in any way -- reuses velocity.py's VectorField exactly as
they do (read-only).

Why tracer particles instead of another streamline/LIC variant: every
static, single-snapshot-per-frame technique (streamplot, LIC, quiver) asks
the viewer to reconstruct motion from a frozen picture. A moving dot with a
short fading trail shows it directly -- this is also the technique NIST's
own Smokeview (the tool actually built for visualizing this exact kind of
FDS output) leads with, ahead of streamlines or LIC, for exactly that
reason (animated tracer particles + shaded contours + flow vectors).

Real advection, not decoration: each particle is pushed every render tick
by the real, bilinearly-sampled (u, w) velocity at its current position
(measure.probe_value's scalar formula, vectorized locally for a whole
particle pool at once -- duplicated rather than batching through that
scalar function, same "small utility duplication, zero cross-panel
coupling" precedent every sibling panel here already follows). This is a
different mechanism from cinema/particles.py's EmberParticles, which is
purely decorative (fixed buoyancy + jitter, never actually reads the
velocity field for its own motion) -- that module's structure-of-arrays
pool/step()/render_arrays() shape is reused as an architectural precedent
only, not its physics.

Seeded within the room's own bounds (schematic.ROOM_X/ROOM_Z), same
reasoning as streamline_panel.py's _seed_points: the FDS door-corridor/
ambient-air buffer outside the room isn't what this analysis is about.
Unlike streamline_panel's *fixed* grid (needed there for frame-to-frame
stability of a redrawn-from-scratch line), particles persist and carry
their own state across ticks, so seeding is a one-time random scatter
within the room, not a grid -- unnaturally regular starting positions
would read as a grid pattern once the dots started moving.

Room-boundary containment (visual-clarity fix): a live particle is
contained by the room's own walls and ceiling, not the far-away FDS
domain edge -- it may only leave through an actual opening. The ceiling
(ROOM_Z[1]) is solid except where a VOD/VOC vent is currently *open*; the
left wall (ROOM_X[0]) is solid except across the doorway's z-span; the
right wall and floor are always solid. `_open_vent_spans_and_duct_top()`
and `_door_z_span()` read the open x-span(s), the ceiling slab's real top
face, and the door opening's height straight off room_overlay_geometry()
-- the same geometry the drawn room outline itself uses -- so containment
is driven by the scenario's actual door/vod/voc state, never hardcoded. A
particle that does cross an open vent is allowed to continue only as far
as that real slab top face (the opening's true physical depth); one that
crosses the doorway continues a small fixed _DOOR_EXIT_DEPTH into the
(unmodelled) corridor. Past that it's treated as gone and respawned, not
left wandering outside the room.

Trail rendering is clipped separately, and more tightly: a trail segment
only draws where both endpoints sit inside the room rectangle, or inside
a small TRAIL_VENT_MARGIN_FRAC allowance directly outside an open vent's
span or the doorway (see _render()) -- a render-only tolerance so a trail
streaming out an opening doesn't get guillotined exactly at the room
line. It never widens an opening, changes the containment bound above, or
touches velocity sampling -- only which already-computed trail segments
get drawn.

Row-0-is-ceiling / dt convention: same as every sibling panel (see
streamline_panel.py's own module docstring for the flip reasoning).
Advection uses real time (dt = 1/fps): a particle moves exactly as far as
the real flow would carry it in one playback tick, not an arbitrary
per-render step size. Where the field is fast enough that a single Euler
step would overshoot several grid cells (notably the ~4 m/s HVAC extract
inlet), that tick is transparently split into sub-steps -- same field,
same total dt (see _ParticlePool.step / _CFL_SUBSTEP_CELLS).
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, plot_fg_color
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
from schematic import room_overlay_geometry, ROOM_X, ROOM_Z
import velocity as vel

DEFAULT_N_PARTICLES = 150
DEFAULT_CMAP = "viridis"  # perceptually uniform sequential -- never jet

# Same fixed speed-color calibration streamline_panel.py/lic_flow_panel.py
# already validated on this dataset (duplicated, not imported -- see
# module docstring) -- so a given color means the same speed across every
# flow-visualization tab, not just within this one.
SPEED_REF_MS = 1.2
SPEED_GAMMA = 0.45

# Trail length (positions kept per particle, oldest to newest) and per-
# particle lifespan range (frames) before a staggered respawn -- randomized
# per particle so the whole pool never visibly "resets" at once, matching
# cinema/particles.py's EmberParticles' own age/life convention.
TRAIL_LEN = 8
LIFE_MIN_FRAMES = 30
LIFE_MAX_FRAMES = 90

# Adaptive sub-stepping (see _ParticlePool.step). Plain forward-Euler
# advection at the playback tick (dt = 1/fps = 0.25 s) lets a particle in
# a fast jet -- e.g. a ~4 m/s HVAC extract inlet, ~10x the room's typical
# ~0.35 m/s -- leap ~90 grid cells in a single step. That is pure
# numerical overshoot: it renders as long straight diagonal streaks and a
# tangled knot at the vent, not as real flow (the jet field itself is
# smooth and coherent). The tick is split into k sub-steps so no sub-step
# advances a particle more than _CFL_SUBSTEP_CELLS grid cells, re-sampling
# the frozen-for-this-frame field each time -- identical field, identical
# total dt, just resolved. k is capped at _MAX_SUBSTEPS for cost; measured
# on the HVAC scenario the cap binds nearly every frame (pool-wide vmax is
# the ~3.8 m/s extract jet), a natural-fire scenario sits around k = 7.
_CFL_SUBSTEP_CELLS = 0.5
_MAX_SUBSTEPS = 40

# Door/vent colors match views.py's/streamline_panel.py's own convention.
_DOOR_COLOR = "#38BDF8"
_VENT_STATE_COLORS = {"open": "#22C55E", "closed": "#94A3B8", "HVAC": "#F59E0B"}

# Render-only tolerance (module docstring) -- NOT the particle's own
# opening-exit bound (that's the real slab top face / _DOOR_EXIT_DEPTH,
# see _ceiling_limit and _wall_x_limits). A trail may render up to this
# fraction of the particle's own allowed excursion past the ceiling line
# (over an open vent's span) or past the left wall line (over the
# doorway's z-span), so a trail streaming out an opening doesn't get cut
# exactly at the room line. Kept < 1.0 so the margin never exceeds the
# real physical depth a particle could actually have reached.
TRAIL_VENT_MARGIN_FRAC = 0.6

# Horizontal analogue of the ceiling's vent gap. Unlike an open ceiling
# vent (which opens onto the ceiling slab's real modelled top face,
# _open_vent_spans_and_duct_top's duct_top_z), the doorway opens onto an
# unmodelled corridor, so this is a small fixed depth a particle may run
# past the left wall line through the door's z-span before it's treated
# as gone -- set to the ceiling slab depth (0.02 m) for visual
# consistency with the vent allowance.
_DOOR_EXIT_DEPTH = 0.02


def _open_vent_spans_and_duct_top(entry) -> tuple:
    """(open_vent_x_spans, duct_top_z) for `entry`'s scenario -- read
    straight off room_overlay_geometry(entry.door, entry.vod, entry.voc),
    the same call the drawn room outline itself uses, not a hardcoded
    vent position/state. `open_vent_x_spans` is a deduplicated list of
    (x0, x1) for every VOD/VOC opening currently in the "open" state
    (HVAC and closed are both solid -- see module docstring); `duct_top_z`
    is the ceiling slab's real top face z (where a particle that's
    actually passed through an open vent has cleared the opening),
    read off the geometry's own vent-segment z-coordinates rather than a
    separate private constant."""
    geometry = room_overlay_geometry(entry.door, entry.vod, entry.voc)
    vents = geometry["vents"]
    if not vents:
        return [], None
    open_spans = sorted({(seg[0], seg[2]) for seg, state in vents if state == "open"})
    duct_top_z = max(seg[1] for seg, _state in vents)
    return open_spans, duct_top_z


def _door_z_span(entry) -> tuple:
    """(z0, z1) of the doorway opening in the room's left wall for
    `entry`'s scenario -- read straight off room_overlay_geometry()'s own
    "door" segment (the same geometry the drawn outline uses), so it
    tracks narrow vs wide door without a hardcoded height. None if the
    scenario has no door segment."""
    door = room_overlay_geometry(entry.door, entry.vod, entry.voc).get("door")
    if not door:
        return None
    _dx0, dz0, _dx1, dz1 = door
    return (min(dz0, dz1), max(dz0, dz1))


def _sample_uw(u_frame: np.ndarray, w_frame: np.ndarray, extent: tuple,
               x: np.ndarray, z: np.ndarray) -> tuple:
    """Bilinearly-interpolated (u, w) at every (x[i], z[i]), clamped to the
    domain edges -- same formula as measure.py's probe_value, vectorized
    for a whole particle pool at once instead of one point per call (see
    module docstring for why this isn't imported/batched through that
    scalar function)."""
    from scipy.ndimage import map_coordinates
    x0, x1, z0, z1 = extent
    n_z, n_x = u_frame.shape
    col = (x - x0) / (x1 - x0) * (n_x - 1) if x1 != x0 else np.zeros_like(x)
    row = (z1 - z) / (z1 - z0) * (n_z - 1) if z1 != z0 else np.zeros_like(z)
    row = np.clip(row, 0.0, n_z - 1)
    col = np.clip(col, 0.0, n_x - 1)
    coords = np.stack([row, col])
    u = map_coordinates(u_frame.astype(float), coords, order=1, mode="nearest")
    w = map_coordinates(w_frame.astype(float), coords, order=1, mode="nearest")
    return u, w


class _ParticlePool:
    """Structure-of-arrays tracer pool: step() advects by the real field
    and respawns expired/escaped particles, positions() returns the
    current (x, z) plus a fading trail for rendering. Pure NumPy, no Qt --
    same "step()/render-ready arrays" shape as cinema/particles.py's
    EmberParticles, different physics (see module docstring)."""

    def __init__(self, n: int, room_bounds: tuple, open_vent_spans: list = (),
                 duct_top_z: float = None, door_z_span: tuple = None, seed: int = 0):
        self._room_x0, self._room_x1, self._room_z0, self._room_z1 = room_bounds
        # Room-boundary containment (module docstring): open_vent_spans,
        # duct_top_z and door_z_span all come from the scenario's real
        # geometry (_open_vent_spans_and_duct_top / _door_z_span), not
        # hardcoded here. No open vents (or no geometry) -> duct_top_z
        # falls back to the room's own ceiling, i.e. solid everywhere;
        # door_z_span None -> the left wall is solid its whole height.
        self._open_vent_spans = list(open_vent_spans)
        self._duct_top_z = duct_top_z if duct_top_z is not None else self._room_z1
        self._door_z_span = door_z_span
        self._rng = np.random.default_rng(seed)
        self.n = n
        self._last_substeps = 1        # k used by the most recent step() -- diagnostic
        self.pos = self._spawn_positions(n)
        self.trail = np.repeat(self.pos[:, None, :], TRAIL_LEN, axis=1)
        self.age = np.zeros(n, dtype=np.float32)
        self.life = self._rng.uniform(LIFE_MIN_FRAMES, LIFE_MAX_FRAMES, size=n).astype(np.float32)

    def _ceiling_limit(self, x: np.ndarray) -> np.ndarray:
        """Per-particle z upper bound: the room ceiling everywhere, except
        over an open vent's x-span, where it's the ceiling slab's real top
        face (duct_top_z) -- a particle may cross through the opening's
        true physical depth before being treated as gone (see module
        docstring)."""
        limit = np.full(x.shape, self._room_z1, dtype=np.float64)
        for vx0, vx1 in self._open_vent_spans:
            limit = np.where((x >= vx0) & (x <= vx1), self._duct_top_z, limit)
        return limit

    def _wall_x_limits(self, z: np.ndarray) -> tuple:
        """Per-particle (x_lo, x_hi) bounds: the room's own side walls
        everywhere, except across the doorway's z-span on the left wall,
        where a particle may cross _DOOR_EXIT_DEPTH past the wall line
        (the horizontal analogue of _ceiling_limit's open-vent allowance)
        before it's treated as gone. The right wall is always solid."""
        x_lo = np.full(z.shape, self._room_x0, dtype=np.float64)
        x_hi = np.full(z.shape, self._room_x1, dtype=np.float64)
        if self._door_z_span is not None:
            dz0, dz1 = self._door_z_span
            through_door = (z >= dz0) & (z <= dz1)
            x_lo = np.where(through_door, self._room_x0 - _DOOR_EXIT_DEPTH, x_lo)
        return x_lo, x_hi

    def _spawn_positions(self, n: int) -> np.ndarray:
        xs = self._rng.uniform(self._room_x0, self._room_x1, size=n)
        zs = self._rng.uniform(self._room_z0, self._room_z1, size=n)
        return np.column_stack([xs, zs]).astype(np.float64)

    def reseed(self) -> None:
        self.pos = self._spawn_positions(self.n)
        self.trail = np.repeat(self.pos[:, None, :], TRAIL_LEN, axis=1)
        self.age[:] = 0.0
        self.life = self._rng.uniform(LIFE_MIN_FRAMES, LIFE_MAX_FRAMES, size=self.n).astype(np.float32)

    def step(self, u_frame: np.ndarray, w_frame: np.ndarray, extent: tuple, dt: float) -> None:
        # Bounds are the room's own walls/ceiling now, per-particle (see
        # _wall_x_limits / _ceiling_limit), not the far-off domain edge --
        # a solid border blocks at the real room line, an open vent's span
        # allows through to the slab's real top face and the doorway's
        # z-span allows _DOOR_EXIT_DEPTH into the corridor, matching the
        # module docstring.
        ex0, ex1, ez0, ez1 = extent
        n_z, n_x = u_frame.shape
        cell = min((ex1 - ex0) / (n_x - 1), (ez1 - ez0) / (n_z - 1))

        # Adaptive sub-stepping (see the _CFL_SUBSTEP_CELLS comment): pick
        # k from the fastest particle so no sub-step moves more than
        # _CFL_SUBSTEP_CELLS cells, then advect k times, re-sampling the
        # (frozen-for-this-frame) field at each new sub-position. Same
        # field, same total dt -- a slow particle (k == 1) is unchanged.
        u, w = _sample_uw(u_frame, w_frame, extent, self.pos[:, 0], self.pos[:, 1])
        vmax = float(np.hypot(u, w).max())
        k = 1
        if vmax * dt > _CFL_SUBSTEP_CELLS * cell > 0.0:
            k = min(_MAX_SUBSTEPS, int(np.ceil(vmax * dt / (_CFL_SUBSTEP_CELLS * cell))))
        self._last_substeps = k
        sub_dt = dt / k
        for s in range(k):
            if s > 0:
                u, w = _sample_uw(u_frame, w_frame, extent, self.pos[:, 0], self.pos[:, 1])
            self.pos[:, 0] += sub_dt * u
            self.pos[:, 1] += sub_dt * w
        self.age += 1.0

        ceiling_limit = self._ceiling_limit(self.pos[:, 0])
        x_lo, x_hi = self._wall_x_limits(self.pos[:, 1])
        out_of_room = ((self.pos[:, 0] < x_lo) | (self.pos[:, 0] > x_hi)
                       | (self.pos[:, 1] < self._room_z0) | (self.pos[:, 1] > ceiling_limit))
        expired = self.age >= self.life
        respawn = out_of_room | expired
        if np.any(respawn):
            n_respawn = int(np.sum(respawn))
            self.pos[respawn] = self._spawn_positions(n_respawn)
            self.age[respawn] = 0.0
            self.life[respawn] = self._rng.uniform(
                LIFE_MIN_FRAMES, LIFE_MAX_FRAMES, size=n_respawn).astype(np.float32)
            # A respawned particle's trail must not draw a stray line back
            # to its old (pre-respawn) position -- reset its whole history
            # to the new spot instead of rolling a jump into it.
            self.trail[respawn] = self.pos[respawn][:, None, :]

        # Roll the trail buffer and append the (possibly just-respawned)
        # current position as the newest sample.
        self.trail = np.roll(self.trail, -1, axis=1)
        self.trail[:, -1, :] = self.pos


class TracerFlowPanel(QtWidgets.QWidget):
    """Analysis-page tab: a room-seeded pool of tracer particles advected
    by the validated U/W-VELOCITY field, each drawn as a short fading
    trail. Independent of VelocityPanel/StreamlinePanel/LICFlowPanel -- no
    shared state, no probes, nothing here writes to any of them."""

    def __init__(self, provider, manifest: list, fps: int,
                 n_particles: int = DEFAULT_N_PARTICLES, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._by_index = {e.case_index: e for e in self._manifest}
        self._fps = max(1, fps)
        self._n_particles = n_particles
        self._loaded = False
        self._fields: dict = {}         # case_index -> vel.VectorField (successfully computed)
        self._gate_reasons: dict = {}   # case_index -> str
        self._pools: dict = {}          # case_index -> _ParticlePool
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()
        self._last_stepped_index: dict = {}  # case_index -> frame index the pool was last advected to

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Velocity (tracer particles)")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        header = QtWidgets.QHBoxLayout()
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Tracer particles scenario")
        self.scenario_combo.setToolTip("Which scenario's vector field to compute and display")
        header.addWidget(self.scenario_combo)
        header.addStretch(1)
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Particles"))
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setAccessibleName("Number of tracer particles")
        self.count_spin.setToolTip("How many particles to seed and advect through the flow")
        self.count_spin.setRange(20, 400)
        self.count_spin.setSingleStep(10)
        self.count_spin.setValue(n_particles)
        controls.addWidget(self.count_spin)
        self.reseed_button = QtWidgets.QPushButton("Reseed")
        self.reseed_button.setAccessibleName("Reseed tracer particles")
        self.reseed_button.setToolTip("Reset every particle to a fresh random position in the room")
        controls.addWidget(self.reseed_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Velocity tracer particles canvas")
        layout.addWidget(self.canvas, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.count_spin.valueChanged.connect(self._on_count_changed)
        self.reseed_button.clicked.connect(self._on_reseed_clicked)

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        was_loaded = self._loaded
        self.ensure_loaded()
        if was_loaded:
            self._render()

    def set_bus(self, bus) -> None:
        """Follow the shared playback frame (Selection.time_s), same
        set_bus precedent as streamline_panel.py/lic_flow_panel.py -- this
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

    def _on_count_changed(self, value: int) -> None:
        self._n_particles = value
        case_index = self.scenario_combo.currentData()
        if case_index is not None and case_index in self._pools:
            del self._pools[case_index]  # rebuilt at this new count on next render
        self._render()

    def _on_reseed_clicked(self) -> None:
        case_index = self.scenario_combo.currentData()
        pool = self._pools.get(case_index)
        if pool is not None:
            pool.reseed()
        self._render()

    # -------------------------------------------------- vector field access
    def _ensure_field(self, case_index: int):
        """Lazily compute (and cache) the VectorField for `case_index` --
        identical convention to StreamlinePanel._ensure_field, duplicated
        rather than shared so this panel has zero coupling to its
        internals (module docstring)."""
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

    def _pool_for(self, case_index: int, extent: tuple) -> _ParticlePool:
        pool = self._pools.get(case_index)
        if pool is None:
            room_x0, room_x1 = max(extent[0], min(ROOM_X)), min(extent[1], max(ROOM_X))
            room_z0, room_z1 = max(extent[2], min(ROOM_Z)), min(extent[3], max(ROOM_Z))
            entry = self._by_index.get(case_index)
            if entry is not None:
                open_spans, duct_top_z = _open_vent_spans_and_duct_top(entry)
                door_span = _door_z_span(entry)
            else:
                open_spans, duct_top_z, door_span = [], None, None
            pool = _ParticlePool(self._n_particles, (room_x0, room_x1, room_z0, room_z1),
                                  open_vent_spans=open_spans, duct_top_z=duct_top_z,
                                  door_z_span=door_span, seed=case_index)
            self._pools[case_index] = pool
        return pool

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
            ax.set_title("Tracer particles unavailable -- needs U/W velocity data",
                        fontsize=9, color="#B00020")
            ax.set_xticks([]); ax.set_yticks([])
            self.status.setText(reason)
            self.canvas.draw_idle()
            return
        self.status.setText("")

        frame_index = min(max(self._current_index, 0), field.n_frames - 1)
        x0, x1, z0, z1 = field.extent

        # Row 0 of u/w is the ceiling (z1) -- flipped vertically (reindex
        # only, W's sign untouched) so physical z increases upward, same
        # convention as every sibling flow panel.
        u_frame = np.flip(field.u[frame_index], axis=0)
        w_frame = np.flip(field.w[frame_index], axis=0)

        pool = self._pool_for(case_index, field.extent)
        # Advance by exactly one real playback tick per new frame_index --
        # never re-advect on a redraw that didn't actually move the clock
        # (e.g. this panel becoming visible again, or a spin/reseed
        # control triggering a render), and never take more than one step
        # for a seek/jump (an interactive tool, not a strict physical
        # replay -- see module docstring).
        if self._last_stepped_index.get(case_index) != frame_index:
            pool.step(u_frame, w_frame, field.extent, dt=1.0 / self._fps)
            self._last_stepped_index[case_index] = frame_index

        speed = np.hypot(*_sample_uw(u_frame, w_frame, field.extent, pool.pos[:, 0], pool.pos[:, 1]))
        norm = PowerNorm(gamma=SPEED_GAMMA, vmin=0.0, vmax=SPEED_REF_MS)
        head_colors = plt_cmap(DEFAULT_CMAP)(norm(speed))

        # Fading trail: TRAIL_LEN-1 segments per particle, oldest (alpha
        # near 0) to newest (alpha 1) -- same LineCollection technique
        # views.py's room outline already uses, not a new pattern.
        n = pool.n
        segs = np.stack([pool.trail[:, :-1, :], pool.trail[:, 1:, :]], axis=2).reshape(-1, 2, 2)
        alphas = np.tile(np.linspace(0.05, 0.9, TRAIL_LEN - 1), n)
        seg_colors = np.repeat(head_colors, TRAIL_LEN - 1, axis=0)
        seg_colors[:, 3] = alphas

        # Render clip (module docstring): draw a segment only where BOTH
        # its endpoints sit inside the room rectangle, or inside the small
        # TRAIL_VENT_MARGIN_FRAC allowance just outside an open vent's span
        # (above the ceiling) or the doorway's z-span (left of the wall) --
        # this is what stops a trail from drawing over solid
        # ceiling/walls/the corridor, independent of the pool's own
        # (looser, real-physical-depth) containment bounds.
        tx, tz = pool.trail[:, :, 0], pool.trail[:, :, 1]
        in_room = ((tx >= pool._room_x0) & (tx <= pool._room_x1)
                  & (tz >= pool._room_z0) & (tz <= pool._room_z1))
        in_vent_margin = np.zeros_like(in_room)
        vent_gap = pool._duct_top_z - pool._room_z1
        trail_margin_z = pool._room_z1 + TRAIL_VENT_MARGIN_FRAC * vent_gap
        for vx0, vx1 in pool._open_vent_spans:
            in_vent_margin |= ((tx >= vx0) & (tx <= vx1)
                               & (tz > pool._room_z1) & (tz <= trail_margin_z))
        in_door_margin = np.zeros_like(in_room)
        if pool._door_z_span is not None:
            dz0, dz1 = pool._door_z_span
            trail_margin_x = pool._room_x0 - TRAIL_VENT_MARGIN_FRAC * _DOOR_EXIT_DEPTH
            in_door_margin |= ((tz >= dz0) & (tz <= dz1)
                               & (tx < pool._room_x0) & (tx >= trail_margin_x))
        point_visible = in_room | in_vent_margin | in_door_margin
        seg_visible = (point_visible[:, :-1] & point_visible[:, 1:]).reshape(-1)

        trails = LineCollection(segs[seg_visible], colors=seg_colors[seg_visible],
                                linewidths=1.3, zorder=5)
        ax.add_collection(trails)
        ax.scatter(pool.pos[:, 0], pool.pos[:, 1], c=head_colors, s=10, zorder=6, edgecolors="none")

        # Room outline (walls/door/vents) from the real &HOLE-derived
        # geometry -- same source/colors streamline_panel.py just added.
        entry = self._by_index.get(case_index)
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

        sm = plt_ScalarMappable(cmap=DEFAULT_CMAP, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Speed (m/s)")

        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        ax.set_aspect("auto")
        ax.set_xticks([]); ax.set_yticks([])
        t_s = frame_index / self._fps
        ax.set_title(f"Tracer particles · {pool.n} · t={t_s:.1f}s", fontsize=8)

        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()


def plt_cmap(name):
    from matplotlib import colormaps
    return colormaps[name]


def plt_ScalarMappable(cmap, norm):
    from matplotlib.cm import ScalarMappable
    return ScalarMappable(cmap=cmap, norm=norm)
