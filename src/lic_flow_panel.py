"""LIC (Line Integral Convolution) flow panel: a third, independent
visualization of the validated U/W-VELOCITY field, alongside VelocityPanel's
quiver/streamlines and StreamlinePanel's matplotlib streamplot. Does not
touch either of those two modules or their panels in any way -- reuses
velocity.py's VectorField exactly as they do (read-only), plus TEMPERATURE
via QuantityProvider.get(), the same accessor every other slice view uses.

Why LIC instead of another quiver/streamplot variant: a dense flow texture
that shows direction *everywhere* on the grid, continuously, without the
arrow-density tradeoffs quiver has (too sparse looks like dots, too dense
looks like static) or streamplot's seed-placement/tangle tuning
(streamline_panel.py's own history -- feature seeding, MAXLENGTH/MINLENGTH,
forward/backward arrow splitting -- shows how much hand-tuning streamplot
needed on this exact dataset). LIC has one real free parameter (kernel
length, see KERNEL_HALF_LEN) instead of that whole tuning surface.

Three composited layers, back to front:
  1. Speed-magnitude color (the single filled field) -- pcolormesh-
     equivalent (ax.imshow) of speed = hypot(u, w), viridis + PowerNorm,
     the same fixed SPEED_REF_MS/gamma calibration streamline_panel.py
     already validated on this dataset (duplicated, not imported -- this
     module's own zero-coupling precedent, matching every sibling panel).
  2. LIC texture -- a semi-transparent grayscale ax.imshow on top (see
     LIC_ALPHA), so color still reads through the texture: color = how
     fast, texture = which way.
  3. Temperature isotherms -- thin ax.contour LINES (never contourf: a
     second filled field on top of the speed color would mud into a wash,
     confirmed directly during this panel's own tuning pass), using
     CONTOUR_OVERLAY_LEVELS["TEMPERATURE"], plus the room geometry
     overlay on top of that, thinned/faded the same way every sibling
     panel's geometry is.

LIC algorithm (compute_lic_texture, module-level so it's independently
testable): standard Cabral & Leedom convolution. Each output texel's value
is a Hanning-weighted average of a (blurred, see below) white-noise texture
sampled along a short streamline traced through that texel (forward AND
backward, KERNEL_HALF_LEN steps each way, step size STEP_CELLS native grid
cells). Advection uses a unit direction vector (physical (u, w) converted to
native-grid index units via dx/dz, then normalized) -- constant step size
regardless of local speed, because LIC exists to show *direction*; speed is
already carried by layer 1. A stagnant cell (speed ~ 0) doesn't advect --
it keeps resampling its own noise value, which is the correct degenerate
case (a dead zone should look flat/undirected, not extrapolate a direction
that doesn't exist).

Density/legibility tuning (visual-clarity follow-up: the first version read
as busy, fine-grained marble -- the eye couldn't settle on plume/ceiling-jet/
recirculation structure): two separate levers, and both turned out to
matter, not just one --
  - KERNEL_HALF_LEN alone, even swept to 3x (30), was NOT enough: longer
    streaks help along the flow direction but do nothing to the noise's
    high-frequency detail *across* it, so the texture still read speckly at
    every kernel length tried with the original per-pixel-random noise (see
    lic_sweep_grid.png from the tuning pass).
  - NOISE_BLUR_SIGMA (new): a light Gaussian blur applied to the white-noise
    base ONCE, before advection, coarsens the underlying grain in every
    direction, not just along streamlines. Combined with a longer kernel,
    this is what actually reads as large-scale flow structure instead of
    marble. Still generated/blurred once per scenario and cached (see
    flicker-safety below) -- the blur is baked into the same one-time
    _generate_noise() call, not a per-frame cost.
  Chosen KERNEL_HALF_LEN=20 (2x the original 10) + NOISE_BLUR_SIGMA=1.2
  after comparing 10/15/20/30 x {raw, blurred} noise on scenario 0 at
  t=72s: 20+blur was the point where the plume, ceiling jet, and the room's
  recirculation eddies all read as distinct shapes rather than a single
  smeared blob (30+blur started losing the smaller eddies into one pool).

Flicker-safety (the property that makes or breaks this panel for
playback): the (now blurred) white-noise input texture is generated ONCE
per scenario and cached (self._noise_cache), exactly like
streamline_panel.py's fixed start_points cache exists to stop matplotlib's
per-call auto-reseeding from making the rendered pattern jump between
frames. Only the *advection* recomputes per frame here, driven by the
smoothly-evolving U/W field -- the noise base underneath never changes for
a given scenario, so consecutive frames' textures are smeared/advected
versions of the same fixed pattern, not independently reseeded ones. The
blur is deterministic and applied inside the same one-time generation this
cache already covers, so it doesn't introduce any new per-frame randomness.

Row-0-is-ceiling convention: exactly the same as streamline_panel.py --
U/W/TEMPERATURE have row 0 at the physical ceiling; each frame is flipped
vertically (row reindexing only, no sign changes) so `z` is increasing for
ax.imshow(origin="lower")/ax.contour alike. This panel's own
compute_lic_texture() works entirely in that *flipped, display* index space
(row increases with z, matching the imshow/contour it feeds) -- deliberately
NOT velocity.quiver_grid's raw row-0-is-ceiling convention (this panel draws
no quiver and calls quiver_grid nowhere), so the "double-flip" bug class a
quiver layer sampled from the wrong convention could hit (each arrow drawn
at its physically mirrored z, paired with a different row's velocity)
cannot occur here: there is only one convention in this file, used
consistently front to back.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import PowerNorm
from scipy.ndimage import map_coordinates, gaussian_filter
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
from schematic import room_overlay_geometry
from slice_key import SliceKey
from config import CONTOUR_OVERLAY_LEVELS
import velocity as vel

# The primary free parameter (see module docstring's density/legibility
# section): half the kernel length in advection steps -- the full Hanning
# kernel is 2*KERNEL_HALF_LEN+1 samples. STEP_CELLS (native grid cells per
# step) is a fixed implementation detail, not something a user would tune.
# Together they set the streak length: KERNEL_HALF_LEN * STEP_CELLS native
# cells each direction.
#
# Visual-clarity follow-up: swept 10 (original) / 15 / 20 / 30 (1.5x/2x/3x)
# against scenario 0 at t=72s, each with and without NOISE_BLUR_SIGMA below
# -- kernel length alone, even at 30, still read as fine speckle (the noise
# itself has no across-flow correlation for a longer kernel to smooth out).
# 20 + blur was the clearest large-scale read (plume/ceiling-jet/eddies all
# distinct); 30 + blur started merging the smaller eddies into one pool.
KERNEL_HALF_LEN = 20
STEP_CELLS = 0.5

# Second lever (see module docstring): a light Gaussian blur on the raw
# white-noise base, applied once at generation time (still cached, see
# _generate_noise/_noise_for) -- coarsens the grain in every direction, not
# just along streamlines, which kernel length alone can't do. 1.2 (native
# grid cells) was the sweep's pick: enough to stop the marble/speckle read,
# not so much the texture loses per-eddy shape.
NOISE_BLUR_SIGMA = 1.2

# Fixed noise seed per scenario (see module docstring's flicker-safety
# section): NOISE_SEED_BASE + case_index, not a single shared seed, so
# switching scenarios doesn't show the literal same texture reused.
NOISE_SEED_BASE = 1000

# Raw LIC output (a Hanning-weighted average of noise in [0, 1)) has low
# variance -- central-limit averaging over ~2*KERNEL_HALF_LEN+1 samples
# washes out contrast. A per-frame percentile stretch (not a fixed
# min/max) restores visible streaks regardless of scenario/frame.
LIC_CONTRAST_LOW_PCT = 2.0
LIC_CONTRAST_HIGH_PCT = 98.0

# Semi-transparent grayscale overlay (module docstring, layer 2): alpha low
# enough that layer 1's speed color still reads through the texture.
#
# Visual-clarity follow-up: 0.55 with a full [0, 1] contrast stretch let the
# texture's darkest strokes go near-black, which -- averaged over the whole
# image -- muddied viridis into a purplish grey. Two changes together fixed
# it: LIC_ALPHA down to 0.40 (less overlay overall), and LIC_GRAY_FLOOR
# raised so the texture's darkest pixel is never fully black (it still
# darkens the color underneath for contrast, just not to mud). Texture
# keeps carrying direction; color keeps reading clearly for speed.
LIC_ALPHA = 0.40
LIC_GRAY_FLOOR = 0.25
LIC_CMAP = "gray"

# Layer 1 (speed color): same fixed reference/gamma streamline_panel.py
# measured directly against this dataset (room circulation ~0.03-0.15 m/s,
# strongest-candle plume peak ~1.17 m/s) -- duplicated, not imported, this
# module's own zero-coupling precedent.
SPEED_REF_MS = 1.2
SPEED_CMAP = "viridis"
SPEED_GAMMA = 0.45

# Layer 3 (isotherms): thin lines, not a fill -- see module docstring.
ISOTHERM_COLOR = "white"
ISOTHERM_LINEWIDTH = 0.8
ISOTHERM_ALPHA = 0.85

# Layer 4 (room geometry): same thinned/faded/receded-behind-the-flow
# values every sibling panel settled on -- duplicated rather than imported
# (this module's own "zero coupling to other velocity views" precedent).
_WALL_COLOR = "#FFFFFF"
_DOOR_COLOR = "#38BDF8"
_VENT_STATE_COLORS = {"open": "#22C55E", "closed": "#94A3B8", "HVAC": "#F59E0B"}
_WALL_LINEWIDTH = 1.4
_DOOR_LINEWIDTH = 1.6
_VENT_LINEWIDTH = 2.2
_GEOMETRY_ALPHA = 0.7
_GEOMETRY_ZORDER = 3   # above speed color (0), LIC texture (1), isotherms (2)

_TEMPERATURE_KEY = SliceKey("TEMPERATURE", 1, 0)

_EPS = 1e-9


def _generate_noise(shape: tuple, seed: int, blur_sigma: float = NOISE_BLUR_SIGMA) -> np.ndarray:
    """Fixed noise texture -- the flicker-safety input (see module
    docstring). Deterministic from (shape, seed, blur_sigma) alone: raw
    per-pixel white noise, then a one-time Gaussian coarsening (see
    NOISE_BLUR_SIGMA) so LIC has more than a single pixel of correlation to
    smooth into a stroke. The blur happens here, inside the same call
    _noise_for() caches, so it costs nothing per frame."""
    raw = np.random.default_rng(seed).random(shape)
    return gaussian_filter(raw, sigma=blur_sigma) if blur_sigma > 0 else raw


def compute_lic_texture(u_frame: np.ndarray, w_frame: np.ndarray, noise: np.ndarray,
                         extent: tuple, kernel_half_len: int = KERNEL_HALF_LEN,
                         step_cells: float = STEP_CELLS) -> np.ndarray:
    """Standard LIC: for every texel, a Hanning-weighted average of `noise`
    sampled along a short forward+backward streamline traced from that
    texel through (u_frame, w_frame). All three arrays share (n_z, n_x);
    `extent` is (x0, x1, z0, z1) in the same flipped/display index
    convention u_frame/w_frame are already in (see module docstring) --
    row increases with z, col increases with x."""
    n_z, n_x = u_frame.shape
    x0, x1, z0, z1 = extent
    dx = (x1 - x0) / max(n_x - 1, 1)
    dz = (z1 - z0) / max(n_z - 1, 1)

    kernel = np.hanning(2 * kernel_half_len + 1)
    kernel_sum = kernel.sum()

    def sample_noise(row, col):
        return map_coordinates(noise, [row, col], order=1, mode="nearest")

    def sample_field(row, col):
        u = map_coordinates(u_frame, [row, col], order=1, mode="nearest")
        w = map_coordinates(w_frame, [row, col], order=1, mode="nearest")
        return u, w

    def unit_index_dir(u, w):
        # Physical (u, w) -> unit direction in native-grid INDEX units (not
        # physical units), so a `step_cells`-length step always covers the
        # same number of grid cells regardless of local speed or dx/dz.
        idx_col, idx_row = u / dx, w / dz
        mag = np.hypot(idx_col, idx_row)
        safe = mag > _EPS
        denom = np.where(safe, mag, 1.0)
        return np.where(safe, idx_row / denom, 0.0), np.where(safe, idx_col / denom, 0.0)

    row0, col0 = np.mgrid[0:n_z, 0:n_x].astype(np.float64)
    accum = kernel[kernel_half_len] * sample_noise(row0, col0)

    for sign in (1, -1):
        row, col = row0.copy(), col0.copy()
        for s in range(1, kernel_half_len + 1):
            u, w = sample_field(row, col)
            dr, dc = unit_index_dir(u, w)
            row = np.clip(row + sign * step_cells * dr, 0.0, n_z - 1)
            col = np.clip(col + sign * step_cells * dc, 0.0, n_x - 1)
            accum = accum + kernel[kernel_half_len + sign * s] * sample_noise(row, col)

    return accum / kernel_sum


def _contrast_stretch(texture: np.ndarray, lo_pct: float = LIC_CONTRAST_LOW_PCT,
                       hi_pct: float = LIC_CONTRAST_HIGH_PCT) -> np.ndarray:
    lo, hi = np.percentile(texture, [lo_pct, hi_pct])
    if hi <= lo:
        return np.clip(texture, 0.0, 1.0)
    return np.clip((texture - lo) / (hi - lo), 0.0, 1.0)


class LICFlowPanel(QtWidgets.QWidget):
    """Analysis-page tab: speed-color background + LIC direction texture +
    temperature isotherms + room geometry, one frame. Independent of
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
        self._noise_cache: dict = {}     # case_index -> fixed noise texture, see module docstring
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Velocity (LIC)")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        header = QtWidgets.QHBoxLayout()
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("LIC flow scenario")
        self.scenario_combo.setToolTip(
            "Which scenario's speed/LIC/temperature composite to display")
        header.addWidget(self.scenario_combo)
        header.addStretch(1)
        layout.addLayout(header)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("LIC flow canvas")
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
        identical convention to every sibling panel's _ensure_field,
        duplicated rather than shared (this module's own zero-coupling
        precedent)."""
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
        scenario missing this specific plane, so this guards the same way."""
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

    def _noise_for(self, case_index: int, shape: tuple) -> np.ndarray:
        """The fixed per-scenario noise texture (see module docstring's
        flicker-safety section) -- generated once, reused for every frame
        of this scenario for as long as this panel instance lives."""
        cached = self._noise_cache.get(case_index)
        if cached is not None and cached.shape == shape:
            return cached
        noise = _generate_noise(shape, NOISE_SEED_BASE + case_index)
        self._noise_cache[case_index] = noise
        return noise

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
            ax.set_title("LIC flow unavailable -- needs U/W velocity and TEMPERATURE data",
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
        # no sign change) so z is increasing for imshow(origin="lower")/
        # contour alike -- see module docstring's coordinate-convention note.
        u_frame = np.flip(field.u[frame_index], axis=0)
        w_frame = np.flip(field.w[frame_index], axis=0)
        temp_frame = np.flip(temp[frame_index], axis=0)

        n_z, n_x = u_frame.shape
        x = np.linspace(x0, x1, n_x)
        z = np.linspace(z0, z1, n_z)

        # Layer 1: speed-magnitude color, the single filled field.
        speed = np.hypot(u_frame, w_frame)
        norm = PowerNorm(gamma=SPEED_GAMMA, vmin=0.0, vmax=SPEED_REF_MS)
        im = ax.imshow(speed, extent=(x0, x1, z0, z1), origin="lower", aspect="auto",
                        cmap=SPEED_CMAP, norm=norm, zorder=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Speed (m/s)")

        # Layer 2: LIC texture -- fixed per-scenario noise (flicker-safe,
        # see module docstring), advected through THIS frame's field.
        # LIC_GRAY_FLOOR keeps the darkest stroke short of pure black (see
        # LIC_ALPHA's comment) so it darkens, but never muds, layer 1.
        noise = self._noise_for(case_index, u_frame.shape)
        texture = _contrast_stretch(compute_lic_texture(u_frame, w_frame, noise, field.extent))
        texture = LIC_GRAY_FLOOR + (1.0 - LIC_GRAY_FLOOR) * texture
        ax.imshow(texture, extent=(x0, x1, z0, z1), origin="lower", aspect="auto",
                  cmap=LIC_CMAP, vmin=0.0, vmax=1.0, alpha=LIC_ALPHA,
                  interpolation="bilinear", zorder=1)

        # Layer 3: temperature isotherms -- thin lines, never a fill (see
        # module docstring).
        levels = CONTOUR_OVERLAY_LEVELS.get("TEMPERATURE", [])
        if levels:
            ax.contour(x, z, temp_frame, levels=levels, colors=ISOTHERM_COLOR,
                      linewidths=ISOTHERM_LINEWIDTH, alpha=ISOTHERM_ALPHA, zorder=2)

        # Layer 4: room outline -- thinned/faded, same lesson every sibling
        # panel's own visual-clarity passes already settled: context, not
        # focus.
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
        ax.set_title(f"LIC (speed + direction) + isotherms · t={t_s:.1f}s", fontsize=8)

        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()
