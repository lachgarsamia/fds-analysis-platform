"""PlotView abstraction (M2.2): decouples the rendering backend from the
widgets that use it, so a multi-view grid can hold several independent
view-cells, and later cell types (DifferenceView/EnsembleView in M2.3, a
possible pyqtgraph backend behind M2.4's gate) can all implement the same
minimal interface instead of each reaching into main_window.py's
matplotlib internals directly. Single-view mode (MainWindow) is just a
1-cell user of this same interface -- not a special case.
"""

import re
from typing import Protocol

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import LineCollection
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, TimeSeriesStrip
from config import QUANTITY_DISPLAY, FRAMES_PER_SECOND
from registry import get_quantity
from theme import RADIUS
from cinema.pipeline import EffectsPipeline
from cinema.interp import lerp_frames
from cinema.particles import EmberParticles
from cinema.velocity_arrows import sample_points, compute_deltas
import smoke_density as smd

# Sub-frame interpolation (FireLab roadmap Phase 2.1d): visual refresh
# rate while cinematic mode is on, decoupled from the data's native
# FRAMES_PER_SECOND sample rate.
_INTERP_HZ = 30
_INTERP_INTERVAL_MS = round(1000 / _INTERP_HZ)
_NOMINAL_TICK_MS = 1000.0 / FRAMES_PER_SECOND

# Dark cinema backdrop (FireLab roadmap Phase 2): the FireLUT's alpha ramp
# only reads as "fire floating in a dark room" against a near-black axes
# background, not the plot area's normal white (MplCanvas.PLOT_BG).
CINEMA_BG = "#0B0D12"

# Room overlay vent colors (Live polish): matches schematic.py's own
# open/closed/HVAC semantics (palette.success/text_disabled/warning),
# but as fixed hex values -- the overlay sits on a science heatmap, not
# app chrome, so it uses the same "always-legible regardless of theme"
# convention as the heatmap's other fixed-color overlays (device markers,
# hover ring), not the app's light/dark palette.
_VENT_STATE_COLORS = {"open": "#22C55E", "closed": "#94A3B8", "HVAC": "#F59E0B"}

# Room overlay base line widths at ui_scale=1.0 -- SliceView.set_ui_scale
# multiplies these, so both init_plot (construction) and set_ui_scale
# (later changes) agree on the same unscaled numbers.
_ROOM_WALL_LW = 1.4
_ROOM_DOOR_LW = 2.6
_ROOM_VENT_LW = 4.0

# Room overlay legibility (colormap expressiveness follow-up): the room
# boundary/door/vents are static overlays drawn on top of *every*
# quantity's colormap (inferno, viridis, gray_r, the fds_flow "jet"), so
# no single fixed line color can stay visible everywhere -- room_walls'
# white dashes vanish into inferno's pale-yellow flame plume exactly
# where the fire is, and a green/cyan line would just as easily vanish
# into viridis's own pale end. Fixed by casing each line in a contrasting
# stroke via matplotlib.patheffects.withStroke: a dark casing ("#14171F",
# this app's existing fixed dark overlay color -- see the device-marker
# legend) drawn *underneath* the artist's own color, so whichever of the
# two the current background washes out, the other still reads. Never
# touches the artist's own foreground color -- room_vents keeps its
# open/closed/HVAC state color, it's just outlined now.
_ROOM_OUTLINE_CASING = "#14171F"


def _cased_line_effect(base_linewidth: float) -> list:
    """A patheffects stack for a line of `base_linewidth`: a wider dark
    stroke underneath, then the artist's own color drawn normally on top
    -- see _ROOM_OUTLINE_CASING above. The casing is deliberately only
    ~1pt wider on each side (not a heavy outline) so it reads as a thin
    border, not a different, thicker line."""
    return [path_effects.withStroke(linewidth=base_linewidth + 2.0, foreground=_ROOM_OUTLINE_CASING),
            path_effects.Normal()]

# Virtual device marker shapes, one per device kind (Analysis UX +
# reliability pass): heat_detector (instant threshold) and sprinkler (RTI
# thermal-lag ODE) are independent devices with independently different,
# both-correct response models -- a sprinkler is *supposed* to lag a heat
# detector at the same point. Previously every kind drew as the same
# diamond, distinguished only by active/idle fill color, so two correctly-
# disagreeing devices were visually indistinguishable from "the alarm
# system is broken." Duplicated here (not imported from devices.py) for
# the same view-layer/data-layer boundary EnsemblePickerDialog's
# FACTOR_LABELS already keeps -- just the 3 literal kind strings devices.py
# uses, not a real dependency.
_DEVICE_MARKER_SHAPES = {"thermocouple": "o", "heat_detector": "^", "sprinkler": "s"}

# Legend labels for the same 3 kinds (Analysis final-polish pass): the
# marker-shape distinction above was previously only explained inside the
# separate Devices analysis tab (device_panel.py's model_note caption) --
# nothing on the Live Viewer itself, where the markers are actually
# watched during playback, said what a triangle vs. a square means. A
# small always-available legend closes that gap without touching
# devices.py's activation math (already correct: heat detectors trip an
# instant temperature threshold, sprinklers lag it via a genuine RTI
# thermal-element ODE -- see devices.py -- so the two disagreeing is
# expected physics, not a fault, and is never forced to agree here).
_DEVICE_KIND_LABELS = {
    "thermocouple": "Thermocouple",
    "heat_detector": "Heat detector (instant threshold)",
    "sprinkler": "Sprinkler (RTI thermal lag)",
}


class PlotView(Protocol):
    """Minimum interface every view-cell type must implement. Concrete
    classes (SliceView below) expose additional setup/backend-specific
    methods beyond this -- this is the common surface a grid container or
    the M2.3 comparison views can rely on without knowing the backend."""

    def widget(self) -> QtWidgets.QWidget: ...
    def show_frame(self, frame: np.ndarray) -> None: ...
    def set_cmap(self, name: str) -> None: ...
    def set_clim(self, vmin: float, vmax: float) -> None: ...
    def set_title(self, text: str) -> None: ...


class SliceView:
    """matplotlib-backed single-quantity heatmap cell.

    Blitting internally (M1.3.3, unchanged behavior after this extraction):
    show_frame() only re-renders the image artist, not axes/colorbar
    chrome. Anything that changes what's underneath the animated artist
    (clim/colormap/interpolation/title/theme/resize) does a full draw +
    recapture -- see the setters below, all of which call
    canvas.capture_background() themselves so callers don't have to
    remember to.

    Physical extent (M2.6): if `extent=(x0, x1, z0, z1)` is passed to
    init_plot(), the image is drawn with matplotlib's own `extent=`
    parameter instead of bare pixel-index axes -- matplotlib's default
    `origin='upper'` then places array row 0 at z1 (top) and the last row
    at z0 (bottom). This matters because load_data.py already flips the
    array vertically (np.flip(axis=1)) before any view ever sees it, so
    row 0 already *is* the physical ceiling -- extent + origin='upper'
    lines up with that flip for free, both for what's drawn and for what
    mouse events report back (event.xdata/event.ydata become true
    physical coordinates, not pixel indices), without this class having
    to duplicate the flip's own arithmetic anywhere. Verified against the
    real dataset (not just reasoned about): the row/col of a known
    peak-difference pixel from M2.3's investigation maps back to the same
    physical (x, z) matplotlib itself reports via this convention.
    """

    def __init__(self, parent=None):
        self.canvas = MplCanvas(parent)
        self.ax = None
        self.heatmap = None
        self.colorbar = None
        self._extent = None
        self._isotherm_levels: list = []
        self._isotherms_enabled = False
        self._contour_artist = None
        self._probe_callback = None
        self._motion_cid = None
        # Velocity overlay (GUI modernization pass, item 6): a second,
        # independently-tracked contour -- VELOCITY speed bands drawn on
        # top of a TEMPERATURE heatmap. Kept fully separate from
        # _contour_artist/_isotherm_levels above (own artist, own frame,
        # own enabled flag) so the two overlays can coexist and never
        # fight over the same state; styled distinctly (see
        # _redraw_velocity_overlay) so they're never visually confused.
        self._velocity_overlay_levels: list = []
        self._velocity_overlay_enabled = False
        self._velocity_contour_artist = None
        self._velocity_frame = None
        # Real soot-density smoke overlay (continuous soot-density
        # visualization pass): a second, always-present-but-empty imshow
        # over the temperature heatmap, same "always present, empty by
        # default" convention as device_scatters/ember_scatter below --
        # updated via set_data()/set_alpha() (not remove+recreate like the
        # isotherm/velocity contour artists), so it blit-updates cheaply.
        # Continuous opacity, driven by the real SOOT DENSITY field
        # (smoke_density.py) -- never a threshold. Meaningful only for a
        # "slice" cell showing TEMPERATURE at a plane SOOT DENSITY is
        # actually registered for (see main_window.py's
        # _soot_overlay_frame_for_cell); off by default.
        self.soot_overlay = None
        self.soot_colorbar = None
        self._soot_overlay_enabled = False
        # Cinematic fire rendering (FireLab roadmap Phase 2, task 1):
        # off by default, so every existing "science mode" call site
        # (research views, tests) is pixel-unaffected.
        self._cinematic_enabled = False
        self._cinema_pipeline: EffectsPipeline = None
        self._last_frame = None
        # Ceiling-obstruction display mask (colormap expressiveness follow-
        # up): boolean (n_z, n_x), True where a cell is a dead solid-
        # interior grid point (schematic.ceiling_obstruction_mask) that
        # should render as background, not a colored (mis)reading -- color
        # only, never applied to _last_frame, so hover/isotherms/min-max
        # keep reporting the true raw value at every cell, masked or not.
        # None (default, and for every quantity this doesn't apply to)
        # means "no masking", identical to today's behavior.
        self._ceiling_mask = None
        # Sub-frame interpolation state: blends _interp_from -> _interp_to
        # over the interval between real show_frame() calls, ticking a
        # timer independent of TimeController's own (data-rate) clock.
        self._interp_timer = QtCore.QTimer(self.canvas)
        self._interp_timer.timeout.connect(self._interp_tick)
        self._interp_from = None
        self._interp_to = None
        self._interp_phase = 1.0
        self._interp_bloom_intensity = 1.0
        self._interp_velocity_frame = None
        self._interp_soot_frame = None
        self._interp_soot_ceiling = None
        # Ember particles (FireLab roadmap Phase 2.1g): a second,
        # independently blit-tracked artist -- see widgets.py's
        # multi-artist blit_update() extension.
        self.ember_scatter = None
        self._ember_sim: EmberParticles = None
        # Velocity flow arrows: same lazy, blit-tracked-artist treatment
        # as embers. Sample grid positions (_arrow_rows/_arrow_cols) are
        # fixed once computed; only direction/magnitude change per frame.
        self.velocity_quiver = None
        self._arrow_rows = None
        self._arrow_cols = None
        # Virtual device markers (V6-M2): same blit-tracked-artist treatment
        # as embers/arrows -- fixed-position, per-frame-recolored scatters,
        # never recreated. No color-mapping (literal facecolors only), so
        # (unlike the heatmap) they need no cmap/clim bookkeeping. One
        # scatter per device kind (Analysis UX + reliability pass) so kind
        # is visible as marker shape, not just active/idle color -- see
        # _DEVICE_MARKER_SHAPES.
        self.device_scatters: dict = {}
        self._device_legend = None
        self._device_markers_present = False
        # True velocity vectors (V6-M3): a *different* artist from the
        # cinema-mode `velocity_quiver` above (that one is a heuristic
        # direction guess from |v| + a temperature gradient, explicitly not
        # science-grade -- see cinema/velocity_arrows.py). This quiver/line
        # pair is the real, gated U/W vector field: off/empty until a panel
        # supplies data, positions held fixed by density (only U/V/segments
        # refresh per frame -- see set_vector_field).
        self.true_vector_quiver = None
        self._true_vector_xy = None
        self.streamline_collection = None
        self._streamline_colors = None
        # Linked hover (V6-M4): a small always-present ring, empty until a
        # panel (e.g. the Context Panel) hovers a device/probe row. Never
        # touches selection_bus -- a pure visual highlight, not a selection.
        self.hover_highlight = None
        # Room overlay (Live polish): the enclosed room's physical boundary
        # (schematic.py's room_overlay_geometry/ROOM_X/ROOM_Z, the same
        # single source of truth the sidebar diagram already draws door/
        # vent openings from), so it reads clearly on the heatmap regardless
        # of how the current quantity's colormap happens to render near-
        # zero/wall-adjacent cells. Three artists, not one plain rectangle,
        # so the door gap and vent states are visible too, not just a closed
        # box: room_walls (solid boundary minus the door gap), room_door
        # (the opening, its own color), room_vents (2 short segments,
        # colored by open/closed/HVAC). Only meaningful on the y-normal
        # plane those constants describe; set_room_outline(None) hides all
        # three for any other plane.
        self.room_walls = None
        self.room_door = None
        self.room_vents = None
        # A one-word label on the door segment (colormap expressiveness
        # follow-up): the door line sits at the room's left wall, with
        # real exterior domain space beyond it -- unlabeled, that empty
        # region reads as a rendering gap rather than "outside the room,
        # on purpose". Empty/no position until set_room_outline() places
        # it, same "always present, empty by default" convention as the
        # three artists above.
        self.room_door_label = None

    def widget(self) -> QtWidgets.QWidget:
        return self.canvas

    def init_plot(self, first_frame: np.ndarray, cmap: str, interpolation: str,
                   vmin: float, vmax: float, colorbar_label: str, extent: tuple = None,
                   ceiling_mask=None):
        """One-time setup: axes, image artist, colorbar. Call once per
        SliceView instance before show_frame(). `extent`, if given, is
        (x0, x1, z0, z1) in physical meters -- see the class docstring for
        why this is the load-bearing piece for M2.6's probe/isotherms.
        `ceiling_mask`, if given, is applied to this first frame the same
        way set_ceiling_mask()/show_frame() apply it to every later one."""
        self._extent = extent
        self._ceiling_mask = ceiling_mask
        # Heatmap + colorbar as two explicit GridSpec columns (live-cell
        # re-proportioning pass), not the previous add_subplot(111) +
        # implicit fig.colorbar(fraction=..., pad=...) shrink-the-parent-ax
        # approach -- that left the colorbar's box (and its tick label
        # text, which renders *outside* the box) squeezed into whatever
        # sliver sat between the heatmap's right edge and the figure's own
        # right=0.95 boundary, clipping tick labels whenever that sliver
        # was narrower than the label text (confirmed directly: "1.0"/
        # "0.2"-style labels rendered as bare "1."/"0."). width_ratios and
        # the left/right margins below are all fractions of the figure, so
        # the split holds at any window size instead of only looking right
        # at whatever size this was tuned at -- right=0.83 in particular
        # reserves real figure width *outside* the colorbar's own box for
        # its tick labels and rotated unit label to render into.
        gs = self.canvas.fig.add_gridspec(1, 2, width_ratios=(18, 1), wspace=0.6,
                                          left=0.03, right=0.83, top=0.97, bottom=0.05)
        self.ax = self.canvas.fig.add_subplot(gs[0, 0])
        self._colorbar_ax = self.canvas.fig.add_subplot(gs[0, 1])
        self.ax.set_facecolor(MplCanvas.PLOT_BG)
        imshow_kwargs = dict(cmap=cmap, interpolation=interpolation, aspect="auto")
        if extent is not None:
            imshow_kwargs["extent"] = extent
        self.heatmap = self.ax.imshow(self._masked_for_display(first_frame), **imshow_kwargs)
        # Real soot-density smoke overlay (continuous soot-density
        # visualization pass): a second image over the heatmap, initially
        # fully transparent (alpha=0 everywhere) -- same shape/extent as
        # the primary heatmap so it's pixel-aligned by construction. Uses
        # SOOT DENSITY's own registry colormap (gray_r), not a new color
        # convention. zorder above the heatmap, below devices/room/hover
        # so those stay legible through the smoke.
        soot_q = get_quantity("SOOT DENSITY")
        self.soot_overlay = self.ax.imshow(
            np.zeros_like(first_frame, dtype=np.float32), cmap=soot_q.cmap,
            vmin=0.0, vmax=smd.MIN_CEILING_MG_M3,
            alpha=np.zeros_like(first_frame, dtype=np.float32),
            aspect="auto", zorder=2, **({"extent": extent} if extent is not None else {}))
        # Ember particles (Phase 2.1g): always present, empty/invisible
        # outside cinematic mode -- zorder puts it above the heatmap.
        # Deliberately no c=... at construction: passing a color arg (even
        # an empty list) puts the collection into scalar-mappable mode,
        # where draw() recomputes (and clobbers) facecolor from the
        # colormap on every redraw -- set_facecolor() below would get
        # silently overwritten back to empty on the very next blit.
        self.ember_scatter = self.ax.scatter([], [], s=[], zorder=5)
        # Virtual device markers (V6-M2): scientific style -- a small fixed
        # marker per placed device, above embers/quiver, shaped by device
        # kind (Analysis UX + reliability pass -- see _DEVICE_MARKER_SHAPES)
        # so a heat detector and a sprinkler are visually distinguishable at
        # a glance, not just by their (independently, correctly differing)
        # active/idle fill color. No c=... at construction for the same
        # scalar-mappable reason as ember_scatter.
        self.device_scatters = {
            kind: self.ax.scatter([], [], s=70, marker=shape, zorder=7,
                                  edgecolors="#14171F", linewidths=1.0)
            for kind, shape in _DEVICE_MARKER_SHAPES.items()
        }
        # Device-kind legend (Analysis final-polish pass): a static overlay
        # artist added once, same convention as the colorbar above -- not
        # in _animated_artists(), so it costs nothing per-frame. Built from
        # the scatters themselves so its shapes always match what's
        # actually drawn. Hidden until set_device_markers() sees at least
        # one placed device (nothing to explain on a scenario with none).
        self._device_legend = self.ax.legend(
            handles=[self.device_scatters[k] for k in _DEVICE_MARKER_SHAPES],
            labels=[_DEVICE_KIND_LABELS[k] for k in _DEVICE_MARKER_SHAPES],
            loc="lower left", fontsize=6, framealpha=0.55, facecolor="#14171F",
            labelcolor="white", borderpad=0.4, handletextpad=0.4)
        self._device_legend.set_visible(False)
        # True velocity streamlines (V6-M3): empty until a panel supplies
        # segments; the quiver itself is created lazily on first real data
        # (see set_vector_field) since its arrow positions are fixed for a
        # given density and matplotlib's Quiver has no clean "empty" state.
        self.streamline_collection = LineCollection([], linewidths=1.2, zorder=8)
        self.ax.add_collection(self.streamline_collection)
        # Linked hover (V6-M4): a hollow ring, empty until set_hover_highlight
        # is called -- same "always present, empty by default" convention as
        # ember_scatter/device_scatter above.
        self.hover_highlight = self.ax.scatter([], [], s=220, marker="o", facecolors="none",
                                               edgecolors="#FDE047", linewidths=2.0, zorder=10)
        # Room overlay: empty until set_room_outline() gives it real
        # segments, same "always present, empty by default" convention.
        self.room_walls = LineCollection([], linewidths=_ROOM_WALL_LW, linestyle="--",
                                         colors="#FFFFFF", zorder=6,
                                         path_effects=_cased_line_effect(_ROOM_WALL_LW))
        self.ax.add_collection(self.room_walls)
        self.room_door = LineCollection([], linewidths=_ROOM_DOOR_LW, colors="#38BDF8", zorder=6,
                                        path_effects=_cased_line_effect(_ROOM_DOOR_LW))
        self.ax.add_collection(self.room_door)
        self.room_vents = LineCollection([], linewidths=_ROOM_VENT_LW, zorder=6,
                                         path_effects=_cased_line_effect(_ROOM_VENT_LW))
        self.ax.add_collection(self.room_vents)
        self.room_door_label = self.ax.text(
            0, 0, "", fontsize=6, color="#38BDF8", ha="left", va="center", zorder=6,
            path_effects=_cased_line_effect(1.0))
        self.room_door_label.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # cax=self._colorbar_ax (not the previous fraction=/pad=, which
        # implicitly shrinks self.ax to make room) -- the GridSpec above
        # already reserves this axes' box and the figure margin beyond it,
        # so the colorbar just draws into the box it's given instead of
        # renegotiating space with the heatmap every time.
        self.colorbar = self.canvas.fig.colorbar(self.heatmap, cax=self._colorbar_ax)
        self.colorbar.set_label(colorbar_label)
        self.heatmap.set_clim(vmin=vmin, vmax=vmax)
        self.canvas.capture_background()

    def show_frame(self, frame: np.ndarray, velocity_frame: np.ndarray = None,
                    next_frame: np.ndarray = None, bloom_intensity: float = 1.0,
                    soot_frame: np.ndarray = None, soot_ceiling: float = None) -> None:
        """velocity_frame (GUI modernization pass, item 6): this cell's
        VELOCITY data at the same timestep -- meaningful when the velocity
        overlay is on (drives the contour overlay) and/or cinematic mode
        is on (drives the smoke layer's Tier 2 advection, FireLab roadmap
        Phase 2.1f; Tier 1's fixed drift is used when this is None). None
        otherwise, the default for every pre-existing caller.

        next_frame/bloom_intensity (FireLab roadmap Phase 2.1c/d, cinematic
        mode only): next_frame is the frame at the following timestep --
        already resident in the same cached array MainWindow just indexed
        `frame` out of, so this costs nothing to look up -- used to blend
        smoothly toward it between real ticks (see _interp_tick). None
        means no lookahead available (e.g. at the end of the series);
        interpolation is simply skipped that tick. bloom_intensity scales
        the pipeline's bloom/flicker strength (typically the scenario's
        current HRR normalized to its own peak); ignored outside cinematic
        mode.

        soot_frame/soot_ceiling (continuous soot-density visualization
        pass): this cell's real SOOT DENSITY data at the same timestep,
        and the normalization ceiling MainWindow computed once for this
        (scenario, plane). Drives two *different* things depending on
        mode, deliberately kept distinct (never both at once, to avoid
        double-rendering smoke): outside cinematic mode, the separate
        scientific soot_overlay artist (see set_soot_overlay_enabled) --
        the actual SOOT DENSITY field, continuously mapped to opacity.
        Inside cinematic mode, the fire pipeline's own smoke layer (see
        cinema/smoke.py) -- a visually-enhanced (smoothed) rendering of
        that same real field, not a separate/fabricated effect. Both None
        outside either case, the default for every pre-existing caller."""
        self._last_frame = frame
        if self._cinematic_enabled:
            self._interp_bloom_intensity = bloom_intensity
            self._interp_velocity_frame = velocity_frame
            self._interp_soot_frame = soot_frame
            self._interp_soot_ceiling = soot_ceiling
            if next_frame is not None:
                self._interp_from = frame
                self._interp_to = next_frame
                self._interp_phase = 0.0
                if not self._interp_timer.isActive():
                    self._interp_timer.start(_INTERP_INTERVAL_MS)
            else:
                self._interp_timer.stop()
                self._interp_from = None
                self._interp_to = None
            display_data = self._cinema_pipeline.render(
                frame, hrr_intensity=bloom_intensity, velocity_frame=velocity_frame,
                soot_frame=soot_frame, soot_ceiling=soot_ceiling)
            self._update_ember_scatter(frame, velocity_frame)
            self._update_velocity_arrows(frame, velocity_frame)
            # The cinema pipeline's own smoke layer (above) is the smoke
            # representation while cinematic mode is on -- the separate
            # scientific overlay artist stays cleared so the two never
            # composite on top of each other.
            self._clear_soot_overlay()
        else:
            display_data = self._masked_for_display(frame)
            if soot_frame is not None and soot_ceiling is not None:
                self._update_soot_overlay(soot_frame, soot_ceiling)
        self.heatmap.set_data(display_data)
        self._velocity_frame = velocity_frame
        overlay_active = (
            (self._isotherms_enabled and self._isotherm_levels)
            or (self._velocity_overlay_enabled and self._velocity_overlay_levels and velocity_frame is not None)
        )
        if overlay_active:
            # Contours redrawn per frame, full draw (blit bypass while
            # active) -- matplotlib has no cheap "update contour data"
            # primitive the way set_data() gives the image artist, and
            # ROADMAP.md's M2.6 spec explicitly accepts this cost at this
            # grid size rather than asking for a fast-path that doesn't
            # exist in matplotlib's contour API.
            self._redraw_isotherms()
            self._redraw_velocity_overlay()
            self.canvas.draw_idle()
            self.canvas.capture_background()
        else:
            self.canvas.blit_update(self._animated_artists())

    def _animated_artists(self) -> list:
        artists = [self.heatmap, self.soot_overlay, self.ember_scatter,
                  *self.device_scatters.values(),
                  self.streamline_collection, self.hover_highlight,
                  self.room_walls, self.room_door, self.room_vents, self.room_door_label]
        if self.velocity_quiver is not None:
            artists.append(self.velocity_quiver)
        if self.true_vector_quiver is not None:
            artists.append(self.true_vector_quiver)
        return artists

    def set_room_outline(self, geometry) -> None:
        """`geometry` is schematic.room_overlay_geometry()'s return dict
        (walls minus the door gap, the door opening, the two vents with
        their open/closed/HVAC state) in the same physical-meter
        coordinates as this view's extent, or None to hide -- MainWindow
        only ever passes real geometry on the y-normal plane ROOM_X/ROOM_Z
        describe, matching this app's gating convention of never drawing
        something that isn't actually known for the current plane."""
        if geometry is None:
            self.room_walls.set_segments([])
            self.room_door.set_segments([])
            self.room_vents.set_segments([])
            self.room_door_label.set_visible(False)
            self.ax.set_ylim(auto=True)
            self.ax.autoscale(True, axis="y")
            self.canvas.capture_background()
            return
        self.room_walls.set_segments([[(x0, z0), (x1, z1)] for x0, z0, x1, z1 in geometry["walls"]])
        # Layout-tightening pass: the y-normal mesh domain reaches well above
        # the real ceiling (ambient buffer air FDS needs but nothing this app
        # visualizes), so the un-cropped view was mostly empty sky above the
        # room. Crop the visible range to the room's own wall bounds (floor to
        # the ceiling slab's top face) plus a small margin -- not the full
        # `extent` itself (that stays untouched: probes/pixel math/colorbar
        # all still address the real mesh coordinates, only the camera moves).
        wall_zs = [z for _x0, z0, _x1, z1 in geometry["walls"] for z in (z0, z1)]
        z_bottom, z_top = min(wall_zs), max(wall_zs)
        margin = (z_top - z_bottom) * 0.05
        self.ax.set_ylim(z_bottom - margin, z_top + margin)
        dx0, dz0, dx1, dz1 = geometry["door"]
        self.room_door.set_segments([[(dx0, dz0), (dx1, dz1)]])
        # "Door" (colormap expressiveness follow-up): the door line sits
        # right where the room's real exterior domain space begins (see
        # ROOM_X/ROOM_Z in schematic.py) -- named so that empty region
        # reads as "outside the room" rather than a rendering gap. A small
        # fixed offset to the right of the line, in the same physical
        # (meter) units as everything else this method places, keeps it
        # just inside the room instead of overlapping the door line itself.
        self.room_door_label.set_position((dx0 + 0.02, (dz0 + dz1) / 2))
        self.room_door_label.set_text("Door")
        self.room_door_label.set_visible(True)
        vent_segs, vent_colors = [], []
        for (x0, z0, x1, z1), state in geometry["vents"]:
            vent_segs.append([(x0, z0), (x1, z1)])
            vent_colors.append(_VENT_STATE_COLORS.get(state, "#94A3B8"))
        self.room_vents.set_segments(vent_segs)
        self.room_vents.set_color(vent_colors)
        # set_ylim above changes what's underneath the animated artists
        # (MplCanvas's own docstring: anything that does must recapture) --
        # skipping this left the blit cache holding a stale, differently-
        # zoomed snapshot (sometimes the *previous quantity's* heatmap and
        # colorbar), which the next blit_update() would restore and then
        # paint the new, differently-scaled frame on top of -- a doubled,
        # misaligned-looking plot.
        self.canvas.capture_background()

    def set_ui_scale(self, scale: float) -> None:
        """View -> UI Scale (accessibility zoom): delegates to the canvas's
        DPI scaling (see MplCanvas.set_dpi_scale), which scales everything
        in the figure together -- colorbar ticks/label, the room overlay's
        line widths, any other point-sized artist -- instead of hand-tuning
        one overlay's line widths in isolation (which would now double-
        scale on top of the DPI change if left in place)."""
        self.canvas.set_dpi_scale(scale)

    def set_device_markers(self, markers: list) -> None:
        """markers: [(x, z, color, kind), ...] physical positions (V6-M2
        Virtual Device Network) -- MainWindow recomputes this cheaply every
        tick from each device's already-cached results (state_at()), never
        a recompute here. Bucketed by kind into device_scatters (Analysis
        UX + reliability pass) so each kind keeps its own marker shape;
        unrecognized kinds are silently skipped (never guessed). No-op
        (every kind's markers cleared) when this cell has no physical
        extent to place a point on."""
        by_kind: dict = {kind: ([], []) for kind in self.device_scatters}
        if markers and self._extent is not None:
            for x, z, color, kind in markers:
                if kind in by_kind:
                    offsets, colors = by_kind[kind]
                    offsets.append((x, z))
                    colors.append(color)
        for kind, scatter in self.device_scatters.items():
            offsets, colors = by_kind[kind]
            scatter.set_offsets(offsets if offsets else np.empty((0, 2)))
            scatter.set_facecolor(colors)
        # Device-kind legend: only visible once something is actually
        # placed (nothing to explain on a scenario with no devices), and
        # only redrawn on the rare present/absent transition -- this is
        # called every tick, but the legend isn't in _animated_artists()
        # (a static overlay, like the colorbar), so a plain set_visible()
        # wouldn't survive the next blit_update() without an explicit full
        # draw + background recapture, same as the isotherm/overlay toggle.
        present = any(offsets for offsets, _ in by_kind.values())
        if present != self._device_markers_present:
            self._device_markers_present = present
            if self._device_legend is not None:
                self._device_legend.set_visible(present)
                self.canvas.draw_idle()
                self.canvas.capture_background()

    def set_vector_field(self, quiver: tuple = None, streamlines: list = None) -> None:
        """V6-M3: `quiver` is (xs, zs, us, ws) or None to clear; `streamlines`
        is a list of [(x, z), ...] polylines or None/empty to clear. Quiver
        arrow *positions* are recreated only when they change (a control
        change, e.g. density) -- per-frame, only U/V (and the streamline
        segments) update, matching the ember/device-marker cost profile."""
        if not streamlines:
            self.streamline_collection.set_segments([])
        else:
            self.streamline_collection.set_segments(streamlines)
            if self._streamline_colors is not None and len(self._streamline_colors) == len(streamlines):
                self.streamline_collection.set_color(self._streamline_colors)
        if quiver is None or self._extent is None:
            if self.true_vector_quiver is not None:
                self.true_vector_quiver.set_UVC(
                    np.zeros(len(self._true_vector_xy[0])), np.zeros(len(self._true_vector_xy[0])))
            return
        xs, zs, us, ws = quiver
        same_grid = (self._true_vector_xy is not None
                    and len(xs) == len(self._true_vector_xy[0])
                    and np.allclose(xs, self._true_vector_xy[0])
                    and np.allclose(zs, self._true_vector_xy[1]))
        if self.true_vector_quiver is None or not same_grid:
            if self.true_vector_quiver is not None:
                self.true_vector_quiver.remove()
            self.true_vector_quiver = self.ax.quiver(
                xs, zs, us, ws, color="#14171F", angles="xy", scale_units="xy",
                width=0.004, zorder=9)
            self._true_vector_xy = (xs, zs)
        else:
            self.true_vector_quiver.set_UVC(us, ws)

    def set_streamline_colors(self, colors: list) -> None:
        """One color per streamline (V6-M3 color-by) -- applied on the next
        set_vector_field() call, since LineCollection needs colors and
        segments set together for a clean per-line mapping."""
        self._streamline_colors = list(colors) if colors else None

    def set_hover_highlight(self, point) -> None:
        """Linked hover (V6-M4): highlight a physical (x, z) point (e.g. a
        device/probe row hovered in the Context Panel) without touching
        selection_bus -- `point=None` clears it. Caller must also call
        redraw_overlays_now() to blit immediately (hover isn't tied to a
        TimeController tick)."""
        if point is None or self._extent is None:
            self.hover_highlight.set_offsets(np.empty((0, 2)))
            return
        self.hover_highlight.set_offsets([point])

    def redraw_overlays_now(self) -> None:
        """Blit the current device markers / vector field / hover highlight
        immediately (V6-M2/V6-M3/V6-M4) -- for the placed/edited/deleted/
        hovered case, outside a TimeController tick, where nothing else is
        about to call show_frame() and trigger the blit."""
        self.canvas.blit_update(self._animated_artists())

    def set_cmap(self, name: str) -> None:
        self.heatmap.set_cmap(mpl.colormaps[name])
        self.canvas.capture_background()

    def set_clim(self, vmin: float, vmax: float) -> None:
        self.heatmap.set_clim(vmin=vmin, vmax=vmax)
        self.canvas.capture_background()

    def set_extent(self, extent) -> None:
        """Update the plotted plane's physical extent in place (M2.2):
        needed when a cell switches to a quantity on a *different* plane
        (e.g. a SOOT doorway slice vs the standard side view), whose
        physical box and array shape differ from the current one. Unlike
        M2.6's assumption that a cell's extent is fixed for its lifetime,
        an `.s3d` any-plane quantity can legitimately change it. Keeps
        probe/isotherm coordinate mapping correct, since both read
        self._extent."""
        self._extent = extent
        if extent is not None:
            self.heatmap.set_extent(extent)
        self.canvas.capture_background()

    def set_interpolation(self, name: str) -> None:
        self.heatmap.set_interpolation(name)
        self.canvas.capture_background()

    def set_colorbar_label(self, text: str) -> None:
        self.colorbar.set_label(text)
        self.canvas.capture_background()

    def set_colorbar_offset(self, offset: float = 0.0) -> None:
        """Relabel the colorbar's tick numbers as (raw value - offset)
        without touching the underlying data or clim -- e.g. TEMPERATURE's
        colorbar reads as rise-above-ambient (0..150) while the heatmap
        array itself, and every hover/min-max readout that reads it, stay
        true absolute °C. offset=0 restores plain (untranslated) ticks."""
        if offset:
            self.colorbar.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, pos: f"{v - offset:g}"))
        else:
            self.colorbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        self.canvas.capture_background()

    def set_ceiling_mask(self, mask) -> None:
        """Set/clear the ceiling-obstruction display mask (see the
        _ceiling_mask attribute) -- takes effect on the next show_frame()/
        init_plot(), doesn't itself force a redraw. `mask` is a boolean
        (n_z, n_x) array matching this view's current frame shape, or
        None to disable masking (every quantity this doesn't apply to)."""
        self._ceiling_mask = mask

    def _masked_for_display(self, frame: np.ndarray) -> np.ndarray:
        """`frame`, or a copy with _ceiling_mask cells set to NaN for
        color-mapping only -- imshow renders NaN via the colormap's "bad"
        color (transparent by default), showing the canvas's own white
        background through, same as an empty SOOT DENSITY field. Never
        mutates `frame` itself -- callers keep using the original for
        _last_frame/hover/isotherms."""
        if self._ceiling_mask is None or self._ceiling_mask.shape != frame.shape:
            return frame
        display = frame.astype(float, copy=True)
        display[self._ceiling_mask] = np.nan
        return display

    def set_title(self, text: str) -> None:
        self.ax.set_title(text)
        self.canvas.capture_background()

    def capture_background(self) -> None:
        """Force a full redraw + recapture -- for changes that don't go
        through one of the setters above (e.g. a theme restyle)."""
        self.canvas.capture_background()

    # ---------------------------------- cinematic fire rendering (Phase 2.1)
    def set_cinematic_mode(self, enabled: bool, vmin: float = None, vmax_init: float = None) -> None:
        """Toggle FireLab's cinematic rendering (FireLUT + alpha + filmic
        tone map + auto-exposure, see cinema/pipeline.py) on this cell.

        `vmin`/`vmax_init` seed the pipeline's fixed lower bound and the
        auto-exposure EMA's starting point -- required when enabling,
        ignored when disabling. Science mode (enabled=False, the default)
        is untouched: show_frame() falls back to handing matplotlib the
        raw array through the normal cmap/clim/colorbar path exactly as
        before this feature existed.
        """
        if enabled == self._cinematic_enabled:
            return
        self._cinematic_enabled = enabled
        if enabled:
            if vmin is None or vmax_init is None:
                raise ValueError("vmin and vmax_init are required to enable cinematic mode")
            self._cinema_pipeline = EffectsPipeline(vmin, vmax_init)
            blank_shape = self.heatmap.get_array().shape[:2]
            self._ember_sim = EmberParticles(blank_shape)
            self.ax.set_facecolor(CINEMA_BG)
            self.colorbar.ax.set_visible(False)
            # capture_background() below does a full draw right now, before
            # the first cinematic show_frame() has run -- the heatmap still
            # holds whatever non-RGBA science-mode frame (and cmap) it last
            # displayed. Left alone, that stale frame gets baked into the
            # cached blit background, and every future transparent (ambient)
            # cinema pixel would let it show through instead of the near-
            # black facecolor, since blitting composites new draws with
            # alpha over the cached background, not over a blank facecolor.
            # A fully transparent placeholder frame sidesteps that entirely.
            self.heatmap.set_data(np.zeros(blank_shape + (4,), dtype=np.uint8))
        else:
            self._cinema_pipeline = None
            self._ember_sim = None
            self.ember_scatter.set_offsets(np.empty((0, 2)))
            self.ember_scatter.set_sizes([])
            self.ember_scatter.set_facecolor([])
            if self.velocity_quiver is not None:
                self.velocity_quiver.remove()
                self.velocity_quiver = None
            self._arrow_rows = None
            self._arrow_cols = None
            self._interp_timer.stop()
            self._interp_from = None
            self._interp_to = None
            self.ax.set_facecolor(MplCanvas.PLOT_BG)
            self.colorbar.ax.set_visible(True)
        self.canvas.capture_background()

    def _index_to_display_xy(self, rows, cols):
        """(row, col) array-index coordinates -> (x, y) display
        coordinates matching the heatmap's own coordinate system
        (physical meters if extent is set, else raw pixel indices) --
        shared by ember offsets and velocity arrows, same convention
        _redraw_isotherms' own xs/zs construction already uses. rows/cols
        may be non-integer (e.g. an arrow's head position)."""
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        if self._extent is not None and self._last_frame is not None:
            x0, x1, z0, z1 = self._extent
            n_z, n_x = self._last_frame.shape
            xs = x0 + (cols / max(n_x - 1, 1)) * (x1 - x0)
            ys = z1 - (rows / max(n_z - 1, 1)) * (z1 - z0)  # row 0 = z1 (top), matching origin='upper'
            return xs, ys
        return cols, rows

    def _update_ember_scatter(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray) -> None:
        self._ember_sim.step(temperature_frame, self._cinema_pipeline.vmin, velocity_frame)
        offsets, sizes, colors = self._ember_sim.render_arrays()
        if len(offsets):
            xs, ys = self._index_to_display_xy(offsets[:, 1], offsets[:, 0])
            offsets = np.column_stack([xs, ys])
        self.ember_scatter.set_offsets(offsets)
        self.ember_scatter.set_sizes(sizes)
        self.ember_scatter.set_facecolor(colors)

    def _update_velocity_arrows(self, temperature_frame: np.ndarray, velocity_frame) -> None:
        """Sparse directional flow arrows (heuristic direction -- see
        cinema/velocity_arrows.py's own docstring for why: the stored
        VELOCITY slice is speed magnitude only, no true vector)."""
        if velocity_frame is None:
            if self.velocity_quiver is not None:
                zeros = np.zeros(len(self._arrow_rows))
                self.velocity_quiver.set_UVC(zeros, zeros)
            return
        if self._arrow_rows is None:
            self._arrow_rows, self._arrow_cols = sample_points(temperature_frame.shape)
        d_row, d_col = compute_deltas(temperature_frame, velocity_frame, self._arrow_rows, self._arrow_cols)
        tail_x, tail_y = self._index_to_display_xy(self._arrow_rows, self._arrow_cols)
        head_x, head_y = self._index_to_display_xy(self._arrow_rows + d_row, self._arrow_cols + d_col)
        u, v = head_x - tail_x, head_y - tail_y
        if self.velocity_quiver is None:
            self.velocity_quiver = self.ax.quiver(
                tail_x, tail_y, u, v, color="#CFE8FF", alpha=0.8,
                angles="xy", scale_units="xy", scale=1.0, width=0.004, zorder=6,
            )
        else:
            self.velocity_quiver.set_UVC(u, v)

    @property
    def cinematic_enabled(self) -> bool:
        return self._cinematic_enabled

    @property
    def cinema_pipeline_cost_ms(self) -> float:
        """Last render() call's cost in ms -- 0.0 if cinematic mode is off
        or no frame has been rendered yet. For bench/telemetry use."""
        return self._cinema_pipeline.last_cost_ms if self._cinema_pipeline else 0.0

    def _interp_tick(self) -> None:
        """Sub-frame interpolation timer callback (~30 Hz while cinematic
        mode is on and a lookahead frame is available): advances the blend
        phase toward _interp_to and blits the result, independent of
        TimeController's own (data-rate) tick. Never touches _last_frame
        or overlays -- those stay tied to the last *real* frame only."""
        if not self._cinematic_enabled or self._interp_to is None:
            self._interp_timer.stop()
            return
        self._interp_phase = min(1.0, self._interp_phase + _INTERP_INTERVAL_MS / _NOMINAL_TICK_MS)
        blended = lerp_frames(self._interp_from, self._interp_to, self._interp_phase)
        rgba = self._cinema_pipeline.render(
            blended, hrr_intensity=self._interp_bloom_intensity, velocity_frame=self._interp_velocity_frame,
            soot_frame=self._interp_soot_frame, soot_ceiling=self._interp_soot_ceiling)
        self._update_ember_scatter(blended, self._interp_velocity_frame)
        self._update_velocity_arrows(blended, self._interp_velocity_frame)
        self.heatmap.set_data(rgba)
        self.canvas.blit_update(self._animated_artists())
        if self._interp_phase >= 1.0:
            self._interp_timer.stop()

    # -------------------------------------------------- isotherms (M2.6.2)
    def set_isotherm_levels(self, levels: list) -> None:
        self._isotherm_levels = list(levels)
        if self._isotherms_enabled:
            self._redraw_isotherms()
            self.canvas.capture_background()

    def set_isotherms_enabled(self, enabled: bool) -> None:
        if enabled == self._isotherms_enabled:
            return
        self._isotherms_enabled = enabled
        if enabled:
            self._redraw_isotherms()
        else:
            self._clear_isotherms()
        self.canvas.capture_background()

    @property
    def isotherms_enabled(self) -> bool:
        return self._isotherms_enabled

    def _clear_isotherms(self) -> None:
        if self._contour_artist is not None:
            self._contour_artist.remove()
            self._contour_artist = None

    def _redraw_isotherms(self) -> None:
        self._clear_isotherms()
        if not self._isotherm_levels or self.heatmap is None:
            return
        # Raw temperature data, not the RGBA the heatmap artist may hold
        # while cinematic mode is on (see show_frame()/_last_frame).
        frame = self._last_frame if self._last_frame is not None else self.heatmap.get_array()
        levels = sorted(set(self._isotherm_levels))
        if self._extent is not None:
            x0, x1, z0, z1 = self._extent
            n_z, n_x = frame.shape
            xs = np.linspace(x0, x1, n_x)
            zs = np.linspace(z1, z0, n_z)  # row 0 = z1 (top), matching origin='upper'
            self._contour_artist = self.ax.contour(
                xs, zs, frame, levels=levels, colors="white", linewidths=0.8)
        else:
            self._contour_artist = self.ax.contour(
                frame, levels=levels, colors="white", linewidths=0.8)

    # ---------------------------------------- velocity overlay (item 6)
    def set_velocity_overlay_levels(self, levels: list) -> None:
        self._velocity_overlay_levels = list(levels)

    def set_velocity_overlay_enabled(self, enabled: bool) -> None:
        if enabled == self._velocity_overlay_enabled:
            return
        self._velocity_overlay_enabled = enabled
        if not enabled:
            self._clear_velocity_overlay()
            self.canvas.capture_background()
        # else: left to the next show_frame(..., velocity_frame=...) call
        # to actually draw it -- there's no "current" velocity frame to
        # redraw from until MainWindow supplies one.

    @property
    def velocity_overlay_enabled(self) -> bool:
        return self._velocity_overlay_enabled

    def _clear_velocity_overlay(self) -> None:
        if self._velocity_contour_artist is not None:
            self._velocity_contour_artist.remove()
            self._velocity_contour_artist = None

    def _redraw_velocity_overlay(self) -> None:
        self._clear_velocity_overlay()
        if not self._velocity_overlay_enabled or not self._velocity_overlay_levels or self._velocity_frame is None:
            return
        frame = self._velocity_frame
        levels = sorted(set(self._velocity_overlay_levels))
        # Deliberately distinct from _redraw_isotherms's white solid lines
        # (dashed, accent blue) so a viewer never confuses a temperature
        # hazard-band line with a velocity speed-band line when both
        # happen to be visible at once.
        style = dict(colors="#4FA8E8", linewidths=1.1, linestyles="dashed")
        if self._extent is not None:
            x0, x1, z0, z1 = self._extent
            n_z, n_x = frame.shape
            xs = np.linspace(x0, x1, n_x)
            zs = np.linspace(z1, z0, n_z)
            self._velocity_contour_artist = self.ax.contour(xs, zs, frame, levels=levels, **style)
        else:
            self._velocity_contour_artist = self.ax.contour(frame, levels=levels, **style)

    # ------------------------ real soot-density smoke overlay ("smoke" pass)
    def set_soot_overlay_enabled(self, enabled: bool) -> None:
        if enabled == self._soot_overlay_enabled:
            return
        self._soot_overlay_enabled = enabled
        if enabled:
            if self.soot_colorbar is None and self.soot_overlay is not None:
                # Lazily created only while the overlay is actually on, so
                # a plain temperature view's layout is byte-for-byte
                # unchanged when the overlay is off (never reserves this
                # space by default).
                self.soot_colorbar = self.canvas.fig.colorbar(
                    self.soot_overlay, fraction=0.04, pad=0.10)
                self.soot_colorbar.set_label(
                    f"{get_quantity('SOOT DENSITY').label} ({get_quantity('SOOT DENSITY').unit})")
        else:
            self._clear_soot_overlay()
            if self.soot_colorbar is not None:
                self.soot_colorbar.remove()
                self.soot_colorbar = None
        self.canvas.capture_background()
        # else (enabling): left to the next show_frame(..., soot_frame=...)
        # call to actually draw it -- same "no current frame to redraw
        # from yet" reasoning as the velocity overlay above.

    @property
    def soot_overlay_enabled(self) -> bool:
        return self._soot_overlay_enabled

    def _clear_soot_overlay(self) -> None:
        if self.soot_overlay is None:
            return
        self.soot_overlay.set_alpha(np.zeros(self.soot_overlay.get_array().shape, dtype=np.float32))

    def _update_soot_overlay(self, soot_frame: np.ndarray, soot_ceiling: float) -> None:
        """One real SOOT DENSITY frame + its (externally computed, stable-
        across-playback) normalization ceiling -- a continuous scalar->
        opacity mapping (smoke_density.py), never a threshold. Ceiling is
        supplied rather than computed here (same "MainWindow computes the
        scale, the view just displays it" convention already used for the
        primary heatmap's own vmin/vmax) so it's computed once per
        (scenario, plane) by the caller, not re-derived from the whole
        array on every tick."""
        if self.soot_overlay is None or not self._soot_overlay_enabled or soot_frame is None:
            return
        self.soot_overlay.set_clim(0.0, soot_ceiling)
        self.soot_overlay.set_data(soot_frame)
        self.soot_overlay.set_alpha(smd.soot_alpha(soot_frame, soot_ceiling))

    # ------------------------------------------------------ probe (M2.6.1)
    def enable_probe(self, callback) -> None:
        """callback(x, z, value) is called on mouse move within the axes
        with physical coordinates and the frame value at that point;
        callback(None, None, None) when the mouse leaves the axes (or
        isn't over data). Meaningful physical (x, z) requires extent to
        have been set via init_plot(extent=...) -- without it, x/z fall
        back to raw pixel indices (still functional, just not "meters")."""
        self._probe_callback = callback
        if self._motion_cid is None:
            self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def disable_probe(self) -> None:
        if self._motion_cid is not None:
            self.canvas.mpl_disconnect(self._motion_cid)
            self._motion_cid = None
        self._probe_callback = None

    def _on_mouse_move(self, event) -> None:
        if self._probe_callback is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self._probe_callback(None, None, None)
            return
        value = self.value_at(event.xdata, event.ydata)
        self._probe_callback(event.xdata, event.ydata, value)

    def value_at(self, x: float, z: float):
        """The frame value nearest physical (x, z), or None if out of
        bounds / no extent set / nothing plotted yet. Inverse of the
        row/col -> (x, z) mapping documented on the class -- verified
        against the same M2.3 ground-truth pixel used to verify that
        forward mapping (see the class docstring)."""
        if self.heatmap is None or self._extent is None:
            return None
        x0, x1, z0, z1 = self._extent
        frame = self._last_frame if self._last_frame is not None else self.heatmap.get_array()
        n_z, n_x = frame.shape
        if x1 == x0 or z1 == z0:
            return None
        col = int(round((x - x0) / (x1 - x0) * (n_x - 1)))
        row = int(round((z1 - z) / (z1 - z0) * (n_z - 1)))
        if 0 <= row < n_z and 0 <= col < n_x:
            return float(frame[row, col])
        return None


class DifferenceView:
    """PlotView cell type "A - B" (M2.3.1): shows `store.get(A,key)[i] -
    store.get(B,key)[i]` for two scenarios sharing one quantity.

    Composes a SliceView internally rather than reimplementing rendering --
    the only things that actually differ from a plain slice are *what
    frame data* gets shown (a difference, computed by the caller and handed
    to show_frame() same as SliceView) and the display defaults: a
    diverging colormap (RdBu_r, since 0 = "no difference" is a real,
    physically meaningful center point here, unlike a sequential cmap's
    arbitrary floor) and a symmetric clim (+-max(|delta|)) so equal color
    intensity on either side of the diverging cmap's center means equal
    magnitude of difference regardless of sign.

    Verified against real data (not just synthetic arrays) before this was
    wired into any UI: for the c1_d0/c1_d1 door-width scenario pair (the
    ROADMAP's own example), the dominant difference signal turned out to
    be near the candle/plume, not the doorway -- see ROADMAP.md's M2.3
    section for the full finding. The math here is unaffected either way;
    this docstring just flags that "physically sensible structure" was
    checked against ground truth, not assumed from the spec's example.
    """

    DEFAULT_CMAP = "RdBu_r"

    def __init__(self, parent=None):
        self._inner = SliceView(parent)
        self.case_a = None
        self.case_b = None
        self.quantity_key = None
        self._symmetric_clim_cache = {}

    # Read-only passthroughs so this duck-types as SliceView-shaped for
    # MainWindow's `heatmap`/`colorbar`/`canvas`/`ax` delegating properties
    # (M2.2.1) if a cell showing this view type ever becomes the grid's
    # active cell -- those properties assume that shape regardless of
    # which PlotView implementation the active cell currently holds.
    @property
    def heatmap(self):
        return self._inner.heatmap

    @property
    def colorbar(self):
        return self._inner.colorbar

    @property
    def canvas(self):
        return self._inner.canvas

    @property
    def ax(self):
        return self._inner.ax

    def widget(self) -> QtWidgets.QWidget:
        return self._inner.widget()

    def init_plot(self, first_frame: np.ndarray, interpolation: str,
                   vmin: float, vmax: float, colorbar_label: str, cmap: str = None,
                   extent: tuple = None):
        self._inner.init_plot(first_frame, cmap=cmap or self.DEFAULT_CMAP,
                               interpolation=interpolation, vmin=vmin, vmax=vmax,
                               colorbar_label=colorbar_label, extent=extent)

    def show_frame(self, frame: np.ndarray) -> None:
        """`frame` is already the difference array (A - B) -- computed by
        the caller (compute_diff below, or MainWindow directly), not
        fetched here. Matches SliceView's own "doesn't fetch data, just
        renders what it's given" split."""
        self._inner.show_frame(frame)

    def set_cmap(self, name: str) -> None:
        self._inner.set_cmap(name)

    def set_clim(self, vmin: float, vmax: float) -> None:
        self._inner.set_clim(vmin, vmax)

    def set_extent(self, extent) -> None:
        self._inner.set_extent(extent)

    def set_interpolation(self, name: str) -> None:
        self._inner.set_interpolation(name)

    def set_colorbar_label(self, text: str) -> None:
        self._inner.set_colorbar_label(text)

    def set_title(self, text: str) -> None:
        self._inner.set_title(text)

    def set_ceiling_mask(self, mask) -> None:
        """Delegates through same as the setters above -- MainWindow's
        _sync_cell_ceiling_mask is called unconditionally on type change
        (see GridCell._swap_view), always with None here (_ceiling_mask_for
        only ever computes a real mask for a "slice" cell), but the method
        still has to exist so that call doesn't AttributeError."""
        self._inner.set_ceiling_mask(mask)

    def capture_background(self) -> None:
        self._inner.capture_background()

    # M2.6: probe/isotherms delegate straight through, same as the setters
    # above -- a difference cell's "temperature" reading at a point is just
    # its displayed delta value, and isotherms over a delta field are a
    # legitimate (if unusual) way to see the zero-crossing boundary.
    def set_isotherm_levels(self, levels: list) -> None:
        self._inner.set_isotherm_levels(levels)

    def set_isotherms_enabled(self, enabled: bool) -> None:
        self._inner.set_isotherms_enabled(enabled)

    @property
    def isotherms_enabled(self) -> bool:
        return self._inner.isotherms_enabled

    # Velocity overlay (item 6) never actually applies to this cell type
    # (main_window.py's _apply_contour_overlay_state only turns it on for
    # a "slice" cell showing TEMPERATURE), but every cell_type gets these
    # calls unconditionally the same way it already does for the isotherm
    # setters above -- delegate straight through rather than crashing on
    # a missing attribute (PyQt5 aborts the process on an unhandled
    # exception raised inside a Qt signal handler).
    def set_velocity_overlay_levels(self, levels: list) -> None:
        self._inner.set_velocity_overlay_levels(levels)

    def set_velocity_overlay_enabled(self, enabled: bool) -> None:
        self._inner.set_velocity_overlay_enabled(enabled)

    @property
    def velocity_overlay_enabled(self) -> bool:
        return self._inner.velocity_overlay_enabled

    # Real soot-density smoke overlay: same "never actually applies to
    # this cell type, but delegate straight through rather than crash on
    # a missing attribute" reasoning as the velocity overlay above -- see
    # main_window.py's _apply_smoke_overlay_state, which (like the
    # velocity overlay) only ever turns this on for a "slice" cell showing
    # TEMPERATURE.
    def set_soot_overlay_enabled(self, enabled: bool) -> None:
        self._inner.set_soot_overlay_enabled(enabled)

    @property
    def soot_overlay_enabled(self) -> bool:
        return self._inner.soot_overlay_enabled

    def enable_probe(self, callback) -> None:
        self._inner.enable_probe(callback)

    def disable_probe(self) -> None:
        self._inner.disable_probe()

    def value_at(self, x: float, z: float):
        return self._inner.value_at(x, z)

    @staticmethod
    def compute_diff(data_a: np.ndarray, data_b: np.ndarray, index: int) -> np.ndarray:
        return data_a[index] - data_b[index]

    def symmetric_clim(self, data_a: np.ndarray, data_b: np.ndarray, cache_key,
                        n_samples: int = 20) -> tuple:
        """+-max(|delta|) sampled across up to n_samples frames evenly
        spread over the shared timeline (not necessarily every frame --
        cheap at this dataset's size, but sampling is the general pattern
        that stays cheap as datasets grow, matching the spec's wording).
        Cached per (case_a, case_b, key) via `cache_key` so switching back
        to an already-seen A/B/quantity combo doesn't rescan the arrays."""
        if cache_key in self._symmetric_clim_cache:
            return self._symmetric_clim_cache[cache_key]
        n = min(data_a.shape[0], data_b.shape[0])
        sample_indices = np.linspace(0, n - 1, min(n, n_samples), dtype=int)
        diffs = data_a[sample_indices] - data_b[sample_indices]
        vmax = float(np.max(np.abs(diffs)))
        clim = (-vmax, vmax)
        self._symmetric_clim_cache[cache_key] = clim
        return clim


class EnsembleView:
    """PlotView cell type showing a composite statistic (mean/std/min/max)
    across a *selection* of scenarios sharing one quantity, at each
    timeline index (M2.3.2). Composes a SliceView internally, same split
    as DifferenceView: only what frame data means and the display
    defaults differ, not how blitting/colorbar/axes work.

    mean/min/max keep the quantity's own absolute-value display
    conventions (same cmap/vmin as a plain SliceView of that quantity --
    they're still readings of the quantity itself, just composited across
    scenarios). std is different in kind -- always >= 0, with no natural
    quantity-specific floor -- so it gets its own sequential colormap and
    a data-derived vmax (std_vmax below), labeled "sigma(<quantity>)" per
    spec rather than reusing the quantity's own unit-labeled cmap.
    """

    STATS = ("mean", "std", "min", "max")

    def __init__(self, parent=None):
        self._inner = SliceView(parent)
        self.case_indices: list = []
        self.quantity_key = None
        self.stat = "mean"
        self._std_vmax_cache = {}

    # See DifferenceView's identical passthroughs for why these exist.
    @property
    def heatmap(self):
        return self._inner.heatmap

    @property
    def colorbar(self):
        return self._inner.colorbar

    @property
    def canvas(self):
        return self._inner.canvas

    @property
    def ax(self):
        return self._inner.ax

    def widget(self) -> QtWidgets.QWidget:
        return self._inner.widget()

    def init_plot(self, first_frame: np.ndarray, cmap: str, interpolation: str,
                   vmin: float, vmax: float, colorbar_label: str, extent: tuple = None):
        self._inner.init_plot(first_frame, cmap=cmap, interpolation=interpolation,
                               vmin=vmin, vmax=vmax, colorbar_label=colorbar_label,
                               extent=extent)

    def show_frame(self, frame: np.ndarray) -> None:
        """`frame` is already the composite statistic array -- computed by
        the caller (compute_composite below, or MainWindow directly), not
        fetched here."""
        self._inner.show_frame(frame)

    def set_cmap(self, name: str) -> None:
        self._inner.set_cmap(name)

    def set_clim(self, vmin: float, vmax: float) -> None:
        self._inner.set_clim(vmin, vmax)

    def set_extent(self, extent) -> None:
        self._inner.set_extent(extent)

    def set_interpolation(self, name: str) -> None:
        self._inner.set_interpolation(name)

    def set_colorbar_label(self, text: str) -> None:
        self._inner.set_colorbar_label(text)

    def set_title(self, text: str) -> None:
        self._inner.set_title(text)

    def set_ceiling_mask(self, mask) -> None:
        """See DifferenceView's identical passthrough for why this has to
        exist even though it's always called with None here."""
        self._inner.set_ceiling_mask(mask)

    def capture_background(self) -> None:
        self._inner.capture_background()

    # M2.6: probe/isotherms delegate straight through -- see DifferenceView's
    # identical passthroughs for the reasoning.
    def set_isotherm_levels(self, levels: list) -> None:
        self._inner.set_isotherm_levels(levels)

    def set_isotherms_enabled(self, enabled: bool) -> None:
        self._inner.set_isotherms_enabled(enabled)

    @property
    def isotherms_enabled(self) -> bool:
        return self._inner.isotherms_enabled

    # Velocity overlay (item 6) never actually applies to this cell type
    # (main_window.py's _apply_contour_overlay_state only turns it on for
    # a "slice" cell showing TEMPERATURE), but every cell_type gets these
    # calls unconditionally the same way it already does for the isotherm
    # setters above -- delegate straight through rather than crashing on
    # a missing attribute (PyQt5 aborts the process on an unhandled
    # exception raised inside a Qt signal handler).
    def set_velocity_overlay_levels(self, levels: list) -> None:
        self._inner.set_velocity_overlay_levels(levels)

    def set_velocity_overlay_enabled(self, enabled: bool) -> None:
        self._inner.set_velocity_overlay_enabled(enabled)

    @property
    def velocity_overlay_enabled(self) -> bool:
        return self._inner.velocity_overlay_enabled

    # Real soot-density smoke overlay: same "never actually applies to
    # this cell type, but delegate straight through rather than crash on
    # a missing attribute" reasoning as the velocity overlay above -- see
    # main_window.py's _apply_smoke_overlay_state, which (like the
    # velocity overlay) only ever turns this on for a "slice" cell showing
    # TEMPERATURE.
    def set_soot_overlay_enabled(self, enabled: bool) -> None:
        self._inner.set_soot_overlay_enabled(enabled)

    @property
    def soot_overlay_enabled(self) -> bool:
        return self._inner.soot_overlay_enabled

    def enable_probe(self, callback) -> None:
        self._inner.enable_probe(callback)

    def disable_probe(self) -> None:
        self._inner.disable_probe()

    def value_at(self, x: float, z: float):
        return self._inner.value_at(x, z)

    @staticmethod
    def compute_composite(arrays: list, index: int, stat: str) -> np.ndarray:
        """arrays: one (n_times, n_z, n_x) array per selected scenario
        (from ScenarioStore.get(), already loaded/mmap'd -- this is a pure
        in-memory reduction, no I/O, microseconds-cheap per the spec).
        Stacks frame `index` from each array along a new leading axis and
        reduces along it."""
        if stat not in EnsembleView.STATS:
            raise ValueError(f"unknown ensemble stat {stat!r}, expected one of {EnsembleView.STATS}")
        stacked = np.stack([a[index] for a in arrays], axis=0)
        return getattr(np, stat)(stacked, axis=0)

    @staticmethod
    def cmap_for(stat: str, quantity_cmap: str) -> str:
        return "viridis" if stat == "std" else quantity_cmap

    @staticmethod
    def label_for(stat: str, quantity_label: str, unit: str) -> str:
        if stat == "std":
            return f"σ({quantity_label}) ({unit})"
        prefix = {"mean": "Mean", "min": "Min", "max": "Max"}[stat]
        return f"{prefix} {quantity_label} ({unit})"

    def std_vmax(self, arrays: list, cache_key, n_samples: int = 20) -> float:
        """Data-derived vmax for the std statistic's colorbar (std has no
        natural quantity-specific floor to borrow, unlike mean/min/max).
        Sampled over up to n_samples frames and cached per cache_key, same
        pattern as DifferenceView.symmetric_clim."""
        if cache_key in self._std_vmax_cache:
            return self._std_vmax_cache[cache_key]
        n = min(a.shape[0] for a in arrays)
        sample_indices = np.linspace(0, n - 1, min(n, n_samples), dtype=int)
        stds = [self.compute_composite(arrays, i, "std") for i in sample_indices]
        vmax = float(np.max(stds)) if stds else 0.0
        self._std_vmax_cache[cache_key] = vmax
        return vmax


class EnsemblePickerDialog(QtWidgets.QDialog):
    """Modal checklist of manifest entries for building an EnsembleView
    selection (M2.3.3), with quick factor-filter buttons ("2 candles",
    "Wide door", ...) that bulk-check every matching scenario instead of
    requiring individual clicks through all 24 -- the "checklist ... with
    factor filters ('all vod=open')" the spec asks for.

    `manifest_entries` is duck-typed: anything with .case_index/.folder/
    .candles/.door/.vod/.voc (manifest.ScenarioEntry satisfies this) --
    views.py doesn't import manifest.py to avoid a view-layer -> data-layer
    dependency, same boundary SliceView/DifferenceView/EnsembleView keep.
    """

    FACTOR_LABELS = {
        'candles': {0: '1 candle', 1: '2 candles'},
        'door': {0: 'Narrow door', 1: 'Wide door'},
        'vod': {0: 'Vent 1 open', 1: 'Vent 1 closed', 2: 'Vent 1 HVAC'},
        'voc': {0: 'Vent 2 open', 1: 'Vent 2 closed'},
    }
    FACTORS = ('candles', 'door', 'vod', 'voc')
    # Same pattern manifest.py's own scan_scenarios() matches -- a generic
    # (non-factorial) study's folders never match it, and its 4 factor
    # fields are meaningless placeholders (always zeroed), so decoding them
    # would show the same misleading label for every scenario. Duplicated
    # here rather than imported (see class docstring: this dialog
    # deliberately avoids a view-layer -> data-layer dependency).
    _FACTORIAL_FOLDER_RE = re.compile(r'^c\d+_d\d+_vod\d+_voc\d+$')

    def __init__(self, manifest_entries: list, initial_selection: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select scenarios for ensemble")
        self._entries = list(manifest_entries)
        self._by_case_index = {e.case_index: e for e in self._entries}

        layout = QtWidgets.QVBoxLayout(self)

        if self._entries:
            filter_row = QtWidgets.QHBoxLayout()
            filter_row.addWidget(QtWidgets.QLabel("Quick filters:"))
            for factor in self.FACTORS:
                values = sorted({getattr(e, factor) for e in self._entries})
                for v in values:
                    label = self.FACTOR_LABELS.get(factor, {}).get(v, f"{factor}={v}")
                    btn = QtWidgets.QPushButton(label)
                    btn.setToolTip(f"Check every scenario with {factor}={v}")
                    btn.clicked.connect(lambda _checked, f=factor, val=v: self._apply_filter(f, val))
                    filter_row.addWidget(btn)
            filter_row.addStretch(1)
            layout.addLayout(filter_row)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAccessibleName("Ensemble scenario checklist")
        initial = set(initial_selection)
        for entry in self._entries:
            item = QtWidgets.QListWidgetItem(self._label_for(entry))
            item.setToolTip(entry.folder)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if entry.case_index in initial else QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, entry.case_index)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        select_row = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select all")
        select_all_btn.clicked.connect(lambda: self._set_all(QtCore.Qt.Checked))
        select_none_btn = QtWidgets.QPushButton("Select none")
        select_none_btn.clicked.connect(lambda: self._set_all(QtCore.Qt.Unchecked))
        select_row.addWidget(select_all_btn)
        select_row.addWidget(select_none_btn)
        select_row.addStretch(1)
        layout.addLayout(select_row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _label_for(self, entry) -> str:
        if not self._FACTORIAL_FOLDER_RE.match(entry.folder):
            return entry.folder
        return " · ".join(self.FACTOR_LABELS[f].get(getattr(entry, f), f"{f}={getattr(entry, f)}")
                          for f in self.FACTORS)

    def _apply_filter(self, factor: str, value):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            entry = self._by_case_index[item.data(QtCore.Qt.UserRole)]
            if getattr(entry, factor) == value:
                item.setCheckState(QtCore.Qt.Checked)

    def _set_all(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def selected_case_indices(self) -> list:
        return [
            self.list_widget.item(i).data(QtCore.Qt.UserRole)
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == QtCore.Qt.Checked
        ]


class GridCell(QtWidgets.QWidget):
    """One cell of a ViewGrid: a type-dependent header (scenario/quantity
    pickers) above a PlotView. Clicking the cell makes it the grid's
    active cell; right-clicking opens a context menu to change its type
    (M2.3.3): "Slice" (one scenario), "Difference" (two scenarios, A-B),
    or "Ensemble" (a selection of scenarios, mean/std/min/max composite).

    Doesn't know about SimulationController/ScenarioStore semantics beyond
    "a list of (label, case_index) options", "a list of (label, SliceKey)
    options", and (for the ensemble picker only) manifest entries with
    .candles/.door/.vod/.voc -- MainWindow supplies all of that and reacts
    to this cell's signals, keeping data-fetching out of the view layer
    (same split as SliceView: this widget only knows how to display a
    frame it's handed, not how to load or compute one).
    """

    CELL_TYPES = ("slice", "difference", "ensemble")
    TYPE_LABELS = {"slice": "Slice", "difference": "Difference (A − B)", "ensemble": "Ensemble (statistic)"}

    activated = QtCore.pyqtSignal(object)                       # self
    scenario_selected = QtCore.pyqtSignal(object, int)          # self, case_index (slice type)
    quantity_selected = QtCore.pyqtSignal(object, object)        # self, SliceKey (any type)
    type_changed = QtCore.pyqtSignal(object, str)                # self, new cell_type
    difference_scenarios_changed = QtCore.pyqtSignal(object, int, int)  # self, case_a, case_b
    ensemble_changed = QtCore.pyqtSignal(object, list, str)       # self, case_indices, stat

    def __init__(self, scenario_options: list, quantity_options: list,
                 manifest_entries: list = None, parent=None):
        """scenario_options: [(label, case_index), ...]; quantity_options:
        [(label, SliceKey), ...]. Both may be empty/single-item (demo mode
        has no manifest) -- combos disable themselves in that case, same
        convention as MainWindow's own quantity combo (M2.1).
        manifest_entries: full manifest.ScenarioEntry-shaped list, needed
        only for the ensemble picker's factor filters -- [] in demo mode."""
        super().__init__(parent)
        self._scenario_options = list(scenario_options)
        self._quantity_options = list(quantity_options)
        self._manifest_entries = list(manifest_entries) if manifest_entries else []

        self.cell_type = "slice"
        self.view = SliceView(self)
        self._install_activation_filter()
        self.case_index = self._scenario_options[0][1] if self._scenario_options else 0
        self.case_index_a = self.case_index
        self.case_index_b = (self._scenario_options[1][1] if len(self._scenario_options) > 1
                              else self.case_index)
        self.ensemble_case_indices: list = []
        self.ensemble_stat = "mean"
        self.quantity_key = self._quantity_options[0][1] if self._quantity_options else None
        # Model-evaluation mode (M3.2.5): None means "use the normal,
        # single global ScenarioStore" for every existing cell, unchanged.
        # Set only by MainWindow._open_browser_model_eval() to route a
        # "slice" cell's data (store_override) or a "difference" cell's B
        # operand (store_override_b) through a PredictionSource instead --
        # ground truth always still comes from the real store.
        self.store_override = None
        self.store_override_b = None
        self._is_active = False
        self._accent = "#0B5FA5"

        # Plain QWidget subclasses don't paint QSS background-color/
        # border-radius unless this attribute is set (Qt's documented
        # "Customizing QWidget using Style Sheets" caveat) -- without it,
        # this cell falls back to the OS's native/light widget background
        # regardless of the app's theme (visible as a stray pale rectangle
        # under the dark theme).
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self._outer_layout = QtWidgets.QVBoxLayout(self)
        self._outer_layout.setContentsMargins(2, 2, 2, 2)
        self._outer_layout.setSpacing(2)

        self._header_layout = QtWidgets.QHBoxLayout()
        self._header_layout.setSpacing(4)
        self._outer_layout.addLayout(self._header_layout)
        # Stretch factors (live-cell re-proportioning pass): both the
        # heatmap and the strip below it have an Expanding vertical size
        # policy, so QVBoxLayout actually honors these ratios -- 3:1 gives
        # the strip a real, legible vertical share (title + ticks + x-axis
        # label + caption all fit) without shrinking the heatmap to a
        # sliver, and holds at any window size since it's a ratio, not a
        # pixel count. Only takes effect while the strip is visible; a
        # hidden widget (TEMPERATURE, which never gets a strip) claims no
        # layout space regardless of its stretch factor, so the heatmap
        # alone still fills the whole cell exactly as before.
        self._outer_layout.addWidget(self.view.widget(), 3)
        # Time-series strip (colormap expressiveness follow-up): hidden
        # unless MainWindow's set_timeseries() gives it real data -- only
        # DYNAMIC PRESSURE/TEMPERATURE RISE on a "slice" cell use it (never
        # the raw TEMPERATURE heatmap, and never difference/ensemble cells,
        # which have no single scenario's series to show). Below the
        # heatmap, not inside SliceView, since it's cell-level UI, not a
        # PlotView concern -- DifferenceView/EnsembleView never need it.
        self.timeseries_strip = TimeSeriesStrip(self)
        self.timeseries_strip.setVisible(False)
        self._outer_layout.addWidget(self.timeseries_strip, 1)

        self._build_slice_header()
        self._restyle()

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def mousePressEvent(self, event):
        self.activated.emit(self)
        super().mousePressEvent(event)

    def _install_activation_filter(self) -> None:
        """A click almost anywhere in a cell lands on its plot canvas (the
        canvas fills nearly the whole cell; mousePressEvent above only
        fires for the ~2px margin around it), and FigureCanvasQTAgg's own
        mousePressEvent consumes the click without forwarding it to this
        widget -- so "click a cell to make it active" previously only
        worked from that sliver of margin, not from clicking the heatmap
        itself. An event filter on the canvas catches the click before the
        canvas processes it, without changing that processing (the filter
        doesn't consume the event, just observes it)."""
        self.view.widget().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.view.widget() and event.type() == QtCore.QEvent.MouseButtonPress:
            self.activated.emit(self)
        return super().eventFilter(obj, event)

    def set_active(self, is_active: bool):
        self._is_active = is_active
        self._restyle()

    def apply_accent(self, accent_color: str):
        self._accent = accent_color
        self._restyle()

    def _restyle(self):
        border = f"2px solid {self._accent}" if self._is_active else "2px solid transparent"
        self.setStyleSheet(f"GridCell {{ border: {border}; border-radius: {RADIUS['lg']}px; }}")

    # -------------------------------------------------- type switching (M2.3.3)
    def _show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        group = QtWidgets.QActionGroup(self)
        for type_key in self.CELL_TYPES:
            action = menu.addAction(self.TYPE_LABELS[type_key])
            action.setCheckable(True)
            action.setChecked(self.cell_type == type_key)
            group.addAction(action)
            action.triggered.connect(lambda _checked, t=type_key: self._set_cell_type(t))
        menu.exec_(self.mapToGlobal(pos))

    def _set_cell_type(self, cell_type: str):
        if cell_type == self.cell_type:
            return
        self.cell_type = cell_type
        self._clear_header()
        if cell_type == "slice":
            self._build_slice_header()
        elif cell_type == "difference":
            self._build_difference_header()
        elif cell_type == "ensemble":
            self._build_ensemble_header()
        self._swap_view(cell_type)
        self.type_changed.emit(self, cell_type)

    def set_cell_type(self, cell_type: str):
        """Programmatic counterpart to the context-menu action."""
        self._set_cell_type(cell_type)

    def _swap_view(self, cell_type: str):
        old_widget = self.view.widget()
        self._outer_layout.removeWidget(old_widget)
        old_widget.setParent(None)
        if cell_type == "slice":
            self.view = SliceView(self)
        elif cell_type == "difference":
            self.view = DifferenceView(self)
        elif cell_type == "ensemble":
            self.view = EnsembleView(self)
        self._install_activation_filter()
        # insertWidget(1, ...), not addWidget (which appends to the end of
        # the layout): the timeseries_strip already occupies the slot after
        # this one from construction, and plain addWidget would put the new
        # view widget *after* it -- reordering the strip above the heatmap
        # the first time a cell switches type and back. Same 3:1 stretch
        # ratio as the constructor's own addWidget call.
        self._outer_layout.insertWidget(1, self.view.widget(), 3)

    # Header widget attribute names per type, dropped in _clear_header so a
    # stale reference to a torn-down widget can't linger (and so
    # hasattr(cell, "scenario_combo") reliably reflects "is this cell
    # currently slice-typed", not "has it ever been").
    _HEADER_ATTRS = ("scenario_combo", "scenario_combo_a", "scenario_combo_b",
                      "quantity_combo", "ensemble_select_button", "stat_combo")

    def _clear_header(self):
        while self._header_layout.count():
            item = self._header_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        for attr in self._HEADER_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

    def _make_quantity_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setAccessibleName("Cell quantity")
        combo.setToolTip("Which quantity this cell displays")
        for i, (label, key) in enumerate(self._quantity_options):
            combo.addItem(label)
            # Registry's own plain-language "interpretation" (same text
            # quantities_panel.py shows) as a per-item dropdown tooltip, so
            # a technical label doesn't require visiting that panel separately.
            interpretation = get_quantity(key.quantity).interpretation
            if interpretation:
                combo.setItemData(i, interpretation, QtCore.Qt.ToolTipRole)
        combo.setEnabled(len(self._quantity_options) > 1)
        idx = next((i for i, (_l, k) in enumerate(self._quantity_options) if k == self.quantity_key), 0)
        combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._on_quantity_combo_changed)
        return combo

    def _build_slice_header(self):
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Cell scenario")
        self.scenario_combo.setToolTip("Which scenario this cell displays")
        for label, _case_index in self._scenario_options:
            self.scenario_combo.addItem(label)
        self.scenario_combo.setEnabled(len(self._scenario_options) > 1)
        idx = next((i for i, (_l, c) in enumerate(self._scenario_options) if c == self.case_index), 0)
        self.scenario_combo.setCurrentIndex(idx)
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_combo_changed)

        # Layout-declutter pass: not added to _header_layout -- the sidebar's
        # "Data shown" combo (main_window.py) is the one visible quantity
        # control now, applying to whichever cell is active (click a cell to
        # activate it, then use that combo). self.quantity_combo stays
        # constructed and wired (_on_quantity_combo_changed, set_quantity_
        # silently) since MainWindow reads/drives it directly by attribute,
        # not by it being on screen.
        self.quantity_combo = self._make_quantity_combo()
        self._header_layout.addWidget(self.scenario_combo, 1)

    def _build_difference_header(self):
        self.scenario_combo_a = QtWidgets.QComboBox()
        self.scenario_combo_a.setAccessibleName("Cell scenario A")
        self.scenario_combo_a.setToolTip("First scenario (A) -- the minuend of A minus B")
        self.scenario_combo_b = QtWidgets.QComboBox()
        self.scenario_combo_b.setAccessibleName("Cell scenario B")
        self.scenario_combo_b.setToolTip("Second scenario (B) -- the subtrahend of A minus B")
        for label, _case_index in self._scenario_options:
            self.scenario_combo_a.addItem(label)
            self.scenario_combo_b.addItem(label)
        enabled = len(self._scenario_options) > 1
        self.scenario_combo_a.setEnabled(enabled)
        self.scenario_combo_b.setEnabled(enabled)
        idx_a = next((i for i, (_l, c) in enumerate(self._scenario_options) if c == self.case_index_a), 0)
        idx_b = next((i for i, (_l, c) in enumerate(self._scenario_options) if c == self.case_index_b), 0)
        self.scenario_combo_a.setCurrentIndex(idx_a)
        self.scenario_combo_b.setCurrentIndex(idx_b)
        self.scenario_combo_a.currentIndexChanged.connect(self._on_difference_combo_changed)
        self.scenario_combo_b.currentIndexChanged.connect(self._on_difference_combo_changed)

        # See _build_slice_header's comment -- same combo, same reason it's
        # constructed but not added to _header_layout here.
        self.quantity_combo = self._make_quantity_combo()
        self._header_layout.addWidget(self.scenario_combo_a, 1)
        self._header_layout.addWidget(QtWidgets.QLabel("−"))
        self._header_layout.addWidget(self.scenario_combo_b, 1)
        # "Δ" still flags this as a difference cell (the plot itself uses a
        # diverging colormap, not the quantity's own sequential one) even
        # without its own quantity combo alongside it now.
        self._header_layout.addWidget(QtWidgets.QLabel("Δ"))

    def _build_ensemble_header(self):
        self.ensemble_select_button = QtWidgets.QPushButton(self._ensemble_button_text())
        self.ensemble_select_button.setAccessibleName("Select ensemble scenarios")
        self.ensemble_select_button.setToolTip("Choose which scenarios this cell's statistic is computed over")
        self.ensemble_select_button.clicked.connect(self._open_ensemble_picker)

        self.stat_combo = QtWidgets.QComboBox()
        self.stat_combo.setAccessibleName("Ensemble statistic")
        for stat in EnsembleView.STATS:
            self.stat_combo.addItem(stat.capitalize())
        self.stat_combo.setCurrentIndex(EnsembleView.STATS.index(self.ensemble_stat))
        self.stat_combo.currentIndexChanged.connect(self._on_stat_combo_changed)

        # See _build_slice_header's comment -- same combo, same reason it's
        # constructed but not added to _header_layout here.
        self.quantity_combo = self._make_quantity_combo()
        self._header_layout.addWidget(self.ensemble_select_button, 1)
        self._header_layout.addWidget(self.stat_combo)

    def _ensemble_button_text(self) -> str:
        n = len(self.ensemble_case_indices)
        return f"{n} scenario{'s' if n != 1 else ''} selected…"

    def _open_ensemble_picker(self):
        dialog = EnsemblePickerDialog(self._manifest_entries, self.ensemble_case_indices, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.ensemble_case_indices = dialog.selected_case_indices()
            self.ensemble_select_button.setText(self._ensemble_button_text())
            self.ensemble_changed.emit(self, self.ensemble_case_indices, self.ensemble_stat)

    def _on_stat_combo_changed(self, idx: int):
        if idx < 0 or idx >= len(EnsembleView.STATS):
            return
        self.ensemble_stat = EnsembleView.STATS[idx]
        self.ensemble_changed.emit(self, self.ensemble_case_indices, self.ensemble_stat)

    def _on_difference_combo_changed(self, _idx: int):
        if not self._scenario_options:
            return
        self.case_index_a = self._scenario_options[self.scenario_combo_a.currentIndex()][1]
        self.case_index_b = self._scenario_options[self.scenario_combo_b.currentIndex()][1]
        self.difference_scenarios_changed.emit(self, self.case_index_a, self.case_index_b)

    # -------------------------------------------------- external state sync
    def set_scenario_silently(self, case_index: int):
        """Update case_index/combo without emitting scenario_selected --
        for when this cell's state is being *driven* (e.g. the active cell
        mirroring control-panel changes) rather than user-selected here.
        Only meaningful for "slice"-type cells."""
        self.case_index = case_index
        if self.cell_type != "slice":
            return
        idx = next((i for i, (_l, c) in enumerate(self._scenario_options) if c == case_index), None)
        if idx is not None and idx != self.scenario_combo.currentIndex():
            self.scenario_combo.blockSignals(True)
            self.scenario_combo.setCurrentIndex(idx)
            self.scenario_combo.blockSignals(False)

    def set_quantity_silently(self, key):
        self.quantity_key = key
        idx = next((i for i, (_l, k) in enumerate(self._quantity_options) if k == key), None)
        if idx is not None and idx != self.quantity_combo.currentIndex():
            self.quantity_combo.blockSignals(True)
            self.quantity_combo.setCurrentIndex(idx)
            self.quantity_combo.blockSignals(False)

    def _on_scenario_combo_changed(self, idx: int):
        if idx < 0 or idx >= len(self._scenario_options):
            return
        case_index = self._scenario_options[idx][1]
        self.case_index = case_index
        self.scenario_selected.emit(self, case_index)

    def _on_quantity_combo_changed(self, idx: int):
        if idx < 0 or idx >= len(self._quantity_options):
            return
        key = self._quantity_options[idx][1]
        self.quantity_key = key
        self.quantity_selected.emit(self, key)


class ViewGrid(QtWidgets.QWidget):
    """1x1/1x2/2x2 grid of GridCells (M2.2.2).

    Exactly one visible cell is "active" -- the one MainWindow's control
    panel (candles/door/vod/voc toggles, quantity combo, colormap menu,
    display-scale slider) edits. Other visible cells hold their own
    independent scenario/quantity/clim/colormap, set via their own
    per-cell combos (MainWindow reacts to cell_scenario_selected/
    cell_quantity_selected for those). Growing the grid preserves already-
    built cells' state instead of recreating them; shrinking just hides
    cells rather than destroying them, so switching back doesn't lose
    per-cell selections.
    """

    LAYOUTS = {
        "1x1": (1, 1),
        "1x2": (1, 2),
        "2x1": (2, 1),
        "1x3": (1, 3),
        "2x2": (2, 2),
        "3x3": (3, 3),
    }

    cell_created = QtCore.pyqtSignal(object)             # a newly-instantiated GridCell, needs init_plot
    active_cell_changed = QtCore.pyqtSignal(object)       # the new active GridCell
    cell_scenario_selected = QtCore.pyqtSignal(object, int)      # non-active-or-active cell, case_index
    cell_quantity_selected = QtCore.pyqtSignal(object, object)   # non-active-or-active cell, SliceKey
    cell_type_changed = QtCore.pyqtSignal(object, str)            # cell, new cell_type (M2.3.3)
    cell_difference_scenarios_changed = QtCore.pyqtSignal(object, int, int)  # cell, case_a, case_b
    cell_ensemble_changed = QtCore.pyqtSignal(object, list, str)   # cell, case_indices, stat

    def __init__(self, scenario_options: list, quantity_options: list,
                 manifest_entries: list = None, parent=None):
        super().__init__(parent)
        self._scenario_options = scenario_options
        self._quantity_options = quantity_options
        self._manifest_entries = manifest_entries or []
        self._layout_name = "1x1"
        self._cells: list = []
        self._active_index = 0

        self._grid_layout = QtWidgets.QGridLayout(self)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)

        self._grow_to(1)
        self._place_cells()

    def _grow_to(self, n_cells: int):
        while len(self._cells) < n_cells:
            cell = GridCell(self._scenario_options, self._quantity_options,
                             self._manifest_entries, self)
            cell.activated.connect(self._on_cell_activated)
            # All these signals already carry the emitting cell as their
            # first arg, so this is a straight re-emit -- Qt supports
            # connecting a signal directly to another signal with a
            # matching shape.
            cell.scenario_selected.connect(self.cell_scenario_selected)
            cell.quantity_selected.connect(self.cell_quantity_selected)
            cell.type_changed.connect(self.cell_type_changed)
            cell.difference_scenarios_changed.connect(self.cell_difference_scenarios_changed)
            cell.ensemble_changed.connect(self.cell_ensemble_changed)
            self._cells.append(cell)
            self.cell_created.emit(cell)

    def _place_cells(self):
        rows, cols = self.LAYOUTS[self._layout_name]
        n_visible = rows * cols
        while self._grid_layout.count():
            self._grid_layout.takeAt(0)
        for i, cell in enumerate(self._cells):
            if i < n_visible:
                r, c = divmod(i, cols)
                self._grid_layout.addWidget(cell, r, c)
                cell.setVisible(True)
            else:
                cell.setVisible(False)
        if self._active_index >= n_visible:
            self._active_index = 0
        self._refresh_active_highlight()

    def set_layout(self, layout_name: str):
        if layout_name not in self.LAYOUTS:
            raise ValueError(f"unknown grid layout {layout_name!r}")
        self._layout_name = layout_name
        rows, cols = self.LAYOUTS[layout_name]
        self._grow_to(rows * cols)
        self._place_cells()

    @property
    def layout_name(self) -> str:
        return self._layout_name

    def visible_cells(self) -> list:
        rows, cols = self.LAYOUTS[self._layout_name]
        return self._cells[:rows * cols]

    def active_cell(self) -> GridCell:
        return self._cells[self._active_index]

    def active_view(self) -> SliceView:
        return self.active_cell().view

    def _on_cell_activated(self, cell):
        if cell not in self.visible_cells():
            return
        idx = self._cells.index(cell)
        if idx == self._active_index:
            return
        self._active_index = idx
        self._refresh_active_highlight()
        self.active_cell_changed.emit(cell)

    def _refresh_active_highlight(self):
        for i, cell in enumerate(self._cells):
            cell.set_active(i == self._active_index)

    def apply_accent(self, accent_color: str):
        for cell in self._cells:
            cell.apply_accent(accent_color)
