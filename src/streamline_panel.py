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
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
import velocity as vel

# Visual clarity pass: matplotlib's own streamplot default (1.0) reads
# thin and sparse against this app's real (coherent, post-ignition) U/W
# data -- compared several density/linewidth combos rendered against real
# t=30/60/90s frames (scenario 0) before picking these; 2.5 density got
# visually tangled in the recirculating eddies, this is the balance point.
DEFAULT_DENSITY = 1.8
DEFAULT_CMAP = "viridis"  # perceptually uniform sequential -- never jet
# Linewidth-scales-with-speed formula (see _render): base + scale * ratio,
# ratio = local speed / this frame's peak speed. Same comparison pass --
# bumped from 0.5-2.5px to 1.0-3.5px so the peak-speed jet reads clearly
# bolder than ambient recirculation without the whole plot feeling heavy.
LINEWIDTH_BASE = 1.0
LINEWIDTH_SCALE = 2.5


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
        self._fps = max(1, fps)
        self._cmap = cmap
        self._loaded = False
        self._fields: dict = {}         # case_index -> vel.VectorField (successfully computed)
        self._gate_reasons: dict = {}   # case_index -> str
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()

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

        linewidth = 1.0
        if self.linewidth_check.isChecked():
            peak = float(speed_frame.max())
            if peak > 1e-9:
                linewidth = LINEWIDTH_BASE + LINEWIDTH_SCALE * (speed_frame / peak)

        strm = ax.streamplot(
            x, z, u_frame, w_frame,
            color=speed_frame, cmap=self._cmap,
            density=self.density_spin.value(),
            linewidth=linewidth,
        )
        self.canvas.fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04, label="Speed (m/s)")

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
