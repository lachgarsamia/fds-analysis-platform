"""Space-Time Cube panel (V5-M4/P4; V6-M5 multi-plane), an Analysis-page tab.

The 2D slice stack is an (x, z, t) volume. Rather than a heavy 3D render, this
shows the two cheap space-time cross-sections through a chosen point: the
x-vs-time map at the point's height (how the field propagates horizontally) and
the z-vs-time map at the point's column (vertical development). Both are just
the stored data reshaped -- no interpolation.

V6-M5: a plane selector (axis + offset) chooses which SliceKey the cube reads,
routed through QuantityProvider so an unavailable plane (this dataset has no
X/Z-normal slices, only Y at offset 0 and 15) raises GatedQuantityError and
shows plainly instead of crashing -- never a fabricated reshape. The x/z-time
axis labelling is verified only for the Y-normal plane (row=z, col=x, the
convention every other view uses); a future X/Z-normal dataset would still
read/reshape correctly, but its two in-plane axes are generically labelled
("axis a"/"axis b") since FDS's row/col convention for those directions has
never been observed against real data -- values are always exactly what the
store returns, only that label is a best-effort placeholder pending real data.

V6-M6: a quantity toggle (Temperature / Full FED) alongside the plane
selector. Full FED (tenability.full_fed) reshapes exactly like Temperature --
same (n_t, n_z, n_x) shape -- so the existing locator/x-time/z-time machinery
is unchanged; only the colour scale/labels differ. Requires 'CARBON MONOXIDE
VOLUME FRACTION', gated until the M-SIM re-run: unavailable today, so this
mode is exercised only via the gated-status path (same mechanism the plane
selector already uses), never fabricated.

SelectionBus (M1): scenario_combo is bound by main_window; clicking the locator
publishes the point, and a point selected elsewhere moves the cross-sections.
Reuses the provider (not the raw store) so plane gating is honest, and the
extent/coordinate convention.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity
import hazard_spaces as hz
from slice_key import SliceKey, AXIS_TO_DIRECTION, DIRECTION_TO_AXIS
from timeseries import phys_to_index
from analysis_panel_base import populate_scenario_combo
import tenability as tn

_PLANE_AXES = ("y", "x", "z")   # y first: the app's default/verified plane
_QUANTITY_MODES = ("temperature", "full_fed")
_QUANTITY_LABELS = {"temperature": "Temperature", "full_fed": "Full FED"}


class SpaceTimePanel(QtWidgets.QWidget):
    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}
        self._data = None
        self._extent = None
        self._row = None
        self._col = None
        self._bus = None
        self._loc_ax = None
        self._gate_reason = None
        self._mode = "temperature"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Space-time cube")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Space-time scenario")
        header.addWidget(self.scenario_combo)
        # V6-M5: multi-plane cross-sections. Y is verified (offset 0 or 15,
        # both real on this dataset); X/Z are offered because the engine
        # supports any plane, but cleanly show "gated" here since no X/Z
        # slice exists yet (docs/msim-preparation.md).
        self.plane_combo = QtWidgets.QComboBox()
        self.plane_combo.setAccessibleName("Space-time plane axis")
        self.plane_combo.setToolTip("Which axis the plane is normal to (Y is verified; "
                                    "X/Z are gated on this dataset)")
        for axis in _PLANE_AXES:
            self.plane_combo.addItem(axis.upper(), AXIS_TO_DIRECTION[axis])
        header.addWidget(self.plane_combo)
        self.offset_spin = QtWidgets.QSpinBox()
        self.offset_spin.setAccessibleName("Space-time plane offset")
        self.offset_spin.setToolTip("The plane's mesh-cell offset along its normal axis")
        self.offset_spin.setRange(0, 999)
        header.addWidget(self.offset_spin)
        # V6-M6: Temperature (always real) vs. Full FED (gated until CO
        # exists -- see module docstring).
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Space-time quantity")
        self.quantity_combo.setToolTip("Temperature is always real; Full FED needs CO "
                                       "data, gated on this dataset")
        for mode in _QUANTITY_MODES:
            self.quantity_combo.addItem(_QUANTITY_LABELS[mode], mode)
        header.addWidget(self.quantity_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Click a point on the map. The two panels are its x–time and z–time "
            "cross-sections — the slice stack reshaped along time, no interpolation.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        # Analysis roadmap B6: previously silent about which hazard basis
        # the quantity choice reflects -- the quantity_combo's own tooltip
        # above already says Full FED is gated, but not that the Temperature
        # fallback (what's actually shown, always, on this dataset) is
        # itself a partial screen. Updated per mode in _reload(), same
        # shared hazard_spaces.basis_caption() text every other hazard-
        # rendering panel shows.
        self.hazard_caption = QtWidgets.QLabel(hz.basis_caption())
        self.hazard_caption.setWordWrap(True)
        self.hazard_caption.setProperty("role", "caption")
        layout.addWidget(self.hazard_caption)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        body = QtWidgets.QSplitter()
        self.loc_canvas = MplCanvas(self)
        self.loc_canvas.setAccessibleName("Space-time locator")
        body.addWidget(self.loc_canvas)
        self.xt_canvas = MplCanvas(self)
        self.xt_canvas.setAccessibleName("x-time map")
        body.addWidget(self.xt_canvas)
        self.zt_canvas = MplCanvas(self)
        self.zt_canvas.setAccessibleName("z-time map")
        body.addWidget(self.zt_canvas)
        body.setStretchFactor(0, 2); body.setStretchFactor(1, 3); body.setStretchFactor(2, 3)
        layout.addWidget(body, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.plane_combo.currentIndexChanged.connect(self._reload)
        self.offset_spin.valueChanged.connect(lambda _v: self._reload())
        self.quantity_combo.currentIndexChanged.connect(self._reload)
        self.loc_canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.point is None or self._data is None or self._extent is None:
            return
        self._row, self._col = phys_to_index(self._extent, self._data.shape[1:],
                                             sel.point[0], sel.point[1])
        self._render()

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._reload()

    @property
    def _direction(self) -> int:
        return self.plane_combo.currentData()

    @property
    def _offset(self) -> int:
        return self.offset_spin.value()

    @property
    def _quantity_mode(self) -> str:
        return self.quantity_combo.currentData() or "temperature"

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        mode = self._quantity_mode
        self.hazard_caption.setText(hz.basis_caption(co_based=(mode == "full_fed")))
        ck = (case_index, self._direction, self._offset, mode)
        if ck not in self._cache:
            key = SliceKey("TEMPERATURE", self._direction, self._offset)
            try:
                temp = np.asarray(self._provider.get(case_index, key))
                extent = self._provider.get_extent(case_index, key)
                if mode == "full_fed":
                    # V6-M6: CO is registry-gated (a clean, immediate
                    # GatedQuantityError -- the registry's own gate fires
                    # before any store access) -- never fabricated.
                    co_key = SliceKey("CARBON MONOXIDE VOLUME FRACTION",
                                     self._direction, self._offset)
                    co = np.asarray(self._provider.get(case_index, co_key))
                    data = tn.full_fed(temp, co, self._fps)
                else:
                    data = temp
                self._gate_reason = None
            except Exception as e:
                # V6-M5: an X/Z-normal (or absent-offset) plane -- shown
                # plainly, never a fabricated reshape. Caught broadly, not
                # just GatedQuantityError: a plane can be *declared* in the
                # .smv inventory (so QuantityProvider's own check lets it
                # through) yet still fail to actually load -- e.g. this
                # dataset lists a second Y-offset that raises a plain
                # numpy/IO error deep in the slice reader when read for
                # real. An uncaught exception here would escape this Qt
                # slot, and PyQt5 aborts the process on that (not a
                # crash in the data itself) -- so this is the boundary
                # that must never let one through. self._data stays
                # whatever it was (or None), so _render() can tell the two
                # cases apart.
                self._cache[ck] = (None, None)
                self._gate_reason = str(e)
                self._data, self._extent = None, None
                self._mode = mode
                self.status.setText(f"Gated: {e}")
                self._render()
                return
            self._cache[ck] = (data, extent)
        self._data, self._extent = self._cache[ck]
        self._mode = mode
        if self._data is None:
            self._gate_reason = self._gate_reason or "This plane is not available for this scenario."
            self.status.setText(f"Gated: {self._gate_reason}")
            self._render()
            return
        self.status.setText("")
        if self._row is None or self._row >= self._data.shape[1] or self._col >= self._data.shape[2]:
            self._row = self._data.shape[1] // 2
            self._col = self._data.shape[2] // 2
        self._render()

    def _on_click(self, event) -> None:
        if self._data is None or event.inaxes is not self._loc_ax or event.xdata is None:
            return
        self._row, self._col = phys_to_index(self._extent, self._data.shape[1:],
                                             event.xdata, event.ydata)
        if self._bus is not None:
            self._bus.update(origin=self, point=(float(event.xdata), float(event.ydata)))
        self._render()

    # --------------------------------------------------------------- render
    def _render_gated(self) -> None:
        """V6-M5: clear all three canvases and show the gate reason plainly
        instead of a stale or fabricated cross-section."""
        for canvas in (self.loc_canvas, self.xt_canvas, self.zt_canvas):
            fig = canvas.fig
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_xticks([]); ax.set_yticks([])
            canvas.draw_idle()

    def _display_params(self):
        """(cmap, vmin, vmax) for the current quantity mode (V6-M6):
        Temperature uses the registry's own scale; Full FED uses a fixed
        0..max(1, data) scale with 1.0 = incapacitation (ISO 13571)."""
        if self._mode == "full_fed":
            vmax = max(1.0, float(np.nanmax(self._data)) if self._data.size else 1.0)
            return "RdYlGn_r", 0.0, vmax
        q = get_quantity("TEMPERATURE")
        return q.cmap, q.vmin, q.slider_default

    def _render(self) -> None:
        if self._data is None:
            self._render_gated()
            return
        cmap, vmin, vmax = self._display_params()
        n_t = self._data.shape[0]
        t_end = (n_t - 1) / self._fps
        ext = self._extent or [0, self._data.shape[2], 0, self._data.shape[1]]
        x0, x1, z0, z1 = ext
        # Axis labels: verified (row=z, col=x) only for the app's usual
        # Y-normal plane -- see the module docstring for why other
        # directions get a generic, clearly-unverified label instead of a
        # possibly-wrong physical one. Values are always exactly what the
        # store returned either way.
        if self._direction == 1:
            row_axis_label, col_axis_label = "z (m, floor→ceiling)", "x (m)"
        else:
            axis = DIRECTION_TO_AXIS.get(self._direction, "?")
            row_axis_label = col_axis_label = f"in-plane axis (m, normal={axis}, unverified order)"

        # --- locator (a representative frame + the chosen point) ---
        lfig = self.loc_canvas.fig
        lfig.clear()
        self._loc_ax = lfig.add_subplot(111)
        frame = self._data[int(n_t * 0.6)]
        self._loc_ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax,
                            aspect="auto", extent=ext)
        self._loc_ax.set_xticks([]); self._loc_ax.set_yticks([])
        px = x0 + self._col / max(frame.shape[1] - 1, 1) * (x1 - x0)
        pz = z1 - self._row / max(frame.shape[0] - 1, 1) * (z1 - z0)
        self._loc_ax.plot(px, pz, "x", color="#00E5FF", markersize=9)
        self._loc_ax.set_title("click a point", fontsize=8)
        lfig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
        self.loc_canvas.draw_idle()

        # --- x-time at the point's height ---
        xt = self._data[:, self._row, :]        # (n_t, n_x)
        self._heatmap(self.xt_canvas, xt, (x0, x1, t_end, 0.0), cmap, vmin, vmax,
                      f"axis-b–time at fixed axis-a={pz:.2f} m" if self._direction != 1
                      else f"x–time at z={pz:.2f} m", col_axis_label)
        # --- z-time at the point's column (row 0 = ceiling) ---
        zt = self._data[:, :, self._col]        # (n_t, n_z)
        self._heatmap(self.zt_canvas, zt, (z1, z0, t_end, 0.0), cmap, vmin, vmax,
                      f"axis-a–time at fixed axis-b={px:.2f} m" if self._direction != 1
                      else f"z–time at x={px:.2f} m", row_axis_label)

    @staticmethod
    def _heatmap(canvas, arr, extent, cmap, vmin, vmax, title, xlabel) -> None:
        fig = canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=extent)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("time (s)", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.92, bottom=0.14, left=0.16, right=0.97)
        canvas.draw_idle()
