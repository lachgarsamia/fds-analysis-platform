"""Velocity panel (V6-M3 True Velocity), an Analysis-page tab.

Same locator-canvas + list/rename/delete/jump-to/export pattern as
device_panel.py (V6-M2) and measurement_panel.py: click the map to place a
vector probe (a seed point), see its local speed/direction reading, rename/
delete/jump-to/export it. A whole-plane quiver and/or streamlines overlay
the locator heatmap for the current scenario.

Gated on U-VELOCITY/W-VELOCITY (docs/msim-preparation.md section 3): the
panel tries to compute a VectorField for a scenario once (lazily, cached in
self._fields), and if that raises GatedQuantityError, shows the gate reason
plainly in the caption/status instead of a broken plot -- a probe can still
be placed (it will "light up" automatically once the M-SIM re-run adds the
components, no re-placement needed), but its results are marked gated,
never fabricated.

The background heatmap always shows a real quantity (VELOCITY speed or
TEMPERATURE, both non-gated) regardless of whether the vector overlay
itself is available, so gating never blanks the canvas.

Reuses QuantityProvider (vector reads via get_vector), velocity.py
(VectorField/VectorProbe engine), the registry, the Insight model
(jump-to navigation), and timeseries.write_series_csv (CSV export).
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import SliceKey
from registry import get_quantity
from quantity_provider import GatedQuantityError
import velocity as vel

_COLOR_BY_LABELS = {"speed": "Speed", "direction": "Direction", "temperature": "Temperature overlay"}
_MODE_LABELS = {"quiver": "Quiver", "streamlines": "Streamlines", "both": "Both"}


class VelocityPanel(QtWidgets.QWidget):
    probes_changed = QtCore.pyqtSignal()        # placed/renamed/deleted -- refresh Live Viewer
    probe_activated = QtCore.pyqtSignal(object)  # an Insight -- jump-to navigation

    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._data = None       # background heatmap for the locator canvas
        self._extent = None
        self._fields: dict = {}         # case_index -> vel.VectorField (successfully computed)
        self._gate_reasons: dict = {}   # case_index -> str
        self._probes: list = []
        self._counter = 0
        self._loc_ax = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("True velocity (streamlines / quiver)")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        header = QtWidgets.QHBoxLayout()
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Velocity scenario")
        self.scenario_combo.setToolTip("Which scenario's vector field to compute and display")
        header.addWidget(self.scenario_combo)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setAccessibleName("Vector mode")
        self.mode_combo.setToolTip("Show a whole-plane quiver, per-probe streamlines, or both")
        for m in vel.MODES:
            self.mode_combo.addItem(_MODE_LABELS[m], m)
        header.addWidget(self.mode_combo)
        self.color_by_combo = QtWidgets.QComboBox()
        self.color_by_combo.setAccessibleName("Color by")
        self.color_by_combo.setToolTip("Color streamlines by speed/direction, or show a "
                                       "temperature background instead")
        for c in vel.COLOR_BY:
            self.color_by_combo.addItem(_COLOR_BY_LABELS[c], c)
        header.addWidget(self.color_by_combo)
        # V6-M7: true 3D (colour the quiver by the through-plane V
        # component). Gated on V-VELOCITY, same as U/W -- checking this
        # with no 3D data available leaves the ordinary 2D quiver exactly
        # as it was (a graceful fallback, not an error).
        self.show_3d_check = QtWidgets.QCheckBox("3D (color by V)")
        self.show_3d_check.setAccessibleName("Show 3D vector (color by V)")
        self.show_3d_check.setToolTip("Color quiver arrows by the through-plane V-VELOCITY "
                                      "component. Gated on this dataset -- falls back to the "
                                      "ordinary 2D quiver when V is unavailable.")
        self.show_3d_check.stateChanged.connect(lambda _s: self._render())
        header.addWidget(self.show_3d_check)
        header.addStretch(1)
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Density"))
        self.density_spin = QtWidgets.QSpinBox()
        self.density_spin.setAccessibleName("Vector density")
        self.density_spin.setToolTip("Approximate number of quiver arrows across the plane")
        self.density_spin.setRange(20, 2000)
        self.density_spin.setValue(300)
        self.density_spin.setSingleStep(20)
        controls.addWidget(self.density_spin)
        controls.addWidget(QtWidgets.QLabel("Streamline length (m)"))
        self.length_spin = QtWidgets.QDoubleSpinBox()
        self.length_spin.setAccessibleName("Streamline length")
        self.length_spin.setToolTip("Maximum streamline length, in metres, from each probe")
        self.length_spin.setRange(0.1, 100.0)
        self.length_spin.setValue(3.0)
        controls.addWidget(self.length_spin)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.caption = QtWidgets.QLabel(
            "Click the map to place a vector probe (a seed point). Requires U-VELOCITY/"
            "W-VELOCITY (the true in-plane vector field); today's output only has VELOCITY "
            "(speed magnitude, no direction) -- see docs/msim-preparation.md. A probe placed "
            "now will start showing real streamlines/quiver automatically once that data "
            "exists, with no need to re-place it.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Velocity placement canvas")
        layout.addWidget(self.canvas, 1)

        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("role", "value")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        list_row = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Vector probes list")
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(lambda _i: self._render())
        list_row.addWidget(self.list, 1)
        btns = QtWidgets.QVBoxLayout()
        _TOOLTIPS = {
            "vector-rename": "Rename the selected probe",
            "vector-jump": "Reveal this probe's result across the app (Live Viewer, Context)",
            "vector-export": "Export this probe's speed/direction time series as CSV",
            "vector-delete": "Delete the selected probe",
        }
        for text, slot, name in (
                ("Rename", self._rename, "vector-rename"),
                ("Jump to", self._jump_to, "vector-jump"),
                ("Export CSV", self._export, "vector-export"),
                ("Delete", self._delete, "vector-delete")):
            b = QtWidgets.QPushButton(text)
            b.setAccessibleName(name)
            b.setToolTip(_TOOLTIPS[name])
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        list_row.addLayout(btns)
        layout.addLayout(list_row)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.mode_combo.currentIndexChanged.connect(lambda _i: self._render())
        self.color_by_combo.currentIndexChanged.connect(lambda _i: self._reload_background_and_render())
        self.density_spin.valueChanged.connect(lambda _v: self._render())
        self.length_spin.valueChanged.connect(lambda _v: self._render())
        self.canvas.mpl_connect("button_press_event", self._on_press)

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._reload()

    @property
    def mode(self) -> str:
        return self.mode_combo.currentData() or "quiver"
    

    @property
    def color_by(self) -> str:
        return self.color_by_combo.currentData() or "speed"

    def _background_quantity(self) -> str:
        return "TEMPERATURE" if self.color_by == "temperature" else "VELOCITY"

    def _reload_background_and_render(self) -> None:
        self._reload(
            
        )

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        key = SliceKey(self._background_quantity())
        self._data = np.asarray(self._provider.get(case_index, key))
        self._extent = self._provider.get_extent(case_index, key)
        self._render()

    # -------------------------------------------------- vector field access
    def _ensure_field(self, case_index: int):
        """Lazily compute (and cache) the VectorField for `case_index`.
        Returns the field, or None if gated -- self._gate_reasons records
        why. Never recomputes an already-successful or already-gated
        scenario."""
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
        # V6-M7: optional 3D enhancement (V-VELOCITY) -- tried separately
        # from the 2D (U, W) compute() above, which has already succeeded
        # and is completely unaffected if this fails. Gated today (same
        # registry gate as U/W); the "3D (color by V)" toggle simply has
        # nothing to show until this succeeds.
        try:
            f.compute_v()
        except Exception:  # noqa: BLE001 - GatedQuantityError today; a provider that
            # doesn't even implement get_vector3d (e.g. an older duck-typed
            # fake/test double) must fall back the same way, not crash --
            # the 3D toggle simply has nothing to show either way.
            pass
        return f

    # -------------------------------------------------- session hooks (V6-M3)
    def get_probes(self) -> list:
        return [p.to_dict() for p in self._probes]

    def set_probes(self, data: list) -> None:
        """Restores probes with their cached `results` verbatim -- no
        recompute, so a reopened session reproduces identical numbers."""
        self._probes = [vel.VectorProbe.from_dict(d) for d in (data or []) if isinstance(d, dict)]
        self._counter = len(self._probes)
        self._refresh_list()
        if self._loaded:
            self._render()

    # ----------------------------------------------------------- interaction
    def _on_press(self, event) -> None:
        if self._data is None or event.inaxes is not self._loc_ax or event.xdata is None:
            return
        self._place(event.xdata, event.ydata)

    def _place(self, x: float, z: float) -> None:
        case_index = self.scenario_combo.currentData()
        self._counter += 1
        probe = vel.VectorProbe(id=f"vp-{id(object())}", name=f"VP-{self._counter:02d}",
                                scenario=case_index, position=(float(x), float(z)))
        field = self._ensure_field(case_index)
        if field is None:
            probe.mark_gated(self._gate_reasons[case_index])
            self.status.setText(f"Gated: {self._gate_reasons[case_index]}")
        else:
            probe.compute(field, self._fps)
            self.status.setText("")
        self._probes.append(probe)
        self._refresh_list()
        self.list.setCurrentRow(len(self._probes) - 1)
        self._render()
        self.probes_changed.emit()

    # --------------------------------------------------------------- actions
    def _current(self):
        i = self.list.currentRow()
        return self._probes[i] if 0 <= i < len(self._probes) else None

    def _rename(self) -> None:
        p = self._current()
        if p is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename probe", "Name:", text=p.name)
        if ok and name.strip():
            p.name = name.strip()
            self._refresh_list()
            self._render()
            self.probes_changed.emit()

    def _delete(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self._probes):
            del self._probes[i]
            self._refresh_list()
            self._render()
            self.probes_changed.emit()

    def _jump_to(self) -> None:
        p = self._current()
        if p is None:
            return
        insight = p.summary_insight()
        if insight is not None:
            self.probe_activated.emit(insight)

    def _export(self) -> None:
        p = self._current()
        if p is None or p.results is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export vector probe data", f"{p.name}.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        vel.export_csv(p, path)

    # --------------------------------------------------------------- render
    def _headline(self, p: "vel.VectorProbe") -> str:
        if p.gated:
            return "gated -- awaiting M-SIM re-run"
        r = p.results or {}
        if not r:
            return "not yet computed"
        return f"peak speed {r['max_speed_m_s']:.1f} m/s"

    def _refresh_list(self) -> None:
        self.list.clear()
        for p in self._probes:
            item = QtWidgets.QListWidgetItem(f"{p.name} — {self._headline(p)}")
            item.setToolTip((p.results or {}).get("reason") or (p.results or {}).get("basis", ""))
            self.list.addItem(item)

    def _render(self) -> None:
        if self._data is None:
            return
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._loc_ax = ax
        frame = self._data[int(self._data.shape[0] * 0.6)]
        q = get_quantity(self._background_quantity())
        ax.imshow(frame, cmap=q.cmap, vmin=q.vmin, vmax=q.slider_default,
                  aspect="auto", extent=self._extent if self._extent else None)
        ax.set_xticks([]); ax.set_yticks([])
        case_index = self.scenario_combo.currentData()
        field = self._fields.get(case_index)
        gate_reason = self._gate_reasons.get(case_index)
        if field is not None:
            ax.set_title(f"{_MODE_LABELS[self.mode]} · density {self.density_spin.value()}", fontsize=8)
            frame_index = int(field.n_frames * 0.6)
            if self.mode in ("quiver", "both"):
                if self.show_3d_check.isChecked() and field.has_3d:
                    # V6-M7: colour by the through-plane V component --
                    # only when actually available (has_3d); otherwise this
                    # branch is simply never taken and the plain 2D quiver
                    # below renders exactly as before (graceful fallback).
                    xs, zs, us, ws, vs = field.quiver_at_3d(
                        frame_index, density=self.density_spin.value())
                    q3d = ax.quiver(xs, zs, us, ws, vs, cmap="coolwarm",
                                    angles="xy", scale_units="xy", width=0.004)
                    self.canvas.fig.colorbar(q3d, ax=ax, fraction=0.046, pad=0.04,
                                             label="V-VELOCITY (m/s, through-plane)")
                else:
                    xs, zs, us, ws = field.quiver_at(frame_index, density=self.density_spin.value())
                    ax.quiver(xs, zs, us, ws, color="#14171F", angles="xy",
                             scale_units="xy", width=0.004)
            if self.mode in ("streamlines", "both"):
                for p in self._probes:
                    if p.scenario != case_index or p.gated:
                        continue
                    pts = field.streamline_at(p.position, frame_index,
                                              max_length=self.length_spin.value())
                    if len(pts) > 1:
                        xs2 = [pt[0] for pt in pts]; zs2 = [pt[1] for pt in pts]
                        ax.plot(xs2, zs2, "-", color="#14171F", linewidth=1.2)
        elif gate_reason:
            ax.set_title("Vectors gated -- see caption", fontsize=8, color="#B00020")
        for p in self._probes:
            if p.scenario != case_index:
                continue
            color = "#B00020" if p.gated else "#3DA5FF"
            ax.plot(p.position[0], p.position[1], "o", color=color, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.8)
            ax.annotate(p.name, p.position, fontsize=7, xytext=(4, 4),
                       textcoords="offset points", color=color)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()
        p = self._current()
        if p is not None:
            tail = (p.results or {}).get("reason") if p.gated else (p.results or {}).get("basis", "")
            self.readout.setText(f"{p.name}: {self._headline(p)}  ·  {tail}")
        else:
            self.readout.setText("")
