"""Measurement Tools panel (V4-M7; reduced in the UX consolidation pass),
an Analysis-page tab.

On-field measurement over its own locator canvas (the Live viewer's
cinematic pipeline is left untouched): pick Rectangle or Probe and measure
directly on the field. The rectangle reports physical area plus
min/mean/max of the current quantity; the probe reads the
bilinearly-interpolated value at a point (heatmap probing). Measurements
can be taken at the current time or averaged over a V4-M5 interval, snap
to grid cells, overlay on the canvas, and are labelled/deleted in the
list. They save with the Named Session (V4-M6) and print in its report.

Distance and path (straight-line / polyline length) were removed from the
tool set as generic geometry with no scientific quantity attached -- a
session saved before this change with such entries still loads, still
renders on the locator canvas, and still prints in its report (see
measure.py); only the tool to create new ones is gone.

Static/lazy; reuses the store, the extent/coordinate convention, the
measure.py engine, and the registry.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from slice_key import SliceKey
from analysis_panel_base import populate_scenario_combo
import measure as mz


class MeasurementPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, quantity_options: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._quantity_options = [(label, key) for label, key in quantity_options
                                  if get_quantity(key.quantity).kind == "slice2d"]
        self._fps = max(1, fps)
        self._loaded = False
        self._data = None
        self._extent = None
        self._measurements = []     # list[mz.Measurement]
        self._press = None          # rect drag corner
        self._loc_ax = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Measurement tools")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Measurement scenario")
        header.addWidget(self.scenario_combo)
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Measurement quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)
        layout.addLayout(header)

        tools = QtWidgets.QHBoxLayout()
        self.tool_combo = QtWidgets.QComboBox()
        self.tool_combo.setAccessibleName("Measurement tool")
        self.tool_combo.addItems(["Rectangle", "Probe"])
        tools.addWidget(self.tool_combo)
        self.snap_check = QtWidgets.QCheckBox("Snap to grid")
        tools.addWidget(self.snap_check)
        self.overlay_check = QtWidgets.QCheckBox("Show overlays")
        self.overlay_check.setChecked(True)
        self.overlay_check.stateChanged.connect(lambda _s: self._render())
        tools.addWidget(self.overlay_check)
        tools.addStretch(1)
        layout.addLayout(tools)

        interval_row = QtWidgets.QHBoxLayout()
        self.interval_check = QtWidgets.QCheckBox("Average over interval")
        interval_row.addWidget(self.interval_check)
        interval_row.addWidget(QtWidgets.QLabel("from"))
        self.t0_spin = QtWidgets.QDoubleSpinBox(); self.t0_spin.setSuffix(" s")
        self.t1_spin = QtWidgets.QDoubleSpinBox(); self.t1_spin.setSuffix(" s")
        interval_row.addWidget(self.t0_spin)
        interval_row.addWidget(QtWidgets.QLabel("to"))
        interval_row.addWidget(self.t1_spin)
        interval_row.addStretch(1)
        layout.addLayout(interval_row)

        self.caption = QtWidgets.QLabel(
            "Rectangle: drag a box for area + min/mean/max of the current quantity. "
            "Probe: click a point for its value. Readout below; measurements save with the session.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Measurement canvas")
        layout.addWidget(self.canvas, 1)

        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("role", "value")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Measurement frame")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        list_row = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Measurements list")
        self.list.setMaximumHeight(110)
        list_row.addWidget(self.list, 1)
        btns = QtWidgets.QVBoxLayout()
        for text, slot, name in (("Rename", self._rename, "measure-rename"),
                                 ("Delete", self._delete, "measure-delete"),
                                 ("Clear", self._clear, "measure-clear")):
            b = QtWidgets.QPushButton(text)
            b.setAccessibleName(name)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        list_row.addLayout(btns)
        layout.addLayout(list_row)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.quantity_combo.currentIndexChanged.connect(self._reload)
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        self.frame_slider.valueChanged.connect(self._on_frame)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    @property
    def _key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    @property
    def _tool(self):
        return mz.KINDS[self.tool_combo.currentIndex()]

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest or not self._quantity_options:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._reload()

    # -------------------------------------------------- session hooks (M6/M7)
    def get_measurements(self) -> list:
        return [m.to_dict() for m in self._measurements]

    def set_measurements(self, data: list) -> None:
        self._measurements = [mz.Measurement.from_dict(d) for d in (data or [])
                              if isinstance(d, dict)]
        self._refresh_list()
        if self._loaded:
            self._render()

    # ------------------------------------------------------------- data load
    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        key = self._key
        if case_index is None or key is None:
            return
        self._data = np.asarray(self._store.get(case_index, key))
        self._extent = self._store.get_extent(case_index, key)
        n = self._data.shape[0]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n - 1)
        if self.frame_slider.value() >= n:
            self.frame_slider.setValue(int(n * 0.6))
        self.frame_slider.blockSignals(False)
        t_end = (n - 1) / self._fps
        for spin in (self.t0_spin, self.t1_spin):
            spin.setRange(0.0, t_end)
        self.t1_spin.setValue(t_end)
        self._render()

    def _on_frame(self, _v) -> None:
        self.frame_label.setText(f"t = {self.frame_slider.value() / self._fps:.1f} s")
        self._render()

    def _on_tool_changed(self, _idx) -> None:
        self._press = None
        self._render()

    # ----------------------------------------------------------- interaction
    def _snap(self, x: float, z: float):
        if not self.snap_check.isChecked() or self._extent is None:
            return x, z
        from timeseries import phys_to_index
        x0, x1, z0, z1 = self._extent
        n_z, n_x = self._data.shape[1:]
        r, c = phys_to_index(self._extent, (n_z, n_x), x, z)
        sx = x0 + c / max(n_x - 1, 1) * (x1 - x0)
        sz = z1 - r / max(n_z - 1, 1) * (z1 - z0)
        return sx, sz

    def _on_press(self, event) -> None:
        if self._data is None or event.inaxes is not self._loc_ax or event.xdata is None:
            return
        if self._tool == "rect":
            self._press = self._snap(event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        if self._data is None or event.inaxes is not self._loc_ax or event.xdata is None:
            self._press = None
            return
        x, z = self._snap(event.xdata, event.ydata)
        tool = self._tool
        if tool == "rect":
            if self._press is not None and (abs(x - self._press[0]) > 1e-6
                                            and abs(z - self._press[1]) > 1e-6):
                self._add(mz.Measurement("rect", [self._press, (x, z)]))
            self._press = None
        elif tool == "probe":
            self._add(mz.Measurement("probe", [(x, z)]))

    # --------------------------------------------------------------- compute
    def _interval_indices(self):
        if not self.interval_check.isChecked():
            return None, None
        i0 = int(round(self.t0_spin.value() * self._fps))
        i1 = int(round(self.t1_spin.value() * self._fps))
        n = self._data.shape[0]
        i0 = min(max(i0, 0), n - 1)
        i1 = min(max(i1, i0), n - 1)
        return i0, i1

    def _evaluate(self, m: mz.Measurement) -> str:
        unit = get_quantity(self._key.quantity).unit
        fi = self.frame_slider.value()
        i0, i1 = self._interval_indices()
        when = (f"averaged {self.t0_spin.value():.1f}–{self.t1_spin.value():.1f} s"
                if i0 is not None else f"at t = {fi / self._fps:.1f} s")
        if m.kind == "rect":
            (x0, z0), (x1, z1) = m.points
            st = mz.rect_stats(self._data, self._extent, x0, x1, z0, z1,
                               frame_index=fi, i0=i0, i1=i1)
            return (f"area {st['area']:.4f} m² · min {st['min']:.0f} / mean {st['mean']:.0f} / "
                    f"max {st['max']:.0f} {unit} ({when})")
        # probe
        x, z = m.points[0]
        if i0 is not None:
            val = float(np.mean([mz.probe_value(self._data[k], self._extent, x, z)
                                 for k in range(i0, i1 + 1)]))
        else:
            val = mz.probe_value(self._data[fi], self._extent, x, z)
        return f"{val:.1f} {unit} at ({x:.2f}, {z:.2f}) m ({when})"

    def _add(self, m: mz.Measurement) -> None:
        m.interval = self.interval_check.isChecked()
        m.readout = self._evaluate(m)
        if not m.label:
            m.label = f"{m.kind} {len(self._measurements) + 1}"
        self._measurements.append(m)
        self._refresh_list()
        self.list.setCurrentRow(len(self._measurements) - 1)
        self.readout.setText(f"{m.label}: {m.readout}")
        self._render()

    def _refresh_list(self) -> None:
        self.list.clear()
        for m in self._measurements:
            self.list.addItem(f"{m.label} — {m.readout}")

    def _rename(self) -> None:
        i = self.list.currentRow()
        if not (0 <= i < len(self._measurements)):
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename measurement", "Label:", text=self._measurements[i].label)
        if ok and name.strip():
            self._measurements[i].label = name.strip()
            self._refresh_list()
            self.list.setCurrentRow(i)

    def _delete(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self._measurements):
            del self._measurements[i]
            self._refresh_list()
            self._render()

    def _clear(self) -> None:
        if self._measurements and QtWidgets.QMessageBox.question(
                self, "Clear", "Remove all measurements?") == QtWidgets.QMessageBox.Yes:
            self._measurements = []
            self._refresh_list()
            self._render()

    # --------------------------------------------------------------- render
    def _render(self) -> None:
        if self._data is None:
            return
        fi = min(self.frame_slider.value(), self._data.shape[0] - 1)
        q = get_quantity(self._key.quantity)
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._loc_ax = ax
        ax.imshow(self._data[fi], cmap=q.cmap, vmin=q.vmin, vmax=q.slider_default,
                  aspect="auto", extent=self._extent if self._extent else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {fi / self._fps:.1f} s", fontsize=8)
        if self.overlay_check.isChecked():
            for m in self._measurements:
                self._draw_measurement(ax, m, "#00E5FF")
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()

    @staticmethod
    def _draw_measurement(ax, m: mz.Measurement, color: str) -> None:
        # "distance"/"path" can no longer be created via the tool set, but a
        # session saved before this change may still carry one -- still
        # rendered here so it doesn't silently vanish from a reopened session.
        if m.kind in ("distance", "path"):
            xs = [p[0] for p in m.points]; zs = [p[1] for p in m.points]
            ax.plot(xs, zs, "-o", color=color, markersize=3, linewidth=1.2)
        elif m.kind == "rect":
            from matplotlib.patches import Rectangle
            (x0, z0), (x1, z1) = m.points
            xa, xb = sorted((x0, x1)); za, zb = sorted((z0, z1))
            ax.add_patch(Rectangle((xa, za), xb - xa, zb - za, fill=False,
                                   edgecolor=color, linewidth=1.6))
        elif m.kind == "probe":
            ax.plot(m.points[0][0], m.points[0][1], "x", color=color, markersize=8)
