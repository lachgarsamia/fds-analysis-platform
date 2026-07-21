"""Linked Multi-Quantity Inspection panel (V4-M3), an Analysis-page tab.

Pick a scenario and a time; see the moment across the physics at once: the
temperature field on the left, and -- sharing one time cursor -- the peak
temperature, the heat-release rate, the smoke-layer height, and the peak
air speed on the right. A live readout prints every value at the cursor,
and key linked moments (the temperature peak, the layer's lowest point,
the HRR peak) are listed as navigable, savable Insights, delivering the
brief's example: click a temperature peak, inspect HRR / layer / velocity
at that instant.

Static/lazy; the aligned series are computed once per scenario and
cached; the field and cursor redraw cheaply per frame. Reuses the store,
layer_height, summary_stats' HRR CSV reader, and the registry.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity, AMBIENT_C
from insight import InsightList, Insight
from slice_key import SliceKey
from layer_height import smoke_layer_height_series
from summary_stats import read_hrr_table
import linked_inspection as li


def _fmt_hrr(v: float) -> str:
    """Present HRR in the unit that shows its real magnitude -- a candle
    fire peaks well under 1 kW, so kW-rounded reads as a misleading 0."""
    return f"{v * 1000:.0f} W" if abs(v) < 1.0 else f"{v:.1f} kW"


class LinkedInspectionPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}       # case_index -> dict(series...)
        self._data = None      # temperature frames for the field
        self._extent = None
        self._series = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Inspect this moment")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Linked inspection scenario")
        header.addWidget(self.scenario_combo)
        self.export_button = QtWidgets.QPushButton("Export figure…")
        self.export_button.setAccessibleName("Export linked figure")
        self.export_button.clicked.connect(self._export_figure)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "One moment across the physics. Move the time cursor (or click a "
            "listed moment); every plot and the readout below update together.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.field_canvas = MplCanvas(self)
        self.field_canvas.setAccessibleName("Linked temperature field")
        body.addWidget(self.field_canvas)
        self.plots_canvas = MplCanvas(self)
        self.plots_canvas.setAccessibleName("Linked quantity plots")
        body.addWidget(self.plots_canvas)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("role", "caption")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Linked frame")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        self.insights = InsightList()
        self.insights.setMaximumHeight(110)
        layout.addWidget(self.insights)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.frame_slider.valueChanged.connect(self._on_frame)
        self.insights.insight_activated.connect(self._on_insight)

    def _export_figure(self) -> None:
        from figure_export import export_figure_interactive
        export_figure_interactive(self, self.plots_canvas.fig, "linked_inspection")

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

    def _entry(self, case_index):
        for e in self._manifest:
            if e.case_index == case_index:
                return e
        return None

    def _try_series(self, case_index, quantity):
        """Peak-over-time of a slice quantity, or None if the scenario
        lacks it (guests may not carry VELOCITY)."""
        try:
            data = np.asarray(self._store.get(case_index, SliceKey(quantity)))
        except Exception:
            return None
        return li.peak_over_time(data)

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        self._data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
        self._extent = self._store.get_extent(case_index, SliceKey("TEMPERATURE"))
        if case_index not in self._cache:
            entry = self._entry(case_index)
            hrr = read_hrr_table(entry.path) if entry is not None else None
            hrr_t = hrr.get("Time") if hrr else None
            hrr_v = hrr.get("HRR") if hrr else None
            self._cache[case_index] = {
                "peak_t": li.peak_over_time(self._data),
                "layer": (smoke_layer_height_series(self._data, self._extent, AMBIENT_C)
                          if self._extent is not None else None),
                "speed": self._try_series(case_index, "VELOCITY"),
                "hrr_t": hrr_t, "hrr_v": hrr_v,
            }
        self._series = self._cache[case_index]
        n = self._data.shape[0]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n - 1)
        if self.frame_slider.value() >= n:
            self.frame_slider.setValue(int(np.argmax(self._series["peak_t"])))
        self.frame_slider.blockSignals(False)
        self._render()
        self._build_insights()

    def _on_frame(self, _v) -> None:
        idx = self.frame_slider.value()
        self.frame_label.setText(f"t = {idx / self._fps:.1f} s")
        self._render()

    def _on_insight(self, insight) -> None:
        fi = insight.frame_index(self._fps)
        if fi is not None:
            self.frame_slider.setValue(min(max(fi, 0), self._data.shape[0] - 1))

    def _readout_values(self, t: float) -> dict:
        s = self._series
        out = {"peak_t": float(li.value_at_time(np.arange(len(s["peak_t"])) / self._fps,
                                                 s["peak_t"], t))}
        out["layer"] = (li.value_at_time(np.arange(len(s["layer"])) / self._fps, s["layer"], t)
                        if s["layer"] is not None else None)
        out["speed"] = (li.value_at_time(np.arange(len(s["speed"])) / self._fps, s["speed"], t)
                        if s["speed"] is not None else None)
        out["hrr"] = (li.value_at_time(s["hrr_t"], s["hrr_v"], t)
                      if s["hrr_t"] is not None else None)
        return out

    def _render(self) -> None:
        if self._data is None or self._series is None:
            return
        idx = min(self.frame_slider.value(), self._data.shape[0] - 1)
        t = idx / self._fps
        temp = get_quantity("TEMPERATURE")

        # --- temperature field at the cursor instant ---
        ffig = self.field_canvas.fig
        ffig.clear()
        ax = ffig.add_subplot(111)
        ax.imshow(self._data[idx], cmap=temp.cmap, vmin=temp.vmin,
                  vmax=temp.slider_default, aspect="auto",
                  extent=self._extent if self._extent else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Temperature at t = {t:.1f} s", fontsize=8)
        ffig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
        self.field_canvas.draw_idle()

        # --- linked series, one shared time cursor ---
        s = self._series
        panels = [("Peak T (°C)", np.arange(len(s["peak_t"])) / self._fps, s["peak_t"], "#E8622C")]
        if s["layer"] is not None:
            panels.append(("Smoke layer (m)", np.arange(len(s["layer"])) / self._fps,
                           s["layer"], "#2563EB"))
        if s["hrr_t"] is not None:
            hrr_v = np.asarray(s["hrr_v"], dtype=float)
            if hrr_v.size and np.nanmax(np.abs(hrr_v)) < 1.0:  # sub-kW (candle): plot in W
                panels.append(("HRR (W)", s["hrr_t"], hrr_v * 1000.0, "#C026D3"))
            else:
                panels.append(("HRR (kW)", s["hrr_t"], hrr_v, "#C026D3"))
        if s["speed"] is not None:
            panels.append(("Peak speed (m/s)", np.arange(len(s["speed"])) / self._fps,
                           s["speed"], "#0891B2"))

        fig = self.plots_canvas.fig
        fig.clear()
        axes = fig.subplots(len(panels), 1, sharex=True)
        if len(panels) == 1:
            axes = [axes]
        for ax_i, (label, xs, ys, color) in zip(axes, panels):
            ax_i.plot(xs, ys, color=color, linewidth=1.1)
            ax_i.axvline(t, color="#00E5FF", linewidth=1.0)  # the shared cursor
            ax_i.set_ylabel(label, fontsize=7, labelpad=6)
            ax_i.tick_params(labelsize=6)
        axes[-1].set_xlabel("time (s)", fontsize=8)
        fig.subplots_adjust(top=0.97, bottom=0.12, left=0.22, right=0.97, hspace=0.25)
        self.plots_canvas.draw_idle()

        # --- readout: every value at this one instant ---
        v = self._readout_values(t)
        parts = [f"t = {t:.1f} s", f"peak T = {v['peak_t']:.0f} °C"]
        if v["hrr"] is not None:
            parts.append(f"HRR = {_fmt_hrr(v['hrr'])}")
        if v["layer"] is not None:
            parts.append(f"smoke layer = {v['layer']:.2f} m")
        if v["speed"] is not None:
            parts.append(f"peak speed = {v['speed']:.2f} m/s")
        self.readout.setText("   |   ".join(parts))

    def _moment_insight(self, statement_head: str, idx: int, basis: str) -> Insight:
        """A linked Insight: its statement carries the other quantities'
        values at this same instant, so saving it captures the moment."""
        t = idx / self._fps
        v = self._readout_values(t)
        extras = [f"peak T {v['peak_t']:.0f} °C"]
        if v["hrr"] is not None:
            extras.append(f"HRR {_fmt_hrr(v['hrr'])}")
        if v["layer"] is not None:
            extras.append(f"layer {v['layer']:.2f} m")
        if v["speed"] is not None:
            extras.append(f"speed {v['speed']:.2f} m/s")
        return Insight(f"{statement_head} ({', '.join(extras)}).", category="query",
                       quantity="TEMPERATURE", time_s=t, basis=basis)

    def _build_insights(self) -> None:
        if self._series is None:
            return
        s = self._series
        out = [self._moment_insight("At the temperature peak", int(np.argmax(s["peak_t"])),
                                    "peak of the peak-temperature series")]
        if s["layer"] is not None:
            out.append(self._moment_insight("When the smoke layer is lowest",
                                            int(np.argmin(s["layer"])),
                                            "min of the smoke-layer-height series"))
        if s["hrr_t"] is not None and s["hrr_v"] is not None and len(s["hrr_v"]):
            hrr_time = float(s["hrr_t"][int(np.argmax(s["hrr_v"]))])
            idx = int(round(hrr_time * self._fps))
            idx = min(max(idx, 0), self._data.shape[0] - 1)
            out.append(self._moment_insight("At the HRR peak", idx, "peak of the HRR CSV"))
        self.insights.set_insights(out)
