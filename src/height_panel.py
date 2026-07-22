"""Height-Aware Analysis Workspace (V4-M1), an Analysis-page tab.

Pick a vertical line on the field and read the fire's vertical behaviour:
the temperature-vs-height profile T(z) at that x (scrubbable in time),
and -- over the whole run -- the smoke-layer height, the plume height,
and the ceiling-jet temperature, with the current time marked. Key
readings are listed as navigable Insights.

Static/lazy; the over-time series are computed once per (scenario,
quantity) and cached; the profile recomputes cheaply per (x, frame).
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity, AMBIENT_C
from insight import InsightList, Insight
from layer_height import smoke_layer_height_series
import height_analysis as ha


class HeightPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, quantity_options: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._quantity_options = [(label, key) for label, key in quantity_options
                                  if get_quantity(key.quantity).kind == "slice2d"]
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}       # (case, quantity) -> dict(series...)
        self._data = None
        self._extent = None
        self._series = None
        self._x_col = None
        self._loc_ax = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Height analysis")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Height scenario")
        header.addWidget(self.scenario_combo)
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Height quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)
        self.export_button = QtWidgets.QPushButton("Export figure…")
        self.export_button.setAccessibleName("Export height figure")
        self.export_button.clicked.connect(self._export_figure)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Click the map to choose a vertical line. The profile shows how the "
            "quantity changes with height there; the curves track the smoke layer, "
            "plume, and ceiling over time.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.loc_canvas = MplCanvas(self)
        self.loc_canvas.setAccessibleName("Height locator map")
        body.addWidget(self.loc_canvas)
        self.plot_canvas = MplCanvas(self)
        self.plot_canvas.setAccessibleName("Height plots")
        body.addWidget(self.plot_canvas)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Height frame")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        self.insights = InsightList()
        self.insights.setMaximumHeight(120)
        layout.addWidget(self.insights)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.quantity_combo.currentIndexChanged.connect(self._reload)
        self.frame_slider.valueChanged.connect(self._on_frame)
        self.insights.insight_activated.connect(self._on_insight)
        self.loc_canvas.mpl_connect("button_press_event", self._on_click)

    def _export_figure(self) -> None:
        from figure_export import export_figure_interactive
        export_figure_interactive(self, self.plot_canvas.fig, "height_profile")

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    @property
    def _key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest or not self._quantity_options:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        key = self._key
        if case_index is None or key is None:
            return
        self._data = np.asarray(self._store.get(case_index, key))
        self._extent = self._store.get_extent(case_index, key)
        cache_key = (case_index, key.quantity)
        if cache_key not in self._cache:
            q = get_quantity(key.quantity)
            thr = (q.hazard_levels or (AMBIENT_C * 3,))[0]
            self._cache[cache_key] = {
                "layer": (smoke_layer_height_series(self._data, self._extent, AMBIENT_C)
                          if self._extent is not None else None),
                "plume": (ha.plume_height_series(self._data, self._extent, thr)
                          if self._extent is not None else None),
                "ceiling": ha.ceiling_jet_series(self._data),
                "threshold": thr, "unit": q.unit, "label": q.label,
            }
        self._series = self._cache[cache_key]
        n = self._data.shape[0]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n - 1)
        if self.frame_slider.value() >= n:
            self.frame_slider.setValue(int(n * 0.6))
        self.frame_slider.blockSignals(False)
        if self._x_col is None:
            self._x_col = self._data.shape[2] // 2
        self._render()
        self._build_insights()

    def _on_frame(self, _v) -> None:
        idx = self.frame_slider.value()
        self.frame_label.setText(f"t = {idx / self._fps:.1f} s")
        self._render()

    def _on_click(self, event) -> None:
        if self._data is None or event.inaxes != self._loc_ax or event.xdata is None:
            return
        self._x_col = ha.column_for_x(self._extent, self._data.shape[2], event.xdata)
        self._render()

    def _on_insight(self, insight) -> None:
        fi = insight.frame_index(self._fps)
        if fi is not None:
            self.frame_slider.setValue(min(max(fi, 0), self._data.shape[0] - 1))

    def _render(self) -> None:
        if self._data is None or self._series is None:
            return
        idx = min(self.frame_slider.value(), self._data.shape[0] - 1)
        frame = self._data[idx]
        display = get_quantity(self._key.quantity)
        unit = self._series["unit"]

        # --- locator map with the chosen vertical line ---
        lfig = self.loc_canvas.fig
        lfig.clear()
        self._loc_ax = lfig.add_subplot(111)
        self._loc_ax.imshow(frame, cmap=display.cmap, vmin=display.vmin,
                            vmax=display.slider_default, aspect="auto",
                            extent=self._extent if self._extent else None)
        self._loc_ax.set_xticks([]); self._loc_ax.set_yticks([])
        if self._extent is not None and self._x_col is not None:
            x0, x1, _z0, _z1 = self._extent
            x = x0 + self._x_col / max(frame.shape[1] - 1, 1) * (x1 - x0)
            self._loc_ax.axvline(x, color="#00E5FF", linewidth=2)
        self._loc_ax.set_title(f"t = {idx / self._fps:.1f} s — click to pick x", fontsize=8)
        lfig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
        self.loc_canvas.draw_idle()

        # --- profile + over-time plots ---
        fig = self.plot_canvas.fig
        fig.clear()
        prof_ax = fig.add_subplot(211)
        zs, vals = ha.vertical_profile(frame, self._extent, self._x_col)
        prof_ax.plot(vals, zs, "-", color="#E8622C")
        for level in display.hazard_levels:
            prof_ax.axvline(level, color="#888", linewidth=0.6, linestyle=":")
        if self._series["layer"] is not None:
            prof_ax.axhline(self._series["layer"][idx], color="#2563EB", linewidth=1.0,
                            linestyle="--", label="smoke layer")
            prof_ax.legend(fontsize=6, loc="upper right")
        xlabel = f"{self._series['label']} ({unit})"
        prof_ax.set_xlabel(xlabel, fontsize=8)
        prof_ax.set_ylabel("height above floor  z (m)", fontsize=8)
        # RC polish: a self-explanatory title (what / where / when) so the plot
        # reads on its own -- hot gas stratifies near the ceiling (top), cooler
        # air stays near the floor (bottom).
        px = (self._extent[0] + self._x_col / max(frame.shape[1] - 1, 1)
              * (self._extent[1] - self._extent[0])) if self._extent else self._x_col
        prof_ax.set_title(f"{self._series['label']} vs height  ·  x = {px:.2f} m, "
                          f"t = {idx / self._fps:.1f} s", fontsize=9, fontweight="bold")
        if len(zs):
            prof_ax.set_ylim(zs.min(), zs.max())
        prof_ax.tick_params(labelsize=7)

        time_ax = fig.add_subplot(212)
        times = np.arange(self._data.shape[0]) / self._fps
        if self._series["layer"] is not None:
            time_ax.plot(times, self._series["layer"], color="#2563EB", label="smoke layer (m)")
        if self._series["plume"] is not None:
            time_ax.plot(times, self._series["plume"], color="#E8622C", label="plume height (m)")
        time_ax.set_xlabel("time (s)", fontsize=8)
        time_ax.set_ylabel("height (m)", fontsize=8)
        time_ax.set_title("Layer, plume & ceiling over time", fontsize=8)
        time_ax.tick_params(labelsize=7)
        # ceiling-jet temperature on a twin axis (linked multi-quantity view)
        jet_ax = time_ax.twinx()
        jet_ax.plot(times, self._series["ceiling"], color="#888", linewidth=0.9,
                    label="ceiling temp")
        jet_ax.set_ylabel(f"ceiling {unit}", fontsize=8, color="#888")
        jet_ax.tick_params(labelsize=7, colors="#888")
        time_ax.axvline(idx / self._fps, color="#00E5FF", linewidth=1.0)  # time cursor
        lines = [l for l in (time_ax.get_lines() + jet_ax.get_lines())
                 if not l.get_label().startswith("_")]
        time_ax.legend(lines, [l.get_label() for l in lines], fontsize=6, loc="upper left")
        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.12, right=0.88, hspace=0.85)
        self.plot_canvas.draw_idle()

    def _build_insights(self) -> None:
        if self._series is None:
            return
        s = self._series
        unit = s["unit"]
        out = []
        if s["plume"] is not None:
            i = int(np.argmax(s["plume"]))
            out.append(Insight(f"Plume reaches its greatest height ({s['plume'][i]:.2f} m).",
                               category="query", quantity=self._key.quantity,
                               time_s=i / self._fps, value=float(s["plume"][i]),
                               basis="max of per-frame highest hot cell"))
        if s["layer"] is not None:
            i = int(np.argmin(s["layer"]))
            out.append(Insight(f"Smoke layer descends to its lowest ({s['layer'][i]:.2f} m).",
                               category="query", quantity=self._key.quantity,
                               time_s=i / self._fps, value=float(s["layer"][i]),
                               basis="min of the smoke-layer-height series"))
        ci = int(np.argmax(s["ceiling"]))
        out.append(Insight(f"Ceiling {s['label'].lower()} peaks at {s['ceiling'][ci]:.0f} {unit}.",
                           category="query", quantity=self._key.quantity, time_s=ci / self._fps,
                           value=float(s["ceiling"][ci]), basis="max of the near-ceiling band"))
        self.insights.set_insights(out)
