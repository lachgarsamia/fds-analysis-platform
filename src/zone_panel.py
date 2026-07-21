"""Named Region / Zone Statistics panel (V4-M4), an Analysis-page tab.

Draw a rectangle on the field, name it, and the zone becomes a persistent,
scenario-independent probe: its full stats bundle (mean/max temperature,
time-to-threshold, thermal dose, hazard duration, affected-cell fraction,
energy proxy) shows for the current scenario as scalars and over-time
curves, and one click compares the same zone across every scenario. Zones
are saved in the session (main_window collects get_zones() / restores via
set_zones()), so "apply the doorway zone to Case A and Case B" is one
selection. Findings are navigable, savable Insights (Evidence Notebook).

Static/lazy; the current-scenario bundle is cheap; cross-scenario
comparison is on demand (a button), streaming one scenario at a time.
Reuses the store, timeseries region mapping, tenability's threshold idea,
and the registry.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity, AMBIENT_C
from insight import InsightList, Insight
from slice_key import SliceKey
import zone_stats as zs


class ZonePanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._zones = []          # list[zs.Zone] -- persistent, session-backed
        self._data = None
        self._extent = None
        self._press = None        # first drag corner (x, z)
        self._q = get_quantity("TEMPERATURE")
        self._threshold = (self._q.hazard_levels or (AMBIENT_C * 3,))[0]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Zone statistics")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Zone scenario")
        header.addWidget(self.scenario_combo)
        self.export_button = QtWidgets.QPushButton("Export figure…")
        self.export_button.setAccessibleName("Export zone figure")
        self.export_button.clicked.connect(self._export_figure)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Drag a rectangle on the map to define a named zone. Its stats "
            f"show for this scenario (hazard threshold {self._threshold:.0f} "
            "°C); \"Compare\" evaluates the same zone across every scenario.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.loc_canvas = MplCanvas(self)
        self.loc_canvas.setAccessibleName("Zone locator map")
        body.addWidget(self.loc_canvas)
        self.plot_canvas = MplCanvas(self)
        self.plot_canvas.setAccessibleName("Zone plots")
        body.addWidget(self.plot_canvas)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        zone_row = QtWidgets.QHBoxLayout()
        zone_row.addWidget(QtWidgets.QLabel("Zone:"))
        self.zone_combo = QtWidgets.QComboBox()
        self.zone_combo.setAccessibleName("Zone selector")
        self.zone_combo.setMinimumWidth(140)
        zone_row.addWidget(self.zone_combo)
        for text, slot, name in (("Rename", self._rename_zone, "zone-rename"),
                                 ("Delete", self._delete_zone, "zone-delete"),
                                 ("Compare", self._compare_across_scenarios, "zone-compare")):
            btn = QtWidgets.QPushButton(text)
            btn.setAccessibleName(name)
            btn.clicked.connect(slot)
            zone_row.addWidget(btn)
        zone_row.addStretch(1)
        layout.addLayout(zone_row)

        self.stats_label = QtWidgets.QLabel("Draw a zone to see its statistics.")
        self.stats_label.setProperty("role", "caption")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        self.compare_table = QtWidgets.QTableWidget()
        self.compare_table.setAccessibleName("Zone comparison table")
        self.compare_table.setMaximumHeight(150)
        self.compare_table.setVisible(False)
        layout.addWidget(self.compare_table)

        self.insights = InsightList()
        self.insights.setMaximumHeight(100)
        layout.addWidget(self.insights)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.zone_combo.currentIndexChanged.connect(self._recompute)
        self.insights.insight_activated.connect(self._on_insight)
        self.loc_canvas.mpl_connect("button_press_event", self._on_press)
        self.loc_canvas.mpl_connect("button_release_event", self._on_release)

    def _export_figure(self) -> None:
        from figure_export import export_figure_interactive
        export_figure_interactive(self, self.plot_canvas.fig, "zone_stats")

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

    # -------------------------------------------------------- session hooks
    def get_zones(self) -> list:
        return [z.to_dict() for z in self._zones]

    def set_zones(self, zones: list) -> None:
        self._zones = [zs.Zone.from_dict(d) for d in (zones or []) if isinstance(d, dict)]
        self._refresh_zone_combo()
        if self._loaded:
            self._recompute()

    # ------------------------------------------------------------- data load
    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        self._data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
        self._extent = self._store.get_extent(case_index, SliceKey("TEMPERATURE"))
        self._draw_locator()
        self._recompute()

    def _draw_locator(self) -> None:
        fig = self.loc_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._loc_ax = ax
        if self._data is not None:
            frame = self._data[int(self._data.shape[0] * 0.6)]
            ax.imshow(frame, cmap=self._q.cmap, vmin=self._q.vmin,
                      vmax=self._q.slider_default, aspect="auto",
                      extent=self._extent if self._extent else None)
        ax.set_xticks([]); ax.set_yticks([])
        zone = self._current_zone()
        if zone is not None and self._extent is not None:
            ax.add_patch(plt_rect(zone))
            ax.set_title(f"Zone: {zone.name}", fontsize=8)
        else:
            ax.set_title("Drag to define a zone", fontsize=8)
        fig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
        self.loc_canvas.draw_idle()

    # ------------------------------------------------------------- drawing
    def _on_press(self, event) -> None:
        if event.inaxes is getattr(self, "_loc_ax", None) and event.xdata is not None:
            self._press = (event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        if self._press is None or event.inaxes is not getattr(self, "_loc_ax", None):
            self._press = None
            return
        x0, z0 = self._press
        self._press = None
        if event.xdata is None:
            return
        x1, z1 = event.xdata, event.ydata
        if abs(x1 - x0) < 1e-6 or abs(z1 - z0) < 1e-6:  # a click, not a drag
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Name zone", "Zone name:", text=f"zone {len(self._zones) + 1}")
        if not ok or not name.strip():
            return
        self._zones.append(zs.Zone(name.strip(), float(x0), float(x1), float(z0), float(z1)))
        self._select_zone(len(self._zones) - 1)

    def _refresh_zone_combo(self) -> None:
        self.zone_combo.blockSignals(True)
        self.zone_combo.clear()
        for z in self._zones:
            self.zone_combo.addItem(z.name)
        self.zone_combo.blockSignals(False)

    def _select_zone(self, index: int) -> None:
        """Refresh the combo and recompute for `index`, without relying on
        currentIndexChanged (a no-op when the index does not move, e.g. the
        first zone selecting row 0)."""
        self._refresh_zone_combo()
        self.zone_combo.blockSignals(True)
        self.zone_combo.setCurrentIndex(index)
        self.zone_combo.blockSignals(False)
        self._recompute()

    def _current_zone(self):
        i = self.zone_combo.currentIndex()
        return self._zones[i] if 0 <= i < len(self._zones) else None

    def _rename_zone(self) -> None:
        zone = self._current_zone()
        if zone is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename zone", "Name:", text=zone.name)
        if ok and name.strip():
            zone.name = name.strip()
            self._select_zone(self._zones.index(zone))

    def _delete_zone(self) -> None:
        i = self.zone_combo.currentIndex()
        if 0 <= i < len(self._zones):
            del self._zones[i]
            self._refresh_zone_combo()
            self.compare_table.setVisible(False)
            self._recompute()

    # ------------------------------------------------------------- compute
    def _recompute(self) -> None:
        if not self._loaded or self._data is None:
            return
        self._draw_locator()
        zone = self._current_zone()
        if zone is None or self._extent is None:
            self.stats_label.setText("Draw a zone to see its statistics.")
            self.plot_canvas.fig.clear(); self.plot_canvas.draw_idle()
            self.insights.clear()
            return
        b = zs.zone_bundle(self._data, self._extent, zone, self._fps,
                           self._threshold, AMBIENT_C)
        self._render_stats(zone, b)
        self._render_plot(zone, b)
        self._build_insights(zone, b)

    @staticmethod
    def _fmt_ttt(v):
        return f"{v:.1f} s" if v is not None else "never"

    def _render_stats(self, zone, b) -> None:
        self.stats_label.setText(
            f"<b>{zone.name}</b> ({b['n_cells']} cells, {zone.area():.3f} m²) &nbsp; "
            f"mean {b['mean_temperature']:.0f} °C &nbsp; peak {b['max_temperature']:.0f} °C "
            f"&nbsp; time-to-{self._threshold:.0f}°C {self._fmt_ttt(b['time_to_threshold'])} "
            f"&nbsp; hazard duration {b['hazard_duration']:.1f} s &nbsp; "
            f"thermal dose {b['thermal_dose']:.0f} °C·s &nbsp; "
            f"peak affected {b['peak_affected_fraction'] * 100:.0f}% &nbsp; "
            f"energy proxy {b['energy_proxy']:.1f} °C·s·m²")

    def _render_plot(self, zone, b) -> None:
        fig = self.plot_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(b["times"], b["region_mean"], color="#E8622C", label="zone mean")
        ax.plot(b["times"], b["region_max"], color="#B91C1C", linewidth=0.9, label="zone max")
        ax.axhline(self._threshold, color="#888", linewidth=0.7, linestyle=":",
                   label=f"{self._threshold:.0f} °C")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("temperature (°C)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
        dose_ax = ax.twinx()
        dose_ax.plot(b["times"], b["dose_curve"], color="#2563EB", linewidth=0.9)
        dose_ax.set_ylabel("thermal dose (°C·s)", fontsize=8, color="#2563EB")
        dose_ax.tick_params(labelsize=7, colors="#2563EB")
        fig.subplots_adjust(top=0.95, bottom=0.14, left=0.13, right=0.87)
        self.plot_canvas.draw_idle()

    def _compare_across_scenarios(self) -> None:
        zone = self._current_zone()
        if zone is None:
            return
        cols = [("Scenario", None), ("peak °C", "max_temperature"),
                ("mean °C", "mean_temperature"),
                (f"t→{self._threshold:.0f}°C", "time_to_threshold"),
                ("hazard s", "hazard_duration"), ("dose °C·s", "thermal_dose"),
                ("peak aff %", "peak_affected_fraction")]
        self.compare_table.clear()
        self.compare_table.setColumnCount(len(cols))
        self.compare_table.setRowCount(len(self._manifest))
        self.compare_table.setHorizontalHeaderLabels([c[0] for c in cols])
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for r, entry in enumerate(self._manifest):
                data = np.asarray(self._store.get(entry.case_index, SliceKey("TEMPERATURE")))
                extent = self._store.get_extent(entry.case_index, SliceKey("TEMPERATURE"))
                b = zs.zone_bundle(data, extent, zone, self._fps, self._threshold, AMBIENT_C)
                for c, (_label, key) in enumerate(cols):
                    if key is None:
                        text = entry.folder
                    elif key == "time_to_threshold":
                        text = self._fmt_ttt(b[key])
                    elif key == "peak_affected_fraction":
                        text = f"{b[key] * 100:.0f}"
                    else:
                        text = f"{b[key]:.0f}"
                    self.compare_table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.compare_table.resizeColumnsToContents()
        self.compare_table.setVisible(True)

    def _build_insights(self, zone, b) -> None:
        out = [Insight(
            f"Zone \"{zone.name}\" peaks at {b['max_temperature']:.0f} °C "
            f"(mean {b['mean_temperature']:.0f} °C).",
            category="query", quantity="TEMPERATURE",
            time_s=float(b["times"][int(np.argmax(b["region_max"]))]) if len(b["times"]) else None,
            value=b["max_temperature"], unit="°C",
            basis="max over the zone's cells and frames")]
        if b["time_to_threshold"] is not None:
            out.append(Insight(
                f"Zone \"{zone.name}\" reaches {self._threshold:.0f} °C at "
                f"{b['time_to_threshold']:.1f} s; hazardous for {b['hazard_duration']:.1f} s.",
                category="query", quantity="TEMPERATURE", time_s=b["time_to_threshold"],
                value=b["hazard_duration"], unit="s",
                basis="first zone-max crossing; hazard-duration = frames above threshold / fps"))
        self.insights.set_insights(out)

    def _on_insight(self, insight) -> None:
        pass  # zone stats are over-time; navigation seeks nothing here (no frame slider)


def plt_rect(zone):
    """A matplotlib rectangle patch outlining the zone in physical coords."""
    from matplotlib.patches import Rectangle
    x0, x1 = sorted((zone.x0, zone.x1))
    z0, z1 = sorted((zone.z0, zone.z1))
    return Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False,
                     edgecolor="#00E5FF", linewidth=2)
