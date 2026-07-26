"""Sensitivity Explorer panel (V5-M3), an Analysis-page tab.

Parameter sliders for the four design factors (candles/door/vod/voc) that
interpolate across the existing 24-run factorial. Moving them estimates every
response (a "what-if" table), draws a local response surface for a chosen
response over two factors, and a tornado of each factor's local swing. Every
estimate is labelled "Estimated from Existing Scenarios by interpolation" --
never a new simulation.

SelectionBus (M1): the sliders publish the *nearest existing scenario* so the
Live Viewer and linked panels show a real run; selecting a scenario elsewhere
snaps the sliders to its factor levels. Reuses M2's study table and the
sensitivity engine.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, plot_fg_color
import study_analytics as sa
import sensitivity as se

_SLIDER_STEPS = 10  # slider units per factor level


class SensitivityPanel(QtWidgets.QWidget):
    def __init__(self, summaries: list, manifest: list, parent=None):
        super().__init__(parent)
        self._summaries = sorted(summaries or [], key=lambda s: s.case_index)
        self._table = sa.build_table(self._summaries)
        self._levels = {p: se.factor_levels(self._table, p) for p in sa.PARAMS} \
            if self._table else {p: [0.0] for p in sa.PARAMS}
        self._bus = None
        self._syncing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Sensitivity explorer")
        title.setProperty("role", "section-title")
        layout.addWidget(title)
        self.note = QtWidgets.QLabel("⚠ " + se.ESTIMATE_NOTE)
        self.note.setWordWrap(True)
        self.note.setProperty("role", "caption")
        layout.addWidget(self.note)

        # --- factor sliders ---
        sliders = QtWidgets.QFormLayout()
        self._sliders = {}
        self._value_labels = {}
        for p in sa.PARAMS:
            lo, hi = min(self._levels[p]), max(self._levels[p])
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setAccessibleName(f"Sensitivity {p}")
            s.setRange(0, max(1, int(round((hi - lo) * _SLIDER_STEPS))))
            s.valueChanged.connect(self._on_slider)
            self._sliders[p] = s
            lbl = QtWidgets.QLabel()
            self._value_labels[p] = lbl
            roww = QtWidgets.QWidget(); rl = QtWidgets.QHBoxLayout(roww)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(s, 1); rl.addWidget(lbl)
            sliders.addRow(sa.PARAM_LABELS[p], roww)
        layout.addLayout(sliders)

        self.nearest_label = QtWidgets.QLabel("")
        self.nearest_label.setProperty("role", "caption")
        layout.addWidget(self.nearest_label)

        # --- response + surface-axis selectors ---
        selrow = QtWidgets.QHBoxLayout()
        selrow.addWidget(QtWidgets.QLabel("Response:"))
        self.response_combo = QtWidgets.QComboBox()
        self.response_combo.setAccessibleName("Sensitivity response")
        for key in sa.RESPONSE_KEYS:
            self.response_combo.addItem(sa.RESPONSE_LABEL[key], key)
        self.response_combo.currentIndexChanged.connect(self._update)
        selrow.addWidget(self.response_combo)
        selrow.addWidget(QtWidgets.QLabel("Surface X:"))
        self.fx_combo = QtWidgets.QComboBox()
        self.fy_combo = QtWidgets.QComboBox()
        for p in sa.PARAMS:
            self.fx_combo.addItem(sa.PARAM_LABELS[p], p)
            self.fy_combo.addItem(sa.PARAM_LABELS[p], p)
        self.fx_combo.setCurrentIndex(2)  # vod
        self.fy_combo.setCurrentIndex(1)  # door
        selrow.addWidget(self.fx_combo)
        selrow.addWidget(QtWidgets.QLabel("Y:"))
        selrow.addWidget(self.fy_combo)
        selrow.addStretch(1)
        self.fx_combo.currentIndexChanged.connect(self._update)
        self.fy_combo.currentIndexChanged.connect(self._update)
        layout.addLayout(selrow)

        self.tabs = QtWidgets.QTabWidget()
        self.surface_canvas = MplCanvas(self)
        self.surface_canvas.setAccessibleName("Response surface")
        self.tabs.addTab(self.surface_canvas, "Response surface")
        self.tornado_canvas = MplCanvas(self)
        self.tornado_canvas.setAccessibleName("Tornado")
        self.tabs.addTab(self.tornado_canvas, "Tornado")
        self.whatif_table = QtWidgets.QTableWidget()
        self.whatif_table.setAccessibleName("What-if estimates")
        self.tabs.addTab(self.whatif_table, "What-if (all responses)")
        layout.addWidget(self.tabs, 1)

        self._init_sliders()
        self._update()

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.scenario is None:
            return
        row = next((r for r in self._table if r["case_index"] == sel.scenario), None)
        if row is None:
            return
        self._syncing = True
        try:
            for p in sa.PARAMS:
                self._set_slider_value(p, float(row["params"][p]))
        finally:
            self._syncing = False
        self._update()

    # ----------------------------------------------------------- slider maths
    def _init_sliders(self) -> None:
        if not self._table:
            return
        self._syncing = True
        for p in sa.PARAMS:
            self._set_slider_value(p, float(self._table[0]["params"][p]))
        self._syncing = False

    def _set_slider_value(self, p, value) -> None:
        lo = min(self._levels[p])
        self._sliders[p].blockSignals(True)
        self._sliders[p].setValue(int(round((value - lo) * _SLIDER_STEPS)))
        self._sliders[p].blockSignals(False)

    def _setting(self, p) -> float:
        return min(self._levels[p]) + self._sliders[p].value() / _SLIDER_STEPS

    def _settings(self) -> dict:
        return {p: self._setting(p) for p in sa.PARAMS}

    def _on_slider(self, _v) -> None:
        if not self._syncing:
            self._update()

    # --------------------------------------------------------------- update
    def _update(self) -> None:
        if not self._table:
            return
        settings = self._settings()
        for p in sa.PARAMS:
            self._value_labels[p].setText(f"{settings[p]:.1f}")
        # publish nearest existing scenario to the bus (a real run to view)
        ci, dist = se.nearest_scenario(self._table, settings)
        folder = next((r["folder"] for r in self._table if r["case_index"] == ci), "?")
        self.nearest_label.setText(
            f"Nearest existing run: {folder}"
            + ("  (exact)" if dist < 1e-6 else "  — response between runs is interpolated"))
        if self._bus is not None and ci is not None and not self._syncing:
            self._bus.update(origin=self, scenario=ci)
        self._render_whatif(settings)
        self._render_surface(settings)
        self._render_tornado(settings)

    def _render_whatif(self, settings) -> None:
        preds = se.predict_all(self._table, settings)
        self.whatif_table.clear()
        self.whatif_table.setColumnCount(3)
        self.whatif_table.setRowCount(len(sa.RESPONSE_KEYS))
        self.whatif_table.setHorizontalHeaderLabels(["Response", "Estimated", "Unit"])
        for r, key in enumerate(sa.RESPONSE_KEYS):
            self.whatif_table.setItem(r, 0, QtWidgets.QTableWidgetItem(sa.RESPONSE_LABEL[key]))
            self.whatif_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{preds[key]:.1f}"))
            self.whatif_table.setItem(r, 2, QtWidgets.QTableWidgetItem(sa.RESPONSE_UNIT[key]))
        self.whatif_table.resizeColumnsToContents()

    def _render_surface(self, settings) -> None:
        response = self.response_combo.currentData()
        fx, fy = self.fx_combo.currentData(), self.fy_combo.currentData()
        fig = self.surface_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if fx == fy:
            ax.text(0.5, 0.5, "Choose two different factors.", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color=plot_fg_color())
            self.surface_canvas.draw_idle()
            return
        xs, ys, z = se.response_surface(self._table, response, fx, fy, settings)
        im = ax.pcolormesh(xs, ys, z, cmap="inferno", shading="auto")
        ax.plot(settings[fx], settings[fy], "o", color="#00E5FF", markersize=8,
                markeredgecolor="white")
        ax.set_xlabel(sa.PARAM_LABELS[fx], fontsize=8)
        ax.set_ylabel(sa.PARAM_LABELS[fy], fontsize=8)
        ax.set_title(f"Estimated {sa.RESPONSE_LABEL[response]} "
                     f"({sa.RESPONSE_UNIT[response]}) — interpolated", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.subplots_adjust(top=0.92, bottom=0.14, left=0.12, right=0.98)
        self.surface_canvas.draw_idle()

    def _render_tornado(self, settings) -> None:
        response = self.response_combo.currentData()
        rows = se.tornado(self._table, response, settings)
        base = se.predict(self._table, response, settings)
        fig = self.tornado_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        names = [sa.PARAM_LABELS[p] for p, _lo, _hi, _s in rows][::-1]
        for i, (p, lo, hi, _swing) in enumerate(rows[::-1]):
            ax.barh(i, hi - lo, left=min(lo, hi), color="#E8622C", alpha=0.85)
        ax.axvline(base, color="#00E5FF", linewidth=1.2, label="current estimate")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(f"Estimated {sa.RESPONSE_LABEL[response]} "
                      f"({sa.RESPONSE_UNIT[response]})", fontsize=8)
        ax.set_title("Local factor swing (each factor low→high)", fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.9, bottom=0.16, left=0.2, right=0.96)
        self.tornado_canvas.draw_idle()
