"""Sensitivity Explorer panel (V5-M3), an Analysis-page tab.

Parameter sliders for the four design factors (candles/door/vod/voc) that
interpolate across the existing 24-run factorial, drawing a local response
surface for a chosen response over two factors. Every estimate is labelled
"Estimated from Existing Scenarios by interpolation" -- never a new
simulation.

Tornado (each factor's local swing) and What-if (all-responses table) were
removed as their own tabs (Analysis UX + reliability pass) -- Response
surface already answers "how does this response change near my current
setting" for the two factors that matter most, and the two removed views
were rarely-needed detail on top of that. sensitivity.tornado() and
predict_all() were deleted with them (used only by the removed render
methods); sensitivity.predict() and nearest_scenario() stay -- both are
still used by this panel's own bus-sync (_update, below) and by "Pin
what-if to Knowledge Graph" (_pin_hypothesis).

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
        self._hypotheses: list = []   # pinned what-if estimates (Phase C -> Knowledge Graph)

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
        selrow.addWidget(QtWidgets.QLabel("|"))
        self.pin_button = QtWidgets.QPushButton("Pin what-if to Knowledge Graph")
        self.pin_button.setAccessibleName("sensitivity-pin-hypothesis")
        self.pin_button.setToolTip(
            "Pin the current interpolated estimate as a hypothesis node in the Knowledge Graph")
        self.pin_button.clicked.connect(self._pin_hypothesis)
        selrow.addWidget(self.pin_button)
        layout.addLayout(selrow)

        self.pin_status = QtWidgets.QLabel("")
        self.pin_status.setWordWrap(True)
        self.pin_status.setProperty("role", "caption")
        layout.addWidget(self.pin_status)

        self.tabs = QtWidgets.QTabWidget()
        self.surface_canvas = MplCanvas(self)
        self.surface_canvas.setAccessibleName("Response surface")
        self.tabs.addTab(self.surface_canvas, "Response surface")
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
        self._render_surface(settings)

    def _pin_hypothesis(self) -> None:
        """Pin the current interpolated estimate as a hypothesis node in the
        Knowledge Graph (Analysis-improvement roadmap Phase C) -- the
        graph_panel reads `self._hypotheses` the same way it already reads
        zones/measurements/devices/vector_probes, no new wiring needed."""
        if not self._table:
            return
        settings = self._settings()
        response = self.response_combo.currentData()
        value = se.predict(self._table, response, settings)
        ci, _dist = se.nearest_scenario(self._table, settings)
        setting_text = ", ".join(f"{sa.PARAM_LABELS[p]}={settings[p]:.1f}" for p in sa.PARAMS)
        label = (f"{sa.RESPONSE_LABEL[response]} ≈ {value:.1f} {sa.RESPONSE_UNIT[response]} "
                 f"at ({setting_text})")
        self._hypotheses.append({
            "id": f"whatif-{len(self._hypotheses)}",
            "label": label,
            "settings": dict(settings),
            "response": response,
            "value": value,
            "nearest_scenario": ci,
        })
        self.pin_status.setText(f"Pinned: {label} — see it in the Knowledge Graph tab.")

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
