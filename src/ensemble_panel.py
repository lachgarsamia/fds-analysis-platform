"""Ensemble Spread panel (V5-M5 / Phase 5), an Analysis-page tab.

Shows the min/mean/max (or 25-75% percentile) envelope of a chosen
per-frame metric across every scenario in the factorial, with one or more
chosen scenarios overlaid on top. Labelled as parametric ensemble spread
(not stochastic uncertainty).

Analysis UX + reliability pass: previously only one scenario could be
overlaid on the envelope at a time (via scenario_combo alone), and the
band was always the full min-max spread. Multi-scenario overlay (reusing
views.py's EnsemblePickerDialog -- the exact same "Overlay scenarios…"
picker timeseries.py already uses, not a new widget) lets a researcher
compare 2+ specific scenarios against the envelope at once; the added
25-75% band option is a more robust/less-outlier-sensitive alternative to
min-max, not a replacement for it (ensemble_spread.percentile_envelope()).

SelectionBus (M1): scenario_combo is bound by main_window, so the bus-
synced scenario is always the first overlay curve (same convention
timeseries.py's locator/overlay split already uses) -- additional
overlays are picker-only, local to this panel. Series are computed once
per metric (streaming the store) and cached.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import SliceKey
from analysis_panel_base import populate_scenario_combo
from views import EnsemblePickerDialog
import ensemble_spread as es

_OVERLAY_COLORS = ("#00E5FF", "#E8622C", "#7C3AED", "#22C55E", "#F59E0B")


class EnsemblePanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}     # metric -> {series_by_case, env, percentile_env}
        self._overlay_cases: list = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Ensemble spread")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.setAccessibleName("Ensemble metric")
        for key, label, _u in es.METRICS:
            self.metric_combo.addItem(label, key)
        header.addWidget(self.metric_combo)
        header.addWidget(QtWidgets.QLabel("Band:"))
        self.band_combo = QtWidgets.QComboBox()
        self.band_combo.setAccessibleName("Ensemble band")
        self.band_combo.addItem("Min–max", "minmax")
        self.band_combo.addItem("25–75% (IQR)", "iqr")
        self.band_combo.setToolTip(
            "Min-max shows the full spread; 25-75% is less sensitive to a single "
            "outlier scenario stretching the band")
        header.addWidget(self.band_combo)
        self.scenario_combo = QtWidgets.QComboBox()   # bound to the bus by main_window
        self.scenario_combo.setAccessibleName("Ensemble scenario")
        header.addWidget(self.scenario_combo)
        self.overlay_button = QtWidgets.QPushButton("Overlay scenarios…")
        self.overlay_button.setToolTip(
            "Choose additional scenarios to compare against the envelope")
        self.overlay_button.clicked.connect(self._open_overlay_picker)
        header.addWidget(self.overlay_button)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(es.SPREAD_NOTE)
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Ensemble envelope")
        layout.addWidget(self.canvas, 1)

        self.metric_combo.currentIndexChanged.connect(self._render)
        self.band_combo.currentIndexChanged.connect(self._render)
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)

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
        self._overlay_cases = [self._manifest[0].case_index]
        self._render()

    def _folder_for(self, case_index: int) -> str:
        idx = self.scenario_combo.findData(case_index)
        return self.scenario_combo.itemText(idx) if idx >= 0 else f"scenario {case_index}"

    # ------------------------------------------------------------- controls
    def _on_scenario_changed(self, _idx: int) -> None:
        # The bus-bound scenario is always the first overlay curve (same
        # convention timeseries.py's locator/overlay split uses); keep any
        # additional picker-chosen overlays.
        case_index = self.scenario_combo.currentData()
        if case_index is not None and case_index not in self._overlay_cases:
            self._overlay_cases = [case_index] + [
                c for c in self._overlay_cases if c != case_index]
        self._render()

    def _open_overlay_picker(self) -> None:
        dialog = EnsemblePickerDialog(self._manifest, self._overlay_cases, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected = dialog.selected_case_indices()
            current = self.scenario_combo.currentData()
            if current is not None and current not in selected:
                selected = [current] + selected
            self._overlay_cases = selected
            self._render()

    def _metric_data(self, metric):
        if metric in self._cache:
            return self._cache[metric]
        series = {}
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for entry in self._manifest:
                data = np.asarray(self._store.get(entry.case_index, SliceKey("TEMPERATURE")))
                extent = self._store.get_extent(entry.case_index, SliceKey("TEMPERATURE"))
                series[entry.case_index] = es.per_frame_series(data, extent, metric)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        values = list(series.values())
        self._cache[metric] = {
            "series": series,
            "minmax": es.envelope(values),
            "iqr": es.percentile_envelope(values),
        }
        return self._cache[metric]

    def _render(self) -> None:
        if not self._loaded or not self._manifest:
            return
        metric = self.metric_combo.currentData()
        band_kind = self.band_combo.currentData() or "minmax"
        md = self._metric_data(metric)
        lo, mean, hi = md[band_kind]
        n = len(mean)
        times = np.arange(n) / self._fps
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if n:
            band_label = ("min–max across scenarios" if band_kind == "minmax"
                          else "25–75% across scenarios (IQR)")
            ax.fill_between(times, lo, hi, color="#B0B0B0", alpha=0.4, label=band_label)
            ax.plot(times, mean, color="#333", linewidth=1.2, label="ensemble mean")
            for i, case_index in enumerate(self._overlay_cases):
                if case_index not in md["series"]:
                    continue
                s = md["series"][case_index][:n]
                color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
                ax.plot(times[:len(s)], s, color=color, linewidth=1.6,
                        label=self._folder_for(case_index))
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel(f"{es.METRIC_LABEL[metric]} ({es.METRIC_UNIT[metric]})", fontsize=8)
        ax.set_title("Ensemble spread — " + es.METRIC_LABEL[metric], fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.93, bottom=0.14, left=0.12, right=0.97)
        self.canvas.draw_idle()
