"""Ensemble Spread panel (V5-M5 / Phase 5), an Analysis-page tab.

Shows the min/mean/max envelope of a chosen per-frame metric across every
scenario in the factorial, with the currently-selected scenario drawn on top.
Labelled as parametric ensemble spread (not stochastic uncertainty).

SelectionBus (M1): scenario_combo is bound by main_window, so the highlighted
scenario follows the shared selection both ways. Series are computed once per
metric (streaming the store) and cached.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import SliceKey
import ensemble_spread as es


class EnsemblePanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}     # metric -> {series_by_case, env}

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
        self.scenario_combo = QtWidgets.QComboBox()   # bound to the bus by main_window
        self.scenario_combo.setAccessibleName("Ensemble scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(es.SPREAD_NOTE)
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Ensemble envelope")
        layout.addWidget(self.canvas, 1)

        self.metric_combo.currentIndexChanged.connect(self._render)
        self.scenario_combo.currentIndexChanged.connect(self._render)

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
        env = es.envelope(list(series.values()))
        self._cache[metric] = {"series": series, "env": env}
        return self._cache[metric]

    def _render(self) -> None:
        if not self._loaded or not self._manifest:
            return
        metric = self.metric_combo.currentData()
        md = self._metric_data(metric)
        lo, mean, hi = md["env"]
        n = len(mean)
        times = np.arange(n) / self._fps
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if n:
            ax.fill_between(times, lo, hi, color="#B0B0B0", alpha=0.4,
                            label="min–max across scenarios")
            ax.plot(times, mean, color="#333", linewidth=1.2, label="ensemble mean")
            sel = self.scenario_combo.currentData()
            if sel in md["series"]:
                s = md["series"][sel][:n]
                ax.plot(times[:len(s)], s, color="#00E5FF", linewidth=1.6,
                        label=f"selected: {self.scenario_combo.currentText()}")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel(f"{es.METRIC_LABEL[metric]} ({es.METRIC_UNIT[metric]})", fontsize=8)
        ax.set_title("Ensemble spread — " + es.METRIC_LABEL[metric], fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.93, bottom=0.14, left=0.12, right=0.97)
        self.canvas.draw_idle()
