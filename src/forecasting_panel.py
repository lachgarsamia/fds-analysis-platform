"""Forecasting panel (scientific-visualization completion pass, item 7):
a static, playback-independent view of the trained model's rollout
predictions vs. ground truth for the held-out test scenarios.

Explicitly NOT wired to TimeController -- same "no per-frame redraw, no
connection to playback whatsoever" convention analytics_panel.py's PCA
panel already documents and relies on. This widget renders once per
scenario-combo change, never on a timer/tick.

Data, all already produced offline -- nothing computed here:
  - prediction_store.PredictionSource: predicted TEMPERATURE arrays for
    the held-out test scenarios (ml/rollout.py's
    export_full_scenario_predictions).
  - The real ScenarioStore: ground truth on those same case indices.
  - ml/model_results.json: RMSE vs. lead-time from ml/rollout.py's
    evaluate_model() (ml/metrics.py's evaluate_rollout).
"""

from __future__ import annotations

import json
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from slice_key import DEFAULT_SLICE_KEY
from widgets import MplCanvas
from manifest import scenario_label

_ML_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "model_results.json")


def _peak_curve(data: np.ndarray) -> np.ndarray:
    """Per-frame peak value -- same reduction inspector.py/main_window.py
    already use for the Live Inspector's sparkline, reused here so actual
    and predicted curves are directly comparable on one scale."""
    return data.reshape(data.shape[0], -1).max(axis=1)


class ForecastingPanel(QtWidgets.QWidget):
    """prediction_store may be an unavailable (`is_available=False`)
    PredictionSource -- e.g. nobody has run the ml/ pipeline yet -- in
    which case this shows the same "not available" placeholder convention
    as Dataset/Analysis, rather than an empty or crashing plot."""

    def __init__(self, prediction_store=None, real_store=None, manifest: list = None, parent=None):
        super().__init__(parent)
        self._prediction_store = prediction_store
        self._real_store = real_store
        self._manifest = manifest or []
        self._metrics: dict = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Forecasting")
        title.setProperty("role", "section-title")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Forecasting test scenario")
        self.scenario_combo.setToolTip(
            "Held-out test scenarios the model was never trained on -- "
            "the only scenarios with real predictions to evaluate."
        )
        header_row.addWidget(self.scenario_combo)
        layout.addLayout(header_row)

        available = prediction_store is not None and prediction_store.is_available
        if not available:
            placeholder = QtWidgets.QLabel(
                "No trained-model predictions available -- run ml/train.py then "
                "ml/rollout.py to enable forecasting evaluation here."
            )
            placeholder.setWordWrap(True)
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(placeholder)
            self.scenario_combo.setEnabled(False)
            return

        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas, 1)
        self._curve_ax = self.canvas.fig.add_subplot(121)
        self._metrics_ax = self.canvas.fig.add_subplot(122)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        by_case = {e.case_index: e for e in self._manifest}
        for case_index in prediction_store.case_indices:
            entry = by_case.get(case_index)
            label = scenario_label(entry) if entry is not None else f"scenario {case_index}"
            self.scenario_combo.addItem(label, case_index)
            if entry is not None:
                self.scenario_combo.setItemData(
                    self.scenario_combo.count() - 1, entry.folder, QtCore.Qt.ToolTipRole)
        self.scenario_combo.currentIndexChanged.connect(self._plot_scenario)

        self._metrics = self._load_metrics()
        if self.scenario_combo.count():
            self._plot_scenario(0)

    @staticmethod
    def _load_metrics() -> dict:
        if not os.path.exists(_ML_RESULTS_PATH):
            return {}
        with open(_ML_RESULTS_PATH) as f:
            return json.load(f)

    def _plot_scenario(self, combo_index: int) -> None:
        case_index = self.scenario_combo.itemData(combo_index)
        if case_index is None:
            return
        actual = self._real_store.get(case_index, DEFAULT_SLICE_KEY)
        predicted = self._prediction_store.get(case_index)
        n = min(actual.shape[0], predicted.shape[0])
        actual_curve = _peak_curve(actual[:n])
        predicted_curve = _peak_curve(predicted[:n])
        error_curve = np.abs(actual_curve - predicted_curve)

        self._curve_ax.clear()
        self._curve_ax.plot(actual_curve, label="Actual", color="#2563EB")
        self._curve_ax.plot(predicted_curve, label="Predicted", color="#E8622C", linestyle="--")
        self._curve_ax.plot(error_curve, label="Abs. error", color="#999999", linestyle=":")
        self._curve_ax.set_xlabel("Frame")
        self._curve_ax.set_ylabel("Peak temperature (°C)")
        self._curve_ax.set_title("Actual vs. predicted", fontsize=10, fontweight="bold")
        self._curve_ax.legend(fontsize=8)

        self._metrics_ax.clear()
        rmse_by_lead = self._metrics.get("results", {}).get("fno", {}).get("rmse", {})
        if rmse_by_lead:
            leads = sorted(rmse_by_lead, key=int)
            values = [rmse_by_lead[k] for k in leads]
            self._metrics_ax.plot([int(k) for k in leads], values, marker="o", color="#2563EB")
            self._metrics_ax.set_xlabel("Lead time (steps)")
            self._metrics_ax.set_ylabel("RMSE (°C)")
            self._metrics_ax.set_title("Model evaluation", fontsize=10, fontweight="bold")
        else:
            self._metrics_ax.set_xticks([])
            self._metrics_ax.set_yticks([])

        self.canvas.fig.subplots_adjust(top=0.90, bottom=0.15, left=0.09, right=0.97, wspace=0.35)
        self.canvas.draw_idle()

        rollout_rmse = float(np.sqrt(np.mean(np.square(error_curve)))) if len(error_curve) else 0.0
        self.status_label.setText(
            f"{n} frames -- rollout RMSE (peak-temperature curve) = {rollout_rmse:.1f}°C"
        )
