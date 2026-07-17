"""Difference-over-time (V2 roadmap M1.5): M2.3's DifferenceView computes
one frame's A-B statistics at a time (see inspector.py's
set_difference_stats). This module extends that with the same statistics
across the *whole* timeline, as a curve -- showing when two scenarios
diverge, not just by how much at the current frame.

Pure computation is module-level (vectorized, no I/O); the dialog is a
thin static plot, same isolation convention as forecasting_panel.py
(never wired to TimeController).
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from widgets import MplCanvas


def difference_time_series(data_a: np.ndarray, data_b: np.ndarray) -> dict:
    """Per-frame RMS/max|delta|/mean over the shared timeline -- shape
    (n,) each, n = min(len(data_a), len(data_b))."""
    n = min(data_a.shape[0], data_b.shape[0])
    diff = np.asarray(data_a[:n], dtype=float) - np.asarray(data_b[:n], dtype=float)
    axes = tuple(range(1, diff.ndim))
    return {
        "rms": np.sqrt(np.mean(diff ** 2, axis=axes)),
        "max_abs": np.max(np.abs(diff), axis=axes),
        "mean": np.mean(diff, axis=axes),
    }


class DifferenceOverTimeDialog(QtWidgets.QDialog):
    def __init__(self, data_a: np.ndarray, data_b: np.ndarray, fps: float,
                 label_a: str, label_b: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Difference over time: {label_a} − {label_b}")
        self.resize(560, 380)
        layout = QtWidgets.QVBoxLayout(self)

        series = difference_time_series(data_a, data_b)
        times = np.arange(len(series["rms"])) / max(fps, 1e-6)

        canvas = MplCanvas(self)
        layout.addWidget(canvas, 1)
        ax = canvas.fig.add_subplot(111)
        ax.plot(times, series["rms"], label="RMS |Δ|", color="#2563EB")
        ax.plot(times, series["max_abs"], label="Max |Δ|", color="#E8622C", linestyle="--")
        ax.plot(times, series["mean"], label="Mean Δ", color="#999999", linestyle=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Δ ({unit})" if unit else "Δ")
        ax.set_title(f"{label_a} − {label_b}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        canvas.fig.subplots_adjust(top=0.90, bottom=0.14, left=0.12, right=0.97)
        canvas.draw_idle()

        peak_idx = int(np.argmax(series["max_abs"]))
        summary = QtWidgets.QLabel(
            f"Largest divergence at t = {times[peak_idx]:.1f} s "
            f"(max|Δ| = {series['max_abs'][peak_idx]:.1f}{unit})")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)
