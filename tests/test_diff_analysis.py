"""Tests for difference-over-time (V2 roadmap M1.5)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diff_analysis import DifferenceOverTimeDialog, difference_time_series  # noqa: E402


class TestDifferenceTimeSeries:
    def test_constant_offset_gives_constant_curves(self):
        a = np.full((5, 3, 3), 10.0)
        b = np.full((5, 3, 3), 4.0)
        series = difference_time_series(a, b)
        np.testing.assert_allclose(series["rms"], 6.0)
        np.testing.assert_allclose(series["max_abs"], 6.0)
        np.testing.assert_allclose(series["mean"], 6.0)

    def test_growing_divergence_is_monotonic(self):
        n = 6
        a = np.stack([np.full((2, 2), float(i)) for i in range(n)])
        b = np.zeros((n, 2, 2))
        series = difference_time_series(a, b)
        assert np.all(np.diff(series["max_abs"]) >= 0)
        assert series["max_abs"][-1] == n - 1

    def test_truncates_to_shorter_array(self):
        a = np.ones((10, 2, 2))
        b = np.zeros((4, 2, 2))
        series = difference_time_series(a, b)
        assert len(series["rms"]) == 4

    def test_mean_can_be_signed_while_rms_is_not(self):
        a = np.full((3, 2, 2), -5.0)
        b = np.zeros((3, 2, 2))
        series = difference_time_series(a, b)
        assert np.all(series["mean"] < 0)
        assert np.all(series["rms"] > 0)


class TestDifferenceOverTimeDialog:
    def test_dialog_title_names_both_scenarios(self, qapp):
        n = 5
        a = np.stack([np.full((2, 2), float(i)) for i in range(n)])
        b = np.zeros((n, 2, 2))
        dialog = DifferenceOverTimeDialog(a, b, fps=2.0, label_a="A", label_b="B", unit="°C")
        assert "A" in dialog.windowTitle() and "B" in dialog.windowTitle()
        dialog.deleteLater()

    def test_summary_label_names_peak_time_and_magnitude(self, qapp):
        n = 5
        a = np.stack([np.full((2, 2), float(i)) for i in range(n)])
        b = np.zeros((n, 2, 2))
        dialog = DifferenceOverTimeDialog(a, b, fps=2.0, label_a="A", label_b="B", unit="°C")
        from PyQt5 import QtWidgets
        summary_texts = [w.text() for w in dialog.findChildren(QtWidgets.QLabel)]
        assert any("Largest divergence" in t for t in summary_texts)
        assert any("t = 2.0 s" in t for t in summary_texts)  # frame 4 / fps 2.0
        dialog.deleteLater()
