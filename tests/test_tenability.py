"""Tests for tenability screening (V2 roadmap M3.2)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tenability as tn  # noqa: E402


class TestTimeToUntenableField:
    def test_first_crossing_per_cell(self):
        # 3 frames, 1x2 grid. Cell 0 crosses 60 at frame 1, cell 1 never.
        data = np.array([
            [[20.0, 20.0]],
            [[70.0, 30.0]],
            [[90.0, 40.0]],
        ], dtype=np.float32)
        field = tn.time_to_untenable_field(data, 60.0, fps=2)
        assert field.shape == (1, 2)
        assert field[0, 0] == pytest.approx(0.5)  # frame 1 / fps 2
        assert np.isinf(field[0, 1])

    def test_cell_hot_from_start_is_zero(self):
        data = np.array([[[100.0]], [[100.0]]], dtype=np.float32)
        field = tn.time_to_untenable_field(data, 60.0, fps=4)
        assert field[0, 0] == 0.0

    def test_scalar_is_earliest_crossing_anywhere(self):
        data = np.array([
            [[20.0, 20.0]],
            [[20.0, 70.0]],   # cell 1 crosses at frame 1
            [[90.0, 90.0]],   # cell 0 crosses at frame 2
        ], dtype=np.float32)
        assert tn.time_to_untenable_scalar(data, 60.0, fps=1) == pytest.approx(1.0)

    def test_scalar_none_when_never_reached(self):
        data = np.full((5, 2, 2), 25.0, dtype=np.float32)
        assert tn.time_to_untenable_scalar(data, 60.0, fps=4) is None

    def test_untenable_fraction(self):
        data = np.array([[[70.0, 20.0], [80.0, 30.0]]], dtype=np.float32)  # 1 frame, 2x2
        assert tn.untenable_fraction(data, 60.0, 0) == pytest.approx(0.5)

    def test_threshold_is_strict_exceedance(self):
        data = np.full((2, 1, 1), 60.0, dtype=np.float32)  # exactly at threshold
        assert tn.time_to_untenable_scalar(data, 60.0, fps=4) is None


class FakeEntry:
    def __init__(self, case_index, folder, path):
        self.case_index = case_index
        self.folder = folder
        self.path = path
        self.candles = self.door = self.vod = self.voc = 0


class FakeStore:
    def __init__(self):
        self.data = np.stack([np.full((3, 4), 20.0 + 30.0 * t, dtype=np.float32) for t in range(5)])

    def get(self, case_index, key):
        return self.data

    def get_extent(self, case_index, key):
        return [0.0, 1.0, 0.0, 0.3]


class TestTenabilityPanel:
    def test_ensure_loaded_populates_and_reports(self, qapp):
        from tenability_panel import TenabilityPanel
        panel = TenabilityPanel(FakeStore(), [FakeEntry(0, "case", "/x")], fps=4)
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == 1
        assert "untenable" in panel.stats_label.text().lower()
        panel.deleteLater()

    def test_threshold_change_recomputes(self, qapp):
        from tenability_panel import TenabilityPanel
        panel = TenabilityPanel(FakeStore(), [FakeEntry(0, "case", "/x")], fps=4)
        panel.ensure_loaded()
        before = panel.stats_label.text()
        panel.threshold_spin.setValue(500)  # never reached by the fake data
        assert panel.stats_label.text() != before
        assert "never" in panel.stats_label.text().lower()
        panel.deleteLater()

    def test_disclaimer_states_partial_and_no_co(self, qapp):
        from tenability_panel import TenabilityPanel, _DISCLAIMER
        assert "FED" in _DISCLAIMER
        assert "CO" in _DISCLAIMER
        assert "Partial" in _DISCLAIMER or "partial" in _DISCLAIMER
