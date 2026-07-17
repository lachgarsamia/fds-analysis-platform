"""Tests for the Time-Series Workspace (V2 roadmap M1.1)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from timeseries import (  # noqa: E402
    TimeSeriesPanel, line_profile, phys_to_index, point_series,
    region_series, write_series_csv,
)


class FakeEntry:
    def __init__(self, case_index, folder):
        self.case_index = case_index
        self.folder = folder
        self.path = "/nonexistent"
        self.candles = self.door = self.vod = self.voc = 0


class FakeStore:
    """Synthetic (n_times, n_z, n_x) data with a known linear ramp so
    every probe result is hand-computable."""

    EXTENT = (0.0, 1.0, 0.0, 0.3)

    def __init__(self, n_times=10, n_z=7, n_x=11):
        t = np.arange(n_times, dtype=np.float32)[:, None, None]
        z = np.arange(n_z, dtype=np.float32)[None, :, None]
        x = np.arange(n_x, dtype=np.float32)[None, None, :]
        self.data = t * 100.0 + z * 10.0 + x  # value encodes (t, row, col)

    def get(self, case_index, key):
        return self.data + case_index * 1000.0

    def get_extent(self, case_index, key):
        return self.EXTENT


class TestPureHelpers:
    def test_phys_to_index_corners(self):
        extent = (0.0, 1.0, 0.0, 0.3)
        shape = (7, 11)
        assert phys_to_index(extent, shape, 0.0, 0.3) == (0, 0)      # top-left
        assert phys_to_index(extent, shape, 1.0, 0.0) == (6, 10)     # bottom-right

    def test_phys_to_index_clips_out_of_bounds(self):
        extent = (0.0, 1.0, 0.0, 0.3)
        assert phys_to_index(extent, (7, 11), -5.0, 99.0) == (0, 0)
        assert phys_to_index(extent, (7, 11), 99.0, -5.0) == (6, 10)

    def test_point_series_extracts_expected_ramp(self):
        store = FakeStore()
        series = point_series(store.data, 2, 3)
        expected = np.arange(10) * 100.0 + 23.0
        np.testing.assert_allclose(series, expected)

    def test_region_series_is_mean_over_rectangle(self):
        store = FakeStore()
        series = region_series(store.data, 1, 2, 3, 4)  # rows 1-3, cols 2-4
        expected = np.arange(10) * 100.0 + 20.0 + 3.0   # mean row=2, mean col=3
        np.testing.assert_allclose(series, expected)

    def test_region_series_handles_swapped_corners(self):
        store = FakeStore()
        a = region_series(store.data, 3, 4, 1, 2)
        b = region_series(store.data, 1, 2, 3, 4)
        np.testing.assert_allclose(a, b)

    def test_line_profile_linear_field_is_linear(self):
        store = FakeStore()
        profile = line_profile(store.data, index=0, row0=0, col0=0, row1=6, col1=10,
                                n_samples=5)
        # Field is 10*row + col at t=0; along the diagonal both vary linearly.
        expected = np.linspace(0.0, 70.0, 5)
        np.testing.assert_allclose(profile, expected)

    def test_write_series_csv_round_trip(self, tmp_path):
        path = str(tmp_path / "out.csv")
        x = np.array([0.0, 0.25, 0.5])
        write_series_csv(path, "Time (s)", x, [("a", np.array([1.0, 2.0, 3.0])),
                                                ("b", np.array([4.0, 5.0, 6.0]))])
        lines = open(path).read().strip().splitlines()
        assert lines[0] == "Time (s),a,b"
        assert lines[1] == "0,1,4"
        assert lines[2] == "0.25,2,5"


@pytest.fixture
def panel(qapp):
    store = FakeStore()
    manifest = [FakeEntry(0, "case_a"), FakeEntry(1, "case_b")]
    quantity_options = [("Temperature", "TEMP_KEY")]
    p = TimeSeriesPanel(store, manifest, quantity_options, fps=4)
    p.ensure_loaded()
    yield p
    p.deleteLater()


class TestTimeSeriesPanel:
    def test_ensure_loaded_populates_scenarios_and_locator(self, panel):
        assert panel.scenario_combo.count() == 2
        assert panel._locator_image is not None
        assert panel._loaded

    def test_ensure_loaded_is_idempotent(self, panel):
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == 2

    def test_point_click_plots_curve_and_enables_export(self, panel):
        panel._apply_click(0.5, 0.15)
        assert len(panel._last_curves) == 1
        assert panel._last_curves[0][0] == "case_a"
        assert panel.export_button.isEnabled()
        assert panel._last_x[0] == "Time (s)"
        # fps=4 -> last time = 9/4 s
        assert panel._last_x[1][-1] == pytest.approx(9 / 4)

    def test_line_mode_needs_two_clicks(self, panel):
        panel.mode_combo.setCurrentIndex(1)  # line
        panel._apply_click(0.0, 0.3)
        assert panel._last_curves == []      # only the start point so far
        panel._apply_click(1.0, 0.0)
        assert len(panel._last_curves) == 1
        assert panel._last_x[0].startswith("Distance")
        # Full diagonal of a 1.0 x 0.3 extent
        assert panel._last_x[1][-1] == pytest.approx(np.hypot(1.0, 0.3))

    def test_region_mode_two_clicks_gives_time_axis(self, panel):
        panel.mode_combo.setCurrentIndex(2)  # region
        panel._apply_click(0.1, 0.05)
        panel._apply_click(0.9, 0.25)
        assert len(panel._last_curves) == 1
        assert panel._last_x[0] == "Time (s)"

    def test_overlay_cases_add_curves(self, panel):
        panel._overlay_cases = [0, 1]
        panel._apply_click(0.5, 0.15)
        labels = [label for label, _v in panel._last_curves]
        assert labels == ["case_a", "case_b"]
        # case 1's fake data is offset by exactly +1000
        diff = panel._last_curves[1][1] - panel._last_curves[0][1]
        np.testing.assert_allclose(diff, 1000.0)

    def test_mode_switch_clears_probe_and_disables_export(self, panel):
        panel._apply_click(0.5, 0.15)
        assert panel.export_button.isEnabled()
        panel.mode_combo.setCurrentIndex(1)
        assert panel._probe is None
        assert not panel.export_button.isEnabled()

    def test_export_csv_to_writes_plotted_curves(self, panel, tmp_path):
        panel._apply_click(0.5, 0.15)
        path = str(tmp_path / "curves.csv")
        panel.export_csv_to(path)
        lines = open(path).read().strip().splitlines()
        assert lines[0] == "Time (s),case_a"
        assert len(lines) == 1 + 10  # header + n_times rows

    def test_export_csv_to_without_probe_raises(self, panel, tmp_path):
        with pytest.raises(RuntimeError):
            panel.export_csv_to(str(tmp_path / "nope.csv"))

    def test_frame_slider_visible_only_in_line_mode(self, panel):
        assert not panel._frame_row_widget.isVisibleTo(panel)
        panel.mode_combo.setCurrentIndex(1)
        assert panel._frame_row_widget.isVisibleTo(panel)
