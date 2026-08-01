"""Tests for the Time-Series Workspace (V2 roadmap M1.1)."""

import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import colormaps  # noqa: E402,F401 -- registers "fds_fire"/"fds_flow" etc. with
                   # matplotlib at import time; main_window.py normally
                   # guarantees this via its own import chain before any
                   # panel renders, which a standalone panel test bypasses.
from slice_key import SliceKey  # noqa: E402
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
    # A real SliceKey, not a bare placeholder string: current_key.quantity
    # is read by production code (_reload_locator's colormap lookup), and
    # every real quantity_options list (built in main_window.py) is always
    # (label, SliceKey) pairs.
    quantity_options = [("Temperature", SliceKey("TEMPERATURE"))]
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

    def test_point_probe_publishes_to_the_bus(self, panel):
        """Consolidation Phase 2: the probe (previously local-only)
        publishes Selection.point so another panel following the shared
        selection (e.g. SpaceTimePanel) can follow it."""
        from selection import SelectionBus
        bus = SelectionBus()
        panel.set_bus(bus)
        panel._apply_click(0.5, 0.15)
        assert bus.current.point == (0.5, 0.15)

    def test_region_probe_publishes_normalized_region(self, panel):
        from selection import SelectionBus
        bus = SelectionBus()
        panel.set_bus(bus)
        panel.mode_combo.setCurrentIndex(2)  # region
        panel._apply_click(0.6, 0.05)
        panel._apply_click(0.1, 0.2)
        assert bus.current.region == (0.1, 0.6, 0.05, 0.2)

    def test_line_probe_has_no_matching_field_and_does_not_publish(self, panel):
        from selection import SelectionBus
        bus = SelectionBus()
        panel.set_bus(bus)
        panel.mode_combo.setCurrentIndex(1)  # line
        panel._apply_click(0.0, 0.3)
        panel._apply_click(1.0, 0.0)
        assert bus.current.point is None and bus.current.region is None

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


#: Live-polish follow-up: Time Series/Height now route through field_fn/
#: extent_fn (the same computed-quantity dispatch main_window.py uses
#: elsewhere) instead of calling store.get() directly, so DYNAMIC PRESSURE
#: and TEMPERATURE RISE can be plotted here. These tests use real SliceKeys
#: (quantity options used to be plain opaque strings in this suite's
#: original FakeStore-based fixture above) since current_key.quantity is
#: now read by production code (see the get_quantity fix below).
COMPUTED_QUANTITY_OPTIONS = [
    ("Temperature", SliceKey("TEMPERATURE")),
    ("Dynamic pressure", SliceKey("DYNAMIC PRESSURE")),
    ("Temperature rise (ΔT)", SliceKey("TEMPERATURE RISE")),
    ("Smoke — doorway (x = 0.25 m)", SliceKey("SOOT DENSITY")),
]


def _computed_field_fn(case_index, key):
    return np.random.rand(5, 6, 11).astype(np.float32)


def _computed_extent_fn(case_index, key):
    return (0.0, 1.0, 0.0, 0.3)


@pytest.fixture
def computed_panel(qapp):
    manifest = [FakeEntry(0, "case_a")]
    p = TimeSeriesPanel(None, manifest, COMPUTED_QUANTITY_OPTIONS, fps=4,
                         field_fn=_computed_field_fn, extent_fn=_computed_extent_fn)
    p.ensure_loaded()
    yield p
    p.deleteLater()


class TestTimeSeriesComputedQuantityRouting:
    def test_field_fn_is_used_instead_of_bare_store_get(self, qapp):
        """Regression guard for the store.get() bypass bug class fixed
        this session across main_window.py: a store that raises if ever
        touched directly proves field_fn/extent_fn are the only data path."""
        class ExplodingStore:
            def get(self, *a, **k):
                raise AssertionError("bypassed field_fn, called store.get() directly")

            def get_extent(self, *a, **k):
                raise AssertionError("bypassed extent_fn, called store.get_extent() directly")

        manifest = [FakeEntry(0, "case_a")]
        p = TimeSeriesPanel(ExplodingStore(), manifest, COMPUTED_QUANTITY_OPTIONS, fps=4,
                             field_fn=_computed_field_fn, extent_fn=_computed_extent_fn)
        p.ensure_loaded()
        for i in range(p.quantity_combo.count()):
            p.quantity_combo.setCurrentIndex(i)
        p.deleteLater()

    def test_all_computed_and_native_quantities_selectable(self, computed_panel):
        for i in range(computed_panel.quantity_combo.count()):
            computed_panel.quantity_combo.setCurrentIndex(i)

    def test_locator_uses_the_selected_quantitys_own_colormap(self, computed_panel):
        """Regression test for a real bug found while writing this test:
        _reload_locator was calling get_quantity(self.current_key) --
        passing the whole SliceKey instead of its .quantity string -- so
        the lookup always missed and silently fell back to TEMPERATURE's
        colormap ("fds_fire") no matter what was selected."""
        from registry import get_quantity
        idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                    if k.quantity == "DYNAMIC PRESSURE")
        computed_panel.quantity_combo.setCurrentIndex(idx)
        assert computed_panel._locator_image.get_cmap().name == get_quantity("DYNAMIC PRESSURE").cmap

    def test_temperature_rise_auto_probes_doorway_mid_height(self, computed_panel):
        """Live-polish follow-up: TEMPERATURE RISE's default view is a
        fixed-height point probe at x=0.25 m (doorway), z=0.11 m (this
        scaled model's own geometric mid-height, ROOM_Z=(0.0, 0.22) in
        schematic.py -- not a human "head height" guess), not the raw 2D
        slice, so the layer-descent signal isn't lost in the rest of the
        domain."""
        idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                    if k.quantity == "TEMPERATURE RISE")
        computed_panel.quantity_combo.setCurrentIndex(idx)
        assert computed_panel._mode == "point"
        assert computed_panel._probe == (0.25, 0.11)
        assert len(computed_panel._last_curves) == 1

    def test_temperature_rise_auto_probe_does_not_override_an_existing_pick(self, computed_panel):
        temp_idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                         if k.quantity == "TEMPERATURE")
        rise_idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                         if k.quantity == "TEMPERATURE RISE")
        computed_panel.quantity_combo.setCurrentIndex(temp_idx)
        computed_panel._apply_click(0.6, 0.2)
        computed_panel.quantity_combo.setCurrentIndex(rise_idx)
        assert computed_panel._probe == (0.6, 0.2)


class TestTimeSeriesSafetyNet:
    """PyQt5 aborts the whole process (qFatal -> abort(), no catchable
    Python exception) on an unhandled error inside a slot connected to a
    signal -- quantity_combo's currentIndexChanged is exactly such a slot,
    so _on_quantity_changed must never let an exception escape into Qt's
    event loop (the actual bug found this session: a stale column index
    out of bounds for a narrower quantity's array, in the sibling Height
    panel)."""

    def test_exception_in_field_fn_is_caught_not_raised(self, qapp, caplog):
        def exploding_field_fn(case_index, key):
            if key.quantity == "DYNAMIC PRESSURE":
                raise RuntimeError("simulated resolver failure")
            return _computed_field_fn(case_index, key)

        manifest = [FakeEntry(0, "case_a")]
        p = TimeSeriesPanel(None, manifest, COMPUTED_QUANTITY_OPTIONS, fps=4,
                             field_fn=exploding_field_fn, extent_fn=_computed_extent_fn)
        p.ensure_loaded()
        idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                    if k.quantity == "DYNAMIC PRESSURE")
        with caplog.at_level(logging.ERROR):
            p.quantity_combo.setCurrentIndex(idx)  # must not raise/abort
        assert any(r.levelno >= logging.ERROR for r in caplog.records)
        p.deleteLater()

    def test_recovers_after_a_failed_switch(self, qapp):
        def exploding_field_fn(case_index, key):
            if key.quantity == "DYNAMIC PRESSURE":
                raise RuntimeError("simulated resolver failure")
            return _computed_field_fn(case_index, key)

        manifest = [FakeEntry(0, "case_a")]
        p = TimeSeriesPanel(None, manifest, COMPUTED_QUANTITY_OPTIONS, fps=4,
                             field_fn=exploding_field_fn, extent_fn=_computed_extent_fn)
        p.ensure_loaded()
        bad_idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                        if k.quantity == "DYNAMIC PRESSURE")
        good_idx = next(i for i, (_l, k) in enumerate(COMPUTED_QUANTITY_OPTIONS)
                         if k.quantity == "TEMPERATURE")
        p.quantity_combo.setCurrentIndex(bad_idx)
        p.quantity_combo.setCurrentIndex(good_idx)  # must still work afterwards
        p.deleteLater()
