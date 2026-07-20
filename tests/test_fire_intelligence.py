"""Tests for the V3 Fire Intelligence Layer foundation (Phase 0):
signatures.py, descriptors.py, events.py, insight.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import signatures as sg  # noqa: E402
import descriptors as dc  # noqa: E402
import events as ev  # noqa: E402
from insight import Insight  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402
from slice_key import DEFAULT_SLICE_KEY  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present")

EXTENT = (0.0, 1.0, 0.0, 0.3)


def _ramp_data():
    """A 5-frame field that heats linearly then holds, so every signature
    is hand-computable. Two cells: one crosses thresholds, one stays cool."""
    # shape (5, 1, 2): cell A ramps 20->300, cell B stays at 20.
    a = np.array([20, 100, 200, 300, 300], dtype=np.float32)
    b = np.full(5, 20.0, dtype=np.float32)
    return np.stack([a, b], axis=1).reshape(5, 1, 2)


class TestSignatures:
    def test_peak_and_time_of_peak(self):
        s = sg.compute_signatures(_ramp_data(), EXTENT, fps=2, levels=(100, 300), ambient_c=20.0)
        np.testing.assert_allclose(s.map("peak")[0], [300.0, 20.0])
        # cell A first reaches 300 at frame 3 -> t = 1.5 s at fps 2
        assert s.map("time_of_peak")[0, 0] == pytest.approx(1.5)

    def test_first_crossing_and_duration(self):
        s = sg.compute_signatures(_ramp_data(), EXTENT, fps=2, levels=(100, 300), ambient_c=20.0)
        # strict >100 (app convention): frame 1 is exactly 100, so the
        # first crossing is frame 2 (200) -> t = 1.0 s; cell B is never inf
        assert s.map("first_crossing_100")[0, 0] == pytest.approx(1.0)
        assert np.isinf(s.map("first_crossing_100")[0, 1])
        # frames strictly above 300: none (peak is exactly 300) -> 0 s
        assert s.map("duration_above_300")[0, 0] == 0.0
        # frames strictly above 100: frames 2,3,4 -> 3 * 0.5 s = 1.5 s
        assert s.map("duration_above_100")[0, 0] == pytest.approx(1.5)

    def test_thermal_dose_is_time_integral_of_excess(self):
        s = sg.compute_signatures(_ramp_data(), EXTENT, fps=2, levels=(100,), ambient_c=20.0)
        # cell A excess over ambient: [0,80,180,280,280]; * dt(0.5) summed = 410
        assert s.map("thermal_dose")[0, 0] == pytest.approx(410.0)
        assert s.map("thermal_dose")[0, 1] == 0.0

    def test_at_cell_returns_all_channels(self):
        s = sg.compute_signatures(_ramp_data(), EXTENT, fps=2, levels=(100,), ambient_c=20.0)
        d = s.at_cell(0, 0)
        assert "peak" in d and "thermal_dose" in d and d["peak"] == 300.0


class TestDescriptors:
    def test_spatial_reductions_and_rate(self):
        t = dc.compute_descriptors(_ramp_data(), EXTENT, fps=2, ambient_c=20.0)
        np.testing.assert_allclose(t.column("spatial_max"), [20, 100, 200, 300, 300])
        # d_spatial_max forward diff * fps: [160,200,200,0,0]
        np.testing.assert_allclose(t.column("d_spatial_max"), [160, 200, 200, 0, 0])
        assert t.n_frames == 5

    def test_matrix_shape(self):
        t = dc.compute_descriptors(_ramp_data(), EXTENT, fps=2)
        m = t.as_matrix(["spatial_max", "spatial_mean"])
        assert m.shape == (5, 2)


class TestEvents:
    def test_events_are_time_ordered_and_typed(self):
        t = dc.compute_descriptors(_ramp_data(), EXTENT, fps=2)
        evs = ev.detect_events(t, "TEMPERATURE")
        assert evs, "expected at least ignition + peak"
        times = [e.primary_time() for e in evs]
        assert times == sorted(times)
        assert all(isinstance(e, Insight) for e in evs)
        # ignition precedes the peak
        kinds = [e.statement for e in evs]
        assert any("Ignition" in k for k in kinds)
        assert any("Peak" in k for k in kinds)

    def test_threshold_events_match_first_crossings(self):
        t = dc.compute_descriptors(_ramp_data(), EXTENT, fps=2)
        evs = ev.detect_events(t, "TEMPERATURE")
        crossing_100 = next(e for e in evs if "exceeds 100" in e.statement)
        assert crossing_100.primary_time() == pytest.approx(1.0)  # strict >100 -> frame 2


class TestInsight:
    def test_frame_index_and_primary_time(self):
        i = Insight("x", time_s=1.5)
        assert i.primary_time() == 1.5
        assert i.frame_index(fps=4) == 6

    def test_interval_uses_start(self):
        i = Insight("x", time_s=(2.0, 5.0))
        assert i.primary_time() == 2.0

    def test_none_time(self):
        assert Insight("x").primary_time() is None
        assert Insight("x").frame_index(4) is None


@requires_real_dataset
class TestCrossValidationAgainstSummaryStats:
    """Honesty rule: derived signatures/events must agree with the
    already-trusted per-scenario statistics."""

    def _sim(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        return sim

    def test_signature_peak_equals_summary_max_temp(self):
        from summary_stats import compute_scenario_summary
        sim = self._sim()
        entry = sim.manifest[0]
        summary = compute_scenario_summary(entry, sim.store, sim.timesteps_per_second)
        s = sg.load_signatures(sim.store, entry.case_index, DEFAULT_SLICE_KEY,
                               sim.timesteps_per_second)
        assert float(s.map("peak").max()) == pytest.approx(summary.max_temp_c, abs=0.5)

    def test_event_threshold_time_matches_summary(self):
        from summary_stats import compute_scenario_summary
        sim = self._sim()
        entry = sim.manifest[0]
        summary = compute_scenario_summary(entry, sim.store, sim.timesteps_per_second)
        data = sim.store.get(entry.case_index, DEFAULT_SLICE_KEY)
        extent = sim.store.get_extent(entry.case_index, DEFAULT_SLICE_KEY)
        t = dc.compute_descriptors(data, extent, sim.timesteps_per_second)
        evs = ev.detect_events(t, "TEMPERATURE")
        crossing = next((e for e in evs if "exceeds 100" in e.statement), None)
        if summary.time_to_100c_s is not None:
            assert crossing is not None
            assert crossing.primary_time() == pytest.approx(summary.time_to_100c_s, abs=0.5)

    def test_load_signatures_disk_cache_roundtrip(self, tmp_path):
        sim = self._sim()
        entry = sim.manifest[0]
        cache_dir = str(tmp_path / "sigcache")
        s1 = sg.load_signatures(sim.store, entry.case_index, DEFAULT_SLICE_KEY,
                                sim.timesteps_per_second, cache_dir=cache_dir,
                                source_folder=entry.path)
        s2 = sg.load_signatures(sim.store, entry.case_index, DEFAULT_SLICE_KEY,
                                sim.timesteps_per_second, cache_dir=cache_dir,
                                source_folder=entry.path)
        np.testing.assert_allclose(s1.map("peak"), s2.map("peak"))
