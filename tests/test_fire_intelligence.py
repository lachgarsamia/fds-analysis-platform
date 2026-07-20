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

    def test_first_arrival_isochrone_monotone_outward_from_source(self):
        # A wavefront that reaches column c at frame c (source at column 0),
        # so first-arrival time increases with distance from the source --
        # the property the Fire MRI isochrones visualize.
        n_x = 6
        data = np.full((n_x, 1, n_x), 20.0, dtype=np.float32)
        for t in range(n_x):
            for c in range(n_x):
                if t >= c:
                    data[t, 0, c] = 100.0
        s = sg.compute_signatures(data, (0.0, 1.0, 0.0, 0.1), fps=1, levels=(50,), ambient_c=20.0)
        arrival = s.map("first_crossing_50")[0]
        assert np.all(np.diff(arrival) > 0), "first-arrival time must increase outward"


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


class TestInspectorStory:
    """V3-M2: the Inspector's Fire story list + current-phase line."""

    def test_set_story_populates_and_phase_tracks(self, qapp):
        from inspector import InspectorPanel
        panel = InspectorPanel()
        events = [
            Insight("Ignition.", time_s=0.5),
            Insight("Peak 400 C.", time_s=10.0),
        ]
        panel.set_story(events, fps=4)
        assert panel.story_list.count() == 2
        # before the first event
        panel.set_story_index(0)
        assert "before ignition" in panel.phase_label.text()
        # after the first event, before the second (0.5 s -> frame 2)
        panel.set_story_index(5)
        assert "Ignition" in panel.phase_label.text()
        # after the last event (10 s -> frame 40)
        panel.set_story_index(100)
        assert "Peak" in panel.phase_label.text()
        panel.deleteLater()

    def test_empty_story_clears_phase(self, qapp):
        from inspector import InspectorPanel
        panel = InspectorPanel()
        panel.set_story([], fps=4)
        panel.set_story_index(10)
        assert panel.phase_label.text() == ""
        panel.deleteLater()


import semantic_diff as sd  # noqa: E402


class TestSemanticDiff:
    def _pair(self, peak_a, peak_b):
        # 4-frame ramps to different peaks, 1x3 grid.
        def ramp(peak):
            return np.stack([np.full((1, 3), 20 + (peak - 20) * t / 3.0, dtype=np.float32)
                             for t in range(4)])
        return ramp(peak_a), ramp(peak_b)

    def test_peak_difference_named_and_ranked(self):
        a, b = self._pair(200, 400)
        ins = sd.compare(a, b, (0.0, 1.0, 0.0, 0.3), 2, "TEMPERATURE", "A", "B")
        assert ins  # non-empty, ranked
        stmts = [i.statement for i in ins]
        assert any("B peaks" in s and "higher" in s for s in stmts)

    def test_threshold_timing_difference(self):
        # B reaches 100 sooner because it heats faster.
        def ramp(rate):
            return np.stack([np.full((1, 2), 20 + rate * t, dtype=np.float32) for t in range(10)])
        a, b = ramp(20), ramp(40)  # b hotter faster
        ins = sd.compare(a, b, (0, 1, 0, 0.3), 4, "TEMPERATURE", "A", "B")
        assert any("B reaches 100" in i.statement and "sooner" in i.statement for i in ins)

    def test_spatial_insight_has_location_and_is_navigable(self):
        a, b = self._pair(200, 400)
        ins = sd.compare(a, b, (0.0, 1.0, 0.0, 0.3), 2, "TEMPERATURE", "A", "B")
        top = ins[0]
        assert top.category == "difference"
        assert top.location is not None or top.primary_time() is not None

    def test_identical_scenarios_produce_no_or_trivial_diffs(self):
        a, _ = self._pair(300, 300)
        ins = sd.compare(a, a.copy(), (0, 1, 0, 0.3), 2, "TEMPERATURE", "A", "B")
        # only the (zero-magnitude) spatial "biggest difference" may remain
        assert all("peaks" not in i.statement for i in ins)


@requires_real_dataset
class TestSemanticDiffRealData:
    """DoD: the door-width pair's diff is plume-dominated for TEMPERATURE
    but carries a distinct door effect for VELOCITY (M2.3's finding),
    every difference navigable."""

    def _sim(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        return sim

    def _door_pair(self, sim):
        by = {e.folder: e.case_index for e in sim.manifest}
        return by["c1_d0_vod0_voc0"], by["c1_d1_vod0_voc0"]

    def test_temperature_diff_plume_dominated_and_navigable(self):
        from slice_key import SliceKey
        sim = self._sim()
        ca, cb = self._door_pair(sim)
        ins = sd.compare(sim.store.get(ca, SliceKey("TEMPERATURE")),
                         sim.store.get(cb, SliceKey("TEMPERATURE")),
                         sim.store.get_extent(ca, SliceKey("TEMPERATURE")),
                         sim.timesteps_per_second, "TEMPERATURE", "A", "B")
        assert ins
        top = ins[0]
        # M2.3: the temperature difference is dominated by the candle/plume
        assert top.location is not None and top.location[0] > 0.75
        # every difference is navigable (time or location)
        assert all(i.primary_time() is not None or i.location is not None for i in ins)

    def test_velocity_diff_surfaces_a_door_effect(self):
        from slice_key import SliceKey
        sim = self._sim()
        ca, cb = self._door_pair(sim)
        ins = sd.compare(sim.store.get(ca, SliceKey("VELOCITY")),
                         sim.store.get(cb, SliceKey("VELOCITY")),
                         sim.store.get_extent(ca, SliceKey("VELOCITY")),
                         sim.timesteps_per_second, "VELOCITY", "c1_d0", "c1_d1")
        # the wider door drives a distinct air-speed difference the
        # temperature comparison does not show (a threshold or peak diff)
        assert any(("reaches" in i.statement or "peaks" in i.statement) for i in ins)


import query_engine as qe  # noqa: E402


class TestQueryEngine:
    def _data(self):
        # (4, 2, 3): a hot cell that crosses 100 at frame 2, peak 300 at frame 3.
        d = np.full((4, 2, 3), 20.0, dtype=np.float32)
        d[1, 0, 2] = 80
        d[2, 0, 2] = 150   # crosses 100 at frame 2
        d[3, 0, 2] = 300   # peak
        return d

    EXT = (0.0, 1.0, 0.0, 0.3)

    def test_first_crossing_time_and_location(self):
        ins = qe.execute(qe.Query("first_crossing", "TEMPERATURE", 100.0), self._data(), self.EXT, 2)[0]
        assert ins.primary_time() == pytest.approx(1.0)  # frame 2 / fps 2
        assert ins.location is not None

    def test_first_crossing_never(self):
        ins = qe.execute(qe.Query("first_crossing", "TEMPERATURE", 9999.0), self._data(), self.EXT, 2)[0]
        assert "never" in ins.statement and ins.primary_time() is None

    def test_extreme_value_and_location(self):
        ins = qe.execute(qe.Query("extreme", "TEMPERATURE"), self._data(), self.EXT, 2)[0]
        assert ins.value == pytest.approx(300.0)
        assert ins.primary_time() == pytest.approx(1.5)  # frame 3

    def test_regions_above_count(self):
        ins = qe.execute(qe.Query("regions_above", "TEMPERATURE", 100.0), self._data(), self.EXT, 2)[0]
        assert "1 of" in ins.statement  # exactly one cell ever exceeds 100

    def test_region_restriction_excludes_the_hot_cell(self):
        # the hot cell is at x=1.0 (candle band); restrict to the door (x~0.25)
        ins = qe.execute(qe.Query("first_crossing", "TEMPERATURE", 100.0, region="door"),
                         self._data(), self.EXT, 2)[0]
        assert "never" in ins.statement and "door" in ins.statement

    def test_plume_height(self):
        # a column hot up to a known row -> known height
        d = np.full((3, 4, 2), 20.0, dtype=np.float32)
        d[1, 1, 0] = 200  # row 1 hot at frame 1 (row 0 = ceiling z=0.3)
        ins = qe.execute(qe.Query("plume_height", "TEMPERATURE", 100.0), d, (0, 1, 0, 0.3), 2)[0]
        assert ins.value == pytest.approx(0.2, abs=0.01)  # row 1 of 4 -> z = 0.3 - 1/3*0.3


class TestQueryParser:
    def test_examples_parse_to_valid_kinds(self):
        for ex in qe.EXAMPLE_QUERIES:
            q = qe.parse(ex)
            assert q is not None and q.kind in qe.KINDS

    def test_threshold_and_region_extracted(self):
        q = qe.parse("first time temperature exceeds 300 near the candle")
        assert q.kind == "first_crossing" and q.threshold == 300.0 and q.region == "candle"

    def test_ventilation_maps_to_velocity(self):
        q = qe.parse("regions affected by ventilation")
        assert q.quantity == "VELOCITY" and q.kind == "regions_above"

    def test_gibberish_returns_none(self):
        assert qe.parse("what is the meaning of life") is None
        assert qe.parse("") is None


@requires_real_dataset
class TestQueryRealData:
    """DoD: queries return answers matching hand/known values."""

    def _sim(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        return sim

    def test_first_crossing_matches_summary(self):
        from slice_key import SliceKey
        from summary_stats import compute_scenario_summary
        sim = self._sim()
        e = sim.manifest[0]
        summary = compute_scenario_summary(e, sim.store, sim.timesteps_per_second)
        if summary.time_to_100c_s is None:
            pytest.skip("scenario never reaches 100 C")
        data = sim.store.get(e.case_index, SliceKey("TEMPERATURE"))
        extent = sim.store.get_extent(e.case_index, SliceKey("TEMPERATURE"))
        ins = qe.execute(qe.Query("first_crossing", "TEMPERATURE", 100.0), data, extent,
                         sim.timesteps_per_second)[0]
        assert ins.primary_time() == pytest.approx(summary.time_to_100c_s, abs=0.5)

    def test_hottest_matches_summary_peak(self):
        from slice_key import SliceKey
        from summary_stats import compute_scenario_summary
        sim = self._sim()
        e = sim.manifest[0]
        summary = compute_scenario_summary(e, sim.store, sim.timesteps_per_second)
        data = sim.store.get(e.case_index, SliceKey("TEMPERATURE"))
        extent = sim.store.get_extent(e.case_index, SliceKey("TEMPERATURE"))
        ins = qe.execute(qe.Query("extreme", "TEMPERATURE"), data, extent,
                         sim.timesteps_per_second)[0]
        assert ins.value == pytest.approx(summary.max_temp_c, abs=0.5)
