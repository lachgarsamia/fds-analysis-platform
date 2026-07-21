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


import state_space as ss  # noqa: E402


class TestStateSpace:
    def _evolving(self):
        # 20 frames, a hot spot that grows/moves smoothly -> smooth descriptors.
        d = np.full((20, 5, 8), 20.0, dtype=np.float32)
        for t in range(20):
            d[t, :, :t // 2 + 1] = 20 + 15 * t  # hot region widens with time
        return d

    def test_trajectory_shape_and_temporal_smoothness(self):
        coords, times, evr = ss.scenario_trajectory(self._evolving(), (0, 1, 0, 0.3), 4)
        assert coords.shape == (20, 2) and times.shape == (20,)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        # a smooth evolution -> consecutive frames much closer than the span
        span = np.linalg.norm(coords.max(0) - coords.min(0))
        assert steps.mean() < 0.5 * span

    def test_genome_traits_peak_and_rate(self):
        d = np.full((4, 1, 2), 20.0, dtype=np.float32)
        d[:, 0, 0] = [20, 120, 320, 320]
        g = ss.genome_traits(d, (0, 1, 0, 0.3), 2)
        assert g["peak_temp"] == pytest.approx(320.0)
        assert g["heating_rate"] == pytest.approx(400.0)  # max diff 200 * fps 2

    def test_normalize_genomes_min_max_and_constant(self):
        traits = [
            {"peak_temp": 100, "heating_rate": 5, "smoke_descent": 0, "energy": 1, "spread": 2},
            {"peak_temp": 300, "heating_rate": 5, "smoke_descent": 1, "energy": 3, "spread": 4},
        ]
        norm = ss.normalize_genomes(traits)
        assert norm[0]["peak_temp"] == 0.0 and norm[1]["peak_temp"] == 1.0
        assert norm[0]["heating_rate"] == 0.5 and norm[1]["heating_rate"] == 0.5  # constant

    def test_genome_matrix_shape(self):
        norm = ss.normalize_genomes([
            {"peak_temp": 1, "heating_rate": 1, "smoke_descent": 1, "energy": 1, "spread": 1},
            {"peak_temp": 2, "heating_rate": 2, "smoke_descent": 2, "energy": 2, "spread": 2},
        ])
        m = ss.genome_matrix(norm)
        assert m.shape == (2, len(ss.GENOME_TRAITS))


@requires_real_dataset
class TestStateSpaceRealData:
    """DoD: trajectory temporally ordered; genomes cluster with candle count."""

    def _sim(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        return sim

    def test_trajectory_temporally_ordered(self):
        sim = self._sim()
        e = sim.manifest[0]
        data = sim.store.get(e.case_index, DEFAULT_SLICE_KEY)
        extent = sim.store.get_extent(e.case_index, DEFAULT_SLICE_KEY)
        coords, _t, _evr = ss.scenario_trajectory(data, extent, sim.timesteps_per_second)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        rng = np.random.default_rng(0)
        i, j = rng.integers(0, len(coords), 500), rng.integers(0, len(coords), 500)
        rand = np.linalg.norm(coords[i] - coords[j], axis=1)
        assert steps.mean() < 0.6 * rand.mean()

    def test_genomes_cluster_with_candle_count(self):
        from summary_stats import compute_scenario_summary
        from analytics.clustering import run_clustering, cluster_alignment
        sim = self._sim()
        raw = [ss.genome_traits(sim.store.get(e.case_index, DEFAULT_SLICE_KEY),
                                sim.store.get_extent(e.case_index, DEFAULT_SLICE_KEY),
                                sim.timesteps_per_second,
                                compute_scenario_summary(e, sim.store, sim.timesteps_per_second))
               for e in sim.manifest]
        mat = ss.genome_matrix(ss.normalize_genomes(raw))
        labels = run_clustering(mat, 2)
        align = cluster_alignment(labels, [e.candles for e in sim.manifest])
        assert align >= 0.5  # at least as good as chance; consistent with the PCA finding


import attention as at  # noqa: E402


class TestAttention:
    def test_static_field_is_near_empty(self):
        static = np.full((4, 5, 5), 100.0, dtype=np.float32)  # nothing changes
        sal = at.attention_series(static, fps=4)
        assert sal.shape == (4, 5, 5)
        assert float(sal.max()) == 0.0  # stable -> near-empty

    def test_active_region_glows_stable_stays_dark(self):
        d = np.full((3, 5, 5), 20.0, dtype=np.float32)
        d[1, 2, 2] = 300.0            # a spike in the middle at frame 1
        sal = at.attention_series(d, fps=4)
        assert 0.0 <= sal.min() and sal.max() == pytest.approx(1.0)
        # the spike's neighbourhood is far brighter than a quiet corner
        assert sal[1][1:4, 1:4].max() > sal[1, 0, 0]

    def test_velocity_and_hrr_cues_optional(self):
        temp = np.random.default_rng(0).random((4, 4, 4)).astype(np.float32) * 100
        base = at.attention_series(temp, fps=4)
        vel = np.random.default_rng(1).random((4, 4, 4)).astype(np.float32)
        hrr = np.array([0.0, 1.0, 3.0, 2.0])
        withcues = at.attention_series(temp, vel, hrr, fps=4)
        assert base.shape == withcues.shape  # cues add signal, don't change shape
        assert float(withcues.max()) == pytest.approx(1.0)


@requires_real_dataset
class TestAttentionRealData:
    """DoD: the source/plume region dominates the attention over the run."""

    def test_most_active_region_is_the_source(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        e = sim.manifest[0]
        temp = np.asarray(sim.store.get(e.case_index, DEFAULT_SLICE_KEY))
        extent = sim.store.get_extent(e.case_index, DEFAULT_SLICE_KEY)
        sal = at.attention_series(temp, fps=sim.timesteps_per_second)
        mean_sal = sal.mean(axis=0)
        n_z, n_x = mean_sal.shape
        row, col = np.unravel_index(int(np.argmax(mean_sal)), mean_sal.shape)
        x = extent[0] + col / (n_x - 1) * (extent[1] - extent[0])
        assert x > 0.70  # the candle/plume region, not the cold left side


import cause_explorer as ce  # noqa: E402


class TestCauseExplorer:
    def _sourced_field(self):
        # A single hot source at (0,0); temperature falls off with distance.
        n = 5
        f = np.zeros((n, n), dtype=np.float32)
        for r in range(n):
            for c in range(n):
                f[r, c] = max(20.0, 300.0 - 20.0 * np.hypot(r, c))
        return f

    def test_trace_is_monotonic_and_ends_at_source(self):
        f = self._sourced_field()
        path = ce.trace_to_source(f, 4, 4)
        temps = [f[r, c] for r, c in path]
        assert all(temps[k + 1] >= temps[k] for k in range(len(temps) - 1))
        assert path[-1] == (0, 0)  # the hottest cell / source

    def test_explain_chain_reaches_source_and_labels_association(self):
        f = self._sourced_field()
        insights, path = ce.explain(f, (0.0, 1.0, 0.0, 1.0), time_s=5.0, row=4, col=4)
        assert len(insights) >= 2
        assert "source" in insights[-1].statement.lower()
        # honesty gate: the tracing is labelled association, not causation
        assert any("association" in i.basis.lower() and "not proven causation" in i.basis.lower()
                   for i in insights)
        # navigable: the last step points at the source location
        assert insights[-1].location is not None

    def test_cold_cell_has_no_source_to_trace(self):
        f = np.full((4, 4), 20.0, dtype=np.float32)
        insights, path = ce.explain(f, (0, 1, 0, 1), 0.0, 3, 3)
        assert len(insights) == 1 and "near ambient" in insights[0].statement
        assert len(path) == 1

    def test_local_maximum_is_reported_as_a_source(self):
        f = self._sourced_field()
        insights, _p = ce.explain(f, (0, 1, 0, 1), 0.0, 0, 0)  # pick the source itself
        assert "itself the hottest" in insights[-1].statement.lower()


@requires_real_dataset
class TestCauseRealData:
    """DoD: a hot plume cell traces back toward the fire source; the
    tracing is gated as association, not proven causation."""

    def test_hot_cell_traces_to_the_candle_source(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        e = sim.manifest[0]
        data = np.asarray(sim.store.get(e.case_index, DEFAULT_SLICE_KEY))
        extent = sim.store.get_extent(e.case_index, DEFAULT_SLICE_KEY)
        fi = int(data.shape[0] * 0.6)
        frame = data[fi]
        gr, gc = np.unravel_index(int(np.argmax(frame)), frame.shape)
        # pick a hot (>100 C) cell that is not the global maximum
        hot = np.argwhere((frame > 100) & ~((np.arange(frame.shape[0])[:, None] == gr)
                                             & (np.arange(frame.shape[1])[None, :] == gc)))
        if hot.size == 0:
            pytest.skip("no secondary hot cell to trace")
        pr, pc = hot[np.argmin(hot[:, 0])]
        insights, path = ce.explain(frame, extent, fi / sim.timesteps_per_second, int(pr), int(pc))
        # traces back to the source, near the candle band (x > 0.7)
        src = insights[-1].location
        assert src is not None and src[0] > 0.70
        assert any("not proven causation" in i.basis.lower() for i in insights)


import height_analysis as haz  # noqa: E402


class TestHeightAnalysis:
    EXT = (0.0, 1.0, 0.0, 0.4)  # z 0..0.4

    def test_column_for_x(self):
        assert haz.column_for_x(self.EXT, 11, 0.0) == 0
        assert haz.column_for_x(self.EXT, 11, 1.0) == 10
        assert haz.column_for_x(self.EXT, 11, 0.5) == 5

    def test_vertical_profile_is_floor_first(self):
        # frame (n_z=5, n_x=3), row 0 = ceiling. Column 0 = [100,80,60,40,20]
        # from ceiling to floor -> profile (floor-first) = [20,40,60,80,100].
        frame = np.array([[100, 0, 0], [80, 0, 0], [60, 0, 0], [40, 0, 0], [20, 0, 0]],
                         dtype=np.float32)
        zs, vals = haz.vertical_profile(frame, self.EXT, 0)
        assert zs[0] == pytest.approx(0.0) and zs[-1] == pytest.approx(0.4)  # floor -> ceiling
        np.testing.assert_allclose(vals, [20, 40, 60, 80, 100])  # cool floor, hot ceiling

    def test_plume_height_tracks_highest_hot_cell(self):
        # 2 frames, row 0 = ceiling (z=0.4). Frame 0: hot only near floor
        # (row 4); Frame 1: hot up to row 1 (near ceiling).
        d = np.full((2, 5, 2), 20.0, dtype=np.float32)
        d[0, 4, 0] = 200  # floor (z=0.0)
        d[1, 1, 0] = 200  # near ceiling (z=0.3)
        h = haz.plume_height_series(d, self.EXT, 100.0)
        assert h[0] == pytest.approx(0.0)   # only the floor cell is hot
        assert h[1] == pytest.approx(0.3)   # plume rose to row 1 -> z=0.3

    def test_ceiling_jet_is_near_ceiling_max(self):
        d = np.full((2, 10, 3), 20.0, dtype=np.float32)
        d[0, 0, 0] = 300   # ceiling row, frame 0
        d[1, 9, 0] = 500   # floor row (not the ceiling band), frame 1
        jet = haz.ceiling_jet_series(d, band_frac=0.2)  # top 2 rows
        assert jet[0] == pytest.approx(300.0)
        assert jet[1] == pytest.approx(20.0)  # the 500 is at the floor, ignored


@requires_real_dataset
class TestHeightRealData:
    def test_gas_layer_stratifies_hot_over_cool(self):
        from data_provider import load_simulation_data
        from slice_key import SliceKey
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        e = sim.manifest[0]
        data = np.asarray(sim.store.get(e.case_index, SliceKey("TEMPERATURE")))
        extent = sim.store.get_extent(e.case_index, SliceKey("TEMPERATURE"))
        fi = int(data.shape[0] * 0.6)
        # away from the flame column, the upper gas is warmer than the floor
        col = haz.column_for_x(extent, data.shape[2], 0.5)
        zs, vals = haz.vertical_profile(data[fi], extent, col)
        assert vals[-1] >= vals[0]  # ceiling >= floor (stratification)

    def test_plume_and_ceiling_series_are_finite_and_sane(self):
        from data_provider import load_simulation_data
        from slice_key import SliceKey
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        e = sim.manifest[0]
        data = np.asarray(sim.store.get(e.case_index, SliceKey("TEMPERATURE")))
        extent = sim.store.get_extent(e.case_index, SliceKey("TEMPERATURE"))
        plume = haz.plume_height_series(data, extent, 60.0)
        jet = haz.ceiling_jet_series(data)
        assert np.all(np.isfinite(plume)) and plume.max() <= extent[3] + 1e-6
        assert np.all(np.isfinite(jet)) and jet.max() >= 20.0


import evidence_notebook as enb  # noqa: E402
from insight import Insight as _Insight  # noqa: E402


class TestEvidenceNotebook:
    def _ins(self, s="Peak temperature is 320 C."):
        return _Insight(s, category="query", quantity="TEMPERATURE",
                        time_s=42.0, value=320.0, unit="C", basis="global max")

    def test_add_remove_clear(self):
        nb = enb.EvidenceNotebook()
        assert nb.is_empty()
        nb.add(self._ins()); nb.add(self._ins("second"))
        assert len(nb) == 2 and not nb.is_empty()
        nb.remove(0)
        assert len(nb) == 1 and nb.entries[0].insight.statement == "second"
        nb.clear()
        assert nb.is_empty()

    def test_move_reorders_and_clamps(self):
        nb = enb.EvidenceNotebook()
        nb.add(self._ins("a")); nb.add(self._ins("b")); nb.add(self._ins("c"))
        assert nb.move(0, 1) == 1  # a<->b
        assert [e.insight.statement for e in nb.entries] == ["b", "a", "c"]
        assert nb.move(0, -1) == 0  # already at top: no-op
        assert nb.move(2, 1) == 2   # already at bottom: no-op

    def test_note_and_tags_are_cleaned(self):
        nb = enb.EvidenceNotebook()
        nb.add(self._ins())
        nb.set_note(0, "growth phase")
        nb.set_tags(0, [" plume ", "", "  ", "caseA"])
        assert nb.entries[0].note == "growth phase"
        assert nb.entries[0].tags == ["plume", "caseA"]  # blanks dropped, stripped

    def test_serialization_roundtrip_preserves_everything(self):
        nb = enb.EvidenceNotebook()
        nb.add(_Insight("Layer descends during (12, 40) s.", category="event",
                        quantity="TEMPERATURE", time_s=(12.0, 40.0),
                        location=(0.9, 0.1), region=(0.0, 1.0, 0.0, 0.4),
                        value=0.2, unit="m", basis="interval stat"),
               note="door open", tags=["ventilation"])
        back = enb.EvidenceNotebook.from_list(nb.to_list())
        e = back.entries[0]
        assert e.note == "door open" and e.tags == ["ventilation"]
        assert e.insight.time_s == (12.0, 40.0)  # tuple survives JSON list round-trip
        assert e.insight.location == (0.9, 0.1)
        assert e.insight.region == (0.0, 1.0, 0.0, 0.4)
        assert e.insight.primary_time() == 12.0

    def test_from_list_tolerates_none_and_junk(self):
        assert enb.EvidenceNotebook.from_list(None).is_empty()
        nb = enb.EvidenceNotebook.from_list([{"insight": {"statement": "x"}}, "bad"])
        assert len(nb) == 1 and nb.entries[0].insight.statement == "x"


import linked_inspection as lki  # noqa: E402


class TestLinkedInspection:
    def test_value_at_time_interpolates_and_clamps(self):
        times = np.array([0.0, 1.0, 2.0])
        vals = np.array([10.0, 20.0, 40.0])
        assert lki.value_at_time(times, vals, 0.5) == pytest.approx(15.0)  # interpolate
        assert lki.value_at_time(times, vals, 1.5) == pytest.approx(30.0)
        assert lki.value_at_time(times, vals, -5.0) == pytest.approx(10.0)  # clamp low
        assert lki.value_at_time(times, vals, 99.0) == pytest.approx(40.0)  # clamp high

    def test_value_at_time_empty_is_none(self):
        assert lki.value_at_time([], [], 1.0) is None

    def test_peak_over_time_is_per_frame_spatial_max(self):
        d = np.zeros((3, 4, 5), dtype=np.float32)
        d[0, 1, 1] = 5.0
        d[1, 0, 0] = 12.0
        d[2, 3, 4] = 7.0
        np.testing.assert_allclose(lki.peak_over_time(d), [5.0, 12.0, 7.0])


import zone_stats as zst  # noqa: E402


class TestZoneStats:
    def test_zone_serialization_and_area(self):
        z = zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3)
        assert zst.Zone.from_dict(z.to_dict()) == z
        assert z.area() == pytest.approx(0.2 * 0.3)

    def test_zone_indices_map_both_corners(self):
        # extent (0,1,0,1), 2x2 grid; whole-field zone -> all cells.
        r0, r1, c0, c1 = zst.zone_indices((0.0, 1.0, 0.0, 1.0), (2, 2),
                                          zst.Zone("z", 0.0, 1.0, 0.0, 1.0))
        assert (r0, r1, c0, c1) == (0, 1, 0, 1)

    def test_zone_bundle_hand_computed(self):
        # 4 frames, 2x2, fps=2, threshold=50, ambient=20; uniform per frame.
        data = np.zeros((4, 2, 2), dtype=np.float32)
        data[0] = 10; data[1] = 100; data[2] = 100; data[3] = 10
        zone = zst.Zone("all", 0.0, 1.0, 0.0, 1.0)
        b = zst.zone_bundle(data, (0.0, 1.0, 0.0, 1.0), zone, fps=2,
                            threshold=50.0, ambient=20.0)
        assert b["n_cells"] == 4
        assert b["mean_temperature"] == pytest.approx(55.0)   # 880/16
        assert b["max_temperature"] == pytest.approx(100.0)
        assert b["time_to_threshold"] == pytest.approx(0.5)   # first >50 at frame 1 / fps
        assert b["hazard_duration"] == pytest.approx(1.0)     # 2 frames over / fps
        assert b["peak_affected_fraction"] == pytest.approx(1.0)
        assert b["thermal_dose"] == pytest.approx(80.0)       # sum(clip(mean-20))/fps
        assert b["energy_proxy"] == pytest.approx(80.0)       # dose * area(=1)
        np.testing.assert_allclose(b["dose_curve"], [0.0, 40.0, 80.0, 80.0])

    def test_time_to_threshold_none_when_never_reached(self):
        data = np.full((3, 2, 2), 30.0, dtype=np.float32)
        b = zst.zone_bundle(data, (0.0, 1.0, 0.0, 1.0),
                            zst.Zone("all", 0.0, 1.0, 0.0, 1.0), 2, 50.0, 20.0)
        assert b["time_to_threshold"] is None
        assert b["hazard_duration"] == 0.0

    def test_smoke_accumulation_is_time_integral_of_mean(self):
        soot = np.full((4, 2, 2), 2.0, dtype=np.float32)
        acc = zst.smoke_accumulation(soot, (0.0, 1.0, 0.0, 1.0),
                                     zst.Zone("all", 0.0, 1.0, 0.0, 1.0), fps=2)
        assert acc == pytest.approx(4.0)  # mean 2 per frame * 4 frames / fps


import time_window as twm  # noqa: E402


class TestTimeWindow:
    T = np.array([0.0, 1.0, 2.0, 3.0])
    MEAN = np.array([10.0, 20.0, 30.0, 40.0])
    MAX = np.array([10.0, 20.0, 30.0, 40.0])

    def test_window_indices_inclusive(self):
        assert twm.window_indices(self.T, 1.0, 3.0) == (1, 3)
        assert twm.window_indices(self.T, 3.0, 1.0) == (1, 3)  # order-independent

    def test_interval_stats_hand_computed(self):
        st = twm.interval_stats(self.MEAN, self.MAX, self.T, 1.0, 3.0)
        assert st["mean"] == pytest.approx(30.0)
        assert st["peak"] == pytest.approx(40.0)
        assert st["integral"] == pytest.approx(60.0)   # trapz of [20,30,40] over [1,2,3]
        assert st["slope"] == pytest.approx(10.0)
        assert st["delta"] == pytest.approx(20.0)
        assert st["n_frames"] == 3

    def test_before_after_split(self):
        before, after = twm.before_after_split(self.MEAN, self.MAX, self.T, 1.5)
        assert (before["t0"], before["t1"]) == (0.0, 1.0)
        assert (after["t0"], after["t1"]) == (2.0, 3.0)
        assert before["mean"] == pytest.approx(15.0)   # [10,20]
        assert after["mean"] == pytest.approx(35.0)    # [30,40]

    def test_phase_windows_with_pre_ignition(self):
        w = twm.phase_windows([(2.0, "B"), (1.0, "A")], t_end=5.0)
        assert w == [("Pre-ignition", 0.0, 1.0), ("A", 1.0, 2.0), ("B", 2.0, 5.0)]

    def test_phase_windows_empty(self):
        assert twm.phase_windows([], t_end=5.0) == []
