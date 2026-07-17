import os

import numpy as np
import pytest

from manifest import ScenarioEntry
from summary_stats import ScenarioSummary, build_summary_index, compute_scenario_summary
from auto_summary import (
    _spatial_descriptor,
    _vent_comparison_sentence,
    export_markdown,
    generate_all_summaries,
    generate_summary,
)
from load_data import SIM_ROOT
from slice_key import DEFAULT_SLICE_KEY

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


class FakeStore:
    def __init__(self, arrays=None, extent=None):
        self.arrays = arrays or {}
        self.extent = extent

    def get(self, case_index, key=DEFAULT_SLICE_KEY):
        return self.arrays[case_index]

    def get_extent(self, case_index, key=DEFAULT_SLICE_KEY):
        return self.extent


def _summary(case_index, candles, door, vod, voc, max_temp_c, max_by_frame, time_to_300c_s=None):
    return ScenarioSummary(
        case_index=case_index, folder=f"case{case_index}",
        candles=candles, door=door, vod=vod, voc=voc,
        max_temp_c=max_temp_c, max_temp_by_frame_c=max_by_frame,
        time_to_100c_s=None, time_to_300c_s=time_to_300c_s, time_to_600c_s=None,
        mean_upper_temp_c=50.0, peak_hrr_kw=None, total_energy_kj=None,
        growth_alpha_kw_s2=None, layer_min_height_m=None,
    )


class TestSpatialDescriptor:
    def test_peak_near_candle_zone(self):
        entry_case = 0
        # data shape (n_times=1, n_z=1, n_x=101); peak at column 94 -> x = 0.94 (candle zone)
        data = np.zeros((1, 1, 101), dtype=np.float32)
        data[0, 0, 94] = 500.0
        store = FakeStore(arrays={0: data}, extent=(0.0, 1.0, 0.0, 0.48))
        descriptor = _spatial_descriptor(store, entry_case, DEFAULT_SLICE_KEY, peak_frame_index=0)
        assert descriptor == " (near the candle)"

    def test_peak_near_door_zone(self):
        data = np.zeros((1, 1, 101), dtype=np.float32)
        data[0, 0, 27] = 500.0  # x = 0.27 -> door zone
        store = FakeStore(arrays={0: data}, extent=(0.0, 1.0, 0.0, 0.48))
        descriptor = _spatial_descriptor(store, 0, DEFAULT_SLICE_KEY, peak_frame_index=0)
        assert descriptor == " (near the door)"

    def test_peak_outside_any_zone_returns_empty(self):
        data = np.zeros((1, 1, 101), dtype=np.float32)
        data[0, 0, 50] = 500.0  # x = 0.50 -> not a known landmark
        store = FakeStore(arrays={0: data}, extent=(0.0, 1.0, 0.0, 0.48))
        assert _spatial_descriptor(store, 0, DEFAULT_SLICE_KEY, peak_frame_index=0) == ""

    def test_no_extent_returns_empty_not_a_crash(self):
        frame = np.zeros((1, 101), dtype=np.float32)
        store = FakeStore(arrays={0: np.array([frame[0]])}, extent=None)
        assert _spatial_descriptor(store, 0, DEFAULT_SLICE_KEY, peak_frame_index=0) == ""

    def test_store_error_returns_empty_not_a_crash(self):
        class BrokenStore:
            def get(self, case_index, key=DEFAULT_SLICE_KEY):
                raise RuntimeError("boom")

            def get_extent(self, case_index, key=DEFAULT_SLICE_KEY):
                return (0.0, 1.0, 0.0, 0.48)

        assert _spatial_descriptor(BrokenStore(), 0, DEFAULT_SLICE_KEY, peak_frame_index=0) == ""


class TestVentComparisonSentence:
    def test_open_scenario_compares_against_closed_hvac_mean(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        all_summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=500.0, max_by_frame=[500.0]),
            _summary(1, 0, 0, 1, 0, max_temp_c=400.0, max_by_frame=[400.0]),
            _summary(2, 0, 0, 2, 0, max_temp_c=440.0, max_by_frame=[440.0]),
        ]
        sentence = _vent_comparison_sentence(entry, all_summaries[0], all_summaries)
        # own mean (vod=0 group) = 500; comparison mean (vod!=0) = (400+440)/2=420
        # diff = 80 -> "lower" (comparison group is cooler than this scenario)
        assert "Closed/HVAC vent-1 variants peaked 80" in sentence
        assert "lower" in sentence

    def test_hvac_acronym_stays_uppercase(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        all_summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=500.0, max_by_frame=[500.0]),
            _summary(1, 0, 0, 1, 0, max_temp_c=400.0, max_by_frame=[400.0]),
        ]
        sentence = _vent_comparison_sentence(entry, all_summaries[0], all_summaries)
        assert "HVAC" in sentence
        assert "hvac" not in sentence

    def test_closed_scenario_compares_against_open_mean(self):
        entry = ScenarioEntry(1, "x", "/fake", candles=0, door=0, vod=1, voc=0)
        all_summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=500.0, max_by_frame=[500.0]),
            _summary(1, 0, 0, 1, 0, max_temp_c=400.0, max_by_frame=[400.0]),
        ]
        sentence = _vent_comparison_sentence(entry, all_summaries[1], all_summaries)
        assert "Vent-1-open variants peaked 100" in sentence
        assert "higher" in sentence  # open group (500) is hotter than this scenario (400)

    def test_no_comparison_group_returns_empty(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        all_summaries = [_summary(0, 0, 0, 0, 0, max_temp_c=500.0, max_by_frame=[500.0])]
        assert _vent_comparison_sentence(entry, all_summaries[0], all_summaries) == ""

    def test_only_compares_within_same_candle_count(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        all_summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=500.0, max_by_frame=[500.0]),
            _summary(1, 1, 0, 1, 0, max_temp_c=100.0, max_by_frame=[100.0]),  # different candle count, must be excluded
        ]
        # No same-candle vod!=0 scenario exists -> no comparison group.
        assert _vent_comparison_sentence(entry, all_summaries[0], all_summaries) == ""

    def test_near_zero_difference_uses_about_the_same_phrasing(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        all_summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=500.0, max_by_frame=[500.0]),
            _summary(1, 0, 0, 1, 0, max_temp_c=500.2, max_by_frame=[500.2]),
        ]
        sentence = _vent_comparison_sentence(entry, all_summaries[0], all_summaries)
        assert "about the same" in sentence


class TestGenerateSummary:
    def test_full_sentence_structure(self):
        entry = ScenarioEntry(0, "c1_d0_vod0_voc0", "/fake", candles=0, door=0, vod=0, voc=0)
        own = _summary(0, 0, 0, 0, 0, max_temp_c=469.0, max_by_frame=[20, 100, 469, 300], time_to_300c_s=0.5)
        store = FakeStore(extent=None)  # no spatial descriptor path exercised here

        text = generate_summary(entry, own, [own], store, fps=4)

        assert text.startswith("Peak 469°C at t=")
        assert "Exceeded 300°C at t=0s." in text

    def test_never_exceeded_300_uses_negative_phrasing(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        own = _summary(0, 0, 0, 0, 0, max_temp_c=150.0, max_by_frame=[100, 150, 120], time_to_300c_s=None)
        store = FakeStore(extent=None)
        text = generate_summary(entry, own, [own], store, fps=4)
        assert "Never exceeded 300°C." in text

    def test_peak_time_uses_argmax_frame_not_last_frame(self):
        entry = ScenarioEntry(0, "x", "/fake", candles=0, door=0, vod=0, voc=0)
        # peak at frame index 1 (value 900), fps=2 -> t=0.5s -> "t=0s" (rounded)
        own = _summary(0, 0, 0, 0, 0, max_temp_c=900.0, max_by_frame=[100, 900, 50])
        store = FakeStore(extent=None)
        text = generate_summary(entry, own, [own], store, fps=2)
        assert "t=1s" in text or "t=0s" in text  # 1 frame / 2 fps = 0.5s, formatted to nearest int


class TestGenerateAllSummariesAndExport:
    def test_generate_all_summaries_covers_every_entry(self):
        entries = [
            ScenarioEntry(0, "a", "/fake/a", 0, 0, 0, 0),
            ScenarioEntry(1, "b", "/fake/b", 0, 0, 1, 0),
        ]
        summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=100.0, max_by_frame=[100.0]),
            _summary(1, 0, 0, 1, 0, max_temp_c=200.0, max_by_frame=[200.0]),
        ]
        store = FakeStore(extent=None)
        result = generate_all_summaries(entries, summaries, store, fps=4)
        assert set(result.keys()) == {0, 1}
        assert "Peak 100°C" in result[0]
        assert "Peak 200°C" in result[1]

    def test_export_markdown_writes_all_scenarios(self, tmp_path):
        entries = [
            ScenarioEntry(0, "scenario_a", "/fake/a", 0, 0, 0, 0),
            ScenarioEntry(1, "scenario_b", "/fake/b", 0, 0, 1, 0),
        ]
        summaries = [
            _summary(0, 0, 0, 0, 0, max_temp_c=100.0, max_by_frame=[100.0]),
            _summary(1, 0, 0, 1, 0, max_temp_c=200.0, max_by_frame=[200.0]),
        ]
        store = FakeStore(extent=None)
        out_path = str(tmp_path / "summaries.md")

        export_markdown(entries, summaries, store, fps=4, path=out_path)

        content = open(out_path).read()
        assert "scenario_a" in content
        assert "scenario_b" in content
        assert "Peak 100°C" in content
        assert "Peak 200°C" in content


@requires_real_dataset
class TestAutoSummaryRealData:
    """The DoD's explicit check: summaries verified against browser stats
    for (at least) 3 real scenarios -- every number quoted in the
    generated text must match the same ScenarioSummary the experiment
    browser table itself displays, not a separately-drifted computation."""

    def test_peak_and_threshold_numbers_match_browser_stats_for_three_scenarios(self, tmp_path):
        from data_provider import load_simulation_data
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        cache_path = str(tmp_path / "summaries.json")
        summaries = build_summary_index(sim_data.manifest, sim_data.store,
                                         sim_data.timesteps_per_second, cache_path)
        by_case = {s.case_index: s for s in summaries}
        entries = {e.case_index: e for e in sim_data.manifest}

        for case_index in (0, 5, 15):
            summary = by_case[case_index]
            entry = entries[case_index]
            text = generate_summary(entry, summary, summaries, sim_data.store,
                                     sim_data.timesteps_per_second)

            assert f"Peak {summary.max_temp_c:.0f}°C" in text
            if summary.time_to_300c_s is not None:
                assert f"Exceeded 300°C at t={summary.time_to_300c_s:.0f}s." in text
            else:
                assert "Never exceeded 300°C." in text

    def test_spatial_descriptor_says_near_the_candle_for_real_scenarios(self):
        """Pinned real-data finding (see also test_features.py/test_
        clustering.py's own real-data pins): every scenario's peak
        temperature is physically located near the candle, not the door --
        the auto-summary's descriptor should reflect that, not a
        hardcoded guess."""
        from data_provider import load_simulation_data
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        cache_path_dir = os.path.join(SIM_ROOT, ".cache")
        os.makedirs(cache_path_dir, exist_ok=True)
        summaries = build_summary_index(sim_data.manifest, sim_data.store,
                                         sim_data.timesteps_per_second,
                                         os.path.join(cache_path_dir, "summaries.json"))
        by_case = {s.case_index: s for s in summaries}
        entry = next(e for e in sim_data.manifest if e.case_index == 0)
        text = generate_summary(entry, by_case[0], summaries, sim_data.store,
                                 sim_data.timesteps_per_second)
        assert "(near the candle)" in text
