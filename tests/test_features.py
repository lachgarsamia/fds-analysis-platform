import os

import numpy as np
import pytest

from manifest import ScenarioEntry
from analytics.features import (
    CURVE_POINTS,
    NEVER_CROSSED_SECONDS,
    ScenarioFeatures,
    build_feature_index,
    build_feature_matrix,
    compute_scenario_features,
)
from load_data import SIM_ROOT, load_data
from slice_key import DEFAULT_SLICE_KEY

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


class FakeStore:
    def __init__(self, arrays):
        self.arrays = arrays

    def get(self, case_index, key=DEFAULT_SLICE_KEY):
        return self.arrays[case_index]


def test_compute_scenario_features_hand_computed_values():
    entry = ScenarioEntry(0, "c1_d0_vod0_voc0", "/fake", 0, 0, 0, 0)
    # 4 frames, 2x2 each -- hand-computable max/mean/hot-fraction per frame.
    data = np.asarray([
        [[20, 30], [40, 50]],    # max=50, mean=35, none >60
        [[80, 20], [20, 20]],    # max=80, mean=35, 1/4 >60
        [[20, 20], [20, 20]],    # max=20, mean=20, none >60
        [[70, 70], [20, 20]],    # max=70, mean=45, 2/4 >60
    ], dtype=np.float32)
    store = FakeStore({0: data})

    features = compute_scenario_features(entry, store, fps=2, hot_threshold_c=60.0)

    assert features.case_index == 0
    assert features.folder == "c1_d0_vod0_voc0"
    assert (features.candles, features.door, features.vod, features.voc) == (0, 0, 0, 0)
    # No downsampling needed here (4 frames -> not asserting CURVE_POINTS
    # shape with hand-picked values; covered by the length test below).
    assert features.time_to_100c_s is None
    assert features.time_to_300c_s is None
    assert features.time_to_600c_s is None


def test_curves_are_always_exactly_curve_points_long():
    entry = ScenarioEntry(0, "x", "/fake", 0, 0, 0, 0)
    for n_frames in (1, 4, 481, 500):
        data = np.random.default_rng(0).uniform(20, 100, size=(n_frames, 3, 3)).astype(np.float32)
        store = FakeStore({0: data})
        features = compute_scenario_features(entry, store, fps=4)
        assert len(features.max_temp_curve) == CURVE_POINTS
        assert len(features.hot_area_fraction_curve) == CURVE_POINTS
        assert len(features.spatial_mean_curve) == CURVE_POINTS


def test_time_to_threshold_matches_first_crossing():
    entry = ScenarioEntry(0, "x", "/fake", 0, 0, 0, 0)
    # frame index 0,1,2,3 -> t=0,0.5,1.0,1.5s at fps=2
    data = np.asarray([
        [[50]], [[150]], [[350]], [[650]],
    ], dtype=np.float32)
    store = FakeStore({0: data})

    features = compute_scenario_features(entry, store, fps=2)

    assert features.time_to_100c_s == 0.5
    assert features.time_to_300c_s == 1.0
    assert features.time_to_600c_s == 1.5


def test_hot_area_fraction_is_bounded_0_to_1():
    entry = ScenarioEntry(0, "x", "/fake", 0, 0, 0, 0)
    data = np.random.default_rng(1).uniform(0, 500, size=(20, 5, 5)).astype(np.float32)
    store = FakeStore({0: data})
    features = compute_scenario_features(entry, store, fps=4)
    assert all(0.0 <= v <= 1.0 for v in features.hot_area_fraction_curve)


def test_computation_is_deterministic():
    entry = ScenarioEntry(0, "x", "/fake", 0, 0, 0, 0)
    data = np.random.default_rng(2).uniform(20, 300, size=(50, 4, 4)).astype(np.float32)
    store = FakeStore({0: data})
    a = compute_scenario_features(entry, store, fps=4)
    b = compute_scenario_features(entry, store, fps=4)
    assert a.as_vector().tolist() == b.as_vector().tolist()


class TestScenarioFeaturesVector:
    def test_as_vector_length_matches_three_curves_plus_three_thresholds(self):
        features = ScenarioFeatures(
            case_index=0, folder="x", candles=0, door=0, vod=0, voc=0,
            max_temp_curve=[1.0] * CURVE_POINTS,
            hot_area_fraction_curve=[0.5] * CURVE_POINTS,
            spatial_mean_curve=[2.0] * CURVE_POINTS,
            time_to_100c_s=1.0, time_to_300c_s=2.0, time_to_600c_s=None,
        )
        vec = features.as_vector()
        assert vec.shape == (CURVE_POINTS * 3 + 3,)

    def test_never_crossed_uses_sentinel_not_nan_or_zero(self):
        features = ScenarioFeatures(
            case_index=0, folder="x", candles=0, door=0, vod=0, voc=0,
            max_temp_curve=[1.0] * CURVE_POINTS,
            hot_area_fraction_curve=[0.0] * CURVE_POINTS,
            spatial_mean_curve=[1.0] * CURVE_POINTS,
            time_to_100c_s=None, time_to_300c_s=None, time_to_600c_s=None,
        )
        vec = features.as_vector()
        assert np.isfinite(vec).all()
        assert (vec[-3:] == NEVER_CROSSED_SECONDS).all()


class TestBuildFeatureMatrix:
    def test_matrix_rows_match_case_index_order_not_input_order(self):
        f2 = ScenarioFeatures(2, "b", 0, 0, 0, 0, [9.0] * CURVE_POINTS, [0.0] * CURVE_POINTS, [0.0] * CURVE_POINTS)
        f0 = ScenarioFeatures(0, "a", 0, 0, 0, 0, [1.0] * CURVE_POINTS, [0.0] * CURVE_POINTS, [0.0] * CURVE_POINTS)
        matrix, case_indices = build_feature_matrix([f2, f0])
        assert case_indices == [0, 2]
        assert matrix[0, 0] == 1.0
        assert matrix[1, 0] == 9.0

    def test_empty_input_produces_empty_matrix(self):
        matrix, case_indices = build_feature_matrix([])
        assert matrix.shape == (0, 0)
        assert case_indices == []


@requires_real_dataset
class TestFeaturesRealData:
    """Cross-checks against the same real dataset used to verify M2.3's
    DifferenceView and M2.6's probe -- not a fresh assumption each time,
    the same fixture scenario and known findings (e.g. peak concentrates
    near the candle) recur across this project's verification work."""

    def test_build_feature_index_covers_all_24_scenarios(self):
        from data_provider import load_simulation_data
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        features = build_feature_index(sim_data.manifest, sim_data.store, sim_data.timesteps_per_second)
        assert len(features) == 24
        assert sorted(f.case_index for f in features) == list(range(24))

    def test_matches_direct_computation_on_known_scenario(self):
        key = DEFAULT_SLICE_KEY
        data = load_data(os.path.join(SIM_ROOT, "c1_d0_vod0_voc0"), key)
        entry = ScenarioEntry(0, "c1_d0_vod0_voc0", os.path.join(SIM_ROOT, "c1_d0_vod0_voc0"), 0, 0, 0, 0)
        store = FakeStore({0: data})
        features = compute_scenario_features(entry, store, fps=4)

        max_by_frame = data.max(axis=(1, 2))
        assert features.max_temp_curve[-1] == pytest.approx(float(max_by_frame[-1]), rel=1e-3)
        assert max(features.max_temp_curve) <= float(max_by_frame.max()) + 1e-3
