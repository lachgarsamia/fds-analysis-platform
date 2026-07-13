"""Tests for ml/dataset.py. Run separately from the app's own suite
(pytest ml/tests, not `pytest` from the repo root) -- ml/ has its own
dependency boundary (torch, neuraloperator) the app must never require,
and keeping the test entry points separate mirrors that split precisely,
not just the runtime import graph.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src"))

from dataset import (  # noqa: E402
    SPLIT_SEED,
    TEST_SCENARIOS_COUNT,
    VAL_SCENARIOS_COUNT,
    WINDOW_IN,
    build_dataset,
    build_windows,
    compute_normalization,
    config_hash,
    denormalize,
    load_entries,
    normalize,
    scenario_split,
    train_val_split,
)
from load_data import SIM_ROOT

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


class TestBuildWindows:
    def test_window_count_and_shapes(self):
        data = np.arange(20 * 3 * 4, dtype=np.float32).reshape(20, 3, 4)
        inputs, targets = build_windows(data, window_in=8)
        assert inputs.shape == (12, 8, 3, 4)
        assert targets.shape == (12, 3, 4)

    def test_input_target_alignment(self):
        data = np.arange(20 * 2 * 2, dtype=np.float32).reshape(20, 2, 2)
        inputs, targets = build_windows(data, window_in=8)
        # Window 0: input = frames[0:8], target = frame[8]
        assert np.array_equal(inputs[0], data[0:8])
        assert np.array_equal(targets[0], data[8])
        # Window 5: input = frames[5:13], target = frame[13]
        assert np.array_equal(inputs[5], data[5:13])
        assert np.array_equal(targets[5], data[13])

    def test_too_few_frames_returns_empty(self):
        data = np.zeros((5, 3, 3), dtype=np.float32)
        inputs, targets = build_windows(data, window_in=8)
        assert inputs.shape[0] == 0
        assert targets.shape[0] == 0

    def test_exact_minimum_length_produces_one_window(self):
        data = np.arange(9 * 2 * 2, dtype=np.float32).reshape(9, 2, 2)
        inputs, targets = build_windows(data, window_in=8)
        assert inputs.shape[0] == 1


class TestNormalization:
    def test_normalize_denormalize_round_trip(self):
        data = np.random.default_rng(0).uniform(0, 500, size=(10, 5, 5)).astype(np.float32)
        stats = {"mean": 50.0, "std": 20.0}
        normalized = normalize(data, stats)
        recovered = denormalize(normalized, stats)
        assert np.allclose(recovered, data, atol=1e-3)

    def test_normalize_hand_computed(self):
        data = np.array([[0.0, 10.0], [20.0, 30.0]])
        stats = {"mean": 10.0, "std": 10.0}
        result = normalize(data, stats)
        assert np.allclose(result, [[-1.0, 0.0], [1.0, 2.0]])


class TestConfigHash:
    def test_deterministic(self):
        config = {"lr": 1e-3, "hidden_channels": 32, "n_layers": 4}
        assert config_hash(config) == config_hash(dict(config))

    def test_key_order_does_not_matter(self):
        a = config_hash({"lr": 1e-3, "hidden": 32})
        b = config_hash({"hidden": 32, "lr": 1e-3})
        assert a == b

    def test_different_configs_differ(self):
        a = config_hash({"lr": 1e-3})
        b = config_hash({"lr": 1e-4})
        assert a != b

    def test_returns_short_hex_string(self):
        h = config_hash({"x": 1})
        assert len(h) == 12
        int(h, 16)  # must be valid hex


@requires_real_dataset
class TestRealDataset:
    def test_load_entries_returns_24_scenarios(self):
        entries = load_entries()
        assert len(entries) == 24

    def test_scenario_split_is_20_4_no_overlap(self):
        entries = load_entries()
        train, test = scenario_split(entries)
        assert len(train) == 20
        assert len(test) == TEST_SCENARIOS_COUNT
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(24))

    def test_scenario_split_is_deterministic(self):
        entries = load_entries()
        train1, test1 = scenario_split(entries)
        train2, test2 = scenario_split(entries)
        assert train1 == train2
        assert test1 == test2

    def test_train_val_split_carves_from_train_only(self):
        entries = load_entries()
        train, test = scenario_split(entries)
        fit, val = train_val_split(train)
        assert len(val) == VAL_SCENARIOS_COUNT
        assert len(fit) == len(train) - VAL_SCENARIOS_COUNT
        assert set(fit) | set(val) == set(train)
        assert set(val).isdisjoint(test), "validation scenarios must never be test scenarios"

    def test_normalization_stats_are_physically_plausible(self):
        entries = load_entries()
        train, _test = scenario_split(entries)
        fit, _val = train_val_split(train)
        stats = compute_normalization(entries, fit)
        # Ambient is 20C (config.AMBIENT_C); peaks run a few hundred C
        # (M2.3/M3.1's own findings) -- a sane mean/std should reflect that
        # range, not be some wildly-off number from a units/shape bug.
        assert 15.0 < stats["mean"] < 100.0
        assert 5.0 < stats["std"] < 100.0

    def test_build_dataset_shapes_match_window_math(self):
        entries = load_entries()
        train, _test = scenario_split(entries)
        fit, val = train_val_split(train)
        stats = compute_normalization(entries, fit)
        inputs, targets = build_dataset(entries, val, stats)
        assert inputs.shape[1:] == (WINDOW_IN, 49, 101)
        assert targets.shape[1:] == (49, 101)
        assert inputs.shape[0] == targets.shape[0]
        # 3 val scenarios, each with (n_frames - WINDOW_IN) windows.
        assert inputs.shape[0] > 0

    def test_normalized_dataset_has_near_zero_mean(self):
        entries = load_entries()
        train, _test = scenario_split(entries)
        fit, val = train_val_split(train)
        stats = compute_normalization(entries, fit)
        inputs, _targets = build_dataset(entries, fit, stats)
        # Computed on the SAME scenarios the stats came from -> should be
        # very close to 0, not just "in the right ballpark".
        assert abs(float(inputs.mean())) < 0.1
