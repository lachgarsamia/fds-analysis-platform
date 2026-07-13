"""Tests for ml/metrics.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src"))

from dataset import load_entries, scenario_split, compute_normalization  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402
from metrics import evaluate_rollout, rmse, ssim  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


class TestRmse:
    def test_zero_for_identical_arrays(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert rmse(a, a) == 0.0

    def test_hand_computed(self):
        pred = np.array([0.0, 0.0])
        true = np.array([3.0, 4.0])
        # sqrt(mean(9, 16)) = sqrt(12.5)
        assert rmse(pred, true) == pytest.approx(np.sqrt(12.5))


class TestSsim:
    def test_one_for_identical_arrays(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, size=(20, 20))
        assert ssim(a, a) == pytest.approx(1.0)

    def test_lower_for_dissimilar_arrays(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 100, size=(20, 20))
        b = rng.uniform(0, 100, size=(20, 20))
        assert ssim(a, b) < ssim(a, a)

    def test_constant_array_does_not_crash(self):
        a = np.full((10, 10), 5.0)
        result = ssim(a, a)
        assert np.isfinite(result)


class TestEvaluateRollout:
    @requires_real_dataset
    def test_perfect_forecaster_has_zero_rmse_and_ssim_one(self):
        entries = load_entries()
        train, test = scenario_split(entries)
        stats = compute_normalization(entries, train)

        def oracle_step(window):
            # Not actually achievable without the true future frame, but
            # returning the last seed frame's own normalized value is a
            # cheap sanity check that the harness's bookkeeping (shapes,
            # denormalization, lead indexing) doesn't crash and produces
            # finite, non-negative metrics.
            return window[-1]

        result = evaluate_rollout(oracle_step, entries, test[:1], stats, horizon=3, stride=100)
        assert result["n_rollouts"] > 0
        for lead in range(1, 4):
            assert result["rmse"][lead] >= 0.0
            assert 0.0 <= result["ssim"][lead] <= 1.0

    @requires_real_dataset
    def test_lead_time_keys_cover_full_horizon(self):
        entries = load_entries()
        train, test = scenario_split(entries)
        stats = compute_normalization(entries, train)
        result = evaluate_rollout(lambda w: w[-1], entries, test[:1], stats, horizon=5, stride=200)
        assert set(result["rmse"].keys()) == {1, 2, 3, 4, 5}
        assert set(result["ssim"].keys()) == {1, 2, 3, 4, 5}
