"""Tests for ml/baselines.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src"))

from baselines import linear_extrapolation_step, persistence_step, run_baselines  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


class TestPersistenceStep:
    def test_repeats_last_frame(self):
        window = np.arange(8 * 3 * 3, dtype=np.float32).reshape(8, 3, 3)
        result = persistence_step(window)
        assert np.array_equal(result, window[-1])

    def test_does_not_mutate_input(self):
        window = np.arange(8 * 2 * 2, dtype=np.float32).reshape(8, 2, 2)
        original = window.copy()
        result = persistence_step(window)
        result[0, 0] = 999.0
        assert np.array_equal(window, original)


class TestLinearExtrapolationStep:
    def test_hand_computed_constant_trend(self):
        # Last two frames differ by a constant +1 everywhere -> next frame
        # should be last + 1.
        window = np.zeros((8, 2, 2), dtype=np.float32)
        window[-2] = 5.0
        window[-1] = 6.0
        result = linear_extrapolation_step(window)
        assert np.allclose(result, 7.0)

    def test_zero_trend_equals_persistence(self):
        window = np.full((8, 2, 2), 3.0, dtype=np.float32)
        result = linear_extrapolation_step(window)
        assert np.allclose(result, 3.0)

    def test_negative_trend_extrapolates_down(self):
        window = np.zeros((8, 2, 2), dtype=np.float32)
        window[-2] = 10.0
        window[-1] = 8.0
        result = linear_extrapolation_step(window)
        assert np.allclose(result, 6.0)


@requires_real_dataset
class TestRunBaselinesReal:
    """Real-data verification per this project's established discipline:
    baseline numbers must be checked against the actual 24-scenario
    dataset, not assumed. Pinned loosely (not exact-value asserts) since
    these are legitimate measured results, not hand-computed constants --
    a future change to the split, stats, or FDS data should be free to
    shift them without breaking the suite, but a gross regression
    (e.g. a shape/units bug making RMSE 1000x too large) should still fail.
    """

    def test_both_baselines_produce_finite_positive_rmse(self):
        output = run_baselines(stride=100)
        for name, result in output["results"].items():
            for lead, value in result["rmse"].items():
                assert value is not None and np.isfinite(value) and value > 0, \
                    f"{name} lead={lead} rmse={value}"

    def test_persistence_rmse_is_physically_plausible(self):
        # Ambient ~20C, peaks a few hundred C (M2.3/M3.1 findings) -- a
        # sane one-step RMSE should be a modest fraction of that range,
        # not near-zero (bug) or near-thousands (units bug).
        output = run_baselines(stride=100)
        persistence_rmse_lead1 = output["results"]["persistence"]["rmse"][1]
        assert 0.1 < persistence_rmse_lead1 < 200.0

    def test_test_scenarios_match_scenario_split(self):
        from dataset import load_entries, scenario_split
        entries = load_entries()
        _train, expected_test = scenario_split(entries)
        output = run_baselines(stride=200)
        assert output["test_scenarios"] == expected_test
