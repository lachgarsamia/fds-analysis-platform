"""Tests for ml/rollout.py.

compare_to_baselines() is pure (no torch/real data) and tested directly
with fabricated inputs. evaluate_model()/export_full_scenario_predictions()
need a real trained checkpoint -- guarded, skipped if ml/checkpoints/ is
empty (e.g. before ml/train.py has ever been run)."""

import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src"))

from dataset import WINDOW_IN, load_entries  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402
from rollout import compare_to_baselines, evaluate_model, export_full_scenario_predictions  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)

_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT_DIR = os.path.join(_ML_DIR, "checkpoints")

requires_checkpoint = pytest.mark.skipif(
    not os.path.isdir(_CHECKPOINT_DIR) or not glob.glob(os.path.join(_CHECKPOINT_DIR, "*.pt")),
    reason="no trained checkpoint in ml/checkpoints/ -- run ml/train.py first",
)


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


class TestCompareToBaselines:
    def test_fno_beating_persistence_at_lead_4_is_detected(self, tmp_path):
        baseline_path = tmp_path / "baseline_results.json"
        _write_json(baseline_path, {
            "results": {"persistence": {"rmse": {"1": 8.0, "4": 8.6, "8": 9.2}}},
        })
        model_results = {"results": {"fno": {"rmse": {1: 9.0, 4: 5.0, 8: 5.5}}}}

        comparison = compare_to_baselines(model_results, str(baseline_path))

        assert comparison["by_lead"][4]["fno_beats_persistence"] is True
        assert comparison["beats_persistence_at_lead_4_or_more"] is True
        assert 4 in comparison["leads_where_fno_beats_persistence_at_4_plus"]

    def test_fno_never_beating_persistence_is_honestly_reported(self, tmp_path):
        baseline_path = tmp_path / "baseline_results.json"
        _write_json(baseline_path, {
            "results": {"persistence": {"rmse": {"1": 8.0, "4": 8.6, "8": 9.2}}},
        })
        model_results = {"results": {"fno": {"rmse": {1: 20.0, 4: 25.0, 8: 30.0}}}}

        comparison = compare_to_baselines(model_results, str(baseline_path))

        assert comparison["beats_persistence_at_lead_4_or_more"] is False
        assert comparison["leads_where_fno_beats_persistence_at_4_plus"] == []

    def test_beating_only_at_lead_below_4_does_not_count(self, tmp_path):
        baseline_path = tmp_path / "baseline_results.json"
        _write_json(baseline_path, {
            "results": {"persistence": {"rmse": {"1": 8.0, "2": 8.0, "4": 8.6}}},
        })
        # Beats persistence at lead=1 (below the DoD's >=4 threshold) but not lead=4.
        model_results = {"results": {"fno": {"rmse": {1: 5.0, 2: 9.0, 4: 9.0}}}}

        comparison = compare_to_baselines(model_results, str(baseline_path))

        assert comparison["by_lead"][1]["fno_beats_persistence"] is True
        assert comparison["beats_persistence_at_lead_4_or_more"] is False


@requires_real_dataset
@requires_checkpoint
class TestRolloutReal:
    def _latest_checkpoint(self):
        checkpoints = sorted(glob.glob(os.path.join(_CHECKPOINT_DIR, "*.pt")), key=os.path.getmtime)
        return checkpoints[-1]

    def test_evaluate_model_covers_full_horizon_on_test_scenarios(self):
        checkpoint_path = self._latest_checkpoint()
        result = evaluate_model(checkpoint_path, stride=100)
        assert len(result["test_scenarios"]) == 4
        rmse = result["results"]["fno"]["rmse"]
        assert set(rmse.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
        assert all(v is not None and np.isfinite(v) for v in rmse.values())

    def test_export_predictions_matches_ground_truth_shape(self, tmp_path, monkeypatch):
        import rollout as rollout_module
        monkeypatch.setattr(rollout_module, "PREDICTIONS_DIR", str(tmp_path))
        checkpoint_path = self._latest_checkpoint()

        manifest = export_full_scenario_predictions(checkpoint_path)

        entries = load_entries()
        by_case = {e.case_index: e for e in entries}
        for case_index_str, info in manifest["cases"].items():
            case_index = int(case_index_str)
            pred_path = os.path.join(str(tmp_path), f"{case_index}.npy")
            predicted = np.load(pred_path)
            assert predicted.shape[0] == info["n_frames"]
            # Seed frames must be an exact copy of ground truth, not a
            # model prediction -- the model never sees fewer than
            # WINDOW_IN real frames as input.
            from load_data import load_data
            from slice_key import DEFAULT_SLICE_KEY
            ground_truth = load_data(by_case[case_index].path, DEFAULT_SLICE_KEY)
            assert np.array_equal(predicted[:WINDOW_IN], ground_truth[:WINDOW_IN])
            assert np.isfinite(predicted).all()
