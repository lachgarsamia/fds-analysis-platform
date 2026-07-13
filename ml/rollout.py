"""Autoregressive rollout evaluation + full-scenario prediction export for
M3.2 (3.2.4).

Two things: (1) reuses ml/metrics.py's evaluate_rollout() harness -- the
exact same one used for the persistence/linear-extrapolation baselines --
to produce the trained FNO's RMSE/SSIM-vs-lead-time curve on the 4
held-out TEST scenarios, for a fair, apples-to-apples comparison against
ml/baseline_results.json (the model must beat persistence at some lead
time or this reports that honestly, per the milestone's own
non-negotiable). (2) exports a full-length prediction array per test
scenario to predictions/<case_index>.npy, matching the real scenario's
own (n_frames, H, W) shape exactly -- frames [0:WINDOW_IN) are copied
straight from ground truth (the model's own required seed, never itself
a prediction) and frames [WINDOW_IN:) are the model's own autoregressive
rollout, denormalized back to physical units -- so the app's existing
SliceView/DifferenceView machinery can load and display this array
exactly like a real scenario's TEMPERATURE data (3.2.5).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ML_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from dataset import WINDOW_IN, denormalize, load_entries, normalize  # noqa: E402
from load_data import load_data  # noqa: E402
from metrics import evaluate_rollout  # noqa: E402
from model import build_fno, get_device, model_forecast_step  # noqa: E402
from slice_key import DEFAULT_SLICE_KEY  # noqa: E402

PREDICTIONS_DIR = os.path.join(_ML_DIR, "..", "predictions")


def load_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = build_fno(
        n_modes=ckpt["config"]["n_modes"],
        hidden_channels=ckpt["config"]["hidden_channels"],
        n_layers=ckpt["config"]["n_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def evaluate_model(checkpoint_path: str, stride: int = 20) -> dict:
    """Model's RMSE/SSIM-vs-lead-time on the TEST scenarios, via the same
    evaluate_rollout() harness as the baselines -- directly comparable."""
    device = get_device()
    model, ckpt = load_checkpoint(checkpoint_path)
    model = model.to(device)
    entries = load_entries()
    step_fn = model_forecast_step(model, device)
    result = evaluate_rollout(step_fn, entries, ckpt["test_cases"], ckpt["stats"], stride=stride)
    return {"test_scenarios": ckpt["test_cases"], "stats": ckpt["stats"], "results": {"fno": result}}


def compare_to_baselines(model_results: dict, baseline_results_path: str) -> dict:
    """Per-lead-time fno vs. persistence RMSE, plus the DoD's explicit
    yes/no: does the model beat persistence at ANY lead time >= 4? Written
    out as its own report rather than only printed, so the finding --
    positive or negative -- is preserved as an artifact, not just a
    console line."""
    with open(baseline_results_path) as f:
        baselines = json.load(f)
    fno_rmse = model_results["results"]["fno"]["rmse"]
    persistence_rmse = baselines["results"]["persistence"]["rmse"]

    by_lead = {}
    beats_persistence_leads = []
    for lead_str, p_rmse in persistence_rmse.items():
        lead = int(lead_str)
        f_rmse = fno_rmse.get(lead)
        if f_rmse is None:
            continue
        beats = f_rmse < p_rmse
        by_lead[lead] = {"fno_rmse": f_rmse, "persistence_rmse": p_rmse, "fno_beats_persistence": beats}
        if beats and lead >= 4:
            beats_persistence_leads.append(lead)

    return {
        "by_lead": by_lead,
        "beats_persistence_at_lead_4_or_more": bool(beats_persistence_leads),
        "leads_where_fno_beats_persistence_at_4_plus": beats_persistence_leads,
    }


def export_full_scenario_predictions(checkpoint_path: str, quantity_key=DEFAULT_SLICE_KEY) -> dict:
    """Writes predictions/<case_index>.npy for every TEST scenario: a full
    (n_frames, H, W) array, ground-truth seed frames followed by the
    model's own autoregressive rollout for the rest of the scenario's
    real length (not just a fixed evaluation horizon)."""
    device = get_device()
    model, ckpt = load_checkpoint(checkpoint_path)
    model = model.to(device)
    stats = ckpt["stats"]
    step_fn = model_forecast_step(model, device)

    entries = load_entries()
    by_case = {e.case_index: e for e in entries}
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    manifest = {"checkpoint": os.path.basename(checkpoint_path), "config_hash": ckpt["config_hash"], "cases": {}}
    for case_index in ckpt["test_cases"]:
        data = load_data(by_case[case_index].path, quantity_key)
        n_times = data.shape[0]
        normalized = normalize(data, stats)

        predicted = np.empty_like(data)
        predicted[:WINDOW_IN] = data[:WINDOW_IN]

        window = normalized[:WINDOW_IN].copy()
        for t in range(WINDOW_IN, n_times):
            pred_norm = step_fn(window)
            pred_frame = denormalize(pred_norm, stats)
            predicted[t] = pred_frame
            window = np.concatenate([window[1:], pred_norm[np.newaxis]], axis=0)

        out_path = os.path.join(PREDICTIONS_DIR, f"{case_index}.npy")
        np.save(out_path, predicted)
        manifest["cases"][str(case_index)] = {
            "folder": by_case[case_index].folder,
            "n_frames": int(n_times),
        }

    with open(os.path.join(PREDICTIONS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _latest_checkpoint() -> str:
    checkpoint_dir = os.path.join(_ML_DIR, "checkpoints")
    checkpoints = sorted(
        (os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")),
        key=os.path.getmtime,
    )
    if not checkpoints:
        raise SystemExit("no checkpoints found in ml/checkpoints/ -- run ml/train.py first")
    return checkpoints[-1]


def main():
    checkpoint_path = _latest_checkpoint()
    print(f"using checkpoint: {checkpoint_path}")

    model_results = evaluate_model(checkpoint_path)
    out_path = os.path.join(_ML_DIR, "model_results.json")
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"wrote {out_path}")
    for lead in sorted(model_results["results"]["fno"]["rmse"]):
        rmse = model_results["results"]["fno"]["rmse"][lead]
        ssim = model_results["results"]["fno"]["ssim"][lead]
        print(f"  lead={lead}: rmse={rmse:.3f}  ssim={ssim:.4f}")

    baseline_results_path = os.path.join(_ML_DIR, "baseline_results.json")
    if os.path.exists(baseline_results_path):
        comparison = compare_to_baselines(model_results, baseline_results_path)
        comparison_path = os.path.join(_ML_DIR, "comparison_report.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"wrote {comparison_path}")
        print(f"beats persistence at lead>=4: {comparison['beats_persistence_at_lead_4_or_more']}")

    manifest = export_full_scenario_predictions(checkpoint_path)
    print(f"exported predictions for {len(manifest['cases'])} test scenarios")


if __name__ == "__main__":
    main()
