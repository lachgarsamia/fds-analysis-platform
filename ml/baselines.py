"""Persistence + linear-extrapolation baselines for M3.2 (3.2.2).

Non-negotiable per spec: these run BEFORE the FNO model, and the model
must beat them (at some documented lead time) or the eventual report says
so honestly. Both forecasters are one-step functions plugged into
ml/metrics.py's evaluate_rollout(), so their multi-step curves are
produced by the exact same autoregressive-rollout harness the FNO model
will use in 3.2.4 -- an apples-to-apples comparison, not two different
methodologies compared after the fact.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ML_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataset import compute_normalization, load_entries, scenario_split  # noqa: E402
from metrics import evaluate_rollout  # noqa: E402


def persistence_step(window: np.ndarray) -> np.ndarray:
    """Forecasts "no change": repeats the most recent frame."""
    return window[-1].copy()


def linear_extrapolation_step(window: np.ndarray) -> np.ndarray:
    """Per-pixel linear trend from the last two frames of the window,
    extrapolated one step forward. No motion/flow estimation (pure
    per-pixel delta) -- deliberately the simplest "trend-aware" baseline
    above plain persistence."""
    delta = window[-1] - window[-2]
    return window[-1] + delta


def run_baselines(stride: int = 20) -> dict:
    """Evaluates both baselines on the held-out test scenarios (the same
    20/4 scenario_split() used everywhere else in M3.2) and returns a
    dict of results keyed by baseline name."""
    entries = load_entries()
    train, test = scenario_split(entries)
    # Normalization stats are computed on TRAIN only (never test), matching
    # what the FNO model will do in 3.2.3 -- baselines don't strictly need
    # normalized stats (they're scale-invariant to it), but evaluate_rollout
    # requires stats to convert back to physical units for RMSE/SSIM.
    stats = compute_normalization(entries, train)

    results = {}
    for name, step_fn in [
        ("persistence", persistence_step),
        ("linear_extrapolation", linear_extrapolation_step),
    ]:
        results[name] = evaluate_rollout(step_fn, entries, test, stats, stride=stride)

    return {"test_scenarios": test, "stats": stats, "results": results}


def main():
    output = run_baselines()
    out_path = os.path.join(_ML_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    for name, res in output["results"].items():
        print(f"\n{name} (n_rollouts={res['n_rollouts']}):")
        for lead in sorted(res["rmse"]):
            print(f"  lead={lead}: rmse={res['rmse'][lead]:.3f}  ssim={res['ssim'][lead]:.4f}")


if __name__ == "__main__":
    main()
