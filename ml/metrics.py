"""Shared RMSE/SSIM-vs-lead-time evaluation harness for M3.2.

Used identically for the persistence/linear-extrapolation baselines
(3.2.2) and the trained FNO model (3.2.4) so their curves are directly,
fairly comparable -- same rollout starts, same scenarios, same metrics,
the only thing that differs is the one-step forecaster function each
supplies.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from skimage.metrics import structural_similarity

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ML_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataset import WINDOW_IN, denormalize, normalize  # noqa: E402
from load_data import load_data  # noqa: E402
from slice_key import DEFAULT_SLICE_KEY  # noqa: E402

# Rollout start points are `stride` frames apart within each scenario --
# many more sample points than one rollout per scenario (481 frames / 20
# ~= 24 starts per scenario, x4 test scenarios ~= 96 rollouts total),
# for a curve that reflects more than 4 single trajectories.
DEFAULT_STRIDE = 20
DEFAULT_HORIZON = 8


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def ssim(pred: np.ndarray, true: np.ndarray) -> float:
    data_range = float(true.max() - true.min())
    if data_range == 0:
        data_range = 1.0
    return float(structural_similarity(pred, true, data_range=data_range))


def evaluate_rollout(forecast_step_fn, entries: list, case_indices: list, stats: dict,
                      horizon: int = DEFAULT_HORIZON, seed_len: int = WINDOW_IN,
                      quantity_key=DEFAULT_SLICE_KEY, stride: int = DEFAULT_STRIDE) -> dict:
    """forecast_step_fn(window: (seed_len, H, W) normalized) -> (H, W)
    normalized one-step-ahead prediction. Rolled out autoregressively for
    `horizon` steps (each prediction feeds back in as the new most-recent
    frame) from every valid rollout start `stride` frames apart, across
    every scenario in case_indices.

    Returns {"rmse": {lead: mean_rmse}, "ssim": {lead: mean_ssim},
    "n_rollouts": int} -- lead times are 1-indexed (lead=1 is the first
    autoregressive step past the seed window).
    """
    by_case = {e.case_index: e for e in entries}
    rmse_by_lead = {t: [] for t in range(1, horizon + 1)}
    ssim_by_lead = {t: [] for t in range(1, horizon + 1)}
    n_rollouts = 0

    for case_index in case_indices:
        data = load_data(by_case[case_index].path, quantity_key)
        normalized = normalize(data, stats)
        n_times = data.shape[0]
        last_start = n_times - seed_len - horizon
        if last_start <= 0:
            continue
        for start in range(0, last_start, stride):
            window = normalized[start:start + seed_len].copy()
            n_rollouts += 1
            for lead in range(1, horizon + 1):
                pred_norm = forecast_step_fn(window)
                true_frame = data[start + seed_len + lead - 1]
                pred_frame = denormalize(pred_norm, stats)
                rmse_by_lead[lead].append(rmse(pred_frame, true_frame))
                ssim_by_lead[lead].append(ssim(pred_frame, true_frame))
                window = np.concatenate([window[1:], pred_norm[np.newaxis]], axis=0)

    return {
        "rmse": {t: float(np.mean(v)) if v else None for t, v in rmse_by_lead.items()},
        "ssim": {t: float(np.mean(v)) if v else None for t, v in ssim_by_lead.items()},
        "n_rollouts": n_rollouts,
    }
