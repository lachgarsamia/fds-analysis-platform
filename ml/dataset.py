"""Sliding-window dataset builder for M3.2's forecasting model (3.2.1).

Lives outside src/ entirely (ml/ has its own dependencies -- torch,
neuraloperator -- that the app itself must never require, per the
milestone's "keeps torch out of the app's required deps" split). Reuses
src/'s manifest.py + load_data.py directly rather than re-implementing
scenario discovery -- neither of those modules imports torch, and reusing
them means "scenario 5" means the exact same thing here as in the app,
which matters once predictions/<case>.npy needs to line up with
ScenarioStore.get(case_index) in 3.2.5, not just as a convenience.

Modeling choice worth being explicit about (the spec's own wording is a
little compressed): "sliding windows (k=8 in -> 1..8 out, autoregressive
rollout later)" is read here as *single-step* supervision during training
(8 frames in, the 9th frame out) with multi-step (1..8-ahead) evaluation
produced *later*, in 3.2.4, by feeding the model's own predictions back in
autoregressively -- not as multi-step-output supervision during training
itself. 3.2.4's rollout task only makes sense as a separate step if
training itself was single-step; documented here so the choice is visible
and correctable if that reading is wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ML_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from manifest import get_manifest  # noqa: E402
from load_data import load_data, SIM_ROOT  # noqa: E402
from slice_key import SliceKey, DEFAULT_SLICE_KEY  # noqa: E402

# k=8 input frames -> 1 predicted frame (see module docstring on the
# single-step-training / multi-step-eval-via-rollout split).
WINDOW_IN = 8

# 24 scenarios total: 20 train / 4 test, BY SCENARIO (spec's explicit
# "not by time -- no leakage" requirement) -- an entire scenario's full
# time series is either wholly train or wholly test, never split across
# the boundary. 3 of the 20 "train" scenarios are further carved out as a
# validation set for early stopping (train_val_split below), so the 4 TEST
# scenarios are never touched until final evaluation.
TEST_SCENARIOS_COUNT = 4
VAL_SCENARIOS_COUNT = 3

# Fixed so every run (dataset build, training, rollout) partitions
# scenarios identically without passing the split around by hand --
# documented in ml/README.md for reproducibility.
SPLIT_SEED = 42


def scenario_split(entries: list, seed: int = SPLIT_SEED) -> tuple:
    """Deterministic (train_case_indices, test_case_indices) -- 20/4."""
    case_indices = sorted(e.case_index for e in entries)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(case_indices)
    test = sorted(int(c) for c in shuffled[:TEST_SCENARIOS_COUNT])
    train = sorted(int(c) for c in shuffled[TEST_SCENARIOS_COUNT:])
    return train, test


def train_val_split(train_case_indices: list, seed: int = SPLIT_SEED) -> tuple:
    """Further split TRAIN scenarios into (fit, val) for early stopping.
    val is never used for the final RMSE/SSIM reporting -- that's the 4
    scenarios scenario_split() held out, untouched until 3.2.4."""
    rng = np.random.default_rng(seed + 1)
    shuffled = rng.permutation(train_case_indices)
    val = sorted(int(c) for c in shuffled[:VAL_SCENARIOS_COUNT])
    fit = sorted(int(c) for c in shuffled[VAL_SCENARIOS_COUNT:])
    return fit, val


def compute_normalization(entries: list, case_indices: list,
                           quantity_key: SliceKey = DEFAULT_SLICE_KEY) -> dict:
    """mean/std over ONLY the given scenarios (the training-fit set) --
    computed once and reused for val/test so no statistic ever depends on
    data the model isn't allowed to have seen."""
    by_case = {e.case_index: e for e in entries}
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for ci in case_indices:
        data = load_data(by_case[ci].path, quantity_key).astype(np.float64)
        total_sum += data.sum()
        total_sq_sum += (data ** 2).sum()
        total_count += data.size
    mean = total_sum / total_count
    variance = total_sq_sum / total_count - mean ** 2
    return {"mean": float(mean), "std": float(np.sqrt(max(variance, 1e-12)))}


def normalize(data: np.ndarray, stats: dict) -> np.ndarray:
    return (data - stats["mean"]) / stats["std"]


def denormalize(data: np.ndarray, stats: dict) -> np.ndarray:
    return data * stats["std"] + stats["mean"]


def build_windows(data: np.ndarray, window_in: int = WINDOW_IN) -> tuple:
    """(n_windows, window_in, H, W) inputs and (n_windows, H, W) targets,
    stride 1, over one scenario's full time series. 481 frames is small
    enough not to need subsampling."""
    n_times = data.shape[0]
    n_windows = n_times - window_in
    if n_windows <= 0:
        empty_in = np.zeros((0, window_in) + data.shape[1:], dtype=np.float32)
        empty_out = np.zeros((0,) + data.shape[1:], dtype=np.float32)
        return empty_in, empty_out
    inputs = np.stack([data[i:i + window_in] for i in range(n_windows)])
    targets = np.stack([data[i + window_in] for i in range(n_windows)])
    return inputs.astype(np.float32), targets.astype(np.float32)


def build_dataset(entries: list, case_indices: list, stats: dict,
                   quantity_key: SliceKey = DEFAULT_SLICE_KEY,
                   window_in: int = WINDOW_IN) -> tuple:
    """Normalized (inputs, targets) concatenated across every scenario in
    case_indices. Each scenario contributes its own sliding windows; a
    window's 8 input frames and 1 target frame always come from the same
    scenario (never spanning two different simulations)."""
    by_case = {e.case_index: e for e in entries}
    all_inputs, all_targets = [], []
    for ci in case_indices:
        data = load_data(by_case[ci].path, quantity_key)
        data = normalize(data, stats)
        inputs, targets = build_windows(data, window_in)
        all_inputs.append(inputs)
        all_targets.append(targets)
    if not all_inputs:
        return (np.zeros((0, window_in), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)


def config_hash(config: dict) -> str:
    """Short, deterministic hash of a training config dict -- 3.2.3's
    "save best checkpoint + config hash", so a checkpoint file can be
    traced back to exactly the hyperparameters that produced it."""
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def load_entries():
    """The manifest, scanning the real fds/sim/ dataset -- callers should
    check len(entries) == 24 (or handle fewer gracefully) since ml/
    scripts assume the real dataset is present, unlike the app's demo
    fallback."""
    return get_manifest(SIM_ROOT)
