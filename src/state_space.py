"""Fire State-Space + Fire Genome (V3-M5, Fire Intelligence Layer).

Two views of a scenario's descriptor data (descriptors.py):

- **State-space trajectory**: every *frame* embedded into 2D (PCA over
  the per-frame descriptors), so a run traces a path through fire-state
  space -- ignition, growth, steady burning, decay -- and regime changes
  show as turns in the path. Answers "when did the regime change?"
  visually rather than by staring at a temperature curve.

- **Fire Genome**: a short per-scenario fingerprint (peak temperature,
  heating rate, smoke descent, energy released, spread speed) normalized
  across the ensemble, so many runs can be compared and clustered at a
  glance.

Pure NumPy/sklearn, Qt-free, deterministic. Reuses the descriptor engine
and the existing PCA (analytics/clustering.py). Latent axes are honestly
labelled as learned combinations, with their explained-variance shares.
"""

from __future__ import annotations

import numpy as np

from descriptors import compute_descriptors
from analytics.clustering import run_pca

# The genome traits, in a fixed order (key, human label). Each is computed
# per scenario, then min-max normalized across the ensemble to 0..1 bars.
GENOME_TRAITS = (
    ("peak_temp", "Peak temperature"),
    ("heating_rate", "Heating rate"),
    ("smoke_descent", "Smoke descent"),
    ("energy", "Energy released"),
    ("spread", "Spread speed"),
)


# Frames to average over before embedding: FDS turbulence makes the raw
# per-frame descriptors jitter, which buries the run's actual evolution in
# noise. A short moving average keeps the trajectory a smooth path (so
# regime changes read as turns, not scatter) without altering the science.
_SMOOTH_WINDOW = 7


def _smooth_columns(matrix: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or matrix.shape[0] <= window:
        return matrix
    kernel = np.ones(window) / window
    return np.column_stack([np.convolve(matrix[:, c], kernel, mode="same")
                            for c in range(matrix.shape[1])])


def scenario_trajectory(data: np.ndarray, extent, fps: int, smooth_window: int = _SMOOTH_WINDOW):
    """Embed a scenario's per-frame descriptors into 2D. Returns
    (coords (n_frames, 2), times (n_frames,), explained_variance_ratio).
    Descriptors are lightly time-smoothed first (see _SMOOTH_WINDOW)."""
    fps = max(1, fps)
    table = compute_descriptors(data, extent, fps)
    matrix = _smooth_columns(table.as_matrix(), smooth_window)
    result = run_pca(matrix, n_components=2)
    return result.coords, table.times, result.explained_variance_ratio


def genome_traits(data: np.ndarray, extent, fps: int, summary=None) -> dict:
    """Raw (un-normalized) genome traits for one scenario, computed from
    its descriptors (+ optional summary_stats for energy)."""
    fps = max(1, fps)
    table = compute_descriptors(data, extent, fps)
    smax = table.column("spatial_max")
    d_smax = table.column("d_spatial_max")
    layer = table.column("layer_height")
    hot = table.column("hot_area_fraction")
    return {
        "peak_temp": float(smax.max()),
        "heating_rate": float(d_smax.max()) if d_smax.size else 0.0,
        "smoke_descent": float(layer[0] - layer.min()) if layer.size else 0.0,
        "energy": float(getattr(summary, "total_energy_kj", None) or 0.0),
        "spread": float(np.max(np.diff(hot)) * fps) if hot.size > 1 else 0.0,
    }


def normalize_genomes(traits_list: list) -> list:
    """Min-max normalize each trait across the ensemble to 0..1. A trait
    that is constant across scenarios maps to 0.5 (no information to rank
    on, so neither high nor low)."""
    if not traits_list:
        return []
    keys = [k for k, _ in GENOME_TRAITS]
    lo = {k: min(t[k] for t in traits_list) for k in keys}
    hi = {k: max(t[k] for t in traits_list) for k in keys}
    out = []
    for t in traits_list:
        norm = {}
        for k in keys:
            span = hi[k] - lo[k]
            norm[k] = 0.5 if span <= 0 else (t[k] - lo[k]) / span
        out.append(norm)
    return out


def genome_matrix(normalized: list) -> np.ndarray:
    """(n_scenarios, n_traits) matrix from normalized genomes -- for
    clustering the ensemble by fingerprint."""
    keys = [k for k, _ in GENOME_TRAITS]
    if not normalized:
        return np.zeros((0, len(keys)))
    return np.array([[g[k] for k in keys] for g in normalized])
