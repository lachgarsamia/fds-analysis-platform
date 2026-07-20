"""Factor-Effect Field Maps (V2 roadmap M3.1, feature F2 -- the Phase 3
flagship). ANOVA lifted to field level: for a factorial ensemble, the
*main effect* of a design factor (candles / door / vod / voc) is the
mean field of the high-level group minus the mean field of the low-level
group, averaged over every other factor -- a per-cell, per-frame
diverging field showing *where* and *when* that factor changes the fire,
not just by how much overall. Optionally, the 2-factor *interaction*
field measures non-additivity between two factors.

Pure computation (module level, Qt-free, testable). Memory: rather than
holding all N scenarios' arrays at once, group means are accumulated by
streaming through the store one scenario at a time, so a main-effect
result is a single (n_times, n_z, n_x) field (~9.5 MB here), not the
whole dataset.

Scope (M3.1): computed over the `.sf` quantities (TEMPERATURE, VELOCITY);
the same math applies to any quantity, but running it over the 24 `.s3d`
SOOT planes would decode the whole volumetric dataset, deferred.
"""

from __future__ import annotations

import numpy as np

FACTORS = ('candles', 'door', 'vod', 'voc')
FACTOR_LABELS = {'candles': 'Candles', 'door': 'Door width', 'vod': 'Vent 1 (VOD)', 'voc': 'Vent 2 (VOC)'}


def factor_groups(entries: list, factor: str) -> dict:
    """{level -> [case_index, ...]} grouping entries by their factor level."""
    groups: dict = {}
    for e in entries:
        groups.setdefault(getattr(e, factor), []).append(e.case_index)
    return groups


def factor_levels(entries: list, factor: str) -> list:
    return sorted(factor_groups(entries, factor))


def group_mean_series(store, case_indices: list, quantity_key) -> np.ndarray:
    """Streaming per-frame mean field over a group of scenarios ->
    (n_times, n_z, n_x). Accumulates one scenario at a time so the group's
    raw arrays are never all resident at once. n_times is truncated to the
    shortest member (all equal in this dataset, but defensive)."""
    acc = None
    n_times = None
    count = 0
    for ci in case_indices:
        arr = np.asarray(store.get(ci, quantity_key), dtype=np.float64)
        if acc is None:
            n_times = arr.shape[0]
            acc = np.zeros((n_times,) + arr.shape[1:], dtype=np.float64)
        n_times = min(n_times, arr.shape[0])
        acc[:n_times] += arr[:n_times]
        count += 1
    if acc is None or count == 0:
        raise ValueError("no scenarios in group")
    return (acc[:n_times] / count).astype(np.float32)


def main_effect_series(store, entries: list, factor: str, quantity_key) -> np.ndarray:
    """High-level-minus-low-level mean field over the whole timeline
    (n_times, n_z, n_x). Uses the factor's extreme levels (min vs max)
    for factors with more than two levels (e.g. vod: open vs HVAC).
    Returns None if fewer than two levels are present."""
    groups = factor_groups(entries, factor)
    levels = sorted(groups)
    if len(levels) < 2:
        return None
    low, high = levels[0], levels[-1]
    mean_low = group_mean_series(store, groups[low], quantity_key)
    mean_high = group_mean_series(store, groups[high], quantity_key)
    n = min(mean_low.shape[0], mean_high.shape[0])
    return mean_high[:n] - mean_low[:n]


def interaction_series(store, entries: list, factor_a: str, factor_b: str, quantity_key) -> np.ndarray:
    """2-factor interaction field (non-additivity) over the timeline:
    [mean(a_hi,b_hi) - mean(a_hi,b_lo)] - [mean(a_lo,b_hi) - mean(a_lo,b_lo)],
    using each factor's extreme levels. Returns None if any of the four
    corner groups is empty."""
    la = sorted(factor_groups(entries, factor_a))
    lb = sorted(factor_groups(entries, factor_b))
    if len(la) < 2 or len(lb) < 2:
        return None
    a_lo, a_hi = la[0], la[-1]
    b_lo, b_hi = lb[0], lb[-1]

    def corner(av, bv):
        cases = [e.case_index for e in entries if getattr(e, factor_a) == av and getattr(e, factor_b) == bv]
        return cases

    corners = {(av, bv): corner(av, bv) for av in (a_lo, a_hi) for bv in (b_lo, b_hi)}
    if any(len(c) == 0 for c in corners.values()):
        return None
    means = {k: group_mean_series(store, v, quantity_key) for k, v in corners.items()}
    n = min(m.shape[0] for m in means.values())
    return ((means[(a_hi, b_hi)][:n] - means[(a_hi, b_lo)][:n])
            - (means[(a_lo, b_hi)][:n] - means[(a_lo, b_lo)][:n]))


def _sample_indices(n_times: int, n_samples: int) -> np.ndarray:
    return np.linspace(0, n_times - 1, min(n_times, n_samples), dtype=int)


def effect_magnitude(field_series: np.ndarray, n_samples: int = 20) -> float:
    """Space-time-integrated magnitude of an effect field: the mean over
    sampled frames of each frame's spatial-mean absolute value, in the
    quantity's own units. The ANOVA-style scalar that ranks factors."""
    idx = _sample_indices(field_series.shape[0], n_samples)
    return float(np.mean([np.mean(np.abs(field_series[i])) for i in idx]))


def effect_peak(field_series: np.ndarray, n_samples: int = 20) -> float:
    """Peak absolute effect over sampled frames (the strongest local
    difference the factor produces anywhere, any sampled time)."""
    idx = _sample_indices(field_series.shape[0], n_samples)
    return float(np.max([np.max(np.abs(field_series[i])) for i in idx]))


def symmetric_vmax(field_series: np.ndarray, n_samples: int = 20) -> float:
    """+-vmax for a symmetric diverging colorscale, sampled like the
    difference view's own symmetric_clim (M2.3)."""
    return effect_peak(field_series, n_samples) or 1.0
