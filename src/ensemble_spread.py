"""Ensemble spread (V5-M5 / Phase 5): min/mean/max envelopes across the
existing factorial.

For a chosen per-frame metric (peak T, mean T, hot-area fraction, smoke layer),
this takes the series from every scenario and reports the min / mean / max
envelope over time. This is the spread of the *designed* scenarios -- a
parametric ensemble, NOT a calibrated stochastic uncertainty -- and every UI
surface must say so (SPREAD_NOTE).

Pure NumPy, Qt-free. The panel supplies each scenario's series; this module only
reduces them.
"""

from __future__ import annotations

import numpy as np

from layer_height import smoke_layer_height_series

SPREAD_NOTE = ("Parametric ensemble spread across existing scenarios "
               "(the designed factorial), not calibrated stochastic uncertainty.")

# (key, label, unit)
METRICS = [
    ("spatial_max", "Peak temperature", "°C"),
    ("spatial_mean", "Mean temperature", "°C"),
    ("hot_area_fraction", "Hot-area fraction", "fraction"),
    ("layer_height", "Smoke-layer height", "m"),
]
METRIC_LABEL = {k: l for k, l, _u in METRICS}
METRIC_UNIT = {k: u for k, _l, u in METRICS}


def per_frame_series(data, extent, metric: str, hot_threshold: float = 100.0,
                     ambient: float = 20.0) -> np.ndarray:
    """One scenario's per-frame value of `metric`."""
    arr = np.asarray(data, dtype=float)
    if metric == "spatial_max":
        return arr.max(axis=(1, 2))
    if metric == "spatial_mean":
        return arr.mean(axis=(1, 2))
    if metric == "hot_area_fraction":
        return (arr > hot_threshold).mean(axis=(1, 2))
    if metric == "layer_height":
        return (smoke_layer_height_series(arr, extent, ambient)
                if extent is not None else np.zeros(arr.shape[0]))
    raise KeyError(metric)


def envelope(series_list: list):
    """(min, mean, max) per frame across scenarios, truncated to the shortest
    series so ragged run lengths align."""
    if not series_list:
        return np.array([]), np.array([]), np.array([])
    n = min(len(s) for s in series_list)
    m = np.vstack([np.asarray(s, dtype=float)[:n] for s in series_list])
    return m.min(axis=0), m.mean(axis=0), m.max(axis=0)


def percentile_envelope(series_list: list, lo: float = 25.0, hi: float = 75.0):
    """(p_lo, mean, p_hi) per frame across scenarios -- same (band, mean,
    band) shape as envelope(), just a robust/IQR-style band instead of the
    full min-max spread, so a single outlier scenario doesn't stretch the
    band to cover the whole range (Analysis UX + reliability pass: an
    "Ensemble spread" panel improvement, requested as an alternative,
    less-extreme band alongside min-max, not a replacement for it)."""
    if not series_list:
        return np.array([]), np.array([]), np.array([])
    n = min(len(s) for s in series_list)
    m = np.vstack([np.asarray(s, dtype=float)[:n] for s in series_list])
    return (np.percentile(m, lo, axis=0), m.mean(axis=0),
            np.percentile(m, hi, axis=0))
