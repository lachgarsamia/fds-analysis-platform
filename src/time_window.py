"""Time-Window & Interval Analysis (V4-M5).

Promotes time to a selectable dimension. Given a quantity's per-frame
spatial-mean and spatial-max series, this module computes interval
statistics over any [t0, t1] window (mean, peak, time-integral, linear
trend, net change), a before/after split at a chosen instant, and turns
the Fire Story's detected events into selectable phase windows.

Pure NumPy, Qt-free, deterministic. Every figure is a plain reduction of
the series, so an interval readout is always traceable to the frames it
covers.
"""

from __future__ import annotations

import numpy as np


def window_indices(times, t0: float, t1: float):
    """Inclusive (i0, i1) frame indices covering [min, max] of t0, t1."""
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n == 0:
        return 0, -1
    lo, hi = sorted((float(t0), float(t1)))
    i0 = int(np.searchsorted(times, lo, side="left"))
    i1 = int(np.searchsorted(times, hi, side="right")) - 1
    i0 = min(max(i0, 0), n - 1)
    i1 = min(max(i1, i0), n - 1)
    return i0, i1


def interval_stats(mean_series, max_series, times, t0: float, t1: float) -> dict:
    """Stats for the window [t0, t1]: mean and peak of the field, the
    time-integral of the spatial-mean (deg C . s), the linear trend
    (slope, deg C/s) and net change of the spatial-mean across the
    window."""
    m = np.asarray(mean_series, dtype=float)
    x = np.asarray(max_series, dtype=float)
    t = np.asarray(times, dtype=float)
    i0, i1 = window_indices(times, t0, t1)
    ms, xs, ts = m[i0:i1 + 1], x[i0:i1 + 1], t[i0:i1 + 1]
    integral = float(np.trapz(ms, ts)) if ts.size > 1 else 0.0
    slope = float(np.polyfit(ts, ms, 1)[0]) if ts.size >= 2 else 0.0
    return {
        "t0": float(t[i0]), "t1": float(t[i1]), "n_frames": i1 - i0 + 1,
        "mean": float(ms.mean()) if ms.size else 0.0,
        "peak": float(xs.max()) if xs.size else 0.0,
        "integral": integral,
        "slope": slope,
        "delta": float(ms[-1] - ms[0]) if ms.size else 0.0,
    }


def before_after_split(mean_series, max_series, times, t_split: float):
    """(before, after) interval_stats split at t_split."""
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        empty = interval_stats(mean_series, max_series, times, 0.0, 0.0)
        return empty, empty
    before = interval_stats(mean_series, max_series, times, float(t[0]), t_split)
    after = interval_stats(mean_series, max_series, times, t_split, float(t[-1]))
    return before, after


def phase_windows(event_times_names, t_end: float) -> list:
    """Turn detected events [(time, name), ...] into consecutive selectable
    phase windows [(name, t0, t1)], named by the event that opens each
    interval (a leading "Pre-ignition" window when the first event is after
    t=0). The final window runs to t_end."""
    evs = sorted((float(t), str(n)) for t, n in event_times_names
                 if t is not None and 0.0 <= float(t) <= t_end)
    if not evs:
        return []
    bounds = ([(0.0, "Pre-ignition")] if evs[0][0] > 0 else []) + evs
    windows = []
    for i, (t, name) in enumerate(bounds):
        t_next = bounds[i + 1][0] if i + 1 < len(bounds) else t_end
        if t_next > t:
            windows.append((name, t, t_next))
    return windows
