"""Linked multi-quantity inspection (V4-M3).

The brief's exact loop: click a temperature peak and immediately see the
HRR, smoke-layer height, and velocity *at that instant*, not one quantity
at a time. This module is the pure alignment layer -- it turns each
quantity into a time series and samples them all at one shared time, so
the panel can draw them with a single cursor and read every value at the
moment the researcher selected.

The slice quantities live on the frame time axis (index / fps); HRR comes
from the scenario's CSV on its own real-seconds axis. `value_at_time`
resolves both to the same instant by clamped linear interpolation, so a
reading is always the honest interpolated value, never a guessed one.

Pure NumPy, Qt-free, deterministic.
"""

from __future__ import annotations

import numpy as np


def peak_over_time(data: np.ndarray) -> np.ndarray:
    """Per-frame spatial maximum -- peak temperature, or peak air speed."""
    return np.asarray(data, dtype=float).max(axis=(1, 2))


def value_at_time(times, values, t: float):
    """Linear interpolation of values(times) at time t, clamped to the
    endpoints. `times` must be ascending; returns None if empty. Clamping
    (not extrapolation) keeps an out-of-range cursor honest: it reports
    the nearest real sample, never an invented one."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size == 0 or values.size == 0:
        return None
    return float(np.interp(float(t), times, values))
