"""Smoke-layer height (V2 roadmap M2.3, feature F5): a derived time
series bridging CFD slice data to the zone-model vocabulary fire
protection engineers actually use.

Simplified N-percent/integral method: the domain-mean vertical profile
of excess temperature (above ambient) is integrated from the ceiling
downward; the layer height is where that cumulative integral reaches
half the column's total excess. This is a documented simplification of
the rigorous two-zone (Cooper) method -- domain-mean profile rather than
per-column height averaged afterward -- traded for O(n_times) cost on
data already cached, same honesty convention as config.ISOTHERM_LEVELS'
"general reference points, not derived from this study's data" caveat.
"""

from __future__ import annotations

import numpy as np

# Below this integral (deg C * m), a frame is treated as having no
# distinguishable smoke layer -- height reported as the ceiling (z1).
_MIN_INTEGRAL = 1e-6


def smoke_layer_height_series(data: np.ndarray, extent: tuple, ambient_c: float) -> np.ndarray:
    """Layer height (meters, physical z) for every frame -- shape
    (n_times,). `data` is a cached (n_times, n_z, n_x) slice array; row 0
    is the ceiling (z1) per the app's existing origin='upper' + vertical-
    flip convention (see views.py's SliceView docstring)."""
    x0, x1, z0, z1 = extent
    n_t, n_z, n_x = data.shape
    if n_z < 2:
        return np.full(n_t, z1, dtype=float)

    z_desc = np.linspace(z1, z0, n_z)  # matches data's row order (row 0 = ceiling)
    excess = np.clip(np.asarray(data, dtype=float) - ambient_c, 0.0, None)
    mean_excess = excess.mean(axis=2)  # (n_t, n_z), aligned with z_desc

    dz = np.diff(z_desc)  # negative (descending)
    avg = (mean_excess[:, :-1] + mean_excess[:, 1:]) / 2.0
    increments = avg * dz[None, :]
    cum = np.concatenate([np.zeros((n_t, 1)), np.cumsum(increments, axis=1)], axis=1)
    cum_abs = np.abs(cum)  # (n_t, n_z), non-decreasing along the descent from ceiling

    total_abs = cum_abs[:, -1]
    heights = np.full(n_t, z1, dtype=float)
    for t in range(n_t):
        if total_abs[t] < _MIN_INTEGRAL:
            continue
        heights[t] = float(np.interp(total_abs[t] / 2.0, cum_abs[t], z_desc))
    return heights
