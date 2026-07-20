"""Height-aware analysis (V4-M1, Researcher-Centered Interactive workspace).

Fire dynamics is fundamentally vertical -- stratification, the smoke
layer, the ceiling jet, the plume -- and a 2D heat map cannot express it.
This module turns a scenario's field into the vertical, analysed curves a
fire scientist actually reasons with:

- vertical_profile: the temperature-vs-height profile T(z) at a chosen x,
  the canonical stratification plot;
- plume_height_series: the height the hot core reaches over time;
- ceiling_jet_series: the near-ceiling temperature over time;
- (the smoke-layer / interface height reuses layer_height.py).

Pure NumPy, Qt-free, deterministic. Row 0 of a frame is the ceiling (z1),
matching the app's flipped-array convention.
"""

from __future__ import annotations

import numpy as np

# Near-ceiling band (fraction of domain height from the top) used for the
# ceiling-jet temperature.
CEILING_BAND_FRAC = 0.12


def column_for_x(extent, n_x: int, x: float) -> int:
    """Nearest column index to physical x, clipped into range."""
    if extent is None:
        return int(min(max(round(x), 0), n_x - 1))
    x0, x1, _z0, _z1 = extent
    if x1 == x0:
        return 0
    col = int(round((x - x0) / (x1 - x0) * (n_x - 1)))
    return min(max(col, 0), n_x - 1)


def heights(extent, n_z: int) -> np.ndarray:
    """Physical z of each *row-reversed* index, ascending from floor (z0)
    to ceiling (z1) -- so a profile reads bottom-to-top like a real
    vertical axis."""
    if extent is None:
        return np.arange(n_z, dtype=float)
    _x0, _x1, z0, z1 = extent
    return np.linspace(z0, z1, n_z)


def vertical_profile(frame: np.ndarray, extent, col: int):
    """(z_ascending, values) of the column at `col` for one frame. Row 0 is
    the ceiling, so the values are reversed to read floor-first."""
    n_z, _n_x = frame.shape
    zs = heights(extent, n_z)
    values = np.asarray(frame[:, col], dtype=float)[::-1]  # floor-first
    return zs, values


def plume_height_series(data: np.ndarray, extent, threshold: float) -> np.ndarray:
    """Per-frame highest physical z where the field exceeds `threshold`
    anywhere -- the flame/plume tip height over time. z0 (floor) when
    nothing is hot."""
    arr = np.asarray(data)
    n_t, n_z, _n_x = arr.shape
    zs_desc = heights(extent, n_z)[::-1]  # index 0 = ceiling (z1), matches row order
    hot_rows = (arr > threshold).any(axis=2)  # (n_t, n_z), True where a row has a hot cell
    out = np.full(n_t, zs_desc[-1], dtype=float)  # default floor
    for t in range(n_t):
        rows = np.flatnonzero(hot_rows[t])
        if rows.size:
            out[t] = zs_desc[int(rows.min())]  # smallest row index = highest z
    return out


def ceiling_jet_series(data: np.ndarray, band_frac: float = CEILING_BAND_FRAC) -> np.ndarray:
    """Per-frame maximum temperature in the near-ceiling band (top
    `band_frac` of rows) -- the ceiling-jet strength over time."""
    arr = np.asarray(data)
    n_z = arr.shape[1]
    band = max(1, int(round(band_frac * n_z)))
    return arr[:, :band, :].max(axis=(1, 2))
