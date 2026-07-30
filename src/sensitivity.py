"""Sensitivity estimation (V5-M3): "what-if" from the *existing* runs only.

The candle study is a full factorial (candles×door×vod×voc = 2×2×3×2 = 24),
exactly one run per cell, so the response scalars form a fully-populated grid.
A response at arbitrary (continuous) factor settings is therefore estimated by
**multilinear interpolation across the existing runs** -- never a new
simulation. Every consumer must show ESTIMATE_NOTE alongside these numbers.

Provides: an interpolator over the factor grid, single-response prediction, a
2-factor response surface, and the nearest existing scenario to a setting
(for SelectionBus hand-off). Reuses M2's table + factor axes. Pure
NumPy/scipy, Qt-free.

predict_all() (all-responses table) and tornado() (local factor swing) were
removed (Analysis UX + reliability pass) along with the Sensitivity panel's
What-if/Tornado tabs that were their only callers -- Response surface
already answers "how does this response change near my current setting"
for the two factors that matter most.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import study_analytics as sa

ESTIMATE_NOTE = ("Estimated from Existing Scenarios by interpolation "
                 "(no new simulation).")


def factor_levels(table: list, param: str) -> list:
    """Sorted distinct levels of a factor across the runs."""
    return sorted({float(r["params"][param]) for r in table})


def build_grid(table: list, response: str):
    """({param: levels}, response array over the factor grid). Missing/NaN
    cells are filled with the response mean so the grid stays interpolable
    (documented; the candle factorial is fully populated, so this only bites
    for responses a run lacks, e.g. HRR without a CSV)."""
    levels = {p: factor_levels(table, p) for p in sa.PARAMS}
    idx_of = {p: {lv: i for i, lv in enumerate(levels[p])} for p in sa.PARAMS}
    grid = np.full(tuple(len(levels[p]) for p in sa.PARAMS), np.nan)
    for r in table:
        cell = tuple(idx_of[p][float(r["params"][p])] for p in sa.PARAMS)
        grid[cell] = r["responses"][response]
    if np.isnan(grid).any():
        fill = float(np.nanmean(grid)) if not np.isnan(grid).all() else 0.0
        grid = np.where(np.isnan(grid), fill, grid)
    return levels, grid


def make_interpolator(table: list, response: str):
    levels, grid = build_grid(table, response)
    points = tuple(np.array(levels[p], dtype=float) for p in sa.PARAMS)
    interp = RegularGridInterpolator(points, grid, method="linear",
                                     bounds_error=False, fill_value=None)
    return interp, levels


def _point(settings: dict) -> np.ndarray:
    return np.array([[float(settings[p]) for p in sa.PARAMS]], dtype=float)


def predict(table: list, response: str, settings: dict) -> float:
    interp, _levels = make_interpolator(table, response)
    return float(interp(_point(settings))[0])


def response_surface(table: list, response: str, fx: str, fy: str,
                     settings: dict, n: int = 25):
    """(xs, ys, Z) grid of estimates varying fx, fy over their level ranges,
    holding the other factors at `settings`."""
    interp, levels = make_interpolator(table, response)
    xs = np.linspace(min(levels[fx]), max(levels[fx]), n)
    ys = np.linspace(min(levels[fy]), max(levels[fy]), n)
    z = np.empty((n, n))
    base = {p: float(settings[p]) for p in sa.PARAMS}
    for j, yv in enumerate(ys):
        pts = np.empty((n, len(sa.PARAMS)))
        for i, xv in enumerate(xs):
            s = dict(base); s[fx] = xv; s[fy] = yv
            pts[i] = [s[p] for p in sa.PARAMS]
        z[j] = interp(pts)
    return xs, ys, z


def nearest_scenario(table: list, settings: dict):
    """(case_index, distance) of the existing run whose factor cell is closest
    to `settings` -- for snapping the SelectionBus to a real scenario."""
    best, best_d = None, float("inf")
    for r in table:
        d = sum((float(settings[p]) - float(r["params"][p])) ** 2 for p in sa.PARAMS)
        if d < best_d:
            best_d, best = d, r
    return (best["case_index"] if best else None), (best_d ** 0.5)
