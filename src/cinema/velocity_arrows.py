"""Sparse directional flow-arrow overlay for cinematic mode. The stored
VELOCITY slice is speed magnitude only (no true u/w vector components),
so direction is the same heuristic cinema/smoke.py's Tier 2 advection
already uses: buoyant "up" blended with "away from the hot core"
(-grad T), scaled by the real velocity magnitude.
"""

from __future__ import annotations

import numpy as np

GRID_ROWS = 6
GRID_COLS = 9
UP_BIAS = 0.55
ARROW_LENGTH_CELLS = 2.5   # displacement length, in array-index units
MIN_SPEED_TO_SHOW = 0.05   # m/s -- below this, skip the arrow (near-still air)


def sample_points(shape: tuple):
    """(rows, cols) index arrays for a fixed sample grid -- computed once
    per cell shape and reused every frame, since positions don't move,
    only direction/magnitude do."""
    ny, nx = shape
    rows = np.linspace(3, ny - 4, GRID_ROWS).round().astype(int)
    cols = np.linspace(3, nx - 4, GRID_COLS).round().astype(int)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    return grid_r.ravel(), grid_c.ravel()


def compute_deltas(temperature_frame: np.ndarray, velocity_frame: np.ndarray,
                    sample_rows: np.ndarray, sample_cols: np.ndarray):
    """(d_row, d_col) displacement at each sample point, in array-index
    units -- the caller converts to display coordinates (physical or raw
    pixel, whichever the heatmap itself uses) by sampling two points and
    differencing, not by guessing a sign convention here."""
    grad_y, grad_x = np.gradient(temperature_frame)
    away_y, away_x = -grad_y, -grad_x
    norm = np.hypot(away_y, away_x) + 1e-6
    dir_y, dir_x = away_y / norm, away_x / norm
    dir_y = dir_y * (1.0 - UP_BIAS) - UP_BIAS  # blend in a constant "straight up" bias
    dnorm = np.hypot(dir_y, dir_x) + 1e-6
    dir_y, dir_x = dir_y / dnorm, dir_x / dnorm

    speed = np.clip(velocity_frame, 0.0, None)
    s = speed[sample_rows, sample_cols]
    # Length scales with the real velocity magnitude (clamped so both
    # faint and strong flows stay legible), not a fixed arrow size --
    # 2 m/s matches config.QUANTITY_DISPLAY['VELOCITY']'s own default scale.
    magnitude_factor = np.clip(s / 2.0, 0.3, 1.6)
    d_row = dir_y[sample_rows, sample_cols] * ARROW_LENGTH_CELLS * magnitude_factor
    d_col = dir_x[sample_rows, sample_cols] * ARROW_LENGTH_CELLS * magnitude_factor
    weak = s < MIN_SPEED_TO_SHOW
    d_row[weak] = 0.0
    d_col[weak] = 0.0
    return d_row, d_col
