"""Smoke-layer height (V2 roadmap M2.3, feature F5): a derived time
series bridging CFD slice data to the zone-model vocabulary fire
protection engineers actually use.

Two methods, both operating on the same domain-mean vertical profile of
excess temperature (above ambient):

- "gradient" (DEFAULT): the elevation of maximum temperature change with
  height -- the standard literature definition of thermal interface
  height (Steckler, K.D., Quintiere, J.G., Rinkinen, W.J., "Flow Induced
  by Fire in a Compartment," NBSIR 82-2520, National Bureau of Standards,
  1982), which defines the interface as the position of rapid temperature
  change between the lower and upper portions of the room. Made the
  default (methodology correction) after the half-integral method below
  was identified as concentrating near the fire source itself rather than
  tracking where smoke actually accumulates near the ceiling -- confirmed
  on real data, not just reasoned about (a real-data comparison on
  c1_d0_vod0_voc0 found the half-integral method's mean height sat closer
  to the ceiling than the gradient method's, not further, so the original
  "pulled toward the fire" concern doesn't hold on this dataset either --
  the switch is a literature-alignment correction, not a fix for an
  observed numerical bias).

  Steckler et al. themselves report the interface can only be pinned down
  to within roughly +-8% to +-50% accuracy in some conditions, due to
  diffusion and mixing at the true physical interface -- residual
  frame-to-frame noise in this signal is therefore physically expected,
  not a defect to eliminate entirely. Accordingly this module smooths the
  input profile (median filter, robust to a single-frame outlier row) but
  never clamps, drops, or outlier-rejects an output frame: a genuine sharp
  transient in the underlying data stays visible in the returned series,
  not scrubbed.

  Two boundary cases have no real interface to find, and are reported at
  the room's physical extremes rather than left to an arbitrary numeric
  tie-break: no excess temperature anywhere (no fire yet) reports the
  ceiling (z1), and a uniformly hot column (smoke has filled the entire
  domain down to the floor -- no lower "clear" zone remains to form an
  interface against) reports the floor (z0). Without the second guard,
  `np.gradient` on a perfectly flat profile produces floating-point noise
  instead of exact zeros, and argmax on that noise locks onto an arbitrary
  index -- a numerical artifact, not a location, was in the tests as an
  unstable and physically-meaningless intermediate height.

  The discrete argmax peak is further refined to sub-grid-cell precision
  via standard parabolic interpolation (see _parabolic_peak_refine): the
  physical interface moves continuously, not in discrete jumps between
  grid rows, so snapping to the single steepest raw grid cell is a
  discretization artifact of this implementation, not a property of
  "steepest gradient" as a method.

- "half_integral": the original N-percent/integral method -- the
  domain-mean profile of excess temperature is integrated from the
  ceiling downward; the layer height is where that cumulative integral
  reaches half the column's total excess. A documented simplification of
  the rigorous two-zone (Cooper) method -- domain-mean profile rather than
  per-column height averaged afterward -- traded for O(n_times) cost on
  data already cached, same honesty convention as config.ISOTHERM_LEVELS'
  "general reference points, not derived from this study's data" caveat.
  Kept, not removed, for comparison and for the tests that exercise it
  directly; no panel wires it in by default anymore.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

# Below this integral (deg C * m) / this peak excess (deg C), a frame is
# treated as having no distinguishable smoke layer -- height reported as
# the ceiling (z1), for both methods (matches the "no fire yet" ceiling
# convention every existing caller already relies on).
_MIN_INTEGRAL = 1e-6
_MIN_EXCESS_C = 1e-6

# Grid-cell width of the median filter applied to the excess-temperature
# profile before locating its steepest gradient -- comparable in size to
# the moving-average window it replaces (see module docstring: a median
# filter suppresses a single-frame outlier row a moving average would
# instead smear across its whole window).
_GRADIENT_MEDIAN_WINDOW = 5


def smoke_layer_height_series(data: np.ndarray, extent: tuple, ambient_c: float,
                              method: str = "gradient") -> np.ndarray:
    """Layer height (meters, physical z) for every frame -- shape
    (n_times,). `data` is a cached (n_times, n_z, n_x) slice array; row 0
    is the ceiling (z1) per the app's existing origin='upper' + vertical-
    flip convention (see views.py's SliceView docstring).

    method: "gradient" (default, see module docstring), "gradient_simple"
    (same definition, no smoothing/sub-cell refinement -- see
    _steepest_gradient_simple_height_series), or "half_integral" (the
    original method, kept for comparison/tests)."""
    if method == "gradient":
        return _steepest_gradient_height_series(data, extent, ambient_c)
    if method == "gradient_simple":
        return _steepest_gradient_simple_height_series(data, extent, ambient_c)
    if method == "half_integral":
        return _half_integral_height_series(data, extent, ambient_c)
    raise ValueError(f"unknown method: {method!r} (expected 'gradient', 'gradient_simple', or 'half_integral')")


def _half_integral_height_series(data: np.ndarray, extent: tuple, ambient_c: float) -> np.ndarray:
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


def _parabolic_peak_refine(g: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sub-cell refinement of a discrete argmax peak -- standard parabolic/
    quadratic peak interpolation (used widely for recovering sub-sample
    precision from a discretely-sampled peak, e.g. pitch/FFT peak
    refinement): fits a parabola through the peak `g[row, idx[row]]` and
    its immediate neighbors, returns the analytic vertex's offset from
    `idx`, in index units. A genuine local max always yields an offset in
    (-0.5, 0.5); this is the mathematically valid range for the
    technique, not an output clamp -- it bounds the *interpolation
    fraction*, never the returned height itself (see module docstring:
    no output frame is ever clamped/dropped).

    Falls back to an offset of 0 (no refinement, keep the raw discrete
    index) at either array boundary -- no neighbor exists on one side to
    fit a parabola through there -- and wherever the three points are too
    flat to fit reliably (denominator near zero)."""
    n_rows, n_cols = g.shape
    rows = np.arange(n_rows)
    offset = np.zeros(n_rows, dtype=float)
    interior = (idx > 0) & (idx < n_cols - 1)
    if not np.any(interior):
        return offset
    i = idx[interior]
    r = rows[interior]
    y_minus, y_0, y_plus = g[r, i - 1], g[r, i], g[r, i + 1]
    denom = y_minus - 2.0 * y_0 + y_plus
    safe = np.abs(denom) > 1e-12
    delta = np.zeros(i.shape, dtype=float)
    delta[safe] = 0.5 * (y_minus[safe] - y_plus[safe]) / denom[safe]
    offset[interior] = np.clip(delta, -0.5, 0.5)
    return offset


def _steepest_gradient_simple_height_series(data: np.ndarray, extent: tuple, ambient_c: float) -> np.ndarray:
    """Steckler, Quintiere & Rinkinen, NBSIR 82-2520 (1982): thermal
    interface height = elevation of maximum |d(excess temperature)/dz|,
    unsmoothed, unrefined. Minimal variant of "gradient"; no median
    filter, no sub-cell interpolation. A flat/uniform profile has no
    interface to find, so it's reported at a room extreme instead of an
    arbitrary argmax tie-break: no excess anywhere -> ceiling (z1);
    uniformly hot (fully engulfed) -> floor (z0)."""
    x0, x1, z0, z1 = extent
    n_t, n_z, n_x = data.shape
    if n_z < 2:
        return np.full(n_t, z1, dtype=float)

    z_desc = np.linspace(z1, z0, n_z)
    excess = np.clip(np.asarray(data, dtype=float) - ambient_c, 0.0, None)
    mean_excess = excess.mean(axis=2)

    z_asc = z_desc[::-1]
    mean_excess_asc = mean_excess[:, ::-1]
    grad = np.gradient(mean_excess_asc, z_asc, axis=1)

    peak_excess = mean_excess.max(axis=1)
    excess_range = peak_excess - mean_excess.min(axis=1)
    heights = np.full(n_t, z1, dtype=float)
    has_signal = peak_excess >= _MIN_EXCESS_C
    saturated = has_signal & (excess_range < _MIN_EXCESS_C)
    heights[saturated] = z0
    findable = has_signal & ~saturated
    if np.any(findable):
        idx = np.argmax(grad[findable], axis=1)
        heights[findable] = z_asc[idx]
    return heights


def _steepest_gradient_height_series(data: np.ndarray, extent: tuple, ambient_c: float) -> np.ndarray:
    """Steckler et al. 1982 thermal-interface method: the elevation of
    maximum d(excess temperature)/dz, refined to sub-grid-cell precision
    (see _parabolic_peak_refine) -- the real interface moves continuously,
    not in discrete steps between grid rows, so picking the single
    steepest raw grid cell is a discretization artifact, not a property
    of the method itself. See module docstring for the citation, the
    median-filter noise-handling rationale, and why no output frame is
    ever clamped or dropped."""
    x0, x1, z0, z1 = extent
    n_t, n_z, n_x = data.shape
    if n_z < 2:
        return np.full(n_t, z1, dtype=float)

    z_desc = np.linspace(z1, z0, n_z)  # row 0 = ceiling, matches data's row order
    excess = np.clip(np.asarray(data, dtype=float) - ambient_c, 0.0, None)
    mean_excess = excess.mean(axis=2)  # (n_t, n_z), aligned with z_desc

    # Median filter along z only (window applies per-frame, never across
    # frames) -- robust to a single outlier row a moving average would
    # instead smear across its whole window.
    smoothed = median_filter(mean_excess, size=(1, _GRADIENT_MEDIAN_WINDOW), mode="nearest")

    z_asc = z_desc[::-1]            # ascending floor -> ceiling
    smoothed_asc = smoothed[:, ::-1]
    grad = np.gradient(smoothed_asc, z_asc, axis=1)
    dz = z_asc[1] - z_asc[0]        # uniform grid spacing

    peak_excess = mean_excess.max(axis=1)
    excess_range = peak_excess - mean_excess.min(axis=1)
    heights = np.full(n_t, z1, dtype=float)
    has_signal = peak_excess >= _MIN_EXCESS_C
    # Uniformly hot column: no lower "clear" zone remains to form an
    # interface against, so there is no gradient signal to locate -- the
    # smoke has filled the whole domain (see module docstring).
    saturated = has_signal & (excess_range < _MIN_EXCESS_C)
    heights[saturated] = z0
    findable = has_signal & ~saturated
    if np.any(findable):
        g = grad[findable]
        idx = np.argmax(g, axis=1)
        offset = _parabolic_peak_refine(g, idx)
        heights[findable] = z_asc[idx] + offset * dz
    return heights
