"""Real-soot smoke intensity mapping (continuous soot-density visualization
pass): a continuous SOOT DENSITY scalar -> opacity mapping, not a threshold.

Investigated the actual data distribution before choosing a normalization
(research-readiness audit, "smoke" finding) rather than guessing: across
the real 24-scenario dataset, SOOT DENSITY is effectively bimodal, not a
smooth gradient from near-zero up. ~99.5% of cells are exactly 0 (no soot
reached there); the remaining cells are already several thousand mg/m3 --
there is no meaningful population between 0 and roughly 3900 mg/m3, and
the nonzero range itself spans less than one order of magnitude (observed
~3900-18000 mg/m3, i.e. under 5x). A log/asinh transform exists to rescue
detail buried across many decades near zero; that problem doesn't exist
here, so it would only compress the real, already-modest nonzero range
without adding information. Linear normalization against a data-driven
ceiling communicates the actual concentration honestly: exactly-zero
cells render fully transparent (correct -- there is no soot there), and
the graded opacity among nonzero cells reflects their real relative
concentration.

Pure NumPy, Qt-free, no store/provider access -- callers already have the
loaded SOOT DENSITY array.
"""

from __future__ import annotations

import numpy as np

# Percentile (not the bare max) so one spike cell doesn't set every other
# frame's scale -- matches the same "robust ceiling, not a raw max"
# reasoning already used elsewhere in this app (e.g. cinema/pipeline.py's
# AutoExposure).
DEFAULT_CEILING_PERCENTILE = 99.5

# A floor under the computed ceiling: a scenario with negligible/no soot
# (most frames, most scenarios -- see the module docstring) must not divide
# by a near-zero number and blow tiny values up to near-full opacity.
MIN_CEILING_MG_M3 = 1000.0

# Never fully occlude the temperature field underneath -- the overlay is a
# supplement to the temperature view, not a replacement for it.
MAX_OVERLAY_ALPHA = 0.85


def soot_ceiling(data: np.ndarray, percentile: float = DEFAULT_CEILING_PERCENTILE) -> float:
    """A data-driven normalization ceiling for one scenario's whole SOOT
    DENSITY run (all frames): the given percentile of every finite value,
    floored at MIN_CEILING_MG_M3. Computed once per (scenario, plane) and
    reused for every frame's opacity mapping, so the scale stays stable
    across playback instead of rescaling frame-to-frame."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return MIN_CEILING_MG_M3
    return max(float(np.percentile(finite, percentile)), MIN_CEILING_MG_M3)


def soot_alpha(frame: np.ndarray, ceiling: float, max_alpha: float = MAX_OVERLAY_ALPHA) -> np.ndarray:
    """Continuous per-cell opacity for one frame, linear in concentration:
    0 at zero soot (fully transparent -- there is no data-honesty reason
    to draw anything), rising proportionally to `max_alpha` at `ceiling`
    (clipped beyond, never fully occluding the field underneath). No
    thresholding: every nonzero value produces a proportional, nonzero
    opacity, never a boolean on/off."""
    clean = np.clip(np.nan_to_num(np.asarray(frame, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return np.clip(clean / max(float(ceiling), 1e-9), 0.0, 1.0) * max_alpha


# Percentile pair for soot_display_range: chosen to trim outlier cells at
# both ends of the *nonzero* population (see that function's docstring for
# why the floor matters as much as the ceiling here).
DISPLAY_FLOOR_PERCENTILE = 1.0
DISPLAY_CEILING_PERCENTILE = 99.0


def soot_display_range(data: np.ndarray, floor_percentile: float = DISPLAY_FLOOR_PERCENTILE,
                        ceiling_percentile: float = DISPLAY_CEILING_PERCENTILE) -> tuple:
    """A data-driven (vmin, vmax) for displaying SOOT DENSITY as the
    *primary* heatmap quantity (not the temperature overlay soot_alpha/
    soot_ceiling above serve) -- one scenario's whole run (all frames),
    computed once and reused, same "stable across playback" convention as
    soot_ceiling.

    Investigated directly (Analysis final-polish follow-up, "smoke view
    looks weird" report): a fixed vmin=0 wastes almost the entire color
    ramp on empty space. This dataset's SOOT DENSITY is bimodal -- most
    cells are exactly 0, and wherever soot IS present, concentration
    already sits within a narrow band close to a local ceiling (observed
    directly: one scenario's peak frame had nonzero values ranging only
    ~4100-5043 mg/m3, a real relative spread the fixed 0-3000/10000 scale
    compresses into a sliver at the very top, making distinct real
    concentrations look like one flat blob). Anchoring vmin at a low
    percentile of the *nonzero* population instead of 0 spends the full
    ramp on the range where real variation actually exists; exact zeros
    still render at the low (white, for gray_r) end because imshow clips
    values below vmin to vmin's color, so "no soot" and "trace soot at
    the floor" both read as near-white, which is honest -- neither has
    much soot. Confirmed empirically this reveals genuine variation (e.g.
    a thinner-smoke region vs. a denser one) that a fixed 0-vmax scale
    could not distinguish; it does not manufacture gradient detail that
    isn't in the data -- a scenario whose nonzero population truly is
    one tight cluster will still look mostly flat, honestly.

    Returns (0.0, MIN_CEILING_MG_M3) if there's no nonzero data at all
    (a scenario/plane with no soot ever) -- same floor-safety fallback
    soot_ceiling uses, so an all-zero run just renders all-white rather
    than dividing by a near-zero range."""
    finite = data[np.isfinite(data)]
    nonzero = finite[finite > 0]
    if nonzero.size == 0:
        return 0.0, MIN_CEILING_MG_M3
    floor = float(np.percentile(nonzero, floor_percentile))
    ceiling = float(np.percentile(nonzero, ceiling_percentile))
    if ceiling <= floor:
        ceiling = floor + 1e-6
    return floor, max(ceiling, MIN_CEILING_MG_M3)
