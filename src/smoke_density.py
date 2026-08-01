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
