"""Temperature-based tenability screening (V2 roadmap M3.2).

Occupant tenability in fire has two contributions: toxic-gas dose (mainly
CO, via the Fractional Effective Dose / FED integral) and convected-heat
exposure. This dataset ships neither CO nor CO-derived output, so this
module screens on **convected heat only** -- a temperature threshold
above which exposure is treated as untenable. This is a *partial* hazard
screen, NOT a full FED analysis, and every UI surface that shows it says
so (the roadmap's explicit scientific-honesty requirement: "the app must
not imply full FED without CO"). CO2 from the volumetric `.s3d` data
(feature F8) is a documented future extension, not part of M3.2.

Pure computation, Qt-free and testable.

V6 hook (GATED): full FED. When the M-SIM re-run adds a CO output (registry
'CARBON MONOXIDE VOLUME FRACTION', currently gated), the toxic-gas FED integral
combines with the convected-heat dose here into a full FED, and the partial-
screen disclaimer retires. Add `fed_gas_dose(co_field, fps)` beside the heat
functions and sum the two contributions; no other surface changes. See
docs/msim-preparation.md §3 and ROADMAP-V6.md.
"""

from __future__ import annotations

import numpy as np

# Standard convected-heat tenability limit: sustained exposure to air
# above ~60 C is treated as untenable (matches config.ISOTHERM_LEVELS'
# lowest hazard band). Configurable in the UI.
TENABILITY_THRESHOLD_C = 60.0


def time_to_untenable_field(data: np.ndarray, threshold_c: float, fps: int) -> np.ndarray:
    """Per-cell first-crossing time (seconds) of `threshold_c` -> shape
    (n_z, n_x); np.inf where a cell never exceeds it. `data` is a cached
    (n_times, n_z, n_x) slice array."""
    fps = max(1, fps)
    exceed = np.asarray(data) > threshold_c
    ever = exceed.any(axis=0)
    first_idx = np.argmax(exceed, axis=0)  # 0 where never, masked out below
    times = first_idx.astype(float) / fps
    times[~ever] = np.inf
    return times


def time_to_untenable_scalar(data: np.ndarray, threshold_c: float, fps: int) -> float | None:
    """First time (seconds) *any* cell crosses `threshold_c`, i.e. the
    earliest onset of untenable convected-heat conditions anywhere in the
    slice. None if the threshold is never reached."""
    field = time_to_untenable_field(data, threshold_c, fps)
    finite = field[np.isfinite(field)]
    return float(finite.min()) if finite.size else None


def untenable_fraction(data: np.ndarray, threshold_c: float, frame_index: int) -> float:
    """Fraction of slice cells above `threshold_c` at `frame_index`."""
    frame = np.asarray(data)[frame_index]
    return float(np.mean(frame > threshold_c))
