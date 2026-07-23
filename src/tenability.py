"""Tenability screening: convected-heat-only (V2 roadmap M3.2) and full FED
(V6-M6), Qt-free and testable.

Occupant tenability in fire has two contributions: toxic-gas dose (mainly
CO, via the Fractional Effective Dose / FED integral) and convected-heat
exposure. This dataset's CO output ('CARBON MONOXIDE VOLUME FRACTION') is
registered but gated (docs/msim-preparation.md §3) -- absent until the
M-SIM re-run -- so every caller of the FED functions below reads CO through
QuantityProvider.get(), which raises GatedQuantityError immediately (the
registry's own gate, before any store access) rather than fabricating a gas
dose. When CO is unavailable, callers fall back to the pre-existing
convected-heat-only screen (`time_to_untenable_*`/`untenable_fraction`) and
say so plainly -- the "partial screen" disclaimer only retires once a real
full-FED result exists for that scenario.

Full FED (V6-M6): the standard ISO 13571 / SFPE Handbook (Purser) dose
equations, in their commonly-cited simplified form (no respiratory-minute-
volume or activity-level adjustment -- consistent with this app's existing
level of simplification for the heat-only screen, and always shown with
its basis, never presented as a certified life-safety calculation):

    FED_CO (toxic gas)     : d(FED)/dt = [CO_ppm]^1.036 / 35000  per minute
    FED_heat (convected)   : d(FED)/dt = 1 / t_I(T), t_I(T) = exp(5.1849 - 0.0273*T) minutes
    FED_full               : FED_CO + FED_heat (ISO 13571's combination rule)

FED >= 1.0 conventionally marks incapacitation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Standard convected-heat tenability limit: sustained exposure to air
# above ~60 C is treated as untenable (matches config.ISOTHERM_LEVELS'
# lowest hazard band). Configurable in the UI.
TENABILITY_THRESHOLD_C = 60.0

# Conventional FED incapacitation threshold (ISO 13571): FED >= 1.0.
FED_INCAPACITATION = 1.0


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


# ------------------------------------------------------------- full FED (V6-M6)
def fed_heat_dose(temp_c: np.ndarray, fps: int) -> np.ndarray:
    """Cumulative convected-heat FED per cell, shape (n_t, n_z, n_x) --
    the Purser thermal-tolerance equation (ISO 13571 / SFPE Handbook):
    time-to-incapacitation t_I(T) = exp(5.1849 - 0.0273*T) minutes, so each
    minute's dose increment is 1/t_I(T). Always computable (no gating --
    TEMPERATURE is never gated)."""
    fps = max(1, fps)
    t = np.asarray(temp_c, dtype=float)
    dt_min = 1.0 / (fps * 60.0)
    increment = np.exp(0.0273 * t - 5.1849) * dt_min
    return np.cumsum(increment, axis=0)


def fed_gas_dose(co_ppm: np.ndarray, fps: int) -> np.ndarray:
    """Cumulative toxic-gas (CO) FED per cell, shape (n_t, n_z, n_x) -- the
    Purser CO-FED equation (ISO 13571 / SFPE Handbook): each minute's dose
    increment is [CO_ppm]^1.036 / 35000. `co_ppm` must be a real field
    (read via QuantityProvider, which gates 'CARBON MONOXIDE VOLUME
    FRACTION' until it exists) -- this function never fabricates one."""
    fps = max(1, fps)
    co = np.clip(np.asarray(co_ppm, dtype=float), 0.0, None)
    dt_min = 1.0 / (fps * 60.0)
    increment = (co ** 1.036) / 35000.0 * dt_min
    return np.cumsum(increment, axis=0)


def full_fed(temp_c: np.ndarray, co_ppm: np.ndarray, fps: int) -> np.ndarray:
    """Full Fractional Effective Dose per cell: toxic-gas dose plus
    convected-heat dose (ISO 13571's combination rule). Requires a real CO
    field (see fed_gas_dose) -- the partial, heat-only screen above is what
    every caller falls back to when CO is gated."""
    return fed_gas_dose(co_ppm, fps) + fed_heat_dose(temp_c, fps)


def time_to_fed_field(fed_field: np.ndarray, fps: int,
                      level: float = FED_INCAPACITATION) -> np.ndarray:
    """Per-cell first-crossing time (seconds) of `fed_field` (an already-
    computed cumulative FED array, e.g. from full_fed) reaching `level` --
    shape (n_z, n_x); np.inf where a cell never reaches it. Same
    first-crossing convention as time_to_untenable_field."""
    fps = max(1, fps)
    exceed = np.asarray(fed_field) >= level
    ever = exceed.any(axis=0)
    first_idx = np.argmax(exceed, axis=0)
    times = first_idx.astype(float) / fps
    times[~ever] = np.inf
    return times


def time_to_fed_scalar(fed_field: np.ndarray, fps: int,
                       level: float = FED_INCAPACITATION) -> Optional[float]:
    """First time (seconds) *any* cell's FED reaches `level` (1.0 =
    incapacitation, the ISO 13571 convention). None if never reached."""
    fps = max(1, fps)
    exceed = np.asarray(fed_field) >= level
    any_per_frame = exceed.reshape(exceed.shape[0], -1).any(axis=1)
    if not any_per_frame.any():
        return None
    return float(np.argmax(any_per_frame)) / fps
