"""Hazard Spaces (V5-M4): classify the field into occupant-hazard zones.

Instead of a bare temperature map, each cell is placed in a hazard class from
the registry's temperature bands (60 / 100 / 300 °C) *and* its cumulative
exposure above the warning threshold:

    Safe        T < 60 °C
    Warning     60 ≤ T < 100 °C
    Critical    100 ≤ T < 300 °C
    Untenable   T ≥ 300 °C, or long exposure above 60 °C (a heat-dose proxy)

`flashover_indicator` flags frames whose peak temperature exceeds a ~500 °C
indicator -- an *indicator only*, not a flashover/combustion model.

By default this is a temperature-only, partial hazard screen (no CO/CO2),
and every UI surface must say so. V6-M6: when a real CO field is supplied
(`co_field`, read by the caller through QuantityProvider -- gated until the
M-SIM re-run, see tenability.py), `classify_series` escalates on full FED
(tenability.full_fed) instead of the convected-heat-only exposure proxy --
still Pure NumPy, Qt-free.
"""

from __future__ import annotations

import numpy as np

from registry import get_quantity

CLASS_NAMES = ("Safe", "Warning", "Critical", "Untenable")
CLASS_COLORS = ("#2E7D32", "#F9A825", "#EF6C00", "#B71C1C")  # green / amber / orange / red
DEFAULT_EXPOSURE_LIMIT_S = 30.0
FLASHOVER_INDICATOR_C = 500.0
BASIS = ("temperature thresholds (60/100/300 °C) plus cumulative exposure above "
         "60 °C; temperature-only partial screen (no CO/CO₂)")
FULL_FED_BASIS = ("temperature thresholds (60/100/300 °C) plus full FED (Fractional "
                  "Effective Dose: toxic-gas CO dose + convected-heat dose, ISO 13571 / "
                  "SFPE Handbook); Untenable once FED >= 1.0 (incapacitation)")


def basis_caption(co_based: bool = False) -> str:
    """One-line, human-readable caption naming which hazard-classification
    basis is in effect (Analysis roadmap B6) -- "Basis: {BASIS}" or
    "Basis: {FULL_FED_BASIS}". The one shared source every panel that
    renders a hazard_spaces-derived classification should read this from,
    instead of hand-writing its own paraphrase (hazard_panel.py already
    did exactly this inline -- `"Basis: " + hz.BASIS` -- before this
    function existed to name that pattern so every other caller can share
    the identical wording verbatim, not just the same underlying fact)."""
    return "Basis: " + (FULL_FED_BASIS if co_based else BASIS)


def band_thresholds(quantity: str = "TEMPERATURE"):
    """The three class boundaries (warning, critical, untenable)."""
    levels = get_quantity(quantity).hazard_levels
    return tuple(levels) if levels and len(levels) >= 3 else (60.0, 100.0, 300.0)


def classify_instant(frame, thresholds) -> np.ndarray:
    """Per-cell hazard class (0..3) from instantaneous temperature."""
    t0, t1, t2 = thresholds
    f = np.asarray(frame, dtype=float)
    cls = np.zeros(f.shape, dtype=int)
    cls[f >= t0] = 1
    cls[f >= t1] = 2
    cls[f >= t2] = 3
    return cls


def classify_series(data, thresholds, fps: int,
                    exposure_limit_s: float = DEFAULT_EXPOSURE_LIMIT_S,
                    co_field=None) -> np.ndarray:
    """Per-frame, per-cell hazard class (n_t, n_z, n_x). A cell is escalated
    to Untenable once either:

    - `co_field` is given (V6-M6, a real CO ppm field): its full FED
      (tenability.full_fed) reaches 1.0 (ISO 13571 incapacitation), or
    - `co_field` is None (the default, partial screen): its cumulative time
      above the warning threshold reaches `exposure_limit_s` -- a coarse
      heat-dose proxy, kept exactly as before for backward compatibility.
    """
    arr = np.asarray(data, dtype=float)
    t0, t1, t2 = thresholds
    inst = np.zeros(arr.shape, dtype=int)
    inst[arr >= t0] = 1
    inst[arr >= t1] = 2
    inst[arr >= t2] = 3
    if co_field is not None:
        from tenability import full_fed, FED_INCAPACITATION
        fed = full_fed(arr, co_field, fps)
        escalate = np.where(fed >= FED_INCAPACITATION, 3, 0)
    else:
        exposure = np.cumsum(arr > t0, axis=0) / max(1, fps)     # seconds above warning, per cell
        escalate = np.where(exposure >= exposure_limit_s, 3, 0)
    return np.maximum(inst, escalate)


def class_fractions(class_series: np.ndarray) -> np.ndarray:
    """(n_t, 4): fraction of cells in each class per frame."""
    cs = np.asarray(class_series)
    return np.stack([(cs == c).mean(axis=(1, 2)) for c in range(4)], axis=1)


def worst_class(class_series: np.ndarray) -> np.ndarray:
    """(n_t,): the highest hazard class present in each frame."""
    cs = np.asarray(class_series)
    return cs.reshape(cs.shape[0], -1).max(axis=1)


def critical_fraction(class_series: np.ndarray) -> np.ndarray:
    """(n_t,): fraction of cells at Critical or Untenable per frame."""
    cs = np.asarray(class_series)
    return (cs >= 2).mean(axis=(1, 2))


def flashover_indicator(data, indicator_c: float = FLASHOVER_INDICATOR_C):
    """(indicated per frame, first_frame or None). Indicator only."""
    arr = np.asarray(data, dtype=float)
    peak = arr.reshape(arr.shape[0], -1).max(axis=1)
    indicated = peak >= indicator_c
    first = int(np.argmax(indicated)) if indicated.any() else None
    return indicated, first
