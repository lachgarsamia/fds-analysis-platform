"""Derived quantities (V4-M11): fields computed from the ones already in
the run, so they need no new FDS output.

- Temperature rise (ΔT): T − ambient, isolating the fire's contribution.
- Dynamic pressure: ½ρ|v|², a scalar flow-forcing proxy from the stored
  speed magnitude.

Pure NumPy, Qt-free. Each derived registry entry (registry.py, kind =
"derived") maps to one function here via DERIVED; the Quantities panel
uses them to show a live preview on real data, proving feasibility without
threading a new field through the store.
"""

from __future__ import annotations

import numpy as np

from registry import AMBIENT_C

RHO_AIR = 1.2  # kg/m^3, approximate room-air density for the dynamic-pressure proxy


def temperature_rise(temperature_frame, ambient: float = AMBIENT_C) -> np.ndarray:
    """T − ambient (deg C above room), from a TEMPERATURE field."""
    return np.asarray(temperature_frame, dtype=float) - ambient


def dynamic_pressure(speed_frame, rho: float = RHO_AIR) -> np.ndarray:
    """½ρ|v|² (Pa), from the VELOCITY speed magnitude."""
    v = np.asarray(speed_frame, dtype=float)
    return 0.5 * rho * v * v


# derived quantity name -> (source quantity, function)
DERIVED = {
    "TEMPERATURE RISE": ("TEMPERATURE", temperature_rise),
    "DYNAMIC PRESSURE": ("VELOCITY", dynamic_pressure),
}


def derive(name: str, source_frame) -> np.ndarray:
    """Compute a derived quantity from its source field, or raise KeyError
    for an unknown name."""
    _source, fn = DERIVED[name]
    return fn(source_frame)


def source_quantity(name: str):
    """The base quantity a derived one is computed from, or None."""
    entry = DERIVED.get(name)
    return entry[0] if entry else None


# Floor under any computed display ceiling below, mirroring
# smoke_density.MIN_CEILING_MG_M3's reasoning: a scenario with a
# near-zero field must not divide the color range down to nothing.
MIN_DYNAMIC_PRESSURE_CEILING_PA = 0.05


def display_ceiling(data, percentile: float = 99.0, floor: float = MIN_DYNAMIC_PRESSURE_CEILING_PA) -> float:
    """A data-driven display-scale ceiling for one scenario's whole run of
    a derived quantity (all frames) -- percentile (not the bare max) so one
    outlier cell/frame doesn't set the scale, same "robust ceiling"
    reasoning smoke_density.soot_ceiling already uses for SOOT DENSITY.

    Investigated directly (Live-polish follow-up, "only see a blue
    heatmap" report on DYNAMIC PRESSURE): this quantity's absolute scale
    varies roughly 15x across this dataset depending on ventilation mode
    (~0.6-0.7 Pa for natural ventilation vs. ~8-9.4 Pa for HVAC-forced
    scenarios) -- a single fixed registry default can't fit both regimes,
    so it under-fills the color range for most scenarios (natural
    ventilation is the common case). Computed once per (scenario, key) by
    the caller and reused, same "stable across playback" convention as
    soot_ceiling."""
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return floor
    return max(float(np.percentile(finite, percentile)), floor)
