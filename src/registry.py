"""Quantity registry (V2 roadmap M0.2): one structured source of truth
for every physical quantity the app knows how to display -- its label,
unit, colormap, colour-scale policy, hazard/contour bands, and *kind*
(how it is produced and rendered). Formalizes what was previously two
loosely-coupled dicts in config.py (`QUANTITY_DISPLAY` + `ISOTHERM_LEVELS`),
which are now derived views over this registry so every existing call
site keeps working unchanged.

`kind` discriminates how a quantity flows through the pipeline:
  slice2d  -- a 2D `.sf` slice (TEMPERATURE, VELOCITY)
  volume   -- extracted from volumetric `.s3d` data (SOOT DENSITY; M2.1/M2.2)
  series   -- a per-frame scalar time series (reserved; e.g. HRR)
  derived  -- computed from other quantities (reserved; F7 derived fields)

This module owns no colormap objects (only names) and imports nothing
from config, so config can derive from it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ambient (room) temperature, °C -- the temperature colour scale's fixed
# lower bound (owned here so the registry entry below and config's
# re-export share one definition).
AMBIENT_C = 20.0


@dataclass(frozen=True)
class QuantityInfo:
    name: str            # FDS quantity name, the registry key
    label: str           # plain-language display label
    unit: str
    cmap: str            # matplotlib/registered colormap name
    vmin: float          # fixed colour-scale floor
    slider_min: int
    slider_max: int
    slider_default: int  # initial colour-scale ceiling
    hazard_levels: tuple = ()   # isotherm/contour bands (empty = none)
    kind: str = "slice2d"


QUANTITY_REGISTRY = {
    "TEMPERATURE": QuantityInfo(
        "TEMPERATURE", "Temperature", "°C", "fds_fire", AMBIENT_C,
        slider_min=50, slider_max=1000, slider_default=300,
        hazard_levels=(60, 100, 300), kind="slice2d"),
    "VELOCITY": QuantityInfo(
        "VELOCITY", "Air speed", "m/s", "fds_flow", 0.0,
        slider_min=1, slider_max=10, slider_default=2,
        hazard_levels=(1.0, 2.0, 3.0), kind="slice2d"),
    "SOOT DENSITY": QuantityInfo(
        "SOOT DENSITY", "Smoke (soot)", "mg/m³", "gray_r", 0.0,
        slider_min=100, slider_max=10000, slider_default=3000,
        hazard_levels=(), kind="volume"),
}

DEFAULT_QUANTITY = "TEMPERATURE"


def get_quantity(name: str) -> QuantityInfo:
    """Registry entry for `name`, falling back to TEMPERATURE for an
    unknown quantity (same defensive default the old QUANTITY_DISPLAY
    lookups used)."""
    return QUANTITY_REGISTRY.get(name, QUANTITY_REGISTRY[DEFAULT_QUANTITY])


def display_dict() -> dict:
    """The legacy `config.QUANTITY_DISPLAY` shape (name -> {label, unit,
    cmap, vmin, slider_*}), derived from the registry."""
    return {
        name: {
            "label": q.label, "unit": q.unit, "cmap": q.cmap, "vmin": q.vmin,
            "slider_min": q.slider_min, "slider_max": q.slider_max,
            "slider_default": q.slider_default,
        }
        for name, q in QUANTITY_REGISTRY.items()
    }


def isotherm_dict() -> dict:
    """The legacy `config.ISOTHERM_LEVELS` shape, derived from the
    registry -- only quantities that declare hazard bands appear (SOOT has
    none, so it's absent, matching the pre-registry behaviour)."""
    return {name: list(q.hazard_levels) for name, q in QUANTITY_REGISTRY.items() if q.hazard_levels}
