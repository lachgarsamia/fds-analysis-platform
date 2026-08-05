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


# Why a registered quantity may not yet be usable: the FDS output it needs
# is not in the current run and awaits the M-SIM cluster re-run (V4-M11).
MSIM_GATE = ("Requires the M-SIM cluster re-run (see docs/msim-preparation.md); "
             "the current FDS output does not include this quantity.")


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
    interpretation: str = ""    # one-line "what this means" for the Quantities panel
    gated: bool = False         # True -> data not available yet (see gate_reason)
    gate_reason: str = ""       # why it is gated (shown in the UI, non-breaking)
    expression: str = ""        # V6-M1: the Field Calculator expression (calculated fields)
    calculated: bool = False    # V6-M1: True for a user-defined calculated field


QUANTITY_REGISTRY = {
    "TEMPERATURE": QuantityInfo(
        # Colormap expressiveness pass: "inferno" (was "fds_fire") + a
        # fixed 20-170 °C clim (was 20-300, drifting to whatever the vmax
        # slider/adaptive logic last set). 170 = AMBIENT_C + 150 °C of
        # rise -- verified against the real 24-scenario dataset (see
        # derived_quantities.py's docstring): every scenario's peak is
        # well above this (382-469 °C absolute), so this deliberately
        # saturates the flame plume in exchange for spreading the *room's*
        # ambient-to-hazardous gradient across the visible ramp instead of
        # crushing it into a sliver below the peak. vmin stays exactly
        # AMBIENT_C (not renumbered to 0) so this fixed clim reads as
        # "floored at ambient" without touching any of the other call
        # sites that already treat this vmin as the real physical ambient
        # floor (narration's ambient_c, linked-clim's shared floor).
        "TEMPERATURE", "Temperature", "°C", "inferno", AMBIENT_C,
        slider_min=50, slider_max=1000, slider_default=int(AMBIENT_C + 150),
        hazard_levels=(60, 100, 300), kind="slice2d",
        interpretation="Gas temperature; drives buoyancy, the smoke layer, and the "
                       "convected-heat hazard (60/100/300 °C bands)."),
    "VELOCITY": QuantityInfo(
        "VELOCITY", "Air speed", "m/s", "fds_flow", 0.0,
        slider_min=1, slider_max=10, slider_default=2,
        hazard_levels=(1.0, 2.0, 3.0), kind="slice2d",
        interpretation="Speed magnitude |v| of the flow; direction is not stored "
                       "(the in-plane U/W components are gated)."),
    "SOOT DENSITY": QuantityInfo(
        # Fixed range (colormap expressiveness pass): 0-20000 mg/m3, tuned
        # to the real 24-scenario dataset's measured max-over-run (~19289
        # mg/m3 at the default y=0 plane -- verified directly, not
        # assumed) rather than the old adaptive per-scenario percentile
        # floor/ceiling (main_window._soot_display_range_for, removed) so
        # the same color means the same concentration in every scenario
        # and every frame, not just within one scenario's own playback.
        # vmin=0 (not a data-driven nonzero floor) so an empty/near-zero
        # field renders clean white under "gray_r", not a data-dependent
        # partial shade.
        "SOOT DENSITY", "Smoke (soot)", "mg/m³", "gray_r", 0.0,
        slider_min=100, slider_max=25000, slider_default=20000,
        hazard_levels=(), kind="volume",
        interpretation="Soot mass concentration from the volumetric field; a proxy "
                       "for smoke obscuration."),

    # --- Derived quantities (computable now from the fields above) ----------
    "TEMPERATURE RISE": QuantityInfo(
        "TEMPERATURE RISE", "Temperature rise (ΔT)", "°C", "fds_fire", 0.0,
        slider_min=10, slider_max=1000, slider_default=280,
        hazard_levels=(40, 80, 280), kind="derived",
        interpretation="Temperature above ambient (T − 20 °C); isolates the fire's "
                       "contribution. Derived from TEMPERATURE."),
    "DYNAMIC PRESSURE": QuantityInfo(
        # "viridis" (was "fds_flow", VELOCITY's own colormap -- the two
        # were previously indistinguishable in the View menu).
        #
        # slider_default=1, not the dataset-wide max-over-run (~9.4 Pa,
        # HVAC-forced scenarios): this quantity is genuinely bimodal
        # across the 24-scenario dataset (verified directly, all
        # scenarios, whole run) -- 16/24 scenarios (VOD open/closed,
        # natural ventilation) top out at 0.57-0.69 Pa, while only 8/24
        # (VOD=HVAC) reach 6-9.4 Pa, with nothing in between. A ceiling
        # picked from the dataset max (10, the first calibration here)
        # crushes every natural-ventilation frame -- the common case --
        # into under 7% of the ramp, which is worse than the adaptive
        # per-scenario ceiling this replaced. 1 Pa puts a natural-
        # ventilation scenario's real peak at ~60-70% of the ramp instead,
        # at the cost of an HVAC scenario saturating to the top color for
        # most of its run -- the same "sacrifice the rare extreme for
        # common-case legibility" trade-off already made for TEMPERATURE.
        # 1 is also the finest step this quantity's slider can express: it
        # moves in whole Pa (slider_min=1, no sub-1 granularity), so this
        # is the lowest fixed default actually reachable without a wider
        # slider-scale change (out of scope here). A user comparing an
        # HVAC scenario can still drag the slider up to slider_max.
        #
        # An HVAC-scenario frame pinned to the top color is DELIBERATE,
        # not a bug to "fix" by raising this default back toward the
        # dataset max -- doing that just re-crushes the 16/24 natural-
        # ventilation scenarios this default exists for. The slider is the
        # intended escape hatch for HVAC-detail inspection, by choice, not
        # by omission: a per-ventilation-class default (detect VOD=HVAC
        # from the scenario manifest and switch the ceiling to ~10 Pa for
        # those cases -- same 3 call sites in main_window.py this
        # replaced: _apply_quantity_display_defaults/_init_cell_view/
        # _redraw_cell_now) was considered and deliberately deferred: it
        # reintroduces a conditional ceiling of the same shape just
        # removed (coarser, keyed on a study-specific factor instead of
        # computed live) to solve a problem the slider already solves
        # visibly and on demand. Build it if/when the HVAC-detail-in-the-
        # default-view workflow actually comes up -- not preemptively.
        "DYNAMIC PRESSURE", "Dynamic pressure", "Pa", "viridis", 0.0,
        slider_min=1, slider_max=50, slider_default=1,
        hazard_levels=(), kind="derived",
        interpretation="Flow kinetic pressure ½ρ|v|² (ρ≈1.2 kg/m³); a scalar proxy "
                       "for flow forcing. Derived from VELOCITY (magnitude)."),

    # --- Target quantities, gated on the M-SIM re-run -----------------------
    "U-VELOCITY": QuantityInfo(
        "U-VELOCITY", "Velocity U (x-component)", "m/s", "coolwarm", -5.0,
        slider_min=1, slider_max=10, slider_default=5, kind="slice2d",
        interpretation="Signed in-plane x-velocity; with W, the true vector field "
                       "for streamlines/quiver.", gated=True, gate_reason=MSIM_GATE),
    "W-VELOCITY": QuantityInfo(
        "W-VELOCITY", "Velocity W (z-component)", "m/s", "coolwarm", -5.0,
        slider_min=1, slider_max=10, slider_default=5, kind="slice2d",
        interpretation="Signed in-plane z-velocity; with U, the true vector field "
                       "for streamlines/quiver.", gated=True, gate_reason=MSIM_GATE),
    "V-VELOCITY": QuantityInfo(
        "V-VELOCITY", "Velocity V (y-component)", "m/s", "coolwarm", -5.0,
        slider_min=1, slider_max=10, slider_default=5, kind="slice2d",
        interpretation="Signed through-plane y-velocity; with U/W, the full 3D "
                       "vector field for true 3D flow visualization (V6-M7).",
        gated=True, gate_reason=MSIM_GATE),
    "CARBON MONOXIDE VOLUME FRACTION": QuantityInfo(
        "CARBON MONOXIDE VOLUME FRACTION", "Carbon monoxide (CO)", "ppm", "inferno", 0.0,
        slider_min=100, slider_max=12000, slider_default=1200,
        hazard_levels=(1200, 6000), kind="slice2d",
        interpretation="CO volume fraction; unblocks full FED tenability (today's "
                       "screen is temperature-only, partial).", gated=True, gate_reason=MSIM_GATE),
    "PRESSURE": QuantityInfo(
        "PRESSURE", "Pressure", "Pa", "coolwarm", -20.0,
        slider_min=5, slider_max=100, slider_default=20, kind="slice2d",
        interpretation="Gauge pressure perturbation; vent-driving and doorway flow.",
        gated=True, gate_reason=MSIM_GATE),
    "VISIBILITY": QuantityInfo(
        "VISIBILITY", "Visibility", "m", "gray", 0.0,
        slider_min=1, slider_max=30, slider_default=10,
        hazard_levels=(3.0, 10.0), kind="slice2d",
        interpretation="Distance to which a sign is visible through smoke; a direct "
                       "egress-tenability metric.", gated=True, gate_reason=MSIM_GATE),
    "HEAT FLUX": QuantityInfo(
        "HEAT FLUX", "Heat flux", "kW/m²", "fds_fire", 0.0,
        slider_min=1, slider_max=100, slider_default=20,
        hazard_levels=(2.5, 10.0), kind="slice2d",
        interpretation="Radiative + convective heat flux to a surface; skin-burn and "
                       "ignition thresholds (2.5 / 10 kW/m²).", gated=True, gate_reason=MSIM_GATE),
    "SOOT MASS FRACTION": QuantityInfo(
        "SOOT MASS FRACTION", "Soot mass fraction", "—", "gray_r", 0.0,
        slider_min=1, slider_max=100, slider_default=20, kind="slice2d",
        interpretation="Soot mass per unit gas mass on the read plane; a 2D smoke "
                       "read without the volumetric decode.", gated=True, gate_reason=MSIM_GATE),
}

DEFAULT_QUANTITY = "TEMPERATURE"


def get_quantity(name: str) -> QuantityInfo:
    """Registry entry for `name`, falling back to TEMPERATURE for an
    unknown quantity (same defensive default the old QUANTITY_DISPLAY
    lookups used)."""
    return QUANTITY_REGISTRY.get(name, QUANTITY_REGISTRY[DEFAULT_QUANTITY])


def _is_real(q: "QuantityInfo") -> bool:
    """A directly-readable quantity: has data now (not gated) and comes off
    a slice/volume (not a derived computation). The legacy config views and
    the data-driven discovery only ever concerned these."""
    return (not q.gated) and q.kind in ("slice2d", "volume")


def quantity_status(name: str) -> str:
    """'available' | 'derived' | 'gated' for the Quantities reference (M11)."""
    q = QUANTITY_REGISTRY.get(name)
    if q is None:
        return "gated"
    if q.gated:
        return "gated"
    return "derived" if q.kind == "derived" else "available"


def available_quantities() -> list:
    """Names of directly-readable quantities (data present now)."""
    return [n for n, q in QUANTITY_REGISTRY.items() if _is_real(q)]


def derived_quantity_names() -> list:
    return [n for n, q in QUANTITY_REGISTRY.items() if q.kind == "derived" and not q.gated]


def gated_quantities() -> list:
    return [n for n, q in QUANTITY_REGISTRY.items() if q.gated]


def display_dict() -> dict:
    """The legacy `config.QUANTITY_DISPLAY` shape (name -> {label, unit,
    cmap, vmin, slider_*}), derived from the registry. Only the real,
    directly-readable quantities appear -- gated and derived entries (M11)
    are reference metadata, not display config, so legacy callers are
    unchanged."""
    return {
        name: {
            "label": q.label, "unit": q.unit, "cmap": q.cmap, "vmin": q.vmin,
            "slider_min": q.slider_min, "slider_max": q.slider_max,
            "slider_default": q.slider_default,
        }
        for name, q in QUANTITY_REGISTRY.items() if _is_real(q)
    }


def isotherm_dict() -> dict:
    """The legacy `config.ISOTHERM_LEVELS` shape, derived from the
    registry -- only quantities that declare hazard bands appear (SOOT has
    none, so it's absent, matching the pre-registry behaviour)."""
    return {name: list(q.hazard_levels) for name, q in QUANTITY_REGISTRY.items()
            if q.hazard_levels and _is_real(q)}
