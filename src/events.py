"""Event Detection Engine (V3 Phase 0, Fire Intelligence Layer).

Deterministic detectors over a scenario's frame-descriptor table
(descriptors.py) that emit `Insight` objects: ignition, fastest heating,
hazard-threshold crossings, peak, smoke-layer descent, and stabilization.
These are the "Fire Story" of a run (V3-M2) and the raw events the
semantic diff (V3-M3) compares between runs.

Every detector is a simple, explainable computation on the descriptors --
no learned models -- and every emitted Insight carries the time, value,
and a `basis` naming the descriptor it was derived from.
"""

from __future__ import annotations

import numpy as np

from registry import AMBIENT_C, get_quantity
from insight import Insight

# Ignition: first time the field rises this far above ambient.
IGNITION_DELTA_C = 30.0
# Smoke layer "descending": first drop below this fraction of its start.
LAYER_DESCENT_FRAC = 0.90
# Stabilization: heating rate has fallen below this fraction of its peak.
STABILIZE_FRAC = 0.05


def _first(condition: np.ndarray):
    hits = np.flatnonzero(condition)
    return int(hits[0]) if hits.size else None


def detect_events(desc, quantity: str = "TEMPERATURE", ambient_c: float = AMBIENT_C) -> list:
    """Return the detected events for one scenario as time-ordered
    Insights. `desc` is a descriptors.DescriptorTable."""
    q = get_quantity(quantity)
    unit = q.unit
    times = desc.times
    fps = desc.fps
    smax = desc.column("spatial_max")
    events: list = []

    def at(i, statement, value, basis, category="event"):
        return Insight(statement=statement, category=category, quantity=quantity,
                       time_s=float(times[i]), value=float(value), unit=unit, basis=basis)

    # Ignition -----------------------------------------------------------
    ign = _first(smax > ambient_c + IGNITION_DELTA_C)
    if ign is not None:
        events.append(at(ign, f"Ignition: the field first rises above "
                              f"{ambient_c + IGNITION_DELTA_C:.0f} {unit}.",
                         smax[ign], "first frame where spatial maximum exceeds ambient + 30"))

    # Hazard-threshold crossings ----------------------------------------
    for level in q.hazard_levels:
        i = _first(smax > level)
        if i is not None:
            events.append(at(i, f"Field first exceeds {level:g} {unit}.", level,
                             f"first frame where spatial maximum exceeds {level:g}"))

    # Fastest heating ----------------------------------------------------
    d_smax = desc.column("d_spatial_max")
    if d_smax.size and np.max(d_smax) > 0:
        i = int(np.argmax(d_smax))
        events.append(at(i, f"Fastest heating (~{d_smax[i]:.0f} {unit}/s).", d_smax[i],
                         "frame of maximum rate of change of the spatial maximum"))

    # Peak ---------------------------------------------------------------
    if smax.size:
        i = int(np.argmax(smax))
        events.append(at(i, f"Peak {smax[i]:.0f} {unit}.", smax[i],
                         "frame of the global spatial maximum"))

    # Smoke-layer descent ------------------------------------------------
    layer = desc.column("layer_height")
    if layer.size and layer[0] > 0:
        i = _first(layer < LAYER_DESCENT_FRAC * layer[0])
        if i is not None:
            events.append(at(i, f"Smoke layer begins descending (to {layer[i]:.2f} m).",
                             layer[i], "first frame where smoke-layer height drops below "
                                       "90% of its initial value", category="event"))

    # Stabilization ------------------------------------------------------
    if smax.size:
        peak_i = int(np.argmax(smax))
        rate = np.abs(d_smax)
        peak_rate = float(np.max(rate)) if rate.size else 0.0
        if peak_rate > 0:
            after = np.arange(len(rate)) >= peak_i
            calm = _first(after & (rate < STABILIZE_FRAC * peak_rate))
            if calm is not None:
                events.append(at(calm, "Conditions stabilize (heating slows).", smax[calm],
                                 "first post-peak frame where the heating rate falls below "
                                 "5% of its peak"))

    events.sort(key=lambda e: e.primary_time())
    return events
