"""Named region / zone statistics (V4-M4).

A `Zone` is a named rectangle in *physical* coordinates, so the same zone
(e.g. "doorway") applies to any scenario and is compared across them. For
a zone and a scenario's field this module computes the roadmap's bundle --
mean / max temperature, time-to-threshold, thermal dose (heat-exposure
integral), hazard duration, affected-cell fraction, and an energy proxy --
each as a scalar *and* a curve over time.

Every value is a deterministic reduction with a stated basis; the energy
figure is explicitly a *proxy* (thermal dose x area), not a calorimetric
kW. Smoke accumulation is soot-gated: it is only meaningful with SOOT
DENSITY data, so the panel adds it when that field exists rather than
faking it here. Pure NumPy, Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from timeseries import phys_to_index


@dataclass
class Zone:
    name: str
    x0: float
    x1: float
    z0: float
    z1: float

    def to_dict(self) -> dict:
        return {"name": self.name, "x0": self.x0, "x1": self.x1,
                "z0": self.z0, "z1": self.z1}

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        return cls(str(d.get("name", "zone")), float(d["x0"]), float(d["x1"]),
                   float(d["z0"]), float(d["z1"]))

    def area(self) -> float:
        """Physical area in m^2."""
        return abs(self.x1 - self.x0) * abs(self.z1 - self.z0)


def zone_indices(extent, shape, zone: Zone):
    """(r0, r1, c0, c1) inclusive array bounds for the zone's two physical
    corners, ordered low->high. Reuses the app's phys_to_index convention
    (row 0 = top), so a drawn rectangle maps to the same cells the Live
    probe would report."""
    r_a, c_a = phys_to_index(extent, shape, zone.x0, zone.z0)
    r_b, c_b = phys_to_index(extent, shape, zone.x1, zone.z1)
    r0, r1 = sorted((r_a, r_b))
    c0, c1 = sorted((c_a, c_b))
    return r0, r1, c0, c1


def _first_crossing_time(curve: np.ndarray, threshold: float, fps: int):
    """Seconds at which `curve` first exceeds `threshold`, or None."""
    hits = np.flatnonzero(np.asarray(curve) > threshold)
    return float(hits[0]) / max(1, fps) if hits.size else None


def zone_bundle(data: np.ndarray, extent, zone: Zone, fps: int,
                threshold: float, ambient: float) -> dict:
    """The full per-zone bundle for one scenario. Returns scalars and the
    time curves behind them; times are seconds via `fps`."""
    fps = max(1, fps)
    r0, r1, c0, c1 = zone_indices(extent, np.asarray(data).shape[1:], zone)
    sub = np.asarray(data, dtype=float)[:, r0:r1 + 1, c0:c1 + 1]  # (n_t, h, w)
    n_t = sub.shape[0]
    times = np.arange(n_t) / fps

    region_mean = sub.mean(axis=(1, 2))
    region_max = sub.max(axis=(1, 2))
    affected = (sub > threshold).mean(axis=(1, 2))          # fraction of cells hazardous
    over = region_max > threshold                            # any cell hazardous this frame

    # thermal dose: heat-exposure integral of the region-mean excess over
    # ambient (deg C . s), and its running accumulation.
    excess = np.clip(region_mean - ambient, 0.0, None)
    dose_curve = np.cumsum(excess) / fps
    thermal_dose = float(dose_curve[-1]) if n_t else 0.0

    return {
        "times": times,
        "region_mean": region_mean,
        "region_max": region_max,
        "affected": affected,
        "dose_curve": dose_curve,
        "n_cells": int(sub.shape[1] * sub.shape[2]),
        # --- scalars ---
        "mean_temperature": float(sub.mean()) if n_t else 0.0,
        "max_temperature": float(sub.max()) if n_t else 0.0,
        "time_to_threshold": _first_crossing_time(region_max, threshold, fps),
        "thermal_dose": thermal_dose,                        # deg C . s
        "hazard_duration": float(np.count_nonzero(over)) / fps,
        "peak_affected_fraction": float(affected.max()) if n_t else 0.0,
        "energy_proxy": thermal_dose * zone.area(),          # deg C . s . m^2 (proxy)
    }


def smoke_accumulation(soot_data: np.ndarray, extent, zone: Zone, fps: int) -> float:
    """Soot-gated extra: time integral of the zone's mean soot density
    (accumulated smoke). Only call when SOOT DENSITY data exists."""
    fps = max(1, fps)
    r0, r1, c0, c1 = zone_indices(extent, np.asarray(soot_data).shape[1:], zone)
    sub = np.asarray(soot_data, dtype=float)[:, r0:r1 + 1, c0:c1 + 1]
    return float(sub.mean(axis=(1, 2)).sum()) / fps
