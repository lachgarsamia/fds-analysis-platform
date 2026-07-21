"""On-canvas measurement tools (V4-M7).

Deterministic geometric + value measurement over the 2D slice field:

- distance / polyline: straight-line and multi-segment lengths in metres,
  with the Delta-x / Delta-z components a fire scientist reads for flame
  width, layer depth, or plume reach;
- probe: the bilinearly-interpolated value at a physical point;
- rectangle: physical area plus min / mean / max of the quantity inside
  the box, at one instant or averaged over a V4-M5 interval.

A `Measurement` is the saved unit -- a kind, its physical points, a label,
and the human-readable readout captured when it was measured -- so it
persists in a Named Session (V4-M6) and prints in the session report. Pure
NumPy + scipy, Qt-free. Reuses the extent/coordinate convention
(phys_to_index, row 0 = top).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import map_coordinates

from timeseries import phys_to_index

KINDS = ("distance", "path", "rect", "probe")


def distance(p0, p1) -> float:
    """Straight-line physical distance between two (x, z) points."""
    return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def polyline_length(points):
    """(total_length, [segment_lengths]) for an ordered list of points."""
    pts = list(points)
    segs = [distance(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return float(sum(segs)), segs


def _fractional_index(extent, shape, x: float, z: float):
    """Physical (x, z) -> fractional (row, col) for interpolation. Row 0 is
    the top (z1), matching the app's flipped-array convention."""
    x0, x1, z0, z1 = extent
    n_z, n_x = shape
    col = (x - x0) / (x1 - x0) * (n_x - 1) if x1 != x0 else 0.0
    row = (z1 - z) / (z1 - z0) * (n_z - 1) if z1 != z0 else 0.0
    return row, col


def probe_value(frame: np.ndarray, extent, x: float, z: float) -> float:
    """Bilinearly-interpolated field value at physical (x, z), clamped to
    the domain edges."""
    frame = np.asarray(frame, dtype=float)
    if extent is None:
        r, c = 0.0, 0.0
    else:
        r, c = _fractional_index(extent, frame.shape, x, z)
    r = min(max(r, 0.0), frame.shape[0] - 1)
    c = min(max(c, 0.0), frame.shape[1] - 1)
    return float(map_coordinates(frame, [[r], [c]], order=1, mode="nearest")[0])


def rect_indices(extent, shape, x0, x1, z0, z1):
    """Inclusive (r0, r1, c0, c1) for the box's two physical corners."""
    r_a, c_a = phys_to_index(extent, shape, x0, z0)
    r_b, c_b = phys_to_index(extent, shape, x1, z1)
    r0, r1 = sorted((r_a, r_b))
    c0, c1 = sorted((c_a, c_b))
    return r0, r1, c0, c1


def rect_stats(data: np.ndarray, extent, x0, x1, z0, z1,
               frame_index: int = None, i0: int = None, i1: int = None) -> dict:
    """Area (m^2) and min/mean/max of the field inside the box. Evaluated
    at `frame_index`, or time-averaged over frames [i0, i1] when given."""
    arr = np.asarray(data, dtype=float)
    r0, r1, c0, c1 = rect_indices(extent, arr.shape[1:], x0, x1, z0, z1)
    box = arr[:, r0:r1 + 1, c0:c1 + 1]
    if i0 is not None and i1 is not None:
        field_2d = box[i0:i1 + 1].mean(axis=0)
    else:
        fi = 0 if frame_index is None else min(max(frame_index, 0), arr.shape[0] - 1)
        field_2d = box[fi]
    area = abs(x1 - x0) * abs(z1 - z0)
    return {
        "area": float(area),
        "min": float(field_2d.min()),
        "mean": float(field_2d.mean()),
        "max": float(field_2d.max()),
        "n_cells": int(field_2d.size),
    }


@dataclass
class Measurement:
    kind: str                       # distance | path | rect | probe
    points: list                    # [(x, z), ...] physical coordinates
    label: str = ""
    readout: str = ""               # human-readable result captured at measure time
    interval: bool = False          # measured as an interval average (rect/probe)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "points": [[float(x), float(z)] for x, z in self.points],
                "label": self.label, "readout": self.readout, "interval": self.interval}

    @classmethod
    def from_dict(cls, d: dict) -> "Measurement":
        return cls(kind=str(d.get("kind", "distance")),
                   points=[(float(p[0]), float(p[1])) for p in d.get("points", [])],
                   label=str(d.get("label", "")), readout=str(d.get("readout", "")),
                   interval=bool(d.get("interval", False)))
