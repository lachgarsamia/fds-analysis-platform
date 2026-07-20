"""Cause Explorer (V3-M7, Fire Intelligence Layer -- gated).

Answers "why is this region hot?" with a physics-based chain rather than
AI: from a picked cell it traces *up the temperature gradient*, through
the connected hot region, back to the hottest connected point -- the fire
source -- and emits the chain as an Insight sequence (this region <- hot
gas along the gradient <- the source).

GATE (honesty): true causal tracing follows the *flow field*, which needs
the real U/W-velocity components this dataset does not yet have (the
M-SIM wishlist, docs/msim-preparation.md). Until then this traces by
temperature-gradient ascent and connectivity only, which shows
*association*, NOT proven causation. Every Insight says so, and the panel
carries a prominent disclaimer. Once M-SIM adds real velocity, the same
chain can follow the flow and the wording can be upgraded.

Pure NumPy, Qt-free, deterministic.
"""

from __future__ import annotations

import numpy as np

from registry import AMBIENT_C

# Below ambient + this, a cell has no active heat to trace a source for.
_HOT_DELTA_C = 40.0
_ASSOCIATION = "gradient-ascent association (no velocity field) — not proven causation"


def _phys(extent, n_z, n_x, row, col):
    if extent is None:
        return (float(col), float(row))
    x0, x1, z0, z1 = extent
    x = x0 + col / max(n_x - 1, 1) * (x1 - x0)
    z = z1 - row / max(n_z - 1, 1) * (z1 - z0)
    return (float(x), float(z))


def trace_to_source(frame: np.ndarray, row: int, col: int) -> list:
    """Steepest-ascent path (8-connected) from (row, col) up the
    temperature field to the hottest connected point. Monotonically
    increasing in temperature by construction; ends at a local maximum."""
    n_z, n_x = frame.shape
    path = [(int(row), int(col))]
    visited = {(int(row), int(col))}
    r, c = int(row), int(col)
    for _ in range(n_z * n_x):
        best, best_val = None, float(frame[r, c])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_z and 0 <= cc < n_x and (rr, cc) not in visited \
                        and frame[rr, cc] > best_val:
                    best, best_val = (rr, cc), float(frame[rr, cc])
        if best is None:
            break
        visited.add(best)
        path.append(best)
        r, c = best
    return path


def explain(frame: np.ndarray, extent, time_s: float, row: int, col: int) -> tuple:
    """Return (insights, path_rowcol) explaining why the cell (row, col) is
    hot, as an Insight chain and the traced path (for the UI to draw)."""
    from insight import Insight

    n_z, n_x = frame.shape
    start_val = float(frame[row, col])
    start_loc = _phys(extent, n_z, n_x, row, col)

    if start_val < AMBIENT_C + _HOT_DELTA_C:
        return ([Insight(f"This point is near ambient ({start_val:.0f} °C) — no active heat "
                         f"source to trace here.", category="cause", quantity="TEMPERATURE",
                         time_s=time_s, location=start_loc, value=start_val,
                         basis="picked cell is below ambient + 40 °C")], [(row, col)])

    path = trace_to_source(frame, row, col)
    src_r, src_c = path[-1]
    src_val = float(frame[src_r, src_c])
    src_loc = _phys(extent, n_z, n_x, src_r, src_c)

    insights = [Insight(
        f"This region is hot ({start_val:.0f} °C).", category="cause", quantity="TEMPERATURE",
        time_s=time_s, location=start_loc, value=start_val, basis="picked cell temperature")]

    if len(path) > 2:
        mid_r, mid_c = path[len(path) // 2]
        insights.append(Insight(
            f"The heat rises along the temperature gradient toward the plume.",
            category="cause", quantity="TEMPERATURE", time_s=time_s,
            location=_phys(extent, n_z, n_x, mid_r, mid_c),
            basis=_ASSOCIATION))

    if (src_r, src_c) != (row, col):
        insights.append(Insight(
            f"It traces back to the hottest connected point ({src_val:.0f} °C) — the fire source.",
            category="cause", quantity="TEMPERATURE", time_s=time_s, location=src_loc,
            value=src_val, basis="local maximum of the connected hot region (" + _ASSOCIATION + ")"))
    else:
        insights[-1] = Insight(
            f"This is itself the hottest connected point ({start_val:.0f} °C) — a fire source.",
            category="cause", quantity="TEMPERATURE", time_s=time_s, location=start_loc,
            value=start_val, basis="picked cell is a local maximum")

    return insights, path
