"""Physics Query Engine (V3-M4, Fire Intelligence Layer).

Answers physical questions about a scenario by executing them
*deterministically on the data*, never by generating text. A question is
first turned into a `Query` in a small **closed grammar** (a fixed set of
query kinds with typed parameters); the engine then computes the answer
from the field and returns it as an `Insight` bound to the time and place
that evidences it. There is no free-form reasoning and nothing is ever
asserted that was not computed -- the parser can only ever emit a Query,
and an unrecognized question yields no answer rather than a guess.

Pure NumPy, Qt-free, deterministic. Reuses the extent/coordinate
conventions of the rest of the app (row 0 = z1, the ceiling).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from registry import get_quantity

# Closed set of query kinds -- the grammar. Nothing outside this executes.
KINDS = ("first_crossing", "extreme", "plume_height", "regions_above")

# Named regions as physical bands (x in metres; z as a fraction of the
# domain height, 0 = floor, 1 = ceiling). Derived from this study's
# geometry -- the door/exit at x~0.25-0.29, candles at x~0.84-0.96.
REGIONS = {
    "anywhere": {},
    "door": {"x": (0.24, 0.30)},
    "exit": {"x": (0.24, 0.30)},
    "candle": {"x": (0.84, 0.96)},
    "ceiling": {"z_frac": (0.7, 1.0)},
    "floor": {"z_frac": (0.0, 0.3)},
}

EXAMPLE_QUERIES = (
    "First time temperature exceeds 100 near the exit",
    "Hottest region",
    "Max plume height",
    "Regions affected by ventilation",
)


@dataclass(frozen=True)
class Query:
    kind: str
    quantity: str = "TEMPERATURE"
    threshold: float = None
    region: str = "anywhere"
    which: str = "max"


def _region_mask(n_z: int, n_x: int, extent, region: str) -> np.ndarray:
    """Boolean (n_z, n_x) mask selecting the named region's cells."""
    mask = np.ones((n_z, n_x), dtype=bool)
    spec = REGIONS.get(region, {})
    if extent is None or not spec:
        return mask
    x0, x1, z0, z1 = extent
    xs = np.linspace(x0, x1, n_x)
    zs = np.linspace(z1, z0, n_z)  # row 0 = z1 (ceiling)
    if "x" in spec:
        col_ok = (xs >= spec["x"][0]) & (xs <= spec["x"][1])
        mask &= col_ok[None, :]
    if "z_frac" in spec:
        za = z0 + spec["z_frac"][0] * (z1 - z0)
        zb = z0 + spec["z_frac"][1] * (z1 - z0)
        row_ok = (zs >= za) & (zs <= zb)
        mask &= row_ok[:, None]
    return mask


def _phys(extent, n_z, n_x, row, col):
    if extent is None:
        return (float(col), float(row))
    x0, x1, z0, z1 = extent
    x = x0 + col / max(n_x - 1, 1) * (x1 - x0)
    z = z1 - row / max(n_z - 1, 1) * (z1 - z0)
    return (float(x), float(z))


def execute(query: Query, data: np.ndarray, extent, fps: int) -> list:
    """Execute a Query against a scenario's (n_t, n_z, n_x) field, returning
    Insight answers (usually one)."""
    from insight import Insight

    q = get_quantity(query.quantity)
    unit = q.unit
    fps = max(1, fps)
    arr = np.asarray(data, dtype=np.float64)
    n_t, n_z, n_x = arr.shape
    mask = _region_mask(n_z, n_x, extent, query.region)
    reg_txt = "" if query.region == "anywhere" else f" in the {query.region}"

    if query.kind == "first_crossing":
        thr = query.threshold if query.threshold is not None else (q.hazard_levels or (0,))[0]
        exceed = (arr > thr) & mask[None, :, :]
        per_frame = exceed.any(axis=(1, 2))
        hits = np.flatnonzero(per_frame)
        if hits.size == 0:
            return [Insight(f"{q.label} never exceeds {thr:g} {unit}{reg_txt}.",
                            category="query", quantity=query.quantity,
                            basis=f"no cell{reg_txt} ever exceeds {thr:g}")]
        fi = int(hits[0])
        row, col = np.argwhere(exceed[fi])[0]
        loc = _phys(extent, n_z, n_x, row, col)
        return [Insight(f"{q.label} first exceeds {thr:g} {unit}{reg_txt} at t = {fi / fps:.1f} s.",
                        category="query", quantity=query.quantity, time_s=fi / fps,
                        location=loc, value=float(thr),
                        basis=f"first frame where any cell{reg_txt} exceeds {thr:g}")]

    if query.kind == "extreme":
        filled = np.where(mask[None, :, :], arr, -np.inf)
        fi, row, col = np.unravel_index(int(np.argmax(filled)), arr.shape)
        val = float(arr[fi, row, col])
        loc = _phys(extent, n_z, n_x, row, col)
        return [Insight(f"Highest {q.label.lower()}{reg_txt} is {val:.0f} {unit} at "
                        f"t = {fi / fps:.1f} s.", category="query", quantity=query.quantity,
                        time_s=fi / fps, location=loc, value=val,
                        basis=f"global maximum of the field{reg_txt}")]

    if query.kind == "plume_height":
        thr = query.threshold if query.threshold is not None else (q.hazard_levels or (60,))[0]
        hot = (arr > thr) & mask[None, :, :]
        best_h, best = -1.0, None
        for fi in range(n_t):
            rows = np.flatnonzero(hot[fi].any(axis=1))
            if rows.size == 0:
                continue
            top_row = int(rows.min())  # smallest row index = highest z
            col = int(np.flatnonzero(hot[fi][top_row])[0])
            _x, z = _phys(extent, n_z, n_x, top_row, col)
            if z > best_h:
                best_h, best = z, (fi, top_row, col)
        if best is None:
            return [Insight(f"No {q.label.lower()} above {thr:g} {unit} to form a plume.",
                            category="query", quantity=query.quantity,
                            basis=f"no cell exceeds {thr:g}")]
        fi, row, col = best
        loc = _phys(extent, n_z, n_x, row, col)
        return [Insight(f"Plume reaches its greatest height (z = {best_h:.2f} m) at "
                        f"t = {fi / fps:.1f} s.", category="query", quantity=query.quantity,
                        time_s=fi / fps, location=loc, value=best_h,
                        basis=f"highest cell above {thr:g} {unit} over time")]

    if query.kind == "regions_above":
        thr = query.threshold if query.threshold is not None else (q.hazard_levels or (1,))[0]
        ever = ((arr > thr) & mask[None, :, :]).any(axis=0)
        count = int(ever.sum())
        total = int(mask.sum())
        if count == 0:
            return [Insight(f"No region{reg_txt} is affected ({q.label.lower()} never exceeds "
                            f"{thr:g} {unit}).", category="query", quantity=query.quantity,
                            basis=f"no cell{reg_txt} exceeds {thr:g}")]
        rows, cols = np.nonzero(ever)
        row, col = int(rows.mean()), int(cols.mean())
        loc = _phys(extent, n_z, n_x, row, col)
        return [Insight(f"{count} of {total} cells{reg_txt} are affected "
                        f"({q.label.lower()} exceeds {thr:g} {unit} there).",
                        category="query", quantity=query.quantity, location=loc,
                        value=float(count),
                        basis=f"cells{reg_txt} exceeding {thr:g} at any time")]

    return []


def _find_number(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def parse(text: str):
    """Parse a question into a Query in the closed grammar, or None if it
    matches no known pattern. Keyword-based and deterministic -- it can
    only ever produce a Query, never an answer."""
    t = text.lower().strip()
    if not t:
        return None

    region = "anywhere"
    for name in ("exit", "door", "candle", "ceiling", "floor"):
        if name in t:
            region = name
            break

    ventilation = any(w in t for w in ("ventilation", "airflow", "air speed", "velocity", "air flow"))
    quantity = "VELOCITY" if ventilation else "TEMPERATURE"

    if "plume" in t:
        return Query("plume_height", "TEMPERATURE", _find_number(t), region)
    if ("region" in t or "affected" in t) and ventilation:
        return Query("regions_above", "VELOCITY", _find_number(t), region)
    if any(w in t for w in ("hottest", "highest", "hot spot", "max temp", "maximum temp")):
        return Query("extreme", "TEMPERATURE", None, region)
    if any(w in t for w in ("first", "reach", "exceed", "above", "when")):
        thr = _find_number(t)
        if thr is not None:
            return Query("first_crossing", quantity, thr, region)
    if "region" in t or "affected" in t:
        return Query("regions_above", quantity, _find_number(t), region)
    return None
