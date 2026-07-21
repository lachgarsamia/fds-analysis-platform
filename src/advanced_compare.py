"""Advanced comparison workflows (V4-M8).

Extends the semantic diff (V3-M3) into the three comparison axes the brief
names, each producing navigable Insights:

- **Temporal** -- the cross-over time of a danger metric ("when did Case B
  become more dangerous than Case A?"): the first instant the lead flips
  on peak temperature and on affected (hazardous) area.
- **Spatial** -- where the two runs differ most, reported per region of a
  coarse grid, not just one global cell.
- **Physics** -- the descriptors most *associated* with the difference
  (peak HRR, fire-growth rate, affected area, released energy, smoke-layer
  descent), ranked. This is association, not proven causation, and every
  physics Insight says so (the V3-M7 honesty gate).

Pure NumPy, Qt-free, deterministic. Reuses compute_descriptors and the
scenario summaries; the danger metrics and drivers are plain reductions
with a stated basis, never a judged conclusion.
"""

from __future__ import annotations

import numpy as np

from descriptors import compute_descriptors
from layer_height import smoke_layer_height_series
from registry import get_quantity, AMBIENT_C


def _mag(a: float, b: float) -> float:
    ref = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / ref


def find_crossover(series_a, series_b, fps: int, label_a: str, label_b: str):
    """First instant the lead flips between two time series -> (time,
    leader_before, leader_after), or None if one leads throughout. Ties
    (exactly equal samples) are ignored when establishing the lead."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    signs = np.sign(b - a)
    nz = np.flatnonzero(signs != 0)
    if nz.size < 2:
        return None
    first = signs[nz[0]]
    for k in nz[1:]:
        if signs[k] == -first:
            before = label_b if first > 0 else label_a
            after = label_b if signs[k] > 0 else label_a
            return float(k) / max(1, fps), before, after
    return None


def _region_name(r: int, c: int) -> str:
    row = ("upper", "middle", "lower")[r]      # row 0 = ceiling (top)
    col = ("left", "centre", "right")[c]
    return f"{row}-{col}" if not (r == 1 and c == 1) else "centre"


def region_differences(diff_field: np.ndarray, rows: int = 3, cols: int = 3):
    """Mean magnitude of a 2D difference field per grid block, as
    [(name, value), ...] ranked high-to-low."""
    f = np.asarray(diff_field, dtype=float)
    n_z, n_x = f.shape
    r_edges = np.linspace(0, n_z, rows + 1, dtype=int)
    c_edges = np.linspace(0, n_x, cols + 1, dtype=int)
    out = []
    for r in range(rows):
        for c in range(cols):
            block = f[r_edges[r]:r_edges[r + 1], c_edges[c]:c_edges[c + 1]]
            if block.size:
                out.append((_region_name(r, c), float(np.abs(block).mean())))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def advanced_compare(data_a, data_b, extent, fps: int, quantity: str,
                     label_a: str = "A", label_b: str = "B",
                     summary_a=None, summary_b=None) -> dict:
    """Return {'temporal': [...], 'spatial': [...], 'physics': [...]} of
    Insights (each list ranked most-significant first)."""
    from insight import Insight  # local import keeps this module Qt-free
    fps = max(1, fps)
    a = np.asarray(data_a, dtype=float)
    b = np.asarray(data_b, dtype=float)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    q = get_quantity(quantity)
    unit = q.unit
    threshold = (q.hazard_levels or (AMBIENT_C * 3,))[0]

    desc_a = compute_descriptors(a, extent, fps)
    desc_b = compute_descriptors(b, extent, fps)
    smax_a, smax_b = desc_a.column("spatial_max"), desc_b.column("spatial_max")
    hot_a = (a > threshold).mean(axis=(1, 2))
    hot_b = (b > threshold).mean(axis=(1, 2))

    # --- Temporal: cross-over of danger metrics --------------------------
    temporal = []
    for metric_a, metric_b, name in ((smax_a, smax_b, "peak temperature"),
                                     (hot_a, hot_b, "hazardous area")):
        cross = find_crossover(metric_a, metric_b, fps, label_a, label_b)
        if cross is not None:
            t, before, after = cross
            temporal.append(Insight(
                f"{after} overtakes {before} in {name} at {t:.1f} s.",
                category="difference", quantity=quantity, time_s=t,
                basis=f"first lead-flip of the two {name} time series"))

    # --- Spatial: per-region difference ----------------------------------
    spatial = []
    diff_field = np.abs(a - b).mean(axis=0)   # time-averaged |A - B|
    regions = region_differences(diff_field)
    for name, value in regions[:2]:
        if value > 1e-6:
            spatial.append(Insight(
                f"Heat accumulates most differently in the {name} region "
                f"(mean |A − B| ~ {value:.0f} {unit}).",
                category="difference", quantity=quantity, value=value,
                basis="time-averaged |A − B|, averaged over the region"))

    # --- Physics: ranked associated drivers ------------------------------
    physics = _physics_drivers(desc_a, desc_b, hot_a, hot_b, extent, a, b,
                               summary_a, summary_b, quantity, unit,
                               label_a, label_b, Insight)
    return {"temporal": temporal, "spatial": spatial, "physics": physics}


def _physics_drivers(desc_a, desc_b, hot_a, hot_b, extent, a, b,
                     summary_a, summary_b, quantity, unit, label_a, label_b, Insight):
    """Rank the descriptors most associated with the difference (peak HRR,
    fire-growth rate, affected area, released energy, smoke descent).
    Honestly labelled association, not causation."""
    candidates = []  # (mag, name, higher_label, delta, unit_str)

    def add(name, va, vb, unit_str, fmt="{:.1f}"):
        if va is None or vb is None:
            return
        m = _mag(float(va), float(vb))
        delta = abs(float(vb) - float(va))
        # A large *relative* difference between two near-zero values (e.g. a
        # candle's tiny growth alpha) is not a meaningful driver -- drop it
        # if the delta rounds to zero at the figure we would display.
        if m <= 0.02 or float(fmt.format(delta)) == 0.0:
            return
        higher = label_b if vb > va else label_a
        candidates.append((m, name, higher, delta, unit_str, fmt))

    # fire-growth rate: peak rate of rise of the spatial maximum (deg C/s)
    add("fire-growth rate", float(np.max(desc_a.column("d_spatial_max"))),
        float(np.max(desc_b.column("d_spatial_max"))), f"{unit}/s")
    # affected area: peak hazardous-cell fraction
    add("affected area", float(hot_a.max()) * 100.0, float(hot_b.max()) * 100.0, "%")
    if extent is not None and quantity == "TEMPERATURE":
        add("smoke-layer minimum",
            float(smoke_layer_height_series(a, extent, AMBIENT_C).min()),
            float(smoke_layer_height_series(b, extent, AMBIENT_C).min()), "m", "{:.2f}")
    if summary_a is not None and summary_b is not None:
        add("peak HRR", getattr(summary_a, "peak_hrr_kw", None),
            getattr(summary_b, "peak_hrr_kw", None), "kW", "{:.3f}")
        add("released energy", getattr(summary_a, "total_energy_kj", None),
            getattr(summary_b, "total_energy_kj", None), "kJ")
        add("fire-growth alpha", getattr(summary_a, "growth_alpha_kw_s2", None),
            getattr(summary_b, "growth_alpha_kw_s2", None), "kW/s²", "{:.4f}")

    candidates.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _m, name, higher, delta, unit_str, fmt in candidates[:3]:
        sep = "" if unit_str in ("%", "") else " "
        val = fmt.format(delta) + (unit_str if unit_str == "%" else sep + unit_str)
        out.append(Insight(
            f"Associated driver: {higher} has {name} higher by {val} "
            f"(association with the difference, not a proven cause).",
            category="difference", quantity=quantity, value=delta,
            basis=f"ranked relative difference of {name}; association, not causation"))
    return out
