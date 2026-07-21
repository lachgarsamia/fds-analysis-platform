"""Semantic Fire Diff (V3-M3, Fire Intelligence Layer).

"GitHub diff for CFD": instead of a raw pixel subtraction, compare two
scenarios in the language of fire physics and emit ranked, navigable
Insights -- where the biggest difference is, which peaks higher, which
reaches each hazard threshold sooner, whose smoke layer descends lower,
and which becomes untenable first. Each Insight carries the time (and,
for the spatial one, the location) that evidences it, so a reader can
click through to the difference field at that instant.

Pure NumPy, Qt-free, deterministic. Every statement is templated from
computed values with a traceable basis (the auto_summary honesty rule);
the differences are described, not judged. Reuses the descriptor engine
(V3 Phase 0) and the smoke-layer computation.
"""

from __future__ import annotations

import numpy as np

from registry import get_quantity, AMBIENT_C
from descriptors import compute_descriptors
from layer_height import smoke_layer_height_series

_N_SAMPLES = 24


def _first_crossing_time(series: np.ndarray, fps: int, level: float):
    hits = np.flatnonzero(series > level)
    return float(hits[0] / fps) if hits.size else None


def _mag(a: float, b: float) -> float:
    """Relative magnitude of a difference, for ranking (0..1-ish)."""
    ref = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / ref


def compare(data_a: np.ndarray, data_b: np.ndarray, extent, fps: int, quantity: str,
            label_a: str = "A", label_b: str = "B",
            summary_a=None, summary_b=None) -> list:
    """Return ranked physics differences between two scenarios as
    insight.Insight objects (most significant first)."""
    from insight import Insight  # local import: keep this module Qt-free to import

    q = get_quantity(quantity)
    unit = q.unit
    fps = max(1, fps)
    a = np.asarray(data_a, dtype=np.float64)
    b = np.asarray(data_b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]

    scored = []  # (rank_magnitude, Insight)

    def add(mag, statement, **kw):
        scored.append((mag, Insight(statement=statement, category="difference",
                                    quantity=quantity, unit=unit, **kw)))

    # --- WHERE: the largest sustained difference, and its location -------
    sample = np.linspace(0, n - 1, min(n, _N_SAMPLES), dtype=int)
    abs_diff = np.abs(a[sample] - b[sample])
    mean_field = abs_diff.mean(axis=0)
    row, col = np.unravel_index(int(np.argmax(mean_field)), mean_field.shape)
    peak_mag = float(mean_field[row, col])
    location = None
    if extent is not None:
        x0, x1, z0, z1 = extent
        n_z, n_x = mean_field.shape
        x = x0 + col / max(n_x - 1, 1) * (x1 - x0)
        z = z1 - row / max(n_z - 1, 1) * (z1 - z0)   # row 0 = z1 (top)
        location = (x, z)
    # the sampled frame where that cell differs most -> the time to show
    local = abs_diff[:, row, col]
    t_where = float(sample[int(np.argmax(local))] / fps)
    where_txt = (f" near x={location[0]:.2f} m, z={location[1]:.2f} m" if location else "")
    add(1.0, f"Biggest {q.label.lower()} difference{where_txt} "
             f"(~{peak_mag:.0f} {unit}).", time_s=t_where, location=location,
        value=peak_mag, basis="time-averaged |A − B|, peak cell")

    # --- peak difference -------------------------------------------------
    desc_a = compute_descriptors(a, extent, fps)
    desc_b = compute_descriptors(b, extent, fps)
    smax_a, smax_b = desc_a.column("spatial_max"), desc_b.column("spatial_max")
    peak_a, peak_b = float(smax_a.max()), float(smax_b.max())
    if _mag(peak_a, peak_b) > 0.02:
        higher = label_b if peak_b > peak_a else label_a
        t_peak = float(max(np.argmax(smax_a), np.argmax(smax_b)) / fps)
        add(_mag(peak_a, peak_b),
            f"{higher} peaks {abs(peak_b - peak_a):.0f} {unit} higher.",
            time_s=t_peak, value=abs(peak_b - peak_a),
            basis="difference of the two spatial-maximum peaks")

    # --- threshold-timing differences -----------------------------------
    for level in q.hazard_levels:
        ta = _first_crossing_time(smax_a, fps, level)
        tb = _first_crossing_time(smax_b, fps, level)
        if ta is not None and tb is not None and abs(ta - tb) >= 1.0 / fps:
            sooner = label_b if tb < ta else label_a
            add(_mag(ta, tb),
                f"{sooner} reaches {level:g} {unit} {abs(tb - ta):.1f} s sooner.",
                time_s=min(ta, tb), value=abs(tb - ta),
                basis=f"difference of first-crossing times of {level:g} {unit}")
        elif (ta is None) != (tb is None):
            reached = label_a if ta is not None else label_b
            add(0.6, f"Only {reached} ever reaches {level:g} {unit}.",
                time_s=(ta if ta is not None else tb),
                basis=f"only one scenario crosses {level:g} {unit}")

    # --- smoke-layer descent (temperature, needs geometry) --------------
    if extent is not None and quantity == "TEMPERATURE":
        lo_a = float(smoke_layer_height_series(a, extent, AMBIENT_C).min())
        lo_b = float(smoke_layer_height_series(b, extent, AMBIENT_C).min())
        if abs(lo_a - lo_b) > 0.01:
            lower = label_b if lo_b < lo_a else label_a
            add(_mag(lo_a, lo_b),
                f"{lower}'s smoke layer descends {abs(lo_b - lo_a):.2f} m lower.",
                value=abs(lo_b - lo_a),
                basis="difference of minimum smoke-layer heights")

    # --- tenability / energy (from summaries, if available) -------------
    if summary_a is not None and summary_b is not None:
        ua = getattr(summary_a, "time_to_untenable_s", None)
        ub = getattr(summary_b, "time_to_untenable_s", None)
        if ua is not None and ub is not None and abs(ua - ub) >= 1.0 / fps:
            sooner = label_b if ub < ua else label_a
            add(_mag(ua, ub),
                f"{sooner} becomes untenable {abs(ub - ua):.1f} s sooner.",
                time_s=min(ua, ub), value=abs(ub - ua),
                basis="difference of time-to-untenable (temperature screen)")
        ea = getattr(summary_a, "total_energy_kj", None)
        eb = getattr(summary_b, "total_energy_kj", None)
        if ea is not None and eb is not None and _mag(ea, eb) > 0.05:
            more = label_b if eb > ea else label_a
            add(_mag(ea, eb), f"{more} releases {abs(eb - ea):.1f} kJ more energy.",
                value=abs(eb - ea), basis="difference of total released energy")

    scored.sort(key=lambda t: t[0], reverse=True)
    return [ins for _mag_, ins in scored]


def difference_statements(insights: list) -> list:
    """Plain statement strings (for the comparison report's bullet list)."""
    return [ins.statement for ins in insights]
