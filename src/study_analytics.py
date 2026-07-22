"""Study-level analytics (V5-M2): the 24-run candle factorial as a designed
experiment.

Every scenario summary (summary_stats.ScenarioSummary) already carries its
design factors (candles / door / vod / voc) and its response scalars (peak T,
HRR, energy, growth alpha, layer minimum, threshold times). This module treats
those as a parameter × response table and computes, deterministically and with
a traceable basis:

- factor influence: how much moving each factor moves a response (the spread of
  its per-level group means) -- a scalar-level complement to factor_effects'
  field-level main effects;
- a correlation matrix among responses;
- outlier scores (each scenario's standardized distance from the ensemble);
- study statistics (min/mean/max/std per column);
- min-max normalization for a parallel-coordinates plot.

No store reads and no new simulations -- everything is a reduction of the
already-computed summaries. Pure NumPy, Qt-free.
"""

from __future__ import annotations

import numpy as np

PARAMS = ("candles", "door", "vod", "voc")
PARAM_LABELS = {"candles": "Candles", "door": "Door", "vod": "VOD", "voc": "VOC"}

# (summary attribute, label, unit)
RESPONSE_FIELDS = [
    ("max_temp_c", "Peak T", "°C"),
    ("mean_upper_temp_c", "Mean upper T", "°C"),
    ("peak_hrr_kw", "Peak HRR", "kW"),
    ("total_energy_kj", "Energy", "kJ"),
    ("growth_alpha_kw_s2", "Growth α", "kW/s²"),
    ("layer_min_height_m", "Layer min", "m"),
    ("time_to_100c_s", "t→100°C (arrival)", "s"),
    ("time_to_300c_s", "t→300°C (arrival)", "s"),
    ("time_to_600c_s", "t→600°C (arrival)", "s"),
    ("time_to_untenable_s", "t untenable", "s"),
]
RESPONSE_KEYS = [f[0] for f in RESPONSE_FIELDS]
RESPONSE_LABEL = {f[0]: f[1] for f in RESPONSE_FIELDS}
RESPONSE_UNIT = {f[0]: f[2] for f in RESPONSE_FIELDS}


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else float("nan")


def build_table(summaries: list) -> list:
    """[{case_index, folder, params{...}, responses{...}}, ...] sorted by
    case_index, straight from the scenario summaries."""
    rows = []
    for s in sorted(summaries, key=lambda s: s.case_index):
        rows.append({
            "case_index": s.case_index,
            "folder": s.folder,
            "params": {p: _num(getattr(s, p, None)) for p in PARAMS},
            "responses": {k: _num(getattr(s, k, None)) for k in RESPONSE_KEYS},
        })
    return rows


def column(table: list, key: str, kind: str = "response") -> np.ndarray:
    field = "params" if kind == "param" else "responses"
    return np.array([r[field][key] for r in table], dtype=float)


def factor_influence(table: list, response: str) -> dict:
    """{param: influence} where influence is the range of the response's
    per-level group means -- how much that factor moves the response. 0 when
    the factor has fewer than two usable levels."""
    y = column(table, response)
    out = {}
    for p in PARAMS:
        x = column(table, p, "param")
        means = []
        for level in np.unique(x[~np.isnan(x)]):
            m = np.nanmean(y[x == level])
            if not np.isnan(m):
                means.append(m)
        out[p] = float(max(means) - min(means)) if len(means) >= 2 else 0.0
    return out


def influence_ranking(table: list, response: str) -> list:
    """[(param, influence, share), ...] high-to-low; share sums to 1 (or 0)."""
    infl = factor_influence(table, response)
    total = sum(infl.values())
    ranked = sorted(infl.items(), key=lambda kv: kv[1], reverse=True)
    return [(p, v, (v / total if total else 0.0)) for p, v in ranked]


def correlation_matrix(table: list, keys: list = None) -> np.ndarray:
    """Pairwise Pearson r among response columns (NaN where a pair has <3
    valid points or no variance). Diagonal is 1."""
    keys = keys or RESPONSE_KEYS
    cols = [column(table, k) for k in keys]
    n = len(keys)
    c = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() >= 3 and a[mask].std() > 0 and b[mask].std() > 0:
                c[i, j] = c[j, i] = float(np.corrcoef(a[mask], b[mask])[0, 1])
            else:
                c[i, j] = c[j, i] = np.nan
    return c


def outlier_scores(table: list, keys: list = None) -> np.ndarray:
    """Per-scenario RMS of standardized (z-scored) responses -- its distance
    from the ensemble mean in response space. Higher = more unusual."""
    keys = keys or RESPONSE_KEYS
    z_rows = []
    for k in keys:
        c = column(table, k)
        mask = ~np.isnan(c)
        if not mask.any():
            continue
        mu, sd = np.nanmean(c), np.nanstd(c) or 1.0
        z = np.where(mask, (c - mu) / sd, 0.0)
        z_rows.append(z)
    if not z_rows:
        return np.zeros(len(table))
    return np.sqrt(np.mean(np.vstack(z_rows) ** 2, axis=0))


def study_statistics(table: list, keys: list = None, kind: str = "response") -> dict:
    """{key: {min, mean, max, std, n}} across the scenarios (NaN-safe)."""
    keys = keys or RESPONSE_KEYS
    stats = {}
    for k in keys:
        v = column(table, k, kind)
        mask = ~np.isnan(v)
        stats[k] = {
            "min": float(np.nanmin(v)) if mask.any() else float("nan"),
            "mean": float(np.nanmean(v)) if mask.any() else float("nan"),
            "max": float(np.nanmax(v)) if mask.any() else float("nan"),
            "std": float(np.nanstd(v)) if mask.any() else float("nan"),
            "n": int(mask.sum()),
        }
    return stats


def normalized_axes(table: list, keys: list, kind_map: dict):
    """For parallel coordinates: each axis min-max normalized to [0, 1].
    Returns (n_scenarios, n_axes) with NaN preserved. `kind_map` maps each key
    to 'param' or 'response'."""
    axes = []
    for k in keys:
        v = column(table, k, kind_map[k])
        mask = ~np.isnan(v)
        if mask.any():
            lo, hi = np.nanmin(v), np.nanmax(v)
            span = (hi - lo) or 1.0
            axes.append(np.where(mask, (v - lo) / span, np.nan))
        else:
            axes.append(v)
    return np.column_stack(axes)
