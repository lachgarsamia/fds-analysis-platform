"""Cached per-scenario summary statistics for the experiment browser (M2.5).

The visualizer's main data path is frame-oriented; the browser needs a
scenario-oriented index so users can sort/filter the 24-run ensemble by
scientific summaries without repeatedly scanning the same arrays and HRR CSVs.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from slice_key import DEFAULT_SLICE_KEY
from layer_height import smoke_layer_height_series
from tenability import TENABILITY_THRESHOLD_C
from config import AMBIENT_C


THRESHOLDS_C = (100.0, 300.0, 600.0)


@dataclass(frozen=True)
class ScenarioSummary:
    case_index: int
    folder: str
    candles: int
    door: int
    vod: int
    voc: int
    max_temp_c: float
    max_temp_by_frame_c: list
    time_to_100c_s: float | None
    time_to_300c_s: float | None
    time_to_600c_s: float | None
    mean_upper_temp_c: float
    peak_hrr_kw: float | None
    total_energy_kj: float | None
    # V2 roadmap M1.2: least-squares fire-growth coefficient of the
    # standard HRR = alpha * (t - t0)^2 model, fitted over the growth
    # phase (see fit_growth_alpha). None when no HRR CSV exists or the
    # curve has no usable growth segment.
    growth_alpha_kw_s2: float | None
    # V2 roadmap M2.3: minimum smoke-layer height reached (meters) -- the
    # worst-case (lowest) point of layer_height.smoke_layer_height_series,
    # a single hazard-ranking number for the browser.
    layer_min_height_m: float | None
    # V2 roadmap M3.2: first time (s) any cell crosses the convected-heat
    # tenability threshold (partial hazard screen, temperature only -- no
    # CO/FED). None if never reached.
    time_to_untenable_s: float | None


def _source_files(entries: list) -> list:
    files = []
    for entry in entries:
        files.extend(glob.glob(os.path.join(entry.path, "*.sf")))
        files.extend(glob.glob(os.path.join(entry.path, "*.smv")))
        files.extend(glob.glob(os.path.join(entry.path, "*_hrr.csv")))
    return files


def _cache_is_fresh(cache_path: str, entries: list) -> bool:
    if not os.path.exists(cache_path):
        return False
    sources = _source_files(entries)
    if not sources:
        return False
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime >= max(os.path.getmtime(path) for path in sources)


def _summary_from_payload(payload: dict) -> list:
    return [ScenarioSummary(**item) for item in payload.get("summaries", [])]


def _write_cache(cache_path: str, summaries: list[ScenarioSummary]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        # v4 (V2 roadmap M3.2): added time_to_untenable_s. Old caches fail
        # the version check below and are silently rebuilt.
        "version": 4,
        "thresholds_c": list(THRESHOLDS_C),
        "summaries": [asdict(summary) for summary in summaries],
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_cached_summaries(cache_path: str, entries: list) -> list[ScenarioSummary] | None:
    if not _cache_is_fresh(cache_path, entries):
        return None
    try:
        with open(cache_path, "r") as f:
            payload = json.load(f)
        if payload.get("version") != 4:
            return None
        summaries = _summary_from_payload(payload)
        expected_cases = [e.case_index for e in sorted(entries, key=lambda e: e.case_index)]
        cached_cases = [s.case_index for s in summaries]
        if cached_cases != expected_cases:
            return None
        return summaries
    except (OSError, ValueError, TypeError):
        return None


def _read_hrr_csv(folder: str) -> tuple[np.ndarray, np.ndarray] | None:
    paths = glob.glob(os.path.join(folder, "*_hrr.csv"))
    if not paths:
        return None
    path = paths[0]
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # units row
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            return None
        try:
            time_idx = header.index("Time")
            hrr_idx = header.index("HRR")
        except ValueError:
            return None

        times = []
        hrr = []
        for row in reader:
            if len(row) <= max(time_idx, hrr_idx):
                continue
            try:
                times.append(float(row[time_idx]))
                hrr.append(float(row[hrr_idx]))
            except ValueError:
                continue
    if not times:
        return None
    return np.asarray(times, dtype=float), np.asarray(hrr, dtype=float)


def read_hrr_table(folder: str) -> dict[str, np.ndarray] | None:
    """Every column of the scenario's *_hrr.csv as {name: values} (V2
    roadmap M1.2 -- the energy-budget panel's data source). FDS's format:
    row 1 = units, row 2 = column names, then data. Returns None when no
    CSV exists or it has no data rows. Rows with unparseable fields are
    skipped whole, matching _read_hrr_csv's tolerance."""
    paths = glob.glob(os.path.join(folder, "*_hrr.csv"))
    if not paths:
        return None
    with open(paths[0], newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # units row
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            return None
        rows = []
        for row in reader:
            if len(row) < len(header):
                continue
            try:
                rows.append([float(v) for v in row[: len(header)]])
            except ValueError:
                continue
    if not rows:
        return None
    matrix = np.asarray(rows, dtype=float)
    return {name: matrix[:, i] for i, name in enumerate(header)}


# Growth-phase bounds for fit_growth_alpha: the fitted segment runs from
# where HRR first exceeds this fraction of its peak up to the peak itself.
GROWTH_START_FRACTION = 0.01


def fit_growth_alpha(times: np.ndarray, hrr: np.ndarray) -> float | None:
    """Least-squares alpha for the standard t-squared fire-growth model
    HRR = alpha * (t - t0)^2, fitted over the growth phase (up to the
    curve's peak). t0 (ignition delay) is not known a priori: every
    sample time up to the curve's first exceedance of 1% of peak is
    tried as a t0 candidate, alpha solved in closed form for each
    (alpha = sum(q*tau^2) / sum(tau^4), the exact LSQ solution given
    t0), and the (t0, alpha) pair with the smallest squared residual
    wins. Exact on a true alpha*t^2 curve; recovers alpha under an
    ignition delay. Returns None if the curve never rises or the growth
    segment is degenerate (fewer than 3 samples)."""
    if times.size == 0 or hrr.size == 0:
        return None
    peak_idx = int(np.argmax(hrr))
    peak = float(hrr[peak_idx])
    if peak <= 0.0:
        return None
    above = np.flatnonzero(hrr[: peak_idx + 1] > GROWTH_START_FRACTION * peak)
    if above.size == 0:
        return None
    first_crossing = int(above[0])

    best_alpha = None
    best_sse = None
    for i0 in range(first_crossing + 1):
        if peak_idx - i0 < 2:
            continue
        tau = times[i0: peak_idx + 1] - times[i0]
        q = hrr[i0: peak_idx + 1]
        denom = float(np.sum(tau ** 4))
        if denom == 0.0:
            continue
        alpha = float(np.sum(q * tau ** 2) / denom)
        sse = float(np.sum((q - alpha * tau ** 2) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_alpha = alpha
    return best_alpha


def _first_threshold_time(max_by_frame: np.ndarray, fps: int, threshold_c: float) -> float | None:
    hits = np.flatnonzero(max_by_frame > threshold_c)
    if hits.size == 0:
        return None
    return float(hits[0] / fps)


def compute_scenario_summary(entry, store, fps: int) -> ScenarioSummary:
    data = np.asarray(store.get(entry.case_index, DEFAULT_SLICE_KEY))
    max_by_frame = np.max(data, axis=(1, 2))
    upper = data[:, : max(1, data.shape[1] // 2), :]

    try:
        extent = store.get_extent(entry.case_index, DEFAULT_SLICE_KEY)
    except AttributeError:  # a store without geometry support (e.g. a test double)
        extent = None
    if extent is None:
        layer_min_height_m = None
    else:
        layer_min_height_m = float(np.min(smoke_layer_height_series(data, extent, AMBIENT_C)))

    hrr_data = _read_hrr_csv(entry.path)
    if hrr_data is None:
        peak_hrr_kw = None
        total_energy_kj = None
        growth_alpha_kw_s2 = None
    else:
        hrr_t, hrr_kw = hrr_data
        peak_hrr_kw = float(np.max(hrr_kw))
        total_energy_kj = float(np.trapz(hrr_kw, hrr_t))
        growth_alpha_kw_s2 = fit_growth_alpha(hrr_t, hrr_kw)

    return ScenarioSummary(
        case_index=entry.case_index,
        folder=entry.folder,
        candles=entry.candles,
        door=entry.door,
        vod=entry.vod,
        voc=entry.voc,
        max_temp_c=float(np.max(max_by_frame)),
        max_temp_by_frame_c=[float(v) for v in max_by_frame],
        time_to_100c_s=_first_threshold_time(max_by_frame, fps, 100.0),
        time_to_300c_s=_first_threshold_time(max_by_frame, fps, 300.0),
        time_to_600c_s=_first_threshold_time(max_by_frame, fps, 600.0),
        mean_upper_temp_c=float(np.mean(upper)),
        peak_hrr_kw=peak_hrr_kw,
        total_energy_kj=total_energy_kj,
        growth_alpha_kw_s2=growth_alpha_kw_s2,
        layer_min_height_m=layer_min_height_m,
        # Same max-by-frame first-crossing as the threshold times above,
        # at the convected-heat tenability limit (tenability.py).
        time_to_untenable_s=_first_threshold_time(max_by_frame, fps, TENABILITY_THRESHOLD_C),
    )


def build_summary_index(entries: list, store, fps: int, cache_path: str) -> list[ScenarioSummary]:
    cached = load_cached_summaries(cache_path, entries)
    if cached is not None:
        return cached

    summaries = [
        compute_scenario_summary(entry, store, fps)
        for entry in sorted(entries, key=lambda e: e.case_index)
    ]
    _write_cache(cache_path, summaries)
    return summaries


def fmt_hrr(v: float) -> str:
    """Present HRR in the unit that shows its real magnitude -- a candle
    fire peaks well under 1 kW, so kW-rounded reads as a misleading 0."""
    return f"{v * 1000:.0f} W" if abs(v) < 1.0 else f"{v:.1f} kW"
