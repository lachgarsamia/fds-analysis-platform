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
        "version": 1,
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
        if payload.get("version") != 1:
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


def _first_threshold_time(max_by_frame: np.ndarray, fps: int, threshold_c: float) -> float | None:
    hits = np.flatnonzero(max_by_frame > threshold_c)
    if hits.size == 0:
        return None
    return float(hits[0] / fps)


def compute_scenario_summary(entry, store, fps: int) -> ScenarioSummary:
    data = np.asarray(store.get(entry.case_index, DEFAULT_SLICE_KEY))
    max_by_frame = np.max(data, axis=(1, 2))
    upper = data[:, : max(1, data.shape[1] // 2), :]

    hrr_data = _read_hrr_csv(entry.path)
    if hrr_data is None:
        peak_hrr_kw = None
        total_energy_kj = None
    else:
        hrr_t, hrr_kw = hrr_data
        peak_hrr_kw = float(np.max(hrr_kw))
        total_energy_kj = float(np.trapz(hrr_kw, hrr_t))

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
