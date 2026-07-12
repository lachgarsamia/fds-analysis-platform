import csv
import os

import numpy as np

from manifest import ScenarioEntry
from summary_stats import (
    build_summary_index,
    compute_scenario_summary,
    load_cached_summaries,
)
from slice_key import DEFAULT_SLICE_KEY


class FakeStore:
    def __init__(self, arrays):
        self.arrays = arrays
        self.calls = []

    def get(self, case_index, key=DEFAULT_SLICE_KEY):
        self.calls.append((case_index, key))
        return self.arrays[case_index]


def _write_hrr(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "kW"])
        writer.writerow(["Time", "HRR"])
        writer.writerows(rows)


def test_compute_scenario_summary_hand_computed_values(tmp_path):
    folder = tmp_path / "c1_d0_vod0_voc0"
    folder.mkdir()
    _write_hrr(folder / "c1_d0_vod0_voc0_hrr.csv", [(0, 0), (1, 10), (3, 20)])
    entry = ScenarioEntry(0, folder.name, str(folder), 0, 0, 0, 0)
    data = np.asarray([
        [[20, 30], [40, 50]],
        [[80, 101], [60, 70]],
        [[150, 250], [90, 100]],
        [[310, 650], [100, 120]],
    ], dtype=np.float32)
    store = FakeStore({0: data})

    summary = compute_scenario_summary(entry, store, fps=2)

    assert summary.max_temp_c == 650.0
    assert summary.max_temp_by_frame_c == [50.0, 101.0, 250.0, 650.0]
    assert summary.time_to_100c_s == 0.5
    assert summary.time_to_300c_s == 1.5
    assert summary.time_to_600c_s == 1.5
    assert summary.mean_upper_temp_c == float(np.mean(data[:, :1, :]))
    assert summary.peak_hrr_kw == 20.0
    assert summary.total_energy_kj == 35.0


def test_summary_cache_reused_and_invalidated_by_hrr_mtime(tmp_path):
    folder = tmp_path / "c1_d0_vod0_voc0"
    folder.mkdir()
    hrr_path = folder / "c1_d0_vod0_voc0_hrr.csv"
    sf_path = folder / "c1_d0_vod0_voc0_0001_01.sf"
    smv_path = folder / "c1_d0_vod0_voc0.smv"
    _write_hrr(hrr_path, [(0, 0), (1, 10)])
    sf_path.write_bytes(b"sf")
    smv_path.write_text("smv")
    entry = ScenarioEntry(0, folder.name, str(folder), 0, 0, 0, 0)
    store = FakeStore({0: np.full((2, 2, 2), 120.0, dtype=np.float32)})
    cache_path = str(tmp_path / ".cache" / "summaries.json")

    first = build_summary_index([entry], store, fps=1, cache_path=cache_path)
    assert first[0].peak_hrr_kw == 10.0
    assert store.calls == [(0, DEFAULT_SLICE_KEY)]

    store.calls.clear()
    second = build_summary_index([entry], store, fps=1, cache_path=cache_path)
    assert second == first
    assert store.calls == []

    os.utime(hrr_path, (os.path.getatime(hrr_path), os.path.getmtime(cache_path) + 2))
    assert load_cached_summaries(cache_path, [entry]) is None
    rebuilt = build_summary_index([entry], store, fps=1, cache_path=cache_path)
    assert rebuilt[0].folder == entry.folder
    assert store.calls == [(0, DEFAULT_SLICE_KEY)]
