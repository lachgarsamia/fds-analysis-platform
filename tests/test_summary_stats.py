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


# ---------------------------------------------------------------- V2 M1.2
from summary_stats import fit_growth_alpha, read_hrr_table  # noqa: E402


def _write_full_hrr(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "kW", "kW", "kg/s"])
        writer.writerow(["Time", "HRR", "Q_RADI", "MLR_FUEL"])
        writer.writerows([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, -0.3, 0.01),
                          (2.0, 4.0, -1.2, 0.02)])


def test_read_hrr_table_returns_every_column(tmp_path):
    path = tmp_path / "x_hrr.csv"
    _write_full_hrr(path)
    table = read_hrr_table(str(tmp_path))
    assert set(table) == {"Time", "HRR", "Q_RADI", "MLR_FUEL"}
    np.testing.assert_allclose(table["Q_RADI"], [0.0, -0.3, -1.2])


def test_read_hrr_table_missing_csv_returns_none(tmp_path):
    assert read_hrr_table(str(tmp_path)) is None


def test_fit_growth_alpha_recovers_exact_t_squared_curve():
    times = np.linspace(0.0, 10.0, 41)
    hrr = 0.5 * times ** 2
    alpha = fit_growth_alpha(times, hrr)
    assert alpha == 0.5


def test_fit_growth_alpha_flat_curve_returns_none():
    times = np.linspace(0.0, 10.0, 11)
    assert fit_growth_alpha(times, np.zeros(11)) is None


def test_fit_growth_alpha_ignores_pre_ignition_offset():
    # Curve idles at zero until t=5, then grows as alpha*(t-5)^2.
    times = np.linspace(0.0, 15.0, 61)
    hrr = np.where(times > 5.0, 2.0 * (times - 5.0) ** 2, 0.0)
    alpha = fit_growth_alpha(times, hrr)
    assert abs(alpha - 2.0) / 2.0 < 0.05


def test_compute_scenario_summary_populates_growth_alpha(tmp_path):
    folder = tmp_path / "c1_d0_vod0_voc0"
    folder.mkdir()
    times = np.linspace(0.0, 4.0, 17)
    _write_hrr(folder / "c1_d0_vod0_voc0_hrr.csv",
               [(t, 3.0 * t ** 2) for t in times])
    entry = ScenarioEntry(0, folder.name, str(folder), 0, 0, 0, 0)
    store = FakeStore({0: np.full((2, 2, 2), 25.0, dtype=np.float32)})
    summary = compute_scenario_summary(entry, store, fps=1)
    assert summary.growth_alpha_kw_s2 is not None
    assert abs(summary.growth_alpha_kw_s2 - 3.0) / 3.0 < 0.05


def test_v1_cache_payload_is_rejected_and_rebuilt(tmp_path):
    import json
    folder = tmp_path / "c1_d0_vod0_voc0"
    folder.mkdir()
    (folder / "c1_d0_vod0_voc0.smv").write_text("smv")
    # Cache written after its source, so it would be "fresh" -- rejection
    # below must therefore come from the version check, not staleness.
    cache_path = tmp_path / "summaries.json"
    cache_path.write_text(json.dumps({"version": 1, "summaries": []}))
    entry = ScenarioEntry(0, folder.name, str(folder), 0, 0, 0, 0)
    assert load_cached_summaries(str(cache_path), [entry]) is None
