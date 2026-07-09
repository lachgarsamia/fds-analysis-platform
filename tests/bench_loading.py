"""Standalone timing/RAM benchmark for the M1.2 vectorized parser + disk cache.

Not a pytest test (bench_loading.py doesn't match test_*.py, so pytest won't
collect it). Run directly against the real dataset:

    python3 tests/bench_loading.py

Baseline from ROADMAP.md M1.2: 1.99s cold, N/A warm (no disk cache existed).
"""

import os
import sys
import time
import resource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_data import load_data, SIM_ROOT
from scenario_store import list_scenario_folders, ScenarioStore

BASELINE_COLD_S = 1.99


def peak_rss_mb() -> float:
    """Peak resident set size in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def bench_single_scenario(folder: str, cache_dir: str) -> None:
    print("\n=== Single scenario ===")
    print(f"folder: {os.path.basename(folder)}")

    t0 = time.perf_counter()
    data = load_data(folder)
    cold_elapsed = time.perf_counter() - t0
    print(f"cold parse (no cache):  {cold_elapsed:.4f}s  shape={data.shape}")

    store = ScenarioStore(folders=[folder], cache_size=1, cache_dir=cache_dir)
    store.get(0)  # populate disk cache
    fresh_store = ScenarioStore(folders=[folder], cache_size=1, cache_dir=cache_dir)
    t1 = time.perf_counter()
    fresh_store.get(0)
    warm_elapsed = time.perf_counter() - t1
    print(f"warm read (disk cache): {warm_elapsed:.4f}s")
    print(f"baseline (ROADMAP.md):  {BASELINE_COLD_S:.2f}s cold, N/A warm")


def bench_full_dataset(folders: list, cache_dir: str) -> None:
    print(f"\n=== Full dataset ({len(folders)} scenarios) ===")

    t0 = time.perf_counter()
    for folder in folders:
        load_data(folder)
    cold_total = time.perf_counter() - t0
    print(f"cold parse total:  {cold_total:.3f}s  ({cold_total / len(folders):.4f}s/scenario)")

    store = ScenarioStore(folders=folders, cache_size=len(folders), cache_dir=cache_dir)
    for i in range(len(folders)):
        store.get(i)  # populate disk cache for all scenarios

    fresh_store = ScenarioStore(folders=folders, cache_size=len(folders), cache_dir=cache_dir)
    t1 = time.perf_counter()
    for i in range(len(folders)):
        fresh_store.get(i)
    warm_total = time.perf_counter() - t1
    print(f"warm read total:    {warm_total:.3f}s  ({warm_total / len(folders):.4f}s/scenario)")


def main():
    folders = list_scenario_folders()
    if not folders:
        print(f"No scenarios found under {SIM_ROOT}; nothing to benchmark.")
        return

    cache_dir = os.path.join(SIM_ROOT, ".cache")
    bench_single_scenario(folders[0], cache_dir)
    bench_full_dataset(folders, cache_dir)

    print(f"\npeak RSS: {peak_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
