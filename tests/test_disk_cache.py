"""Tests for the on-disk .npy cache layer in ScenarioStore (M1.2)."""

import os
import shutil
import time
from unittest.mock import patch

import numpy as np
import pytest

import scenario_store as ss_module
from scenario_store import ScenarioStore
from slice_key import SliceKey, DEFAULT_SLICE_KEY


class TestDiskCache:
    """Cold parse -> disk cache write -> warm mmap read, invalidation, corruption fallback."""

    def test_disk_cache_cold_then_warm(self, fixtures_dir, tmp_path):
        cache_dir = str(tmp_path / "cache")

        store_cold = ScenarioStore(folders=[fixtures_dir], cache_size=1, cache_dir=cache_dir)
        t0 = time.perf_counter()
        cold_data = store_cold.get(0)
        cold_elapsed = time.perf_counter() - t0

        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1, "expected exactly one .npy cache file to be written"

        # Fresh store instance so the in-memory LRU cache can't short-circuit the disk read.
        store_warm = ScenarioStore(folders=[fixtures_dir], cache_size=1, cache_dir=cache_dir)
        t1 = time.perf_counter()
        warm_data = store_warm.get(0)
        warm_elapsed = time.perf_counter() - t1

        assert warm_data.shape == cold_data.shape
        assert np.array_equal(np.asarray(warm_data), np.asarray(cold_data))
        assert warm_elapsed < 0.1, f"warm load took {warm_elapsed:.3f}s, expected <0.1s"
        assert warm_elapsed < cold_elapsed, "warm mmap read should be faster than cold parse"

    def test_disk_cache_invalidated_on_stale_source(self, fixtures_dir, tmp_path):
        # Work on an isolated copy so this test never mutates the checked-in fixture.
        scenario_dir = str(tmp_path / "scenario")
        shutil.copytree(fixtures_dir, scenario_dir)
        cache_dir = str(tmp_path / "cache")

        store = ScenarioStore(folders=[scenario_dir], cache_size=1, cache_dir=cache_dir)
        store.get(0)  # populates disk cache

        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1

        # Make a source file newer than the cache to force invalidation.
        source_file = os.path.join(scenario_dir, "c1_d0_vod0_voc0_0001_01.sf")
        future = time.time() + 10
        os.utime(source_file, (future, future))

        real_load_data = ss_module.load_data
        calls = []

        def counting_load_data(folder, key=DEFAULT_SLICE_KEY):
            calls.append(folder)
            return real_load_data(folder, key)

        ss_module.load_data = counting_load_data
        try:
            fresh_store = ScenarioStore(folders=[scenario_dir], cache_size=1, cache_dir=cache_dir)
            fresh_store.get(0)
        finally:
            ss_module.load_data = real_load_data

        assert len(calls) == 1, "stale disk cache should trigger a re-parse"

    def test_disk_cache_corrupted_file_falls_back_to_reparse(self, fixtures_dir, tmp_path):
        cache_dir = str(tmp_path / "cache")

        store = ScenarioStore(folders=[fixtures_dir], cache_size=1, cache_dir=cache_dir)
        original = store.get(0)

        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1
        cache_path = os.path.join(cache_dir, cache_files[0])

        with open(cache_path, "wb") as f:
            f.write(b"not a valid npy file")

        fresh_store = ScenarioStore(folders=[fixtures_dir], cache_size=1, cache_dir=cache_dir)
        data = fresh_store.get(0)  # must not raise

        assert data.shape == original.shape
        assert np.array_equal(np.asarray(data), np.asarray(original))

    def test_disk_cache_disabled_by_default(self, fixtures_dir):
        """cache_dir=None (the default) must not touch the filesystem beyond parsing."""
        store = ScenarioStore(folders=[fixtures_dir], cache_size=1)
        assert store.cache_dir is None
        data = store.get(0)
        assert data.shape == (481, 49, 101)

    def test_disk_cache_filenames_differ_per_slice_key(self, tmp_path):
        """M2.1: two different quantities for the same scenario must land in
        two distinct .npy files, not overwrite each other."""
        cache_dir = str(tmp_path / "cache")
        folder = str(tmp_path / "scenario")
        os.makedirs(folder)
        open(os.path.join(folder, "dummy.sf"), "w").close()
        open(os.path.join(folder, "dummy.smv"), "w").close()

        temp_key = SliceKey("TEMPERATURE", 1, 0)
        vel_key = SliceKey("VELOCITY", 1, 0)

        def fake_load(folder_path, key):
            return np.full((2, 2, 2), 1.0 if key.quantity == "TEMPERATURE" else 2.0, dtype=np.float32)

        with patch("scenario_store.load_data", side_effect=fake_load):
            store = ScenarioStore(folders=[folder], cache_size=2, cache_dir=cache_dir)
            store.get(0, temp_key)
            store.get(0, vel_key)

        cache_files = sorted(os.listdir(cache_dir))
        assert len(cache_files) == 2, f"expected 2 distinct cache files, got {cache_files}"
        assert any("TEMPERATURE" in f for f in cache_files)
        assert any("VELOCITY" in f for f in cache_files)

    def test_disk_cache_filenames_differ_per_soot_plane(self, tmp_path):
        """M2.2: two SOOT planes (same quantity, different plane_pos) must
        land in distinct .npy files; a pre-M2.2 .sf key's filename is
        unchanged (no plane_pos suffix)."""
        cache_dir = str(tmp_path / "cache")
        folder = str(tmp_path / "scenario")
        os.makedirs(folder)
        open(os.path.join(folder, "dummy.smv"), "w").close()
        open(os.path.join(folder, "dummy.s3d"), "w").close()

        side = SliceKey("SOOT DENSITY", 1, 0, 0.0)
        doorway = SliceKey("SOOT DENSITY", 0, 0, 0.25)
        temp = SliceKey("TEMPERATURE", 1, 0)

        def fake_load(folder_path, key):
            return np.full((2, 2, 2), 1.0, dtype=np.float32)

        with patch("scenario_store.load_data", side_effect=fake_load):
            store = ScenarioStore(folders=[folder], cache_size=3, cache_dir=cache_dir)
            store.get(0, side)
            store.get(0, doorway)
            store.get(0, temp)

        cache_files = sorted(os.listdir(cache_dir))
        assert len(cache_files) == 3, f"expected 3 distinct cache files, got {cache_files}"
        # The .sf temperature filename keeps its pre-M2.2 form (no _p suffix).
        assert any(f.endswith("_off0.npy") for f in cache_files)
