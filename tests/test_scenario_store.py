"""Unit tests for the scenario store (lazy loading with LRU cache)."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scenario_store import ScenarioStore
from slice_key import SliceKey, DEFAULT_SLICE_KEY


class TestScenarioStore:
    """Tests for thread-safe LRU cache scenario loading."""

    def test_store_get_returns_array(self):
        """Verify store.get returns a numpy array."""
        # Create fake folder list and mock load_data
        folders = [f"/fake/scenario/{i}" for i in range(3)]
        with patch("scenario_store.load_data") as mock_load:
            mock_load.return_value = np.ones((481, 49, 101), dtype=np.float32)
            store = ScenarioStore(folders=folders, cache_size=2)
            result = store.get(0)
            assert isinstance(result, np.ndarray)

    def test_store_lru_eviction(self):
        """Verify LRU eviction: with cache_size=2, case 0 is evicted when cases 1,2 are loaded."""
        folders = [f"/fake/scenario/{i}" for i in range(3)]
        call_count = {}

        def mock_load(folder_path, key=None):
            # Extract index from path to count loads per scenario
            idx = int(folder_path.split("/")[-1])
            call_count[idx] = call_count.get(idx, 0) + 1
            return np.full((481, 49, 101), float(idx), dtype=np.float32)

        with patch("scenario_store.load_data", side_effect=mock_load):
            store = ScenarioStore(folders=folders, cache_size=2)

            # Load cases 0, 1, 2 (with size=2, case 0 should be evicted)
            store.get(0)
            store.get(1)
            store.get(2)

            # Re-accessing case 0 should trigger a reload (cache miss)
            store.get(0)
            assert call_count[0] == 2, "case 0 should be reloaded after eviction"

    def test_store_cache_hit_no_reload(self):
        """Verify accessing a cached item doesn't trigger reload."""
        folders = [f"/fake/scenario/{i}" for i in range(3)]
        call_count = {}

        def mock_load(folder_path, key=None):
            idx = int(folder_path.split("/")[-1])
            call_count[idx] = call_count.get(idx, 0) + 1
            return np.ones((481, 49, 101), dtype=np.float32)

        with patch("scenario_store.load_data", side_effect=mock_load):
            store = ScenarioStore(folders=folders, cache_size=2)
            store.get(0)
            store.get(0)  # Hit again
            store.get(0)  # Hit again
            assert call_count[0] == 1, "case 0 should only be loaded once"

    def test_store_returns_same_object_on_hit(self):
        """Verify cache hit returns the same numpy array object."""
        folders = [f"/fake/scenario/{i}" for i in range(3)]
        with patch("scenario_store.load_data") as mock_load:
            arr = np.ones((481, 49, 101), dtype=np.float32)
            mock_load.return_value = arr
            store = ScenarioStore(folders=folders, cache_size=2)
            result1 = store.get(0)
            result2 = store.get(0)
            # Should be the same object (not just equal values)
            assert result1 is result2, "cache hit should return identical object"

    def test_store_thread_safety_basic(self):
        """Smoke test: concurrent gets should not crash."""
        import threading

        folders = [f"/fake/scenario/{i}" for i in range(4)]
        call_count = {}

        def mock_load(folder_path, key=None):
            idx = int(folder_path.split("/")[-1])
            call_count[idx] = call_count.get(idx, 0) + 1
            return np.ones((481, 49, 101), dtype=np.float32)

        with patch("scenario_store.load_data", side_effect=mock_load):
            store = ScenarioStore(folders=folders, cache_size=4)

            def hammer():
                for _ in range(10):
                    store.get(0)
                    store.get(1)

            t1 = threading.Thread(target=hammer)
            t2 = threading.Thread(target=hammer)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Each case should be loaded exactly once (lock protecting it)
            assert call_count[0] == 1, "thread-safe load should count exactly once"
            assert call_count[1] == 1, "thread-safe load should count exactly once"

    # ------------------------------------------------------ SliceKey (M2.1)
    def test_store_get_defaults_to_default_slice_key(self):
        """A bare store.get(case) must behave exactly as before M2.1 --
        equivalent to explicitly passing DEFAULT_SLICE_KEY."""
        folders = [f"/fake/scenario/{i}" for i in range(2)]
        with patch("scenario_store.load_data") as mock_load:
            mock_load.return_value = np.ones((481, 49, 101), dtype=np.float32)
            store = ScenarioStore(folders=folders, cache_size=2)
            store.get(0)
            mock_load.assert_called_once_with(folders[0], DEFAULT_SLICE_KEY)

    def test_store_different_keys_cached_independently(self):
        """The same scenario under two different SliceKeys must be two
        distinct cache entries, each loaded on its own first access."""
        folders = ["/fake/scenario/0"]
        temp_key = SliceKey("TEMPERATURE", 1, 0)
        vel_key = SliceKey("VELOCITY", 1, 0)
        calls = []

        def mock_load(folder_path, key):
            calls.append(key)
            return np.full((481, 49, 101), 1.0 if key.quantity == "TEMPERATURE" else 2.0, dtype=np.float32)

        with patch("scenario_store.load_data", side_effect=mock_load):
            store = ScenarioStore(folders=folders, cache_size=2)
            temp_data = store.get(0, temp_key)
            vel_data = store.get(0, vel_key)

            assert len(calls) == 2, "each key should trigger its own load"
            assert not np.array_equal(temp_data, vel_data)

            # Both now cache hits -- no further loads.
            store.get(0, temp_key)
            store.get(0, vel_key)
            assert len(calls) == 2, "cached keys must not be reloaded"

    def test_is_cached_is_per_key(self):
        folders = ["/fake/scenario/0"]
        temp_key = SliceKey("TEMPERATURE", 1, 0)
        vel_key = SliceKey("VELOCITY", 1, 0)
        with patch("scenario_store.load_data") as mock_load:
            mock_load.return_value = np.ones((481, 49, 101), dtype=np.float32)
            store = ScenarioStore(folders=folders, cache_size=2)
            assert not store.is_cached(0, temp_key)
            assert not store.is_cached(0, vel_key)
            store.get(0, temp_key)
            assert store.is_cached(0, temp_key)
            assert not store.is_cached(0, vel_key)
