"""Unit tests for the scenario store (lazy loading with LRU cache)."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scenario_store import ScenarioStore


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

        def mock_load(folder_path):
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

        def mock_load(folder_path):
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

        def mock_load(folder_path):
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
