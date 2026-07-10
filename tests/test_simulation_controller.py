"""Unit tests for SimulationController: scenario-parameter state and the
background prefetch mechanism (M1.4.4). Playback timing itself is
TimeController's responsibility -- see test_time_controller.py."""

import time

import numpy as np
from simulation_controller import SimulationController
from slice_key import SliceKey, DEFAULT_SLICE_KEY


class FakeStore:
    """Minimal fake store for testing without disk I/O."""

    def __init__(self, fail_on: set = frozenset()):
        self.get_calls = []
        self._cached = set()
        self._fail_on = fail_on

    def get(self, case_index, key=DEFAULT_SLICE_KEY):
        self.get_calls.append((case_index, key))
        if case_index in self._fail_on:
            raise RuntimeError(f"simulated load failure for case {case_index}")
        self._cached.add((case_index, key))
        return np.ones((481, 49, 101), dtype=np.float32)

    def is_cached(self, case_index, key=DEFAULT_SLICE_KEY):
        return (case_index, key) in self._cached


class TestSimulationController:
    def test_controller_set_candles(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_candles(1)
        assert ctrl.params.candles == 1

    def test_controller_set_door(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_door(0)
        assert ctrl.params.door == 0

    def test_controller_set_vod(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_vod(2)
        assert ctrl.params.vod == 2

    def test_controller_set_voc(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_voc(1)
        assert ctrl.params.voc == 1

    def test_controller_current_case_index(self, qapp):
        """Verify current_case_index computes the correct linear index."""
        store = FakeStore()
        data_matrix = np.arange(24).reshape(2, 2, 3, 2)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.params.candles = 1
        ctrl.params.door = 0
        ctrl.params.vod = 2
        ctrl.params.voc = 1
        idx = ctrl.current_case_index()
        expected = data_matrix[1, 0, 2, 1]
        assert idx == expected, f"expected case {expected}, got {idx}"

    # -------------------------------------------------- prefetch (M1.4.4)
    def test_is_cached_delegates_to_store(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        assert not ctrl.is_cached(0)
        store.get(0)
        assert ctrl.is_cached(0)

    def test_prefetch_emits_finished_for_the_requested_case(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        results = []
        ctrl.prefetch_finished.connect(results.append)

        ctrl.prefetch(5)
        deadline = time.perf_counter() + 2.0
        while not results and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert results == [5]
        assert store.is_cached(5)

    def test_prefetch_emits_error_on_load_failure(self, qapp):
        store = FakeStore(fail_on={7})
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        errors = []
        ctrl.prefetch_error.connect(lambda case_idx, msg: errors.append((case_idx, msg)))

        ctrl.prefetch(7)
        deadline = time.perf_counter() + 2.0
        while not errors and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert len(errors) == 1
        case_idx, msg = errors[0]
        assert case_idx == 7
        assert "case" in msg.lower()

    def test_prefetch_keeps_concurrent_workers_alive_until_each_finishes(self, qapp):
        """Regression test (unit level) for the QThread lifecycle bug found
        while building M1.4: starting a second prefetch used to overwrite
        the only reference to the first, still-running worker, which Qt
        treats as fatal when the garbage-collected QThread is still active.
        Firing several prefetches back-to-back here must not crash and must
        eventually report all of them."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        results = []
        ctrl.prefetch_finished.connect(results.append)

        for case in (1, 2, 3, 4, 5):
            ctrl.prefetch(case)

        deadline = time.perf_counter() + 3.0
        while len(results) < 5 and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert sorted(results) == [1, 2, 3, 4, 5]
        assert ctrl._prefetch_workers == [], "all finished workers should be cleaned up"

    # ------------------------------------------------ SliceKey plumbing (M2.1)
    def test_is_cached_passes_key_through(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        velocity_key = SliceKey("VELOCITY", 1, 0)

        assert not ctrl.is_cached(0, velocity_key)
        assert not ctrl.is_cached(0)  # default key is a separate cache slot
        store.get(0, velocity_key)
        assert ctrl.is_cached(0, velocity_key)
        assert not ctrl.is_cached(0), "the default-key slot must be unaffected"

    def test_prefetch_passes_key_through_to_store(self, qapp):
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        velocity_key = SliceKey("VELOCITY", 1, 0)
        results = []
        ctrl.prefetch_finished.connect(results.append)

        ctrl.prefetch(3, velocity_key)
        deadline = time.perf_counter() + 2.0
        while not results and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert results == [3]
        assert store.is_cached(3, velocity_key)
        assert not store.is_cached(3), "must not have warmed the default-key slot instead"
