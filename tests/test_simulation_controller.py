"""Unit tests for the simulation controller and worker thread."""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
from simulation_controller import SimulationController


class FakeStore:
    """Minimal fake store for testing without disk I/O."""

    def __init__(self):
        self.get_calls = []

    def get(self, case_index):
        self.get_calls.append(case_index)
        return np.ones((481, 49, 101), dtype=np.float32)


class TestSimulationController:
    """Tests for the controller and cooperative thread stop."""

    def test_controller_start_is_running(self, qapp):
        """Verify start() sets is_running() to True."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.start()
        time.sleep(0.2)  # Let worker spin up
        assert ctrl.is_running(), "controller should be running after start()"
        ctrl.stop()
        time.sleep(0.2)  # Let worker shut down
        assert not ctrl.is_running(), "controller should not be running after stop()"

    def test_controller_stop_kills_worker(self, qapp):
        """Verify stop() cleanly terminates the worker thread (no terminate())."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.start()
        time.sleep(0.2)
        ctrl.stop()
        time.sleep(0.5)  # Wait for cooperative shutdown
        # If terminate() were called, there'd be undefined behavior; this should exit cleanly
        assert not ctrl.is_running()

    def test_controller_set_candles(self, qapp):
        """Verify set_candles propagates to params."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_candles(1)
        assert ctrl.params.candles == 1

    def test_controller_set_door(self, qapp):
        """Verify set_door propagates to params."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_door(0)
        assert ctrl.params.door == 0

    def test_controller_set_vod(self, qapp):
        """Verify set_vod propagates to params."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_vod(2)
        assert ctrl.params.vod == 2

    def test_controller_set_voc(self, qapp):
        """Verify set_voc propagates to params."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_voc(1)
        assert ctrl.params.voc == 1

    def test_controller_set_speed(self, qapp):
        """Verify set_speed propagates to params."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.set_speed(2)
        assert ctrl.params.speed == 2

    def test_controller_current_case_index(self, qapp):
        """Verify current_case_index computes the correct linear index."""
        store = FakeStore()
        # Tiny data_matrix: (2,2,3,2) factorization
        data_matrix = np.arange(24).reshape(2, 2, 3, 2)
        ctrl = SimulationController(store, data_matrix, 4)
        ctrl.params.candles = 1
        ctrl.params.door = 0
        ctrl.params.vod = 2
        ctrl.params.voc = 1
        idx = ctrl.current_case_index()
        expected = data_matrix[1, 0, 2, 1]
        assert idx == expected, f"expected case {expected}, got {idx}"

    def test_controller_current_frame(self, qapp):
        """Verify current_frame returns array without crashing."""
        store = FakeStore()
        data_matrix = np.zeros((2, 2, 3, 2), dtype=int)
        ctrl = SimulationController(store, data_matrix, 4)
        frame = ctrl.current_frame()
        assert frame.shape == (49, 101)
        assert frame.dtype == np.float32
