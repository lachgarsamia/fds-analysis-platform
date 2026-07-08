"""
simulation_controller.py
-------------------------
Owns simulation *state* and *thread lifecycle*, with zero Qt-widget
knowledge. The view (main_window.py) only ever talks to this controller
through signals/slots and plain setters - it never reaches into thread
internals.

This replaces the original design where:
    - `Main` held speed/candle/door/vent values AND wired buttons AND
      managed the worker thread directly (a god-object).
    - Stopping a simulation called `QThread.terminate()`, which can kill a
      thread mid-operation and corrupt state / leak resources.

Here, stopping sets a `threading.Event` that the worker checks every loop
iteration, so the thread always exits its `run()` method cleanly.

Frames are fetched on demand through a ScenarioStore-like object (anything
with a `.get(scenario_index) -> ndarray` method -- see data_provider.py),
not a single preloaded array, so scenario switching still benefits from the
lazy-loading/LRU-cache behavior introduced in the data layer.
"""

import threading
from dataclasses import dataclass

from PyQt5 import QtCore
import numpy as np


@dataclass
class SimulationParameters:
    """All the scenario knobs, in one place instead of scattered attributes."""
    speed: int = 1          # 1-3, playback speed multiplier
    candles: int = 0        # index into candle dimension
    door: int = 1           # index into door dimension
    vod: int = 0            # index into vertical-opening-door dimension
    voc: int = 0            # index into vertical-opening-candle dimension


class _Worker(QtCore.QThread):
    """Advances the simulation timestep and emits frames.

    Cooperative stop: `run()` checks `self._stop_event` every iteration and
    exits promptly instead of being killed via `terminate()`. A ScenarioStore
    cache miss (a ~1-1.5s parse) happens here, on the background thread, so
    it causes a playback hitch rather than freezing the GUI.
    """

    frame_ready = QtCore.pyqtSignal(np.ndarray, int, int)
    error = QtCore.pyqtSignal(str)

    def __init__(self, store, data_matrix, timesteps_per_second: int,
                 start_index: int, params: SimulationParameters):
        super().__init__()
        self._store = store
        self._data_matrix = data_matrix
        self._timesteps_per_second = timesteps_per_second
        self._index = start_index
        self._params = params
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def update_params(self, params: SimulationParameters):
        with self._lock:
            self._params = params

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        try:
            i = self._index
            while not self._stop_event.is_set():
                with self._lock:
                    p = self._params
                sleep_s = 1.0 / (self._timesteps_per_second * max(p.speed, 1))
                # Sleep in small slices so a stop request is honored quickly
                # even at slow playback speeds, instead of one long sleep().
                slept = 0.0
                slice_s = 0.05
                while slept < sleep_s and not self._stop_event.is_set():
                    self.msleep(int(slice_s * 1000))
                    slept += slice_s
                if self._stop_event.is_set():
                    break

                case_idx = self._data_matrix[p.candles, p.door, p.vod, p.voc]
                scenario_data = self._store.get(case_idx)
                frame = scenario_data[i, :, :]
                current_time = int(i / self._timesteps_per_second)
                self.frame_ready.emit(frame, current_time, i)

                i += 1
                if i >= scenario_data.shape[0]:
                    i = 1
        except Exception as e:  # noqa: BLE001 - never let a worker crash silently
            self.error.emit(f"Simulation worker stopped unexpectedly: {e}")


class SimulationController(QtCore.QObject):
    """Public API the view talks to. No widget references live here."""

    frame_ready = QtCore.pyqtSignal(np.ndarray, int, int)
    started = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, store, data_matrix, timesteps_per_second: int = 4):
        super().__init__()
        self.store = store
        self.data_matrix = data_matrix
        self.timesteps_per_second = timesteps_per_second
        self.params = SimulationParameters()
        self.current_index = 1
        self._worker: _Worker = None

    # -- parameter setters (called from the view's button handlers) --------
    def set_speed(self, speed: int):
        self.params.speed = speed
        self._push_params()

    def set_candles(self, value: int):
        self.params.candles = value
        self._push_params()

    def set_door(self, value: int):
        self.params.door = value
        self._push_params()

    def set_vod(self, value: int):
        self.params.vod = value
        self._push_params()

    def set_voc(self, value: int):
        self.params.voc = value
        self._push_params()

    def _push_params(self):
        if self._worker is not None:
            self._worker.update_params(self.params)

    def current_case_index(self) -> int:
        return self.data_matrix[
            self.params.candles, self.params.door, self.params.vod, self.params.voc
        ]

    def current_frame(self):
        """Return the frame for the current index at the current parameter
        combination - used for the "update once while paused" case."""
        return self.store.get(self.current_case_index())[self.current_index, :, :]

    # -- lifecycle -----------------------------------------------------------
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start(self):
        if self.is_running():
            return
        self._worker = _Worker(
            self.store, self.data_matrix, self.timesteps_per_second,
            self.current_index, self.params,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self.started.emit()

    def stop(self):
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(2000)  # bounded wait, no forceful kill

    def restart(self):
        self.stop()
        self.current_index = 1
        self.start()

    def _on_frame(self, frame, current_time, index):
        self.current_index = index
        self.frame_ready.emit(frame, current_time, index)

    def _on_worker_finished(self):
        self.stopped.emit()
