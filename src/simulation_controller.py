"""
simulation_controller.py
-------------------------
Owns simulation *scenario-parameter state* plus the background prefetch
thread, with zero Qt-widget knowledge. The view (main_window.py) only ever
talks to this controller through signals/slots and plain setters - it
never reaches into thread internals.

Playback *timing* (the QTimer that used to live here as a frame-pushing
_Worker) moved to time_controller.py's TimeController in M1.4: once M1.2's
disk cache made an already-warm store.get() ~1-6ms, there was no longer a
reason to drive frames from a background thread -- the GUI thread can pull
them directly on each timer tick. The one thing that genuinely still
belongs on a background thread is a *cold* parse (~55-80ms, M1.2's
measured numbers) when the user switches to a scenario that isn't cached
yet; that's what prefetch()/_PrefetchWorker below do. Cooperative stop
isn't needed for it (it's a one-shot load, not a loop) -- letting each
worker run to completion, kept alive via _prefetch_workers until its own
`finished` signal fires, is enough.

Frames are fetched on demand through a ScenarioStore-like object (anything
with a `.get(scenario_index) -> ndarray` method -- see data_provider.py),
not a single preloaded array, so scenario switching still benefits from the
lazy-loading/LRU-cache behavior introduced in the data layer.
"""

from dataclasses import dataclass

from PyQt5 import QtCore


@dataclass
class SimulationParameters:
    """All the scenario knobs, in one place instead of scattered attributes."""
    candles: int = 0        # index into candle dimension
    door: int = 1           # index into door dimension
    vod: int = 0            # index into vertical-opening-door dimension
    voc: int = 0            # index into vertical-opening-candle dimension


class _PrefetchWorker(QtCore.QThread):
    """One-shot background scenario load (M1.4.4): warms ScenarioStore's
    cache for a scenario switch so the GUI thread never blocks on a cold
    parse. Not used for frame-driving playback -- see time_controller.py,
    which pulls already-cached frames directly on the GUI thread instead.
    """

    finished_ok = QtCore.pyqtSignal(int)  # case_index that finished loading
    error = QtCore.pyqtSignal(int, str)   # case_index that failed, message

    def __init__(self, store, case_index: int):
        super().__init__()
        self._store = store
        self._case_index = case_index

    def run(self):
        try:
            self._store.get(self._case_index)
            self.finished_ok.emit(self._case_index)
        except Exception as e:  # noqa: BLE001 - never let a worker crash silently
            self.error.emit(self._case_index, f"Failed to load scenario: {e}")


class SimulationController(QtCore.QObject):
    """Public API the view talks to. No widget references live here."""

    prefetch_finished = QtCore.pyqtSignal(int)       # case_index (M1.4.4)
    prefetch_error = QtCore.pyqtSignal(int, str)      # case_index, message

    def __init__(self, store, data_matrix, timesteps_per_second: int = 4):
        super().__init__()
        self.store = store
        self.data_matrix = data_matrix
        self.timesteps_per_second = timesteps_per_second
        self.params = SimulationParameters()
        # Keep every in-flight prefetch referenced until it actually
        # finishes: a single overwritten attribute would let a still-running
        # QThread get garbage-collected mid-run when a rapid second toggle
        # change starts another prefetch, which Qt treats as fatal (crashes
        # the whole process, not a Python exception) rather than a quiet bug.
        self._prefetch_workers: list = []

    # -- parameter setters (called from the view's button handlers) --------
    def set_candles(self, value: int):
        self.params.candles = value

    def set_door(self, value: int):
        self.params.door = value

    def set_vod(self, value: int):
        self.params.vod = value

    def set_voc(self, value: int):
        self.params.voc = value

    def current_case_index(self) -> int:
        return self.data_matrix[
            self.params.candles, self.params.door, self.params.vod, self.params.voc
        ]

    # -- prefetch (M1.4.4) ----------------------------------------------------
    def is_cached(self, case_index: int) -> bool:
        return self.store.is_cached(case_index)

    def prefetch(self, case_index: int):
        """Warm the store's cache for case_index on a background thread, so
        the GUI thread never blocks on a cold parse when the view switches
        to an uncached scenario. Fire-and-forget: emits prefetch_finished
        or prefetch_error when done; a stale/superseded request (the user
        switched again before this one finished) is the caller's concern to
        detect (main_window.py checks case_index against what's still
        wanted before acting on the result).

        Multiple prefetches can be in flight at once (rapid toggle changes)
        -- each worker is kept alive in _prefetch_workers until its own
        finished signal fires, rather than living in a single attribute a
        newer call would overwrite out from under a still-running thread."""
        worker = _PrefetchWorker(self.store, case_index)
        self._prefetch_workers.append(worker)
        worker.finished_ok.connect(self.prefetch_finished)
        worker.error.connect(self.prefetch_error)
        worker.finished.connect(lambda w=worker: self._cleanup_prefetch_worker(w))
        worker.start()

    def _cleanup_prefetch_worker(self, worker):
        if worker in self._prefetch_workers:
            self._prefetch_workers.remove(worker)
