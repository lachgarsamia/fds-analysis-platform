"""
time_controller.py
-------------------
Pull-based playback clock (M1.4). A GUI-thread QTimer ticks and emits the
current frame index; views pull frame data themselves via
store.get(case)[index] on each tick, rather than a background worker
pushing frames. This is only safe because M1.2's disk cache makes an
already-warm store.get() call ~1-6ms -- cheap enough to do directly on
the timer tick without stalling the GUI thread, which the old worker-push
design (simulation_controller.py's _Worker) existed specifically to avoid
back when a cache-miss parse cost 1-1.5s.

TimeController owns playback *timing* only -- no scenario-parameter or
store knowledge. `frame_count_fn` is queried on each tick/seek to get the
current scenario's frame count; it must be cheap and must never trigger a
load itself (the caller is responsible for keeping it in sync with an
already-loaded scenario -- see main_window.py's cache-miss handling,
which pauses the timer before switching to an uncached scenario).
"""

from typing import Callable

from PyQt5 import QtCore


class TimeController(QtCore.QObject):
    """play/pause/seek/step/set_speed, emits time_changed(index) on every
    tick or explicit seek."""

    time_changed = QtCore.pyqtSignal(int)       # current frame index
    playing_changed = QtCore.pyqtSignal(bool)

    def __init__(self, frame_count_fn: Callable[[], int], timesteps_per_second: int, parent=None):
        super().__init__(parent)
        self._frame_count_fn = frame_count_fn
        self.timesteps_per_second = timesteps_per_second
        self._speed = 1
        self._index = 0
        self._loop = True
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _interval_ms(self) -> int:
        return max(1, round(1000 / (self.timesteps_per_second * max(self._speed, 1))))

    def play(self):
        if self._timer.isActive():
            return
        n = self._frame_count_fn()
        if n > 0 and self._index >= n - 1 and not self._loop:
            # Sitting at the end with looping off -- restart from the top
            # rather than doing nothing when Play is pressed again.
            self.seek(0)
        self._timer.start(self._interval_ms())
        self.playing_changed.emit(True)

    def pause(self):
        if not self._timer.isActive():
            return
        self._timer.stop()
        self.playing_changed.emit(False)

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def seek(self, index: int):
        n = self._frame_count_fn()
        self._index = max(0, min(index, max(n - 1, 0)))
        self.time_changed.emit(self._index)

    def step(self, delta: int):
        self.seek(self._index + delta)

    def set_speed(self, speed: int):
        self._speed = max(1, speed)
        if self._timer.isActive():
            # Restart with the new interval so the change is audible/visible
            # on the very next tick, not after the current interval finishes.
            self._timer.start(self._interval_ms())

    def set_loop(self, enabled: bool):
        self._loop = enabled

    @property
    def index(self) -> int:
        return self._index

    def restart(self):
        was_playing = self.is_playing()
        self.pause()
        self.seek(0)
        if was_playing:
            self.play()

    def _tick(self):
        n = self._frame_count_fn()
        if n <= 0:
            return
        nxt = self._index + 1
        if nxt >= n:
            if self._loop:
                nxt = 0
            else:
                self.pause()
                return
        self._index = nxt
        self.time_changed.emit(self._index)
