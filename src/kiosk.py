"""Kiosk / attract mode (FireLab roadmap Phase 5): after IDLE_TIMEOUT_MS
of no user input, drift back to Home's hero page; the next real input
returns to Live. Also hides the cursor after a shorter idle delay ("F11
full-screen already exists; add cursor auto-hide" per the roadmap).

A single QObject event filter installed on the QApplication -- not tied
to any one widget -- so it catches input anywhere in the window. Cursor
hiding uses the target widget's own setCursor()/unsetCursor(), not
QApplication's global override-cursor stack, deliberately: that stack is
already used by the busy-cursor mechanism (see main_window.py's
_begin_busy_state/_end_busy_state and the mismatched-push bug it once
had), and a second independent pusher on the same shared stack is exactly
the class of bug that already bit this codebase once.
"""

from __future__ import annotations

from typing import Callable

from PyQt5 import QtCore, QtWidgets

IDLE_TIMEOUT_MS = 3 * 60 * 1000       # attract mode after 3 minutes idle
CURSOR_HIDE_DELAY_MS = 10 * 1000      # cursor itself hides after 10s idle

_WAKE_EVENT_TYPES = frozenset({
    QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonPress, QtCore.QEvent.KeyPress,
    QtCore.QEvent.Wheel, QtCore.QEvent.TouchBegin,
})


class KioskController(QtCore.QObject):
    """on_idle()/on_wake() fire at most once per state transition, not
    once per input event.

    Pass `parent` (e.g. the MainWindow) -- this object installs itself as
    a QApplication-level event filter, and relying on pure Python
    refcounting to keep an unparented QObject alive while it's registered
    as a filter is a real, if intermittent, segfault risk (found directly:
    ~80% crash rate across repeated runs without a parent, 0% with one)."""

    def __init__(self, on_idle: Callable[[], None], on_wake: Callable[[], None],
                 app: QtWidgets.QApplication, cursor_target: QtWidgets.QWidget,
                 idle_timeout_ms: int = IDLE_TIMEOUT_MS,
                 cursor_hide_delay_ms: int = CURSOR_HIDE_DELAY_MS, parent=None):
        super().__init__(parent)
        self._on_idle = on_idle
        self._on_wake = on_wake
        self._cursor_target = cursor_target
        self._idle_timeout_ms = idle_timeout_ms
        self._cursor_hide_delay_ms = cursor_hide_delay_ms
        self._is_idle = False
        self._cursor_hidden = False

        self._idle_timer = QtCore.QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._enter_idle)

        self._cursor_timer = QtCore.QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.timeout.connect(self._hide_cursor)

        self._app = app
        app.installEventFilter(self)
        self._idle_timer.start(self._idle_timeout_ms)
        self._cursor_timer.start(self._cursor_hide_delay_ms)

    def shutdown(self) -> None:
        """Removes this object as the QApplication's event filter and
        stops its timers -- call from the owning window's closeEvent().
        QApplication is process-global and outlives any one window, so an
        event filter left installed after its window closes doesn't just
        leak memory: every filter from every never-explicitly-closed
        window instance keeps running on every subsequent Qt event for
        the rest of the process (found directly: a test suite constructing
        ~40 MainWindows without this went from ~27s to ~470s). Same
        "process-global Qt state must be explicitly unwound" lesson as
        the busy-cursor override-stack cleanup in main_window.py's own
        closeEvent, not a new class of bug."""
        self._app.removeEventFilter(self)
        self._idle_timer.stop()
        self._cursor_timer.stop()

    def eventFilter(self, obj, event) -> bool:
        if event.type() in _WAKE_EVENT_TYPES:
            self._on_activity()
        return False

    def _on_activity(self) -> None:
        self._idle_timer.start(self._idle_timeout_ms)
        self._cursor_timer.start(self._cursor_hide_delay_ms)
        if self._cursor_hidden:
            self._cursor_target.unsetCursor()
            self._cursor_hidden = False
        if self._is_idle:
            self._is_idle = False
            self._on_wake()

    def _enter_idle(self) -> None:
        self._is_idle = True
        self._on_idle()

    def _hide_cursor(self) -> None:
        if not self._cursor_hidden:
            self._cursor_target.setCursor(QtCore.Qt.BlankCursor)
            self._cursor_hidden = True
