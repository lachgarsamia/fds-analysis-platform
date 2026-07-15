"""Live Viewer page (FireLab roadmap Phase 1): hosts the existing 1x1
ViewGrid + TimelineWidget + control panel unchanged.

Built eagerly by MainWindow at startup exactly as before this page system
existed -- this is the app's primary, already-fast, already-tested
surface (M1.2's disk cache already made its own construction cheap), not
a lazy-build candidate like the placeholder pages. Boots hidden behind
Home; showing it is a plain QStackedWidget index change, not a rebuild.
"""

from __future__ import annotations

from PyQt5 import QtWidgets

from pages.base import Page


class LivePage(Page):
    title = "Live Viewer"

    def __init__(self, content: QtWidgets.QWidget, time_controller, parent=None):
        super().__init__(parent)
        self._time_controller = time_controller
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(content)

    def on_leave(self) -> None:
        """Pause playback when navigating away -- a fire looping on a page
        the user can't see is wasted work, and resuming is the same one
        click (Start/Pause) the user already knows."""
        if self._time_controller.is_playing():
            self._time_controller.pause()
