"""Persistent playback transport (UI overhaul, global chrome pass): the
Start/Pause/Restart buttons, speed selector, and timeline scrubber (seek
slider + event markers + time readout + loop toggle) as ONE reusable bar,
pinned in MainWindow's header so it's visible -- and stays in sync -- on
every page, not just Live Viewer.

Replaces two previously separate, independently-built implementations
(the Live Viewer sidebar's transport row + speed section, and Analysis's
own _build_analysis_playback_bar()) that both wrapped TimelineWidget ad
hoc and needed manual dual-update call sites scattered through
main_window.py to keep the two instances in sync. Now there is exactly
one instance living in the header, so there is nothing left to keep in
sync.

Pure UI, no playback logic of its own -- same convention as TimelineWidget
(which this composes): emits signals on user interaction, exposes setters
for MainWindow/TimeController to push state back in.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from widgets import TimelineWidget, ToggleGroup


class PlaybackBar(QtWidgets.QWidget):
    start_clicked = QtCore.pyqtSignal()
    pause_clicked = QtCore.pyqtSignal()
    restart_clicked = QtCore.pyqtSignal()
    play_pause_clicked = QtCore.pyqtSignal()  # forwarded from the scrubber's own play/pause button
    seek_requested = QtCore.pyqtSignal(int)   # user dragged/clicked to this frame index
    loop_toggled = QtCore.pyqtSignal(bool)
    speed_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("playbackBar")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setAccessibleName("Start simulation")
        self.start_button.setToolTip("Start the fire simulation animation (Space)")
        self.start_button.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.start_button)

        self.stop_button = QtWidgets.QPushButton("Pause")
        self.stop_button.setAccessibleName("Pause simulation")
        self.stop_button.setToolTip("Pause the simulation (Space)")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.pause_clicked.emit)
        layout.addWidget(self.stop_button)

        self.restart_button = QtWidgets.QPushButton("Restart")
        self.restart_button.setAccessibleName("Restart simulation from the beginning")
        self.restart_button.setToolTip("Restart the simulation from t=0 (Ctrl+R)")
        self.restart_button.clicked.connect(self.restart_clicked.emit)
        layout.addWidget(self.restart_button)

        # Interactive scrubber (play/pause + seek slider + event markers +
        # time label + loop toggle) -- see widgets.py's TimelineWidget.
        self.timeline = TimelineWidget()
        self.timeline.play_pause_clicked.connect(self.play_pause_clicked.emit)
        self.timeline.seek_requested.connect(self.seek_requested.emit)
        self.timeline.loop_toggled.connect(self.loop_toggled.emit)
        layout.addWidget(self.timeline, 1)

        self.speed_toggle = ToggleGroup(
            [("1x", 1), ("2x", 2), ("3x", 3)], default_index=0,
            accessible_name="Playback speed",
        )
        self.speed_toggle.setToolTip(
            "Controls how fast the simulation plays back -- 2x and 3x speed "
            "up the animation without changing the underlying simulation."
        )
        self.speed_toggle.value_changed.connect(self.speed_changed.emit)
        layout.addWidget(self.speed_toggle)

    # --- state pushed in by the controller (mirrors TimelineWidget's own setters,
    # plus the Start/Pause button enabled-state TimelineWidget doesn't own) -------
    def set_range(self, n_frames: int, fps: int) -> None:
        self.timeline.set_range(n_frames, fps)

    def set_index(self, index: int) -> None:
        self.timeline.set_index(index)

    def set_event_markers(self, markers) -> None:
        self.timeline.set_event_markers(markers)

    def set_loop(self, enabled: bool) -> None:
        self.timeline.set_loop(enabled)

    def set_playing(self, playing: bool) -> None:
        self.timeline.set_playing(playing)
        self.start_button.setEnabled(not playing)
        self.stop_button.setEnabled(playing)
