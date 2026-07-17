"""
widgets.py
----------
Small reusable widgets used by the main window.

- MplCanvas: a matplotlib canvas with an Expanding size policy so the plot
  actually grows/shrinks with the window instead of staying a fixed size.
- ToggleGroup: a segmented-control replacement for the original pattern of
  manually calling `.setEnabled()` on sibling buttons. One accessible,
  keyboard-navigable widget instead of N buttons wired by hand.
"""

from typing import List, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """A matplotlib canvas that expands to fill available layout space.

    Supports blitting for per-frame playback updates: capture_background()
    does one full draw and caches the rendered pixels; blit_update(artist)
    restores that cache and redraws only `artist` on top, which is far
    cheaper than a full draw_idle() when only the image data changed.
    Anything that changes what's *underneath* the animated artist (resize,
    theme, colormap, interpolation, clim) must call capture_background()
    again -- see main_window.py's setters.
    """

    # Plot area background is fixed white regardless of the app's own
    # light/dark theme -- standard practice for scientific heatmaps, so a
    # colormap and its colorbar always read the same true colors rather
    # than being visually tinted by whatever app chrome surrounds them.
    # Explicit (not just matplotlib's own default, which happens to also
    # be white) so this can't silently change if a rcParams default ever
    # does.
    PLOT_BG = "#FFFFFF"

    # Bumped from matplotlib's own default of 100 (GUI modernization pass,
    # item 7) -- crisper on-screen rendering (sharper text, contour lines,
    # colorbar ticks) at typical window/cell sizes. This is a rendering-
    # quality change only: the underlying data is still the FDS mesh's
    # native 49x101 grid (see main_window.py's velocity/interpolation
    # docstrings for the distinction between "renders less blocky" and
    # "the simulation itself ran at finer resolution" -- only the former
    # is true here; the latter would mean re-running simulations at a
    # finer mesh, out of scope, gated on M-SIM).
    DEFAULT_DPI = 150

    def __init__(self, parent=None, dpi: int = DEFAULT_DPI):
        self.fig = Figure(dpi=dpi, facecolor=self.PLOT_BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.updateGeometry()
        # Accessibility: the canvas itself is announced meaningfully.
        self.setAccessibleName("Fire simulation heat map")
        self.setAccessibleDescription(
            "Displays the current temperature field for the selected scenario."
        )
        self._background = None

    def capture_background(self):
        """Full draw + cache the result for subsequent blit_update() calls."""
        self.draw()
        self._background = self.copy_from_bbox(self.fig.bbox)

    def blit_update(self, artist):
        """Fast per-frame redraw: restore the cached background, draw only
        `artist` -- or, since the ember-particle scatter (FireLab roadmap
        Phase 2.1g), each artist in an iterable of artists -- on top, blit
        to screen. Falls back to a full draw+capture if there's no cached
        background yet (e.g. before the first paint)."""
        if self._background is None:
            self.capture_background()
            return
        self.restore_region(self._background)
        artists = artist if isinstance(artist, (list, tuple)) else (artist,)
        for a in artists:
            a.axes.draw_artist(a)
        self.blit(self.fig.bbox)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The cached background no longer matches the new canvas size.
        self._background = None


class ToggleGroup(QtWidgets.QWidget):
    """An accessible, keyboard-navigable segmented control.

    Replaces the pattern of `self.speed_1.setEnabled(False)` /
    `self.speed_2.setEnabled(True)` sprinkled across the code: one widget
    owns the exclusive-selection logic, exposes a single `value_changed`
    signal, and gets proper tab/arrow-key navigation for free from
    QButtonGroup + QPushButton's native focus handling.
    """

    value_changed = QtCore.pyqtSignal(object)  # emits the `value` of the option chosen

    def __init__(self, options: Sequence[Tuple[str, object]], default_index: int = 0,
                 accessible_name: str = "", parent=None):
        """
        options: sequence of (label, value) pairs, e.g. [("1x", 1), ("2x", 2)]
        """
        super().__init__(parent)
        self._buttons: List[QtWidgets.QPushButton] = []
        self._values = [v for _, v in options]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)

        for i, (label, _value) in enumerate(options):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("toggle", "true")
            btn.setFocusPolicy(QtCore.Qt.StrongFocus)
            if accessible_name:
                btn.setAccessibleName(f"{accessible_name}: {label}")
            self.group.addButton(btn, i)
            layout.addWidget(btn)
            self._buttons.append(btn)

        if self._buttons:
            self._buttons[default_index].setChecked(True)

        self.group.idClicked.connect(self._on_clicked)

    def _on_clicked(self, index: int):
        self.value_changed.emit(self._values[index])

    @property
    def value(self):
        """The currently selected option's value."""
        checked_id = self.group.checkedId()
        return self._values[checked_id] if checked_id >= 0 else None

    def set_value(self, value):
        """Programmatically select an option without re-emitting the signal
        (useful when restoring saved state)."""
        if value in self._values:
            idx = self._values.index(value)
            self._buttons[idx].setChecked(True)

    def set_enabled_all(self, enabled: bool):
        for b in self._buttons:
            b.setEnabled(enabled)

    def set_icon(self, icon: QtGui.QIcon, size: int = 14):
        """Apply the same category icon (e.g. a flame or door glyph) to every
        button, alongside its existing text label -- not replacing it, since
        icon-only buttons are harder to parse for screen readers and for
        users unfamiliar with the icon convention."""
        for b in self._buttons:
            b.setIcon(icon)
            b.setIconSize(QtCore.QSize(size, size))


class EventMarkerBar(QtWidgets.QWidget):
    """Thin clickable strip of event markers above the timeline slider
    (V2 roadmap M1.3): each marker is a small triangle at a frame index,
    with a tooltip naming the event (threshold crossing, peak
    temperature, ...) and click-to-seek. Pure UI, like TimelineWidget:
    MainWindow computes the (frame, label) list from summary stats and
    pushes it in via set_markers()."""

    marker_clicked = QtCore.pyqtSignal(int)  # frame index of the clicked marker

    BAR_HEIGHT = 12
    CLICK_TOLERANCE_PX = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markers: List[Tuple[int, str]] = []
        self._n_frames = 0
        self.setFixedHeight(self.BAR_HEIGHT)
        self.setMouseTracking(True)
        self.setAccessibleName("Timeline event markers")

    def set_markers(self, markers: Sequence[Tuple[int, str]]):
        self._markers = list(markers)
        self.update()

    def set_range(self, n_frames: int):
        self._n_frames = n_frames
        self.update()

    @property
    def markers(self) -> List[Tuple[int, str]]:
        return list(self._markers)

    def _x_for_frame(self, frame: int) -> float:
        if self._n_frames <= 1:
            return 0.0
        return frame / (self._n_frames - 1) * max(self.width() - 1, 1)

    def _marker_near(self, x: float):
        best = None
        best_dist = self.CLICK_TOLERANCE_PX + 1
        for frame, label in self._markers:
            dist = abs(self._x_for_frame(frame) - x)
            if dist < best_dist:
                best_dist = dist
                best = (frame, label)
        return best

    def paintEvent(self, event):
        if not self._markers or self._n_frames <= 1:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = self.palette().color(QtGui.QPalette.Highlight)
        painter.setBrush(color)
        painter.setPen(QtGui.QPen(color, 1))
        h = self.height()
        for frame, _label in self._markers:
            x = self._x_for_frame(frame)
            triangle = QtGui.QPolygonF([
                QtCore.QPointF(x - 4, 1),
                QtCore.QPointF(x + 4, 1),
                QtCore.QPointF(x, h - 1),
            ])
            painter.drawPolygon(triangle)

    def mousePressEvent(self, event):
        hit = self._marker_near(event.pos().x())
        if hit is not None:
            self.marker_clicked.emit(hit[0])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        hit = self._marker_near(event.pos().x())
        self.setToolTip(hit[1] if hit is not None else "")
        super().mouseMoveEvent(event)


class TimelineWidget(QtWidgets.QWidget):
    """Playback scrubber: play/pause + draggable position slider + time
    label + loop toggle (M1.4.2). Replaces the old read-only QProgressBar --
    this one is interactive, driven by/driving a TimeController.

    Pure UI: emits signals on user interaction and exposes setters for the
    controller to push state back in (`set_index`/`set_playing`/`set_loop`);
    it holds no playback logic of its own.
    """

    play_pause_clicked = QtCore.pyqtSignal()
    seek_requested = QtCore.pyqtSignal(int)   # user dragged/clicked to this frame index
    loop_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.play_button = QtWidgets.QPushButton("▶")  # ▶
        self.play_button.setFixedWidth(32)
        self.play_button.setAccessibleName("Play or pause playback")
        self.play_button.setToolTip("Play/pause (Space)")
        self.play_button.clicked.connect(self.play_pause_clicked.emit)
        layout.addWidget(self.play_button)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setAccessibleName("Playback position")
        self.slider.setToolTip("Drag to seek to any point in the simulation")
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.seek_requested.emit)
        # Event markers (V2 roadmap M1.3) sit directly above the slider,
        # sharing its width so frame->x positions line up; a marker click
        # is just another seek request.
        self.marker_bar = EventMarkerBar()
        self.marker_bar.marker_clicked.connect(self.seek_requested.emit)
        slider_column = QtWidgets.QVBoxLayout()
        slider_column.setContentsMargins(0, 0, 0, 0)
        slider_column.setSpacing(0)
        slider_column.addWidget(self.marker_bar)
        slider_column.addWidget(self.slider)
        layout.addLayout(slider_column, 1)

        self.time_label = QtWidgets.QLabel("t = 0.0 s / 0.0 s")
        self.time_label.setProperty("role", "value")
        self.time_label.setMinimumWidth(130)
        layout.addWidget(self.time_label)

        self.loop_button = QtWidgets.QPushButton("Loop")
        self.loop_button.setCheckable(True)
        self.loop_button.setChecked(True)
        self.loop_button.setAccessibleName("Loop playback")
        self.loop_button.setToolTip("Restart from the beginning when playback reaches the end")
        self.loop_button.toggled.connect(self.loop_toggled.emit)
        layout.addWidget(self.loop_button)

        self._n_frames = 0
        self._fps = 4

    def set_range(self, n_frames: int, fps: int):
        self._n_frames = n_frames
        self._fps = max(fps, 1)
        self.slider.setRange(0, max(n_frames - 1, 0))
        self.marker_bar.set_range(n_frames)

    def set_event_markers(self, markers):
        """[(frame_index, label), ...] -- see EventMarkerBar."""
        self.marker_bar.set_markers(markers)

    def set_index(self, index: int):
        """Reflect the controller's current index -- skipped while the user
        is actively dragging so we don't fight their gesture."""
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
        total_s = (self._n_frames - 1) / self._fps if self._n_frames > 0 else 0.0
        cur_s = index / self._fps
        self.time_label.setText(f"t = {cur_s:.1f} s / {total_s:.1f} s")

    def set_playing(self, playing: bool):
        self.play_button.setText("⏸" if playing else "▶")  # ⏸ / ▶

    def set_loop(self, enabled: bool):
        self.loop_button.blockSignals(True)
        self.loop_button.setChecked(enabled)
        self.loop_button.blockSignals(False)


class CollapsibleSection(QtWidgets.QWidget):
    """A control-panel "card" (Speed / Candles / Doors / Vents) -- rounded
    corners + a soft drop shadow (theme.apply_card_shadow) stand in for the
    old title+divider grouping, so the sidebar reads as a stack of distinct
    cards rather than lines drawn between undifferentiated rows (Streamlit-
    style redesign pass). This widget is static once built (no per-frame
    repaint), so the real QGraphicsDropShadowEffect is safe here -- see
    apply_card_shadow's docstring for why that's not true everywhere."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("sectionCard")
        outer.addWidget(self.card)

        self._layout = QtWidgets.QVBoxLayout(self.card)
        self._layout.setContentsMargins(16, 14, 16, 16)
        self._layout.setSpacing(10)

        title_label = QtWidgets.QLabel(title)
        title_label.setProperty("role", "section-title")
        title_label.setAccessibleName(f"{title} section")
        self._layout.addWidget(title_label)

    def add_row(self, widget: QtWidgets.QWidget):
        self._layout.addWidget(widget)
