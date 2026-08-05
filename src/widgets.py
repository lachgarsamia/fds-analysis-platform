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

import weakref
from typing import List, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --- Plot theming (RC polish) --------------------------------------------
# One source of truth for matplotlib "chrome" (backgrounds, axes text/ticks/
# spines, gridlines) so every plot adapts to the app's light/dark theme. The
# scientific field colormaps are never touched here -- only the surrounding
# chrome -- so a heatmap reads identically in both themes. New axes inherit
# these via rcParams; existing canvases are re-styled on a theme switch.
_PLOT_THEME = {"bg": "#FFFFFF", "axes": "#FFFFFF", "fg": "#14171F", "grid": "#E2E5EA"}
_CANVASES = []   # weakrefs to live MplCanvas instances


def _style_ax(ax) -> None:
    ax.set_facecolor(_PLOT_THEME["axes"])
    for spine in ax.spines.values():
        spine.set_color(_PLOT_THEME["fg"])
    ax.tick_params(colors=_PLOT_THEME["fg"], which="both")
    ax.xaxis.label.set_color(_PLOT_THEME["fg"])
    ax.yaxis.label.set_color(_PLOT_THEME["fg"])
    if ax.get_title():
        ax.title.set_color(_PLOT_THEME["fg"])


def plot_fg_color() -> str:
    """Current theme's plot foreground color, for ad-hoc ax.text() calls
    (placeholder/guidance messages) that aren't covered by _style_ax's
    spine/tick/label/title restyling."""
    return _PLOT_THEME["fg"]


def set_plot_theme(palette) -> None:
    """Point every plot at the palette's chrome colors. Updates rcParams (so
    axes created afterwards inherit them) and re-styles + redraws existing
    canvases. Field colormaps are unaffected."""
    _PLOT_THEME.update(bg=palette.plot_bg, axes=palette.plot_axes,
                       fg=palette.plot_fg, grid=palette.plot_grid)
    mpl.rcParams.update({
        "figure.facecolor": palette.plot_bg,
        "axes.facecolor": palette.plot_axes,
        "axes.edgecolor": palette.plot_fg,
        "axes.labelcolor": palette.plot_fg,
        "axes.titlecolor": palette.plot_fg,
        "xtick.color": palette.plot_fg,
        "ytick.color": palette.plot_fg,
        "text.color": palette.plot_fg,
        "grid.color": palette.plot_grid,
        "legend.edgecolor": palette.plot_grid,
    })
    for ref in list(_CANVASES):
        canvas = ref()
        if canvas is None:
            continue
        canvas.fig.set_facecolor(palette.plot_bg)
        for ax in canvas.fig.axes:
            _style_ax(ax)
        canvas._background = None   # blit cache is stale after a theme change
        canvas.draw_idle()


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
        self.fig = Figure(dpi=dpi, facecolor=_PLOT_THEME["bg"])
        _CANVASES.append(weakref.ref(self))
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

    def set_dpi_scale(self, scale: float) -> None:
        """View -> UI Scale (accessibility zoom). The canvas widget's own
        on-screen pixel footprint is set by Qt's layout (Expanding size
        policy), not by the figure's DPI -- so raising DPI packs more
        rendered detail into that same footprint, which is exactly what
        "everything reads bigger" means for a matplotlib figure: font
        sizes, line widths, and marker sizes are all specified in points,
        and points-to-pixels is dpi-dependent. Previously nothing in this
        canvas (colorbar ticks/label, room overlay lines, etc.) responded
        to UI Scale at all -- only the Qt-side chrome (QSS fonts/padding)
        did, so the plot itself looked completely unaffected by the
        setting. A full draw (not blit_update) is required since this
        changes what's rendered everywhere, not one animated artist."""
        self.fig.set_dpi(self.DEFAULT_DPI * scale)
        self.capture_background()


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


class TimeSeriesStrip(QtWidgets.QWidget):
    """A small custom-painted per-frame line strip with a moving playback
    cursor -- the same pattern as inspector.py's Inspector-panel sparkline
    (peak temperature over time), generalized here for reuse below a grid
    cell's own heatmap (colormap expressiveness follow-up: DYNAMIC PRESSURE
    and TEMPERATURE RISE's real story is temporal, not spatial, so they
    keep their heatmap and gain this strip underneath it).

    Two differences from the Inspector sparkline this generalizes:
    - Fixed y_range (not auto min/max per scenario) -- so the same height
      on the strip means the same value across every scenario, matching
      why the heatmap ranges themselves were fixed.
    - One or more series, each with its own color, drawn together (for
      TEMPERATURE RISE's three stacked hazard-fraction lines); a single
      series works the same as the Inspector sparkline (DYNAMIC PRESSURE).

    Live-cell re-proportioning pass: title/axis-label/caption text turns
    this from an unlabeled sliver into a small properly-labeled plot.
    Expanding (not Fixed) vertical size policy so a QVBoxLayout stretch
    factor (see GridCell) actually gives it real vertical share instead of
    clamping it to sizeHint(); _plot_rect() below insets the drawn axes box
    by *fractions* of the widget's own current size, not fixed pixels, so
    the title/ticks/caption keep their proportions at any window size.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._series: list = []       # [[v0, v1, ...], ...] -- one list per line
        self._colors: list = []       # one color per line, same order
        self._y_range = (0.0, 1.0)
        self._index = 0
        self._title = ""
        self._y_label = ""
        self._x_label = "Time (s)"
        self._caption = ""
        self._band_labels: list = []  # [(color, text), ...] legend swatches, e.g. hazard bands

    def set_series(self, series: list, colors: list, y_range: tuple, title: str = "",
                    y_label: str = "", caption: str = "", band_labels: list = None) -> None:
        """series: a list of per-frame value lists (all the same length);
        colors: one hex string per series; y_range: (lo, hi) fixed scale,
        from Phase 1's measured range -- never recomputed from `series`
        itself, so a quiet scenario doesn't rescale the axis and read as
        "just as active" as a severe one. title/y_label/caption: static
        text describing what this strip shows. band_labels: optional
        [(color, text), ...] legend entries (e.g. TEMPERATURE RISE's three
        hazard thresholds), drawn next to the title, one per color already
        in `colors`."""
        self._series = [list(s) for s in series]
        self._colors = list(colors)
        self._y_range = y_range
        self._index = 0
        self._title = title
        self._y_label = y_label
        self._caption = caption
        self._band_labels = list(band_labels) if band_labels else []
        self.update()

    def set_index(self, index: int) -> None:
        self._index = index
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _plot_rect(self) -> QtCore.QRectF:
        """The inner axes box the series lines are drawn into, inset from
        the widget's own rect by *fractions* of its current width/height
        (never fixed pixel counts, aside from a small cap on the y-axis
        label column so it doesn't eat the plot at very narrow widths) --
        title row above, x-axis label + caption below, y-axis ticks/label
        to the left. Keeps the same proportions whether this cell is one
        of a 1400px single view or a small grid cell."""
        w, h = float(self.width()), float(self.height())
        top = h * 0.24
        bottom = h * 0.32
        left = min(34.0, w * 0.16)
        right = w * 0.03
        return QtCore.QRectF(left, top, max(w - left - right, 1.0), max(h - top - bottom, 1.0))

    def _paint(self, painter: QtGui.QPainter) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        fg = QtGui.QColor(plot_fg_color())
        muted = QtGui.QColor("#94A3B8")
        plot_rect = self._plot_rect()

        # --- title (+ band legend, same row) --------------------------------
        title_font = QtGui.QFont(self.font())
        title_font.setPointSizeF(max(title_font.pointSizeF() * 0.9, 7.0))
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(fg)
        title_rect = QtCore.QRectF(plot_rect.left(), 1, plot_rect.width(), plot_rect.top() - 2)
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._title)

        if self._band_labels:
            legend_font = QtGui.QFont(self.font())
            legend_font.setPointSizeF(max(legend_font.pointSizeF() * 0.75, 6.0))
            painter.setFont(legend_font)
            metrics = QtGui.QFontMetricsF(legend_font)
            swatch = metrics.height() * 0.6
            x = title_rect.right()
            widths = [metrics.horizontalAdvance(text) for _color, text in self._band_labels]
            total = sum(widths) + len(self._band_labels) * (swatch + 10) + 4
            x -= total
            y_mid = title_rect.center().y()
            for (color, text), tw in zip(self._band_labels, widths):
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(color))
                painter.drawRect(QtCore.QRectF(x, y_mid - swatch / 2, swatch, swatch))
                painter.setPen(fg)
                painter.drawText(QtCore.QRectF(x + swatch + 3, title_rect.top(), tw + 4, title_rect.height()),
                                  QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, text)
                x += swatch + 6 + tw + 10

        if not self._series or not self._series[0] or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        # --- y-axis ticks (min/max of the fixed range) + rotated label -----
        tick_font = QtGui.QFont(self.font())
        tick_font.setPointSizeF(max(tick_font.pointSizeF() * 0.75, 6.0))
        painter.setFont(tick_font)
        painter.setPen(muted)
        lo, hi = self._y_range
        span = max(hi - lo, 1e-9)
        for value, y in ((hi, plot_rect.top()), (lo, plot_rect.bottom())):
            label = f"{value:g}"
            painter.drawText(QtCore.QRectF(0, y - 7, plot_rect.left() - 4, 14),
                              QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, label)
        if self._y_label:
            painter.save()
            painter.translate(9, plot_rect.center().y())
            painter.rotate(-90)
            metrics = QtGui.QFontMetricsF(tick_font)
            elided = metrics.elidedText(self._y_label, QtCore.Qt.ElideRight, int(plot_rect.height()))
            painter.drawText(QtCore.QRectF(-plot_rect.height() / 2, -7, plot_rect.height(), 14),
                              QtCore.Qt.AlignCenter, elided)
            painter.restore()

        # --- plot axes box ---------------------------------------------------
        painter.setPen(QtGui.QPen(muted, 1.0))
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
        painter.drawLine(plot_rect.topLeft(), plot_rect.bottomLeft())

        n = len(self._series[0])

        def point(series: list, i: int) -> QtCore.QPointF:
            x = plot_rect.left() + (i / max(n - 1, 1)) * plot_rect.width()
            v = min(max(series[i], lo), hi)  # clip to the fixed range, never extrapolate the axis
            y = plot_rect.bottom() - ((v - lo) / span) * plot_rect.height()
            return QtCore.QPointF(x, y)

        for series, color in zip(self._series, self._colors):
            path = QtGui.QPainterPath()
            path.moveTo(point(series, 0))
            for i in range(1, len(series)):
                path.lineTo(point(series, i))
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.4))
            painter.drawPath(path)

        # Moving playback cursor, same convention as the Inspector sparkline.
        idx = min(self._index, n - 1)
        painter.setPen(QtGui.QPen(muted, 1.0, QtCore.Qt.DashLine))
        x = plot_rect.left() + (idx / max(n - 1, 1)) * plot_rect.width()
        painter.drawLine(QtCore.QPointF(x, plot_rect.top()), QtCore.QPointF(x, plot_rect.bottom()))

        # --- x-axis label + static caption, below the axes box -------------
        label_font = QtGui.QFont(self.font())
        label_font.setPointSizeF(max(label_font.pointSizeF() * 0.75, 6.0))
        painter.setFont(label_font)
        painter.setPen(muted)
        x_label_rect = QtCore.QRectF(plot_rect.left(), plot_rect.bottom() + 2, plot_rect.width(), 14)
        painter.drawText(x_label_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, self._x_label)

        if self._caption:
            caption_font = QtGui.QFont(self.font())
            caption_font.setPointSizeF(max(caption_font.pointSizeF() * 0.7, 6.0))
            caption_font.setItalic(True)
            painter.setFont(caption_font)
            caption_rect = QtCore.QRectF(plot_rect.left(), x_label_rect.bottom(),
                                          plot_rect.width(), self.height() - x_label_rect.bottom() - 1)
            painter.drawText(caption_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.TextWordWrap, self._caption)
