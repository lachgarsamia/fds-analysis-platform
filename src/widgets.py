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

    def __init__(self, parent=None, dpi: int = 100):
        self.fig = Figure(dpi=dpi)
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
        `artist` on top, blit to screen. Falls back to a full draw+capture
        if there's no cached background yet (e.g. before the first paint)."""
        if self._background is None:
            self.capture_background()
            return
        self.restore_region(self._background)
        artist.axes.draw_artist(artist)
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


class CollapsibleSection(QtWidgets.QWidget):
    """A labeled section with a thin divider - used to group control-panel
    rows (Speed / Candles / Doors / Vents) so the panel reads as organized
    sections rather than an undifferentiated stack of buttons."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        title_label = QtWidgets.QLabel(title)
        title_label.setProperty("role", "section-title")
        title_label.setAccessibleName(f"{title} section")
        self._layout.addWidget(title_label)

        divider = QtWidgets.QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        self._layout.addWidget(divider)

    def add_row(self, widget: QtWidgets.QWidget):
        self._layout.addWidget(widget)
