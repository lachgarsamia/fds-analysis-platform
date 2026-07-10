"""PlotView abstraction (M2.2): decouples the rendering backend from the
widgets that use it, so a multi-view grid can hold several independent
view-cells, and later cell types (DifferenceView/EnsembleView in M2.3, a
possible pyqtgraph backend behind M2.4's gate) can all implement the same
minimal interface instead of each reaching into main_window.py's
matplotlib internals directly. Single-view mode (MainWindow) is just a
1-cell user of this same interface -- not a special case.
"""

from typing import Protocol

import numpy as np
from PyQt5 import QtWidgets
from matplotlib import cm

from widgets import MplCanvas


class PlotView(Protocol):
    """Minimum interface every view-cell type must implement. Concrete
    classes (SliceView below) expose additional setup/backend-specific
    methods beyond this -- this is the common surface a grid container or
    the M2.3 comparison views can rely on without knowing the backend."""

    def widget(self) -> QtWidgets.QWidget: ...
    def show_frame(self, frame: np.ndarray) -> None: ...
    def set_cmap(self, name: str) -> None: ...
    def set_clim(self, vmin: float, vmax: float) -> None: ...
    def set_title(self, text: str) -> None: ...


class SliceView:
    """matplotlib-backed single-quantity heatmap cell.

    Blitting internally (M1.3.3, unchanged behavior after this extraction):
    show_frame() only re-renders the image artist, not axes/colorbar
    chrome. Anything that changes what's underneath the animated artist
    (clim/colormap/interpolation/title/theme/resize) does a full draw +
    recapture -- see the setters below, all of which call
    canvas.capture_background() themselves so callers don't have to
    remember to.
    """

    def __init__(self, parent=None):
        self.canvas = MplCanvas(parent)
        self.ax = None
        self.heatmap = None
        self.colorbar = None

    def widget(self) -> QtWidgets.QWidget:
        return self.canvas

    def init_plot(self, first_frame: np.ndarray, cmap: str, interpolation: str,
                   vmin: float, vmax: float, colorbar_label: str):
        """One-time setup: axes, image artist, colorbar. Call once per
        SliceView instance before show_frame()."""
        self.ax = self.canvas.fig.add_subplot(111)
        self.heatmap = self.ax.imshow(
            first_frame, cmap=cmap, interpolation=interpolation, aspect="auto",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.fig.subplots_adjust(top=0.97, bottom=0.03, left=0.02, right=0.95)
        self.colorbar = self.canvas.fig.colorbar(self.heatmap, fraction=0.04, pad=0.02)
        self.colorbar.set_label(colorbar_label)
        self.heatmap.set_clim(vmin=vmin, vmax=vmax)
        self.canvas.capture_background()

    def show_frame(self, frame: np.ndarray) -> None:
        self.heatmap.set_data(frame)
        self.canvas.blit_update(self.heatmap)

    def set_cmap(self, name: str) -> None:
        self.heatmap.set_cmap(cm.get_cmap(name))
        self.canvas.capture_background()

    def set_clim(self, vmin: float, vmax: float) -> None:
        self.heatmap.set_clim(vmin=vmin, vmax=vmax)
        self.canvas.capture_background()

    def set_interpolation(self, name: str) -> None:
        self.heatmap.set_interpolation(name)
        self.canvas.capture_background()

    def set_colorbar_label(self, text: str) -> None:
        self.colorbar.set_label(text)
        self.canvas.capture_background()

    def set_title(self, text: str) -> None:
        self.ax.set_title(text)
        self.canvas.capture_background()

    def capture_background(self) -> None:
        """Force a full redraw + recapture -- for changes that don't go
        through one of the setters above (e.g. a theme restyle)."""
        self.canvas.capture_background()
