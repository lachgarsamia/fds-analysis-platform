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
from PyQt5 import QtCore, QtWidgets
from matplotlib import cm

from widgets import MplCanvas
from config import QUANTITY_DISPLAY


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


class GridCell(QtWidgets.QWidget):
    """One cell of a ViewGrid: a compact scenario combo + quantity combo
    above a PlotView. Clicking the cell makes it the grid's active cell.

    Doesn't know about SimulationController/ScenarioStore/manifest
    semantics beyond "a list of (label, case_index) options" and "a list
    of (label, SliceKey) options" -- MainWindow supplies both and reacts to
    this cell's signals, keeping data-fetching out of the view layer (same
    split as SliceView: this widget only knows how to display a frame it's
    handed, not how to load one).
    """

    activated = QtCore.pyqtSignal(object)                # self
    scenario_selected = QtCore.pyqtSignal(object, int)    # self, case_index
    quantity_selected = QtCore.pyqtSignal(object, object)  # self, SliceKey

    def __init__(self, scenario_options: list, quantity_options: list, parent=None):
        """scenario_options: [(label, case_index), ...]: quantity_options:
        [(label, SliceKey), ...]. Both may be empty/single-item (demo mode
        has no manifest) -- the combo disables itself in that case, same
        convention as MainWindow's own quantity combo (M2.1)."""
        super().__init__(parent)
        self.view = SliceView(self)
        self._scenario_options = list(scenario_options)
        self._quantity_options = list(quantity_options)
        self.case_index = self._scenario_options[0][1] if self._scenario_options else 0
        self.quantity_key = self._quantity_options[0][1] if self._quantity_options else None
        self._is_active = False
        self._accent = "#0B5FA5"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(4)

        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Cell scenario")
        self.scenario_combo.setToolTip("Which scenario this cell displays")
        for label, _case_index in self._scenario_options:
            self.scenario_combo.addItem(label)
        self.scenario_combo.setEnabled(len(self._scenario_options) > 1)
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_combo_changed)

        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Cell quantity")
        self.quantity_combo.setToolTip("Which quantity this cell displays")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        self.quantity_combo.currentIndexChanged.connect(self._on_quantity_combo_changed)

        header.addWidget(self.scenario_combo, 1)
        header.addWidget(self.quantity_combo, 1)
        layout.addLayout(header)
        layout.addWidget(self.view.widget(), 1)

        self._restyle()

    def mousePressEvent(self, event):
        self.activated.emit(self)
        super().mousePressEvent(event)

    def set_active(self, is_active: bool):
        self._is_active = is_active
        self._restyle()

    def apply_accent(self, accent_color: str):
        self._accent = accent_color
        self._restyle()

    def _restyle(self):
        border = f"2px solid {self._accent}" if self._is_active else "2px solid transparent"
        self.setStyleSheet(f"GridCell {{ border: {border}; border-radius: 3px; }}")

    def set_scenario_silently(self, case_index: int):
        """Update case_index/combo without emitting scenario_selected --
        for when this cell's state is being *driven* (e.g. the active cell
        mirroring control-panel changes) rather than user-selected here."""
        self.case_index = case_index
        idx = next((i for i, (_l, c) in enumerate(self._scenario_options) if c == case_index), None)
        if idx is not None and idx != self.scenario_combo.currentIndex():
            self.scenario_combo.blockSignals(True)
            self.scenario_combo.setCurrentIndex(idx)
            self.scenario_combo.blockSignals(False)

    def set_quantity_silently(self, key):
        self.quantity_key = key
        idx = next((i for i, (_l, k) in enumerate(self._quantity_options) if k == key), None)
        if idx is not None and idx != self.quantity_combo.currentIndex():
            self.quantity_combo.blockSignals(True)
            self.quantity_combo.setCurrentIndex(idx)
            self.quantity_combo.blockSignals(False)

    def _on_scenario_combo_changed(self, idx: int):
        if idx < 0 or idx >= len(self._scenario_options):
            return
        case_index = self._scenario_options[idx][1]
        self.case_index = case_index
        self.scenario_selected.emit(self, case_index)

    def _on_quantity_combo_changed(self, idx: int):
        if idx < 0 or idx >= len(self._quantity_options):
            return
        key = self._quantity_options[idx][1]
        self.quantity_key = key
        self.quantity_selected.emit(self, key)


class ViewGrid(QtWidgets.QWidget):
    """1x1/1x2/2x2 grid of GridCells (M2.2.2).

    Exactly one visible cell is "active" -- the one MainWindow's control
    panel (candles/door/vod/voc toggles, quantity combo, colormap menu,
    display-scale slider) edits. Other visible cells hold their own
    independent scenario/quantity/clim/colormap, set via their own
    per-cell combos (MainWindow reacts to cell_scenario_selected/
    cell_quantity_selected for those). Growing the grid preserves already-
    built cells' state instead of recreating them; shrinking just hides
    cells rather than destroying them, so switching back doesn't lose
    per-cell selections.
    """

    LAYOUTS = {
        "1x1": (1, 1),
        "1x2": (1, 2),
        "2x2": (2, 2),
    }

    cell_created = QtCore.pyqtSignal(object)             # a newly-instantiated GridCell, needs init_plot
    active_cell_changed = QtCore.pyqtSignal(object)       # the new active GridCell
    cell_scenario_selected = QtCore.pyqtSignal(object, int)      # non-active-or-active cell, case_index
    cell_quantity_selected = QtCore.pyqtSignal(object, object)   # non-active-or-active cell, SliceKey

    def __init__(self, scenario_options: list, quantity_options: list, parent=None):
        super().__init__(parent)
        self._scenario_options = scenario_options
        self._quantity_options = quantity_options
        self._layout_name = "1x1"
        self._cells: list = []
        self._active_index = 0

        self._grid_layout = QtWidgets.QGridLayout(self)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(4)

        self._grow_to(1)
        self._place_cells()

    def _grow_to(self, n_cells: int):
        while len(self._cells) < n_cells:
            cell = GridCell(self._scenario_options, self._quantity_options, self)
            cell.activated.connect(self._on_cell_activated)
            # Both signals already carry the emitting cell as their first
            # arg, so this is a straight re-emit -- Qt supports connecting
            # a signal directly to another signal with a matching shape.
            cell.scenario_selected.connect(self.cell_scenario_selected)
            cell.quantity_selected.connect(self.cell_quantity_selected)
            self._cells.append(cell)
            self.cell_created.emit(cell)

    def _place_cells(self):
        rows, cols = self.LAYOUTS[self._layout_name]
        n_visible = rows * cols
        while self._grid_layout.count():
            self._grid_layout.takeAt(0)
        for i, cell in enumerate(self._cells):
            if i < n_visible:
                r, c = divmod(i, cols)
                self._grid_layout.addWidget(cell, r, c)
                cell.setVisible(True)
            else:
                cell.setVisible(False)
        if self._active_index >= n_visible:
            self._active_index = 0
        self._refresh_active_highlight()

    def set_layout(self, layout_name: str):
        if layout_name not in self.LAYOUTS:
            raise ValueError(f"unknown grid layout {layout_name!r}")
        self._layout_name = layout_name
        rows, cols = self.LAYOUTS[layout_name]
        self._grow_to(rows * cols)
        self._place_cells()

    @property
    def layout_name(self) -> str:
        return self._layout_name

    def visible_cells(self) -> list:
        rows, cols = self.LAYOUTS[self._layout_name]
        return self._cells[:rows * cols]

    def active_cell(self) -> GridCell:
        return self._cells[self._active_index]

    def active_view(self) -> SliceView:
        return self.active_cell().view

    def _on_cell_activated(self, cell):
        if cell not in self.visible_cells():
            return
        idx = self._cells.index(cell)
        if idx == self._active_index:
            return
        self._active_index = idx
        self._refresh_active_highlight()
        self.active_cell_changed.emit(cell)

    def _refresh_active_highlight(self):
        for i, cell in enumerate(self._cells):
            cell.set_active(i == self._active_index)

    def apply_accent(self, accent_color: str):
        for cell in self._cells:
            cell.apply_accent(accent_color)
