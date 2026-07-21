"""Space-Time Cube panel (V5-M4/P4, optional), an Analysis-page tab.

The 2D slice stack is an (x, z, t) volume. Rather than a heavy 3D render, this
shows the two cheap space-time cross-sections through a chosen point: the
x-vs-time map at the point's height (how the field propagates horizontally) and
the z-vs-time map at the point's column (vertical development). Both are just
the stored data reshaped -- no interpolation.

SelectionBus (M1): scenario_combo is bound by main_window; clicking the locator
publishes the point, and a point selected elsewhere moves the cross-sections.
Reuses the store and the extent/coordinate convention.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from slice_key import SliceKey
from timeseries import phys_to_index


class SpaceTimePanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}
        self._data = None
        self._extent = None
        self._row = None
        self._col = None
        self._bus = None
        self._loc_ax = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Space-time cube")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Space-time scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Click a point on the map. The two panels are its x–time and z–time "
            "cross-sections — the slice stack reshaped along time, no interpolation.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.loc_canvas = MplCanvas(self)
        self.loc_canvas.setAccessibleName("Space-time locator")
        body.addWidget(self.loc_canvas)
        self.xt_canvas = MplCanvas(self)
        self.xt_canvas.setAccessibleName("x-time map")
        body.addWidget(self.xt_canvas)
        self.zt_canvas = MplCanvas(self)
        self.zt_canvas.setAccessibleName("z-time map")
        body.addWidget(self.zt_canvas)
        body.setStretchFactor(0, 2); body.setStretchFactor(1, 3); body.setStretchFactor(2, 3)
        layout.addWidget(body, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.loc_canvas.mpl_connect("button_press_event", self._on_click)

    # V6 hook (GATED): multi-plane linked cross-sections (XY / XZ / YZ). This
    # panel already reshapes the single stored y-normal plane into x–time and
    # z–time. True XY/XZ/YZ views need FDS to output slices on those additional
    # planes (docs/msim-preparation.md). When present, add plane selectors here
    # and read each plane's SliceKey (direction/offset) -- the store already
    # keys slices by (quantity, direction, offset), so no new data path is
    # needed, only the extra SLCF output.

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.point is None or self._data is None or self._extent is None:
            return
        self._row, self._col = phys_to_index(self._extent, self._data.shape[1:],
                                             sel.point[0], sel.point[1])
        self._render()

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        if case_index not in self._cache:
            data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
            extent = self._store.get_extent(case_index, SliceKey("TEMPERATURE"))
            self._cache[case_index] = (data, extent)
        self._data, self._extent = self._cache[case_index]
        if self._row is None:
            self._row = self._data.shape[1] // 2
            self._col = self._data.shape[2] // 2
        self._render()

    def _on_click(self, event) -> None:
        if self._data is None or event.inaxes is not self._loc_ax or event.xdata is None:
            return
        self._row, self._col = phys_to_index(self._extent, self._data.shape[1:],
                                             event.xdata, event.ydata)
        if self._bus is not None:
            self._bus.update(origin=self, point=(float(event.xdata), float(event.ydata)))
        self._render()

    # --------------------------------------------------------------- render
    def _render(self) -> None:
        if self._data is None:
            return
        q = get_quantity("TEMPERATURE")
        n_t = self._data.shape[0]
        t_end = (n_t - 1) / self._fps
        ext = self._extent or [0, self._data.shape[2], 0, self._data.shape[1]]
        x0, x1, z0, z1 = ext

        # --- locator (a representative frame + the chosen point) ---
        lfig = self.loc_canvas.fig
        lfig.clear()
        self._loc_ax = lfig.add_subplot(111)
        frame = self._data[int(n_t * 0.6)]
        self._loc_ax.imshow(frame, cmap=q.cmap, vmin=q.vmin, vmax=q.slider_default,
                            aspect="auto", extent=ext)
        self._loc_ax.set_xticks([]); self._loc_ax.set_yticks([])
        px = x0 + self._col / max(frame.shape[1] - 1, 1) * (x1 - x0)
        pz = z1 - self._row / max(frame.shape[0] - 1, 1) * (z1 - z0)
        self._loc_ax.plot(px, pz, "x", color="#00E5FF", markersize=9)
        self._loc_ax.set_title("click a point", fontsize=8)
        lfig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
        self.loc_canvas.draw_idle()

        # --- x-time at the point's height ---
        xt = self._data[:, self._row, :]        # (n_t, n_x)
        self._heatmap(self.xt_canvas, xt, (x0, x1, t_end, 0.0), q,
                      f"x–time at z={pz:.2f} m", "x (m)")
        # --- z-time at the point's column (row 0 = ceiling) ---
        zt = self._data[:, :, self._col]        # (n_t, n_z)
        self._heatmap(self.zt_canvas, zt, (z1, z0, t_end, 0.0), q,
                      f"z–time at x={px:.2f} m", "z (m, floor→ceiling)")

    @staticmethod
    def _heatmap(canvas, arr, extent, q, title, xlabel) -> None:
        fig = canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(arr, cmap=q.cmap, vmin=q.vmin, vmax=q.slider_default,
                  aspect="auto", extent=extent)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("time (s)", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.92, bottom=0.14, left=0.16, right=0.97)
        canvas.draw_idle()
