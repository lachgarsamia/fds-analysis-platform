"""Multi-plane Linked Cross-Sections (V6-M7), an Analysis-page tab.

Three simultaneous spatial cross-sections of the same scenario at the same
instant -- XY (z-normal), XZ (y-normal, the app's one real/verified plane),
YZ (x-normal) -- all synced via the SelectionBus: the same scenario, the
same time, and (as far as this app's coordinate model allows) the same
point. This is a different view from spacetime_panel.py's cube (which
reshapes a *single* chosen plane into x-time/z-time maps); here all three
principal planes are shown together, each independently gated.

Reuses the exact V6-M5 plane-gating mechanism: each pane reads through
QuantityProvider, whose _ensure_plane_available check raises
GatedQuantityError for a plane this dataset's .smv doesn't actually
describe -- shown as a plain "gated" placeholder, never a fabricated
cross-section. Today only the XZ (y-normal) plane has real .sf slices; XY
and YZ are honestly gated (docs/msim-preparation.md) and are exercised in
tests via a synthetic multi-plane provider.

Point linking: `Selection.point` is the app's existing (x, z) physical
coordinate (what the XZ pane's crosshair uses directly). `Selection.depth`
(V6-M7, new -- a physical y) is the third spatial coordinate, so a point
clicked in the XY or YZ pane can, in principle, move the crosshair in the
others. (`Selection.height`, a similarly-named but semantically different
pre-existing field, was left alone rather than reused: this app's own
convention treats z as the vertical axis, so a Y-coordinate needed its own
field, not height's.) Scenario is bound generically (bind_to_bus); time/
point/depth use a dedicated bus handler here, matching spacetime_panel.py's
own convention.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from slice_key import SliceKey, AXIS_TO_DIRECTION
from quantity_provider import GatedQuantityError
from timeseries import phys_to_index

# name, direction, (x_axis_label, y_axis_label) -- XZ is the app's one
# verified plane (row=z top-down, col=x, the convention every other view
# uses); XY/YZ axis order is not verified against real data (see module
# docstring) since no X/Z-normal slice has ever existed to check against.
_PLANES = (
    ("XY", AXIS_TO_DIRECTION["z"], "x (m)", "y (m)"),
    ("XZ", AXIS_TO_DIRECTION["y"], "x (m)", "z (m, floor→ceiling)"),
    ("YZ", AXIS_TO_DIRECTION["x"], "y (m)", "z (m, floor→ceiling)"),
)
_QUANTITY = "TEMPERATURE"


class MultiPlanePanel(QtWidgets.QWidget):
    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._bus = None
        self._cache = {}          # (case_index, direction) -> (data, extent) or (None, reason)
        self._point = None        # (x, z) -- the app's existing 2D point
        self._depth = None        # Selection.depth -- the third coordinate (a physical y)
        self._time_s = 0.0
        self._axes = {}           # name -> Axes (for click routing)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Multi-plane cross-sections")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Multi-plane scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "XY / XZ / YZ cross-sections of the same scenario at the same instant, linked "
            "by scenario, time, and point. Only XZ (y-normal) has real data today -- XY/YZ "
            "are honestly gated (see docs/msim-preparation.md) until the M-SIM re-run adds "
            "X/Z-normal slices.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.canvases = {}
        for name, _direction, _xl, _yl in _PLANES:
            canvas = MplCanvas(self)
            canvas.setAccessibleName(f"{name} cross-section")
            canvas.mpl_connect("button_press_event",
                               lambda event, n=name: self._on_click(n, event))
            body.addWidget(canvas)
            self.canvases[name] = canvas
        layout.addWidget(body, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.scenario_combo.currentIndexChanged.connect(self._reload)

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)

    def _on_selection(self, sel, origin) -> None:
        if origin is self:
            return
        changed = False
        if sel.point is not None and sel.point != self._point:
            self._point = sel.point
            changed = True
        if sel.depth is not None and sel.depth != self._depth:
            self._depth = sel.depth
            changed = True
        if sel.time_s is not None and sel.time_s != self._time_s:
            self._time_s = sel.time_s
            changed = (changed or self.isVisible())  # time-only ticks: only redraw if shown
        if changed and self._loaded:
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

    # ------------------------------------------------------------- data
    def _plane(self, case_index: int, direction: int):
        """(data, extent) for `direction` at offset 0, or (None, reason)
        if gated -- cached once per (scenario, direction), never refetched
        per tick. Reuses the exact V6-M5 plane-gating path (QuantityProvider
        raises GatedQuantityError for a plane this dataset doesn't have)."""
        ck = (case_index, direction)
        if ck in self._cache:
            return self._cache[ck]
        key = SliceKey(_QUANTITY, direction, 0)
        try:
            data = np.asarray(self._provider.get(case_index, key))
            extent = self._provider.get_extent(case_index, key)
            result = (data, extent)
        except Exception as e:
            # Broadly caught (not just GatedQuantityError): a plane can be
            # declared in the .smv inventory yet still fail to actually
            # load (see V6-M5's PR notes on the 2nd Y-offset) -- an
            # uncaught exception here would escape a Qt slot and PyQt5
            # aborts the process on that, so this boundary must never let
            # one through.
            result = (None, str(e))
        self._cache[ck] = result
        return result

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        self._render()

    # ----------------------------------------------------------- interaction
    def _on_click(self, name: str, event) -> None:
        if event.inaxes is not self._axes.get(name) or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if name == "XZ":
            point, depth = (x, y), self._depth
        elif name == "XY":
            point, depth = (x, self._point[1] if self._point else 0.0), y
        else:  # YZ: axes are (y, z)
            point, depth = (self._point[0] if self._point else 0.0, y), x
        self._point, self._depth = point, depth
        if self._bus is not None:
            self._bus.update(origin=self, point=point, depth=depth)
        self._render()

    # --------------------------------------------------------------- render
    def _render(self) -> None:
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        q = get_quantity(_QUANTITY)
        any_real = False
        for name, direction, xl, yl in _PLANES:
            canvas = self.canvases[name]
            data, extent = self._plane(case_index, direction)
            fig = canvas.fig
            fig.clear()
            ax = fig.add_subplot(111)
            self._axes[name] = ax
            if data is None:
                ax.set_xticks([]); ax.set_yticks([])
            else:
                any_real = True
                n_t = data.shape[0]
                fi = min(max(int(round(self._time_s * self._fps)), 0), n_t - 1)
                ax.imshow(data[fi], cmap=q.cmap, vmin=q.vmin, vmax=q.slider_default,
                          aspect="auto", extent=extent)
                ax.set_xlabel(xl, fontsize=8)
                ax.set_ylabel(yl, fontsize=8)
                self._draw_crosshair(ax, name)
            ax.set_title(f"{name} (t = {self._time_s:.1f} s)", fontsize=8, fontweight="bold")
            fig.subplots_adjust(top=0.90, bottom=0.14, left=0.14, right=0.97)
            canvas.draw_idle()
        self.status.setText("" if any_real else
                            "All three planes are gated for this scenario -- see caption.")

    def _draw_crosshair(self, ax, name: str) -> None:
        if self._point is None and self._depth is None:
            return
        x = self._point[0] if self._point else None
        z = self._point[1] if self._point else None
        y = self._depth
        if name == "XZ" and x is not None and z is not None:
            ax.plot(x, z, "+", color="#00E5FF", markersize=12, markeredgewidth=2)
        elif name == "XY" and x is not None and y is not None:
            ax.plot(x, y, "+", color="#00E5FF", markersize=12, markeredgewidth=2)
        elif name == "YZ" and y is not None and z is not None:
            ax.plot(y, z, "+", color="#00E5FF", markersize=12, markeredgewidth=2)
