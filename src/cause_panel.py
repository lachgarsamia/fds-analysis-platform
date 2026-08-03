"""Cause Explorer panel (V3-M7, gated), an Analysis-page tab.

Pick a hot point on the field and get a physics-based chain explaining why
it is hot: the temperature-gradient path traced back to the fire source,
drawn on the map and listed as an Insight chain. A prominent disclaimer
states this is association (gradient ascent, no velocity field), not
proven causation, per the V3-M7 gate.

Lazy; `frame_slider` (a QSpinBox -- bind_to_bus only needs value()/
setValue()/maximum()/valueChanged, not a particular widget class) is
wired to the shared SelectionBus playback clock (UX consolidation pass,
item 7), so the map follows Play/Pause/scrub from the Analysis page's
shared transport, same as every other per-frame panel.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from slice_key import DEFAULT_SLICE_KEY
from timeseries import phys_to_index
from insight import InsightList
from analysis_panel_base import populate_scenario_combo
import cause_explorer as ce

_DISCLAIMER = (
    "⚠ Association, not proven causation. This traces the temperature gradient back to the "
    "source; true causal tracing needs the flow field (real U/W-velocity), which this dataset "
    "does not yet have."
)


class CausePanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._data = None
        self._extent = None
        self._ax = None
        # Cached for the Context Panel's synthesized point-story (Analysis-
        # improvement roadmap Phase C): the last traced point/chain, so a
        # nearby selection can reuse it without a fresh store read.
        self._last_point = None
        self._last_insights = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Why is it hot?")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Cause scenario")
        header.addWidget(self.scenario_combo)
        header.addWidget(QtWidgets.QLabel("frame:"))
        self.frame_slider = QtWidgets.QSpinBox()
        self.frame_slider.setAccessibleName("Cause frame")
        header.addWidget(self.frame_slider)
        layout.addLayout(header)

        disclaimer = QtWidgets.QLabel(_DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setProperty("role", "caption")
        layout.addWidget(disclaimer)

        self.hint = QtWidgets.QLabel("Click a hot spot on the map to trace why it is hot.")
        self.hint.setProperty("role", "caption")
        layout.addWidget(self.hint)

        body = QtWidgets.QSplitter()
        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Cause map")
        body.addWidget(self.canvas)
        self.chain = InsightList()
        body.addWidget(self.chain)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        layout.addWidget(body, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        # NOT `.connect(self._render)`: QSpinBox.valueChanged(int) passes its
        # new value positionally into whatever it's connected to, which would
        # land in _render's `trace` parameter -- crashing (`for r, c in
        # trace:` on a bare int) on any nonzero frame. Discard the emitted
        # value so _render always runs with its real default (trace=None).
        self.frame_slider.valueChanged.connect(lambda _v: self._render())
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        self._data = np.asarray(self._store.get(case_index, DEFAULT_SLICE_KEY))
        self._extent = self._store.get_extent(case_index, DEFAULT_SLICE_KEY)
        self.frame_slider.blockSignals(True)
        n = self._data.shape[0]
        self.frame_slider.setRange(0, n - 1)
        self.frame_slider.setValue(min(self.frame_slider.value(), n - 1) if self.frame_slider.value()
                                 else int(n * 0.6))
        self.frame_slider.blockSignals(False)
        self.chain.set_insights([])
        self._render()

    def _render(self, trace=None) -> None:
        if self._data is None:
            return
        idx = min(self.frame_slider.value(), self._data.shape[0] - 1)
        display = get_quantity("TEMPERATURE")
        fig = self.canvas.fig
        fig.clear()
        self._ax = fig.add_subplot(111)
        self._ax.imshow(self._data[idx], cmap=display.cmap, vmin=display.vmin,
                        vmax=display.slider_default, aspect="auto",
                        extent=self._extent if self._extent else None)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title(f"Temperature at t = {idx / self._fps:.1f} s — click a hot spot",
                           fontsize=9)
        if trace and self._extent is not None:
            n_z, n_x = self._data[idx].shape
            xs, zs = [], []
            for r, c in trace:
                x, z = ce._phys(self._extent, n_z, n_x, r, c)
                xs.append(x); zs.append(z)
            self._ax.plot(xs, zs, "-o", color="#00E5FF", markersize=3, linewidth=1.5)
            self._ax.plot(xs[-1], zs[-1], "*", color="#FFD84D", markersize=16)  # the source
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.98)
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if self._data is None or event.inaxes != self._ax or event.xdata is None:
            return
        idx = min(self.frame_slider.value(), self._data.shape[0] - 1)
        frame = self._data[idx]
        if self._extent is None:
            return
        row, col = phys_to_index(self._extent, frame.shape, event.xdata, event.ydata)
        insights, path = ce.explain(frame, self._extent, idx / self._fps, row, col)
        self.chain.set_insights(insights)
        self._last_point = (float(event.xdata), float(event.ydata))
        self._last_insights = insights
        self._render(trace=path)
