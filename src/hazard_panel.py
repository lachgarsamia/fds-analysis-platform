"""Hazard Spaces panel (V5-M4; full FED V6-M6), an Analysis-page tab.

A dynamic hazard map (cells coloured Safe / Warning / Critical / Untenable) and
a hazard timeline (stacked class fractions over time, with the flashover
indicator and a time cursor). Classification is temperature + exposure based
by default; the basis is stated on the panel. scenario_combo / frame_slider
are bound to the SelectionBus (M1) by main_window, so it stays in sync with
every panel.

V6-M6: on each scenario load, the panel tries to read 'CARBON MONOXIDE
VOLUME FRACTION' through QuantityProvider. Where available, classify_series
escalates on full FED (toxic-gas + convected-heat dose) instead of the
temperature-exposure proxy, and the caption states the full-FED basis. Gated
today (this dataset has no CO output) -- QuantityProvider raises
GatedQuantityError immediately, and the panel falls back to the exposure
proxy, unchanged from V5-M4.

Static/lazy; the class series is computed once per scenario. Reuses the
provider, the registry hazard bands, and hazard_spaces.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import SliceKey
from quantity_provider import GatedQuantityError
import hazard_spaces as hz


class HazardPanel(QtWidgets.QWidget):
    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}      # case_index -> dict(classes, fractions, flashover, extent, has_co)
        self._data = None
        self._series = None

        self._cmap = ListedColormap(list(hz.CLASS_COLORS))
        self._norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], self._cmap.N)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Hazard spaces")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Hazard scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel("Basis: " + hz.BASIS + ". Flashover is an "
                                        "indicator only (no combustion model).")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.map_canvas = MplCanvas(self)
        self.map_canvas.setAccessibleName("Hazard map")
        body.addWidget(self.map_canvas)
        self.timeline_canvas = MplCanvas(self)
        self.timeline_canvas.setAccessibleName("Hazard timeline")
        body.addWidget(self.timeline_canvas)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Hazard frame")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setProperty("role", "value")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.frame_slider.valueChanged.connect(self._on_frame)

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

    def _co_field(self, case_index):
        """V6-M6: a real CO ppm field for this scenario, or None if gated
        (GatedQuantityError is the registry's own gate, before any store
        access -- never a broad except-and-hope)."""
        try:
            return np.asarray(self._provider.get(
                case_index, SliceKey("CARBON MONOXIDE VOLUME FRACTION")))
        except GatedQuantityError:
            return None

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        if case_index not in self._cache:
            data = np.asarray(self._provider.get(case_index, SliceKey("TEMPERATURE")))
            extent = self._provider.get_extent(case_index, SliceKey("TEMPERATURE"))
            co = self._co_field(case_index)
            thr = hz.band_thresholds("TEMPERATURE")
            classes = hz.classify_series(data, thr, self._fps, co_field=co)
            self._cache[case_index] = {
                "data": data, "extent": extent, "classes": classes,
                "fractions": hz.class_fractions(classes),
                "flashover": hz.flashover_indicator(data),
                "worst": hz.worst_class(classes),
                "has_co": co is not None,
            }
        self._series = self._cache[case_index]
        self._data = self._series["data"]
        self.caption.setText(
            "Basis: " + (hz.FULL_FED_BASIS if self._series["has_co"] else hz.BASIS)
            + ". Flashover is an indicator only (no combustion model).")
        n = self._data.shape[0]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n - 1)
        if self.frame_slider.value() >= n:
            self.frame_slider.setValue(int(n * 0.6))
        self.frame_slider.blockSignals(False)
        self._render_timeline()
        self._render()

    def _on_frame(self, _v) -> None:
        self.frame_label.setText(f"t = {self.frame_slider.value() / self._fps:.1f} s")
        self._render()

    def _render(self) -> None:
        if self._series is None:
            return
        idx = min(self.frame_slider.value(), self._data.shape[0] - 1)
        fig = self.map_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(self._series["classes"][idx], cmap=self._cmap, norm=self._norm,
                  aspect="auto", extent=self._series["extent"] or None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Hazard classes at t = {idx / self._fps:.1f} s", fontsize=8)
        ax.legend(handles=[Patch(color=c, label=n) for c, n in
                           zip(hz.CLASS_COLORS, hz.CLASS_NAMES)],
                  fontsize=6, loc="upper left", framealpha=0.7)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.map_canvas.draw_idle()

        worst = int(self._series["worst"][idx])
        crit = float((self._series["classes"][idx] >= 2).mean()) * 100.0
        fo, _first = self._series["flashover"]
        self.status_label.setText(
            f"Worst class now: <b>{hz.CLASS_NAMES[worst]}</b> · "
            f"{crit:.0f}% of cells at Critical or worse"
            + ("  ·  flashover indicator reached" if bool(fo[idx]) else ""))

    def _render_timeline(self) -> None:
        fig = self.timeline_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        fr = self._series["fractions"]
        times = np.arange(fr.shape[0]) / self._fps
        ax.stackplot(times, *[fr[:, c] * 100 for c in range(4)],
                     colors=hz.CLASS_COLORS, labels=hz.CLASS_NAMES)
        _fo, first = self._series["flashover"]
        if first is not None:
            ax.axvline(first / self._fps, color="#00E5FF", linewidth=1.2,
                       label="flashover indicator")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("% of cells", fontsize=8)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.96, bottom=0.16, left=0.12, right=0.97)
        self.timeline_canvas.draw_idle()
