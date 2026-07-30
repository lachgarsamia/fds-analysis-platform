"""Physics Attention Map panel (V3-M6), an Analysis-page tab.

Shows the heuristic saliency map (attention.py) for a scenario with a
frame slider: stable regions fade, active physics glows -- "look here".
A prominent disclaimer states it is a saliency cue, not a physical field.

Static/lazy; the attention series is computed once per scenario (from
temperature, and velocity/HRR when available) and cached.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import DEFAULT_SLICE_KEY, SliceKey
from summary_stats import _read_hrr_csv
from analysis_panel_base import populate_scenario_combo
import attention as at

_DISCLAIMER = (
    "⚠ Heuristic saliency — this highlights where the physics is *changing* "
    "(temperature/air-speed change, gradients, heat-release change). It is a "
    "\"look here\" cue, NOT a physical field."
)


class AttentionPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}     # case -> (attention_series, extent)
        self._series = None
        self._ax = None
        self._image = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Attention")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Attention scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        disclaimer = QtWidgets.QLabel(_DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setProperty("role", "caption")
        layout.addWidget(disclaimer)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Attention map")
        layout.addWidget(self.canvas, 1)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Attention frame")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.frame_slider.valueChanged.connect(self._render)

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

    def _hrr_frames(self, entry, n_t) -> np.ndarray:
        hrr = _read_hrr_csv(entry.path)
        if hrr is None:
            return None
        times, kw = hrr
        frame_times = np.arange(n_t) / self._fps
        return np.interp(frame_times, times, kw)

    def _velocity(self, case_index):
        try:
            return np.asarray(self._store.get(case_index, SliceKey("VELOCITY")))
        except Exception:  # noqa: BLE001 - velocity cue is optional
            return None

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        entry = next((e for e in self._manifest if e.case_index == case_index), None)
        if entry is None:
            return
        if case_index not in self._cache:
            temp = np.asarray(self._store.get(case_index, DEFAULT_SLICE_KEY))
            extent = self._store.get_extent(case_index, DEFAULT_SLICE_KEY)
            series = at.attention_series(temp, self._velocity(case_index),
                                         self._hrr_frames(entry, temp.shape[0]), self._fps)
            self._cache[case_index] = (series, extent)
        self._series, self._extent = self._cache[case_index]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, self._series.shape[0] - 1)
        self.frame_slider.blockSignals(False)
        self._render()

    def _render(self) -> None:
        if self._series is None:
            return
        idx = min(self.frame_slider.value(), self._series.shape[0] - 1)
        self.frame_label.setText(f"t = {idx / self._fps:.1f} s")
        fig = self.canvas.fig
        fig.clear()
        self._ax = fig.add_subplot(111)
        self._image = self._ax.imshow(self._series[idx], cmap="inferno", vmin=0.0, vmax=1.0,
                                       aspect="auto", extent=self._extent if self._extent else None)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title("Where the physics is active", fontsize=9, fontweight="bold")
        cbar = fig.colorbar(self._image, ax=self._ax, fraction=0.046, pad=0.02)
        cbar.set_label("saliency (0–1)", fontsize=8)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.canvas.draw_idle()
