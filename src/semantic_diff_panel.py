"""Semantic Fire Diff panel (V3-M3), an Analysis-page tab.

Two scenario selectors and a physics-difference report between them: a
ranked list of computed differences (semantic_diff.compare), each row
clickable to show the A − B field at the instant that evidences it, with
the difference's location flashed. This is the "GitHub diff for CFD"
surface -- physics, not pixels.

Static/lazy, same Analysis-panel convention. Differences are computed
once per (A, B, quantity) and cached.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from insight import InsightList
import semantic_diff as sd


class SemanticDiffPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, quantity_options: list, fps: int,
                 summaries=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._quantity_options = [(label, key) for label, key in quantity_options
                                  if get_quantity(key.quantity).kind == "slice2d"]
        self._fps = max(1, fps)
        self._summaries = {s.case_index: s for s in (summaries or [])}
        self._loaded = False
        self._cache: dict = {}      # (a, b, quantity) -> list[Insight]
        self._insights: list = []
        self._image = None
        self._marker = None
        self._ax = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Semantic diff")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.combo_a = QtWidgets.QComboBox()
        self.combo_a.setAccessibleName("Scenario A")
        self.combo_b = QtWidgets.QComboBox()
        self.combo_b.setAccessibleName("Scenario B")
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Semantic diff quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.combo_a)
        header.addWidget(QtWidgets.QLabel("vs"))
        header.addWidget(self.combo_b)
        header.addWidget(self.quantity_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "How these two runs differ, in physics terms. Click a difference to see "
            "the A − B field at the moment it matters.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.list = InsightList()
        body.addWidget(self.list)
        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Difference field")
        body.addWidget(self.canvas)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        self.combo_a.currentIndexChanged.connect(self._recompute)
        self.combo_b.currentIndexChanged.connect(self._recompute)
        self.quantity_combo.currentIndexChanged.connect(self._recompute)
        self.list.insight_activated.connect(self._show_evidence)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    @property
    def _key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    def ensure_loaded(self) -> None:
        if self._loaded or len(self._manifest) < 2 or not self._quantity_options:
            return
        self._loaded = True
        for combo in (self.combo_a, self.combo_b):
            combo.blockSignals(True)
            for entry in self._manifest:
                combo.addItem(entry.folder, entry.case_index)
            combo.blockSignals(False)
        self.combo_b.setCurrentIndex(min(1, self.combo_b.count() - 1))
        self._recompute()

    def _folder(self, case_index) -> str:
        return next((e.folder for e in self._manifest if e.case_index == case_index),
                    f"scenario {case_index}")

    def _recompute(self) -> None:
        if not self._loaded:
            return
        ca, cb = self.combo_a.currentData(), self.combo_b.currentData()
        key = self._key
        if ca is None or cb is None or key is None or ca == cb:
            self.list.set_insights([])
            self._insights = []
            return
        cache_key = (ca, cb, key.quantity)
        if cache_key not in self._cache:
            data_a = self._store.get(ca, key)
            data_b = self._store.get(cb, key)
            extent = self._store.get_extent(ca, key)
            self._cache[cache_key] = sd.compare(
                data_a, data_b, extent, self._fps, key.quantity,
                self._folder(ca), self._folder(cb),
                self._summaries.get(ca), self._summaries.get(cb))
        self._insights = self._cache[cache_key]
        self.list.set_insights(self._insights)
        # show the top difference by default
        if self._insights:
            self._show_evidence(self._insights[0])

    def _show_evidence(self, insight) -> None:
        ca, cb = self.combo_a.currentData(), self.combo_b.currentData()
        key = self._key
        if ca is None or cb is None or key is None:
            return
        data_a = np.asarray(self._store.get(ca, key))
        data_b = np.asarray(self._store.get(cb, key))
        n = min(data_a.shape[0], data_b.shape[0])
        idx = insight.frame_index(self._fps)
        idx = min(idx if idx is not None else 0, n - 1)
        diff = data_a[idx] - data_b[idx]
        vmax = float(np.max(np.abs(diff))) or 1.0
        extent = self._store.get_extent(ca, key)
        display = get_quantity(key.quantity)

        fig = self.canvas.fig
        fig.clear()
        self._ax = fig.add_subplot(111)
        self._image = self._ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                       aspect="auto", extent=extent if extent else None)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title(f"A − B at t = {idx / self._fps:.1f} s", fontsize=9, fontweight="bold")
        cbar = fig.colorbar(self._image, ax=self._ax, fraction=0.046, pad=0.02)
        cbar.set_label(f"Δ{display.label} ({display.unit})", fontsize=8)
        if insight.location is not None and extent is not None:
            self._ax.plot(insight.location[0], insight.location[1], "o",
                          markersize=12, markerfacecolor="none", markeredgecolor="#00E5FF",
                          markeredgewidth=2)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.canvas.draw_idle()
