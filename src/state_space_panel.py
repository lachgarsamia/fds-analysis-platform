"""State-Space + Fire Genome panel (V3-M5), an Analysis-page tab.

Left: the selected scenario's fire-state trajectory (per-frame 2D
embedding, coloured by time) with the detected regime change-points
(events.py) marked. Right: the scenario's Fire Genome fingerprint,
normalized across the whole ensemble.

Static/lazy; genomes for the whole ensemble are computed once (needed for
normalization), trajectories per scenario on demand and cached.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from widgets import MplCanvas
from slice_key import DEFAULT_SLICE_KEY
from descriptors import compute_descriptors
from events import detect_events
from analysis_panel_base import populate_scenario_combo
import state_space as ss


class StateSpacePanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, summaries=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._summaries = {s.case_index: s for s in (summaries or [])}
        self._loaded = False
        self._genomes = []          # normalized, aligned with self._manifest order
        self._traj_cache = {}       # case -> (coords, times, evr, events)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("State space")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("State-space scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Each point is one moment of the fire; the path is the whole run. Turns mark "
            "regime changes. The bars are this run's fingerprint versus the others.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter()
        self.traj_canvas = MplCanvas(self)
        self.traj_canvas.setAccessibleName("Fire-state trajectory")
        body.addWidget(self.traj_canvas)
        self.genome_canvas = MplCanvas(self)
        self.genome_canvas.setAccessibleName("Fire genome")
        body.addWidget(self.genome_canvas)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        layout.addWidget(body, 1)

        self.scenario_combo.currentIndexChanged.connect(self._render)

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
        self._compute_genomes()
        self._render()

    def _compute_genomes(self) -> None:
        raw = []
        for entry in self._manifest:
            data = self._store.get(entry.case_index, DEFAULT_SLICE_KEY)
            extent = self._store.get_extent(entry.case_index, DEFAULT_SLICE_KEY)
            raw.append(ss.genome_traits(data, extent, self._fps,
                                        self._summaries.get(entry.case_index)))
        self._genomes = ss.normalize_genomes(raw)

    def _trajectory(self, case_index):
        if case_index not in self._traj_cache:
            data = self._store.get(case_index, DEFAULT_SLICE_KEY)
            extent = self._store.get_extent(case_index, DEFAULT_SLICE_KEY)
            coords, times, evr = ss.scenario_trajectory(data, extent, self._fps)
            table = compute_descriptors(data, extent, self._fps)
            events = detect_events(table, DEFAULT_SLICE_KEY.quantity)
            self._traj_cache[case_index] = (coords, times, evr, events)
        return self._traj_cache[case_index]

    def _render(self) -> None:
        if not self._loaded:
            return
        idx = self.scenario_combo.currentIndex()
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        coords, times, evr, events = self._trajectory(case_index)

        # --- trajectory ---
        fig = self.traj_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(coords[:, 0], coords[:, 1], "-", color="#bbbbbb", linewidth=0.8, zorder=1)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=times, cmap="viridis", s=10, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("time (s)", fontsize=8)
        # regime change-points from events
        for ev in events:
            fi = ev.frame_index(self._fps)
            if fi is not None and 0 <= fi < len(coords):
                ax.plot(coords[fi, 0], coords[fi, 1], "o", markersize=9, markerfacecolor="none",
                        markeredgecolor="#E8622C", markeredgewidth=1.6, zorder=3)
        pc1 = evr[0] * 100 if len(evr) else 0
        pc2 = evr[1] * 100 if len(evr) > 1 else 0
        ax.set_xlabel(f"state axis 1 — {pc1:.0f}% of variation", fontsize=8)
        ax.set_ylabel(f"state axis 2 — {pc2:.0f}%", fontsize=8)
        ax.set_title("Fire-state trajectory (○ = regime change)", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.90, bottom=0.14, left=0.12, right=0.98)
        self.traj_canvas.draw_idle()

        # --- genome fingerprint ---
        gfig = self.genome_canvas.fig
        gfig.clear()
        gax = gfig.add_subplot(111)
        if 0 <= idx < len(self._genomes):
            g = self._genomes[idx]
            labels = [lbl for _k, lbl in ss.GENOME_TRAITS]
            values = [g[k] for k, _lbl in ss.GENOME_TRAITS]
            y = np.arange(len(labels))
            gax.barh(y, values, color="#E8622C")
            gax.set_yticks(y)
            gax.set_yticklabels(labels, fontsize=8)
            gax.set_xlim(0, 1)
            gax.invert_yaxis()
            gax.set_xlabel("relative to the ensemble (0–1)", fontsize=8)
            gax.set_title("Fire genome", fontsize=9, fontweight="bold")
            gax.tick_params(labelsize=7)
        gfig.subplots_adjust(top=0.90, bottom=0.14, left=0.32, right=0.96)
        self.genome_canvas.draw_idle()
