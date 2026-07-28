"""Parallel Coordinates panel (Analysis section consolidation Phase 3), a
tab inside the Compare & Discover workspace.

Plots every scenario in the factorial as one line across a shared axis
set (4 factor parameters + 4 response summaries), each axis independently
normalized min->max, so a researcher can visually spot which scenarios
are "extreme" across multiple axes at once and how factor levels co-vary
with responses. Clicking a line selects that scenario (publishes to the
SelectionBus).

Extracted verbatim from study_panel.py's own former "Parallel
coordinates" tab (same _table/_axis_keys/_axis_kind, same rendering/click
logic) -- only the container changed. It now sits with the other
cross-scenario discovery tools it conceptually belongs with (Compare
axes/Ensemble/Ensemble analytics) instead of Study's factor/response
tabs, per the Analysis Section Consolidation audit.

Reads the already-computed scenario summaries; no store reads, no new
simulations. Reuses study_analytics.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from widgets import MplCanvas
import study_analytics as sa


class ParallelCoordinatesPanel(QtWidgets.QWidget):
    def __init__(self, summaries: list, manifest: list, parent=None):
        super().__init__(parent)
        self._summaries = sorted(summaries or [], key=lambda s: s.case_index)
        self._table = sa.build_table(self._summaries)
        self._axis_keys = list(sa.PARAMS) + ["max_temp_c", "peak_hrr_kw",
                                             "total_energy_kj", "layer_min_height_m"]
        self._axis_kind = {k: ("param" if k in sa.PARAMS else "response")
                           for k in self._axis_keys}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Parallel coordinates")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Scenario:"))
        self.scenario_combo = QtWidgets.QComboBox()   # bound to the bus by main_window (M1)
        self.scenario_combo.setAccessibleName("Parallel coordinates scenario")
        for s in self._summaries:
            self.scenario_combo.addItem(s.folder, s.case_index)
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Every scenario as one line across parameters + responses (each axis "
            "independently normalized min→max). Click a line to select that scenario.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.parallel_canvas = MplCanvas(self)
        self.parallel_canvas.setAccessibleName("Parallel coordinates")
        layout.addWidget(self.parallel_canvas, 1)

        self.scenario_combo.currentIndexChanged.connect(self._render_parallel)
        # V6-M4: click a line in the parallel-coordinates plot to select
        # that scenario -- the combo above is already bus-bound generically
        # (bind_to_bus, main_window._build_selection), so moving its index
        # is enough to publish the selection; no direct bus reference needed
        # here.
        self.parallel_canvas.mpl_connect("button_press_event", self._on_parallel_click)
        self._render_parallel()

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """No lazy population needed (the table is built once in
        __init__, no store access) -- present for the "every panel in a
        wrapper has ensure_loaded()" convention CompareDiscoverPanel relies
        on to force all of its children ready on first show."""

    # ------------------------------------------------------------- rendering
    def _render_parallel(self) -> None:
        fig = self.parallel_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if not self._table:
            self.parallel_canvas.draw_idle()
            return
        norm = sa.normalized_axes(self._table, self._axis_keys, self._axis_kind)
        n_axes = len(self._axis_keys)
        xs = np.arange(n_axes)
        sel = self.scenario_combo.currentData()
        for i, row in enumerate(self._table):
            y = norm[i]
            is_sel = row["case_index"] == sel
            ax.plot(xs, y, color=("#00E5FF" if is_sel else "#B0B0B0"),
                    linewidth=(2.2 if is_sel else 0.8),
                    alpha=(1.0 if is_sel else 0.5), zorder=(3 if is_sel else 1))
        labels = [sa.PARAM_LABELS.get(k, sa.RESPONSE_LABEL.get(k, k)) for k in self._axis_keys]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_ylabel("normalized (min→max)", fontsize=8)
        for x in xs:
            ax.axvline(x, color="#ddd", linewidth=0.6, zorder=0)
        fig.subplots_adjust(top=0.97, bottom=0.22, left=0.08, right=0.98)
        self.parallel_canvas.draw_idle()

    def _on_parallel_click(self, event) -> None:
        """Study point -> jump to scenario (V6-M4): find the plotted line
        nearest the click and select its scenario in `scenario_combo`,
        which is already wired to the SelectionBus."""
        if event.inaxes is None or event.xdata is None or not self._table:
            return
        norm = sa.normalized_axes(self._table, self._axis_keys, self._axis_kind)
        axis_idx = min(max(int(round(event.xdata)), 0), len(self._axis_keys) - 1)
        best_i, best_d = None, None
        for i in range(len(self._table)):
            d = abs(norm[i][axis_idx] - event.ydata)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            return
        idx = self.scenario_combo.findData(int(self._table[best_i]["case_index"]))
        if idx >= 0:
            self.scenario_combo.setCurrentIndex(idx)
