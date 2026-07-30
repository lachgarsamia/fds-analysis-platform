"""Parallel Coordinates panel (Analysis section consolidation Phase 3), a
tab inside the Compare & Discover workspace.

Plots every scenario in the factorial as one line across a shared axis
set (4 factor parameters + 4 response summaries), each axis independently
normalized min->max, so a researcher can visually spot which scenarios
are "extreme" across multiple axes at once and how factor levels co-vary
with responses. Clicking a line selects that scenario (publishes to the
SelectionBus).

Analysis UX + reliability pass (redesign, not a restyle -- audited first:
the plot's axes/normalization/interaction were already sound; the actual
problem was every non-selected scenario rendering as an identical,
undifferentiated gray line with no y-axis value ticks, so the plot could
only ever answer "where does the one selected scenario sit," never "how
do factor levels co-vary with responses across the ensemble"). Two
additive fixes, same parallel-coordinates form:
- a "Color by:" combo recolors every line by a chosen factor's level
  (small discrete palette) instead of flat gray, so level separation
  across all axes is visible at a glance;
- each axis now shows its own real min/max value (not just a shared
  0-1 normalized scale), so magnitudes are readable, not just relative
  order.
The existing single-scenario highlight and click-to-select are
unchanged, layered on top.

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
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from widgets import MplCanvas
import study_analytics as sa
from analysis_panel_base import populate_scenario_combo

# Small discrete, colorblind-considerate qualitative palette for "Color
# by:" factor levels -- up to 4 levels (the widest factor, vod, has 3),
# indexed by level rank, not tied to any particular level's meaning.
_LEVEL_COLORS = ("#2563EB", "#E8622C", "#22C55E", "#7C3AED")
_UNSELECTED_ALPHA = 0.75


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
        header.addWidget(QtWidgets.QLabel("Color by:"))
        self.color_by_combo = QtWidgets.QComboBox()
        self.color_by_combo.setAccessibleName("Parallel coordinates color-by factor")
        self.color_by_combo.setToolTip(
            "Recolor every line by this factor's level, to see how it separates "
            "scenarios across all axes at once")
        for p in sa.PARAMS:
            self.color_by_combo.addItem(sa.PARAM_LABELS[p], p)
        header.addWidget(self.color_by_combo)
        header.addWidget(QtWidgets.QLabel("Scenario:"))
        self.scenario_combo = QtWidgets.QComboBox()   # bound to the bus by main_window (M1)
        self.scenario_combo.setAccessibleName("Parallel coordinates scenario")
        populate_scenario_combo(self.scenario_combo, self._summaries)
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Every scenario as one line across parameters + responses, colored by the "
            "chosen factor's level (each axis independently normalized min→max, with its "
            "real range labeled). Click a line to select that scenario.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.parallel_canvas = MplCanvas(self)
        self.parallel_canvas.setAccessibleName("Parallel coordinates")
        layout.addWidget(self.parallel_canvas, 1)

        self.scenario_combo.currentIndexChanged.connect(self._render_parallel)
        self.color_by_combo.currentIndexChanged.connect(self._render_parallel)
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
        color_by = self.color_by_combo.currentData() or sa.PARAMS[0]
        levels = sorted({int(row["params"][color_by]) for row in self._table
                        if not np.isnan(row["params"][color_by])})
        level_color = {lvl: _LEVEL_COLORS[i % len(_LEVEL_COLORS)] for i, lvl in enumerate(levels)}

        # Non-selected lines first (so the selected one always draws on top).
        for i, row in enumerate(self._table):
            if row["case_index"] == sel:
                continue
            lvl = row["params"][color_by]
            color = level_color.get(int(lvl), "#B0B0B0") if not np.isnan(lvl) else "#B0B0B0"
            ax.plot(xs, norm[i], color=color, linewidth=1.1, alpha=_UNSELECTED_ALPHA, zorder=1)
        sel_i = next((i for i, row in enumerate(self._table) if row["case_index"] == sel), None)
        if sel_i is not None:
            ax.plot(xs, norm[sel_i], color="#00E5FF", linewidth=2.4, alpha=1.0, zorder=3)

        labels = [sa.PARAM_LABELS.get(k, sa.RESPONSE_LABEL.get(k, k)) for k in self._axis_keys]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_ylabel("normalized (min→max)", fontsize=8)
        for j, x in enumerate(xs):
            ax.axvline(x, color="#ddd", linewidth=0.6, zorder=0)
            # Each axis's own real min/max (not just the shared 0-1 scale)
            # -- a single shared y-tick can't label every column since each
            # has different units/ranges, so the values are annotated
            # directly at each axis's top/bottom instead.
            key = self._axis_keys[j]
            values = sa.column(self._table, key, self._axis_kind[key])
            finite = values[~np.isnan(values)]
            if finite.size:
                unit = sa.RESPONSE_UNIT.get(key, "")
                fmt = (lambda v: f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
                # x in data coordinates, y in axes-fraction -- the min/max
                # labels sit just outside the normalized [0, 1] plot area
                # regardless of the actual y-limits.
                axis_transform = blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(x, 1.02, fmt(float(finite.max())) + (f" {unit}" if unit else ""),
                       ha="center", va="bottom", fontsize=6, color="#666", transform=axis_transform)
                ax.text(x, -0.02, fmt(float(finite.min())) + (f" {unit}" if unit else ""),
                       ha="center", va="top", fontsize=6, color="#666", transform=axis_transform)
        legend_handles = [Line2D([0], [0], color=level_color[lvl], linewidth=2.0,
                                 label=f"{sa.PARAM_LABELS[color_by]} = {lvl:g}")
                          for lvl in levels]
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=6, loc="upper right", ncol=min(len(levels), 4))
        fig.subplots_adjust(top=0.90, bottom=0.24, left=0.08, right=0.98)
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
