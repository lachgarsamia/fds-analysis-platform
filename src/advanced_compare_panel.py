"""Advanced Comparison panel (V4-M8), an Analysis-page tab.

Pick two scenarios and compare them along the brief's three axes at once:
Temporal (when the danger lead flips), Spatial (which region differs most),
and Physics (the descriptors most associated with the difference). Each
axis is a navigable, savable Insight list; the plot shows the two peak-
temperature curves with the cross-over marked. The physics axis is
explicitly association, not causation.

Static/lazy; results are cached per (A, B, quantity). Reuses the store,
the scenario summaries, and the advanced_compare engine.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, plot_fg_color
from registry import get_quantity, AMBIENT_C
from insight import InsightList
from descriptors import compute_descriptors
import advanced_compare as ac
import semantic_diff as sd


class AdvancedComparePanel(QtWidgets.QWidget):
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
        self._cache = {}
        self._metric = None      # (times, smax_a, smax_b) for the temporal plot
        self._pinned_comparisons: list = []   # Phase C -> session report
        self._bus = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Advanced comparison")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.combo_a = QtWidgets.QComboBox(); self.combo_a.setAccessibleName("Compare A")
        self.combo_b = QtWidgets.QComboBox(); self.combo_b.setAccessibleName("Compare B")
        header.addWidget(QtWidgets.QLabel("A:")); header.addWidget(self.combo_a)
        header.addWidget(QtWidgets.QLabel("B:")); header.addWidget(self.combo_b)
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Compare quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Four comparison axes. Temporal: when the danger lead flips. "
            "Spatial: which region differs most. Physics: the descriptors most "
            "associated with the difference (association, not proven cause). "
            "Semantic diff: the ranked physics-difference report -- click a row "
            "to see the A − B field at the moment it matters.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Comparison timeline")
        self.canvas.setMinimumHeight(150)
        layout.addWidget(self.canvas, 1)

        self.temporal_list = self._axis_list(layout, "Temporal — when the lead flips")
        self.spatial_list = self._axis_list(layout, "Spatial — where they differ")
        self.physics_list = self._axis_list(
            layout, "Physics — associated drivers (not proven cause)")

        # Semantic diff (merged in, Analysis-improvement roadmap Phase A):
        # was its own tab with an identical two-scenario+quantity shape --
        # same "GitHub diff for CFD" ranked-differences report, now a 4th
        # axis here instead of a structurally duplicate sibling tab.
        self.semantic_diff_list = self._axis_list(
            layout, "Semantic diff — ranked differences (click for evidence)")
        self.semantic_diff_canvas = MplCanvas(self)
        self.semantic_diff_canvas.setAccessibleName("Semantic diff evidence field")
        self.semantic_diff_canvas.setMinimumHeight(150)
        layout.addWidget(self.semantic_diff_canvas, 1)
        self._sd_cache: dict = {}

        # Add comparison to session report (Analysis-improvement roadmap
        # Phase C): pins the current pair's ranked semantic-diff statements,
        # reusing report_builder's existing differences rendering.
        report_row = QtWidgets.QHBoxLayout()
        self.add_to_report_button = QtWidgets.QPushButton("Add comparison to session report")
        self.add_to_report_button.setAccessibleName("compare-add-to-session-report")
        self.add_to_report_button.setToolTip(
            "Pin the current A/B comparison's ranked differences into the active session report")
        self.add_to_report_button.clicked.connect(self._add_to_session_report)
        report_row.addWidget(self.add_to_report_button)
        report_row.addStretch(1)
        layout.addLayout(report_row)
        self.report_status = QtWidgets.QLabel("")
        self.report_status.setWordWrap(True)
        self.report_status.setProperty("role", "caption")
        layout.addWidget(self.report_status)

        self.combo_a.currentIndexChanged.connect(self._recompute)
        self.combo_b.currentIndexChanged.connect(self._recompute)
        self.quantity_combo.currentIndexChanged.connect(self._recompute)
        self.temporal_list.insight_activated.connect(self._on_temporal)
        self.semantic_diff_list.insight_activated.connect(self._show_semantic_evidence)

    def _axis_list(self, layout, title: str) -> InsightList:
        label = QtWidgets.QLabel(title)
        label.setProperty("role", "caption")
        layout.addWidget(label)
        lst = InsightList()
        lst.setMaximumHeight(84)
        layout.addWidget(lst)
        return lst

    # ------------------------------------------------------------- lifecycle
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
        self.combo_b.setCurrentIndex(1)  # default to two different scenarios
        self._recompute()

    def _label(self, case_index) -> str:
        return next((e.folder for e in self._manifest if e.case_index == case_index),
                    str(case_index))

    # -------------------------------------------------- session hooks (Phase C)
    def get_comparisons(self) -> list:
        return [dict(c) for c in self._pinned_comparisons]

    def set_comparisons(self, data: list) -> None:
        self._pinned_comparisons = [dict(c) for c in (data or []) if isinstance(c, dict)]

    def set_scenarios(self, case_a, case_b) -> None:
        """Select A and B by case index (V4-M9 comparison hand-off)."""
        self.ensure_loaded()
        ia = self.combo_a.findData(case_a)
        ib = self.combo_b.findData(case_b)
        if ia >= 0:
            self.combo_a.setCurrentIndex(ia)
        if ib >= 0:
            self.combo_b.setCurrentIndex(ib)  # triggers _recompute

    # ------------------------------------------------------------- bus (Phase 2)
    def set_bus(self, bus) -> None:
        """Wires the A/B pair to Selection.comparison -- unlike a single
        scenario_combo, a pair doesn't fit bind_to_bus's generic loop
        (analysis_panel_base.py only looks for `scenario_combo`), so this
        panel gets a small custom set_bus, the same way SensitivityPanel/
        SpaceTimePanel already do for their own non-generic shapes. Reuses
        an existing, previously-unused Selection field rather than adding
        new state."""
        self._bus = bus
        bus.changed.connect(self._on_selection)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.comparison is None:
            return
        ca, cb = sel.comparison
        self.set_scenarios(ca, cb)

    def _publish_comparison(self, ca, cb) -> None:
        if self._bus is not None:
            self._bus.update(origin=self, comparison=(ca, cb))

    # --------------------------------------------------------------- compute
    def _recompute(self) -> None:
        if not self._loaded:
            return
        ca = self.combo_a.currentData()
        cb = self.combo_b.currentData()
        key = self._key
        if ca is None or cb is None or key is None:
            return
        if ca == cb:
            for lst in (self.temporal_list, self.spatial_list, self.physics_list,
                       self.semantic_diff_list):
                lst.set_insights([])
            self._metric = None
            self._render(None)
            return
        self._publish_comparison(ca, cb)
        cache_key = (ca, cb, key.quantity)
        if cache_key not in self._cache:
            data_a = np.asarray(self._store.get(ca, key))
            data_b = np.asarray(self._store.get(cb, key))
            extent = self._store.get_extent(ca, key)
            la, lb = self._label(ca), self._label(cb)
            result = ac.advanced_compare(
                data_a, data_b, extent, self._fps, key.quantity, la, lb,
                summary_a=self._summaries.get(ca), summary_b=self._summaries.get(cb))
            n = min(data_a.shape[0], data_b.shape[0])
            times = np.arange(n) / self._fps
            smax_a = compute_descriptors(data_a[:n], extent, self._fps).column("spatial_max")
            smax_b = compute_descriptors(data_b[:n], extent, self._fps).column("spatial_max")
            self._cache[cache_key] = (result, (times, smax_a, smax_b, la, lb))
        result, self._metric = self._cache[cache_key]
        self.temporal_list.set_insights(result["temporal"])
        self.spatial_list.set_insights(result["spatial"])
        self.physics_list.set_insights(result["physics"])
        self._render(None)

        if cache_key not in self._sd_cache:
            data_a = np.asarray(self._store.get(ca, key))
            data_b = np.asarray(self._store.get(cb, key))
            extent = self._store.get_extent(ca, key)
            self._sd_cache[cache_key] = sd.compare(
                data_a, data_b, extent, self._fps, key.quantity,
                self._label(ca), self._label(cb),
                self._summaries.get(ca), self._summaries.get(cb))
        sd_insights = self._sd_cache[cache_key]
        self.semantic_diff_list.set_insights(sd_insights)
        if sd_insights:
            self._show_semantic_evidence(sd_insights[0])
        else:
            self.semantic_diff_canvas.fig.clear()
            self.semantic_diff_canvas.draw_idle()

    def _show_semantic_evidence(self, insight) -> None:
        """Semantic diff's own evidence render (merged in, Phase A): the
        A - B field at the instant the clicked difference row evidences."""
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

        fig = self.semantic_diff_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        image = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          aspect="auto", extent=extent if extent else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"A − B at t = {idx / self._fps:.1f} s", fontsize=9, fontweight="bold")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(f"Δ{display.label} ({display.unit})", fontsize=8)
        if insight.location is not None and extent is not None:
            ax.plot(insight.location[0], insight.location[1], "o",
                    markersize=12, markerfacecolor="none", markeredgecolor="#00E5FF",
                    markeredgewidth=2)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.semantic_diff_canvas.draw_idle()

    def _add_to_session_report(self) -> None:
        """Pin the current A/B comparison into the active session report
        (Analysis-improvement roadmap Phase C): reuses the semantic-diff
        statements already computed for this pair -- report_builder's
        _comparisons_block renders them with the exact same "Key
        differences" list markup build_comparison_report already uses."""
        ca, cb = self.combo_a.currentData(), self.combo_b.currentData()
        key = self._key
        if ca is None or cb is None or ca == cb or key is None:
            return
        cache_key = (ca, cb, key.quantity)
        differences = [ins.statement for ins in self._sd_cache.get(cache_key, [])]
        la, lb = self._label(ca), self._label(cb)
        self._pinned_comparisons.append({
            "label_a": la, "label_b": lb,
            "case_a": int(ca), "case_b": int(cb),
            "quantity": key.quantity, "differences": differences,
        })
        self.report_status.setText(f"Added {la} vs {lb} to the session report.")

    def _on_temporal(self, insight) -> None:
        self._render(insight.primary_time())

    def _render(self, cursor_t) -> None:
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if self._metric is not None:
            times, smax_a, smax_b, la, lb = self._metric
            unit = get_quantity(self._key.quantity).unit
            ax.plot(times, smax_a, color="#2563EB", linewidth=1.0, label=f"{la} peak")
            ax.plot(times, smax_b, color="#E8622C", linewidth=1.0, label=f"{lb} peak")
            ax.set_ylabel(f"peak {unit}", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            if cursor_t is not None:
                ax.axvline(cursor_t, color="#00E5FF", linewidth=1.4)
        else:
            ax.text(0.5, 0.5, "Pick two different scenarios.", ha="center",
                    va="center", fontsize=9, transform=ax.transAxes, color=plot_fg_color())
        ax.set_xlabel("time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.96, bottom=0.2, left=0.12, right=0.97)
        self.canvas.draw_idle()
