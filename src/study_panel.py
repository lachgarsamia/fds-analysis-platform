"""Study-Level Analytics panel (V5-M2), an Analysis-page tab.

Treats the current experiment as a full factorial. Three views over the
parameter × response table (study_analytics): parallel coordinates
(parameters vs responses, one line per scenario), factor influence (which
factor moves a chosen response most), and correlation + outliers + study
statistics. Selecting a scenario publishes it to the SelectionBus (M1), so the
Live Viewer and every linked panel follow.

Reads the already-computed scenario summaries; no store reads, no new
simulations. Reuses study_analytics and, for the factor axis order,
factor_effects' convention.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
import study_analytics as sa


class StudyPanel(QtWidgets.QWidget):
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
        title = QtWidgets.QLabel("Study analytics")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Scenario:"))
        self.scenario_combo = QtWidgets.QComboBox()   # bound to the bus by main_window (M1)
        self.scenario_combo.setAccessibleName("Study scenario")
        for s in self._summaries:
            self.scenario_combo.addItem(s.folder, s.case_index)
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            f"{len(self._summaries)} scenarios as a factorial. Values are computed "
            "from each run's summary; selecting a scenario syncs the linked panels.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.tabs = QtWidgets.QTabWidget()
        # --- parallel coordinates ---
        self.parallel_canvas = MplCanvas(self)
        self.parallel_canvas.setAccessibleName("Parallel coordinates")
        self.tabs.addTab(self.parallel_canvas, "Parallel coordinates")
        # --- factor influence ---
        infl = QtWidgets.QWidget()
        iv = QtWidgets.QVBoxLayout(infl)
        iv.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Response:"))
        self.response_combo = QtWidgets.QComboBox()
        self.response_combo.setAccessibleName("Influence response")
        for key in sa.RESPONSE_KEYS:
            self.response_combo.addItem(sa.RESPONSE_LABEL[key], key)
        self.response_combo.currentIndexChanged.connect(self._render_influence)
        row.addWidget(self.response_combo)
        row.addStretch(1)
        iv.addLayout(row)
        self.influence_canvas = MplCanvas(self)
        self.influence_canvas.setAccessibleName("Factor influence")
        iv.addWidget(self.influence_canvas, 1)
        self.tabs.addTab(infl, "Factor influence")
        # --- correlation + outliers ---
        corr = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(corr)
        cv.setContentsMargins(0, 0, 0, 0)
        self.corr_canvas = MplCanvas(self)
        self.corr_canvas.setAccessibleName("Correlation matrix")
        cv.addWidget(self.corr_canvas, 1)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setProperty("role", "caption")
        self.stats_label.setTextFormat(QtCore.Qt.RichText)
        cv.addWidget(self.stats_label)
        self.tabs.addTab(corr, "Correlation & outliers")
        layout.addWidget(self.tabs, 1)

        self.scenario_combo.currentIndexChanged.connect(self._render_parallel)
        self._render_all()

    # ------------------------------------------------------------- rendering
    def _render_all(self) -> None:
        self._render_parallel()
        self._render_influence()
        self._render_correlation()

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

    def _render_influence(self) -> None:
        fig = self.influence_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        response = self.response_combo.currentData() or sa.RESPONSE_KEYS[0]
        ranking = sa.influence_ranking(self._table, response)
        names = [sa.PARAM_LABELS[p] for p, _v, _s in ranking]
        shares = [s * 100 for _p, _v, s in ranking]
        ax.barh(range(len(names))[::-1], shares, color="#E8622C")
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("share of factor influence (%)", fontsize=8)
        ax.set_title(f"What moves {sa.RESPONSE_LABEL[response]}?", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.9, bottom=0.16, left=0.28, right=0.96)
        self.influence_canvas.draw_idle()

    def _render_correlation(self) -> None:
        fig = self.corr_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        keys = sa.RESPONSE_KEYS
        c = sa.correlation_matrix(self._table, keys)
        im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        labels = [sa.RESPONSE_LABEL[k] for k in keys]
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=6, rotation=40, ha="right")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Response correlations (Pearson r)", fontsize=9)
        fig.subplots_adjust(top=0.9, bottom=0.24, left=0.2, right=0.98)
        self.corr_canvas.draw_idle()
        self._render_stats()

    def _render_stats(self) -> None:
        scores = sa.outlier_scores(self._table)
        order = np.argsort(scores)[::-1]
        top = [f"{self._table[i]['folder']} ({scores[i]:.2f}σ)" for i in order[:3]]
        stats = sa.study_statistics(self._table)
        lines = ["<b>Most unusual scenarios</b> (standardized distance): " + ", ".join(top),
                 "<b>Study statistics</b>:"]
        for key in sa.RESPONSE_KEYS:
            st = stats[key]
            if st["n"]:
                lines.append(
                    f"&nbsp;&nbsp;{sa.RESPONSE_LABEL[key]}: mean {st['mean']:.1f}, "
                    f"range {st['min']:.1f}–{st['max']:.1f} {sa.RESPONSE_UNIT[key]} (n={st['n']})")
        self.stats_label.setText("<br>".join(lines))
