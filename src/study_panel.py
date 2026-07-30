"""Study-Level Analytics panel (V5-M2), an Analysis-page tab -- the
"Factors & Sensitivity" workspace's anchor panel.

Views over the parameter × response table (study_analytics): factor
influence (which factor moves a chosen response most), a response curve
(one factor x one response, level-by-level), correlation + outliers +
study statistics, and (as folded-in sub-tabs) factor effects' spatial
field and the Sensitivity Explorer's what-if interpolation. Selecting a
scenario publishes it to the SelectionBus (M1), so the Live Viewer and
every linked panel follow.

Sensitivity answers a related but distinct question -- local sensitivity
at an interpolated, possibly-unobserved factor setting (What-if table,
Response surface, Tornado), vs. this panel's own global spread across
*observed* factor levels (Factor influence, Response curve) -- so it's
folded in as one whole sub-tab (its own sliders/3-tab-layout/SelectionBus
wiring completely unchanged), the same "thin slot, not a rewrite" pattern
already used for Factor effects, not split apart into this panel's own
tabs.

Parallel coordinates used to be a tab here too; it was extracted into
parallel_coordinates_panel.py (Analysis section consolidation Phase 3)
since it conceptually belongs with the other cross-scenario discovery
tools (Compare & Discover), not this panel's factor/response tabs.
scenario_combo stays here regardless -- every remaining tab of this
panel's own (i.e. not counting the folded-in Sensitivity/Factor-effects
sub-tabs) is a whole-study view with no per-scenario dependency of its
own, but keeping it preserves the existing "every relevant Analysis panel
exposes scenario selection" convention and its cross-panel sync
(unchanged, still bus-bound).

Reads the already-computed scenario summaries; no store reads, no new
simulations. Reuses study_analytics and, for the factor axis order,
factor_effects' convention.

Correlation & outliers is interactive (Compare/Study UX polish): factor-
level filter combos recompute the matrix/outliers/study-statistics over
just the matching scenario subset (so a correlation's strength can be
checked against whether it holds everywhere or only under certain
conditions), each cell is annotated with its actual r value, and clicking
a cell renders the underlying scatter plot (or, on the diagonal, that
response's distribution) in a second canvas -- a correlation number alone
can hide the real relationship (or a single outlier driving it), so the
plot behind it is one click away rather than requiring a separate tool.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.patches import Rectangle

from widgets import MplCanvas
import study_analytics as sa
from analysis_panel_base import populate_scenario_combo
from manifest import LEVEL_LABELS

# Rule-of-thumb floor, not a formal significance test (this module reports
# r and n plainly rather than a p-value): below this many scenarios, a
# single point can swing r substantially, so a pair this thin is flagged
# rather than shown at face value.
_MIN_RELIABLE_N = 6


class StudyPanel(QtWidgets.QWidget):
    def __init__(self, summaries: list, manifest: list,
                 factor_effects_content: QtWidgets.QWidget = None,
                 sensitivity_content: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self._summaries = sorted(summaries or [], key=lambda s: s.case_index)
        self._table = sa.build_table(self._summaries)

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
        populate_scenario_combo(self.scenario_combo, self._summaries)
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            f"{len(self._summaries)} scenarios as a factorial. Values are computed "
            "from each run's summary; selecting a scenario syncs the linked panels.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.tabs = QtWidgets.QTabWidget()
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
        # --- response curve (UX consolidation pass, Study-Level interpretation) ---
        # Factor influence (above) ranks all four factors' overall spread for
        # one response; this answers the more concrete question a researcher
        # actually asks -- "how does changing ventilation affect
        # temperature?" -- as a level-by-level curve for one factor x one
        # response, picked explicitly rather than inferred.
        curve_widget = QtWidgets.QWidget()
        cw = QtWidgets.QVBoxLayout(curve_widget)
        cw.setContentsMargins(0, 0, 0, 0)
        curve_row = QtWidgets.QHBoxLayout()
        curve_row.addWidget(QtWidgets.QLabel("Factor:"))
        self.curve_factor_combo = QtWidgets.QComboBox()
        self.curve_factor_combo.setAccessibleName("Response-curve factor")
        for p in sa.PARAMS:
            self.curve_factor_combo.addItem(sa.PARAM_LABELS[p], p)
        self.curve_factor_combo.currentIndexChanged.connect(self._render_response_curve)
        curve_row.addWidget(self.curve_factor_combo)
        curve_row.addWidget(QtWidgets.QLabel("Response:"))
        self.curve_response_combo = QtWidgets.QComboBox()
        self.curve_response_combo.setAccessibleName("Response-curve response")
        for key in sa.RESPONSE_KEYS:
            self.curve_response_combo.addItem(sa.RESPONSE_LABEL[key], key)
        self.curve_response_combo.currentIndexChanged.connect(self._render_response_curve)
        curve_row.addWidget(self.curve_response_combo)
        curve_row.addStretch(1)
        cw.addLayout(curve_row)
        self.curve_canvas = MplCanvas(self)
        self.curve_canvas.setAccessibleName("Response curve")
        cw.addWidget(self.curve_canvas, 1)
        self.tabs.addTab(curve_widget, "Response curve")
        # --- correlation + outliers (interactive: filterable + drill-down) ---
        corr = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(corr)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)

        # Scenario-subset filter: recomputes the matrix/outliers/stats over
        # only the scenarios matching the selected factor level(s) -- lets a
        # researcher see whether a correlation holds everywhere or only
        # under certain conditions (e.g. only when the door is open), not
        # just its strength over the whole study.
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter to:"))
        self._corr_filters = {p: None for p in sa.PARAMS}
        self._corr_filter_combos = {}
        for p in sa.PARAMS:
            combo = QtWidgets.QComboBox()
            combo.setAccessibleName(f"Correlation filter: {sa.PARAM_LABELS[p]}")
            combo.addItem(f"All ({sa.PARAM_LABELS[p]})", None)
            levels = sorted({int(row["params"][p]) for row in self._table
                             if not np.isnan(row["params"][p])})
            for lvl in levels:
                combo.addItem(LEVEL_LABELS.get(p, {}).get(lvl, f"{sa.PARAM_LABELS[p]}={lvl}"), lvl)
            self._corr_filter_combos[p] = combo
            filter_row.addWidget(combo)
        filter_row.addStretch(1)
        self.corr_reset_button = QtWidgets.QPushButton("Reset filters")
        self.corr_reset_button.clicked.connect(self._reset_corr_filters)
        filter_row.addWidget(self.corr_reset_button)
        cv.addLayout(filter_row)

        corr_caption = QtWidgets.QLabel(
            "Click a cell to see the underlying scatter plot for that response "
            "pair (or its distribution, on the diagonal) -- never trust a "
            "correlation number without looking at the plot behind it.")
        corr_caption.setWordWrap(True)
        corr_caption.setProperty("role", "caption")
        cv.addWidget(corr_caption)

        corr_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.corr_canvas = MplCanvas(self)
        self.corr_canvas.setAccessibleName("Correlation matrix")
        self.corr_canvas.mpl_connect("button_press_event", self._on_corr_click)
        corr_split.addWidget(self.corr_canvas)
        self.corr_drilldown_canvas = MplCanvas(self)
        self.corr_drilldown_canvas.setAccessibleName("Correlation drill-down")
        corr_split.addWidget(self.corr_drilldown_canvas)
        corr_split.setStretchFactor(0, 3)
        corr_split.setStretchFactor(1, 2)
        cv.addWidget(corr_split, 1)

        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setProperty("role", "caption")
        self.stats_label.setTextFormat(QtCore.Qt.RichText)
        cv.addWidget(self.stats_label)
        self.tabs.addTab(corr, "Correlation & outliers")
        # Connected after every combo is populated, so building the filter
        # row itself doesn't trigger a (redundant, and at this point
        # premature -- self._corr_keys doesn't exist yet) re-render.
        for p, combo in self._corr_filter_combos.items():
            combo.currentIndexChanged.connect(
                lambda _i, f=p, c=combo: self._on_corr_filter_changed(f, c.currentData()))
        # Factor effects (Analysis-improvement roadmap Phase B): the actual
        # spatial diverging-field view, complementing this tab's own
        # scalar "Factor influence" ranking above -- folded in as a sub-
        # tab rather than a structurally-separate top-level one, since
        # this panel already reuses factor_effects' axis-order convention.
        # A thin slot, not a rewrite: the panel keeps its own store access,
        # lazy-load (showEvent), and SelectionBus wiring unchanged.
        if factor_effects_content is not None:
            self.tabs.addTab(factor_effects_content, "Factor effects")
        # Sensitivity Explorer (Analysis section consolidation Phase 5):
        # local sensitivity (what-if interpolation, response surface,
        # tornado) at a chosen factor setting, complementing this panel's
        # global spread across observed levels above -- folded in whole
        # (its own sliders/tabs/SelectionBus wiring unchanged) rather than
        # split apart, since its three views share one set of sliders and
        # aren't independently meaningful.
        if sensitivity_content is not None:
            self.tabs.addTab(sensitivity_content, "Sensitivity")
        layout.addWidget(self.tabs, 1)

        self._render_all()

    # ------------------------------------------------------------- rendering
    def _render_all(self) -> None:
        self._render_influence()
        self._render_response_curve()
        self._render_correlation()

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

    def _render_response_curve(self) -> None:
        fig = self.curve_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        factor = self.curve_factor_combo.currentData() or sa.PARAMS[0]
        response = self.curve_response_combo.currentData() or sa.RESPONSE_KEYS[0]
        curve = sa.response_curve(self._table, response, factor)
        if not curve:
            ax.text(0.5, 0.5, "Not enough data.", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            self.curve_canvas.draw_idle()
            return
        levels = [row["level"] for row in curve]
        means = [row["mean"] for row in curve]
        stds = [row["std"] for row in curve]
        ax.errorbar(levels, means, yerr=stds, marker="o", color="#2563EB",
                    ecolor="#93C5FD", capsize=3, linewidth=1.6)
        unit = sa.RESPONSE_UNIT[response]
        ax.set_xlabel(f"{sa.PARAM_LABELS[factor]} level", fontsize=8)
        ax.set_ylabel(sa.RESPONSE_LABEL[response] + (f" ({unit})" if unit else ""), fontsize=8)
        ax.set_title(f"How does {sa.PARAM_LABELS[factor]} affect "
                    f"{sa.RESPONSE_LABEL[response]}?", fontsize=9, fontweight="bold")
        ax.set_xticks(levels)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.88, bottom=0.16, left=0.16, right=0.96)
        self.curve_canvas.draw_idle()

    # ------------------------------------------------ correlation filtering
    def _filtered_table(self) -> list:
        """self._table restricted to the currently selected factor level(s)
        -- None (the "All ..." combo item) means that factor isn't
        filtered."""
        table = self._table
        for factor, value in self._corr_filters.items():
            if value is None:
                continue
            table = [r for r in table if not np.isnan(r["params"][factor])
                     and int(r["params"][factor]) == value]
        return table

    def _on_corr_filter_changed(self, factor: str, value) -> None:
        self._corr_filters[factor] = value
        self._render_correlation()

    def _reset_corr_filters(self) -> None:
        for combo in self._corr_filter_combos.values():
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._corr_filters = {p: None for p in sa.PARAMS}
        self._render_correlation()

    # ------------------------------------------------------------- rendering
    def _render_correlation(self) -> None:
        table = self._filtered_table()
        fig = self.corr_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        keys = sa.RESPONSE_KEYS
        self._corr_keys = keys
        c = sa.correlation_matrix(table, keys)
        counts = sa.pairwise_n(table, keys)
        im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        labels = [sa.RESPONSE_LABEL[k] for k in keys]
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=6, rotation=40, ha="right")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=6)
        # Exact values on the grid itself -- previously just color, with no
        # way to read the actual r without a separate hover/inspection step.
        # A pair's own n can be thinner than the subset's overall scenario
        # count (a response that's NaN for some scenarios shrinks just the
        # pairs involving it) -- hatched + the n shown flags exactly which
        # cells that applies to, rather than a uniform "n of total" caption
        # implying every cell shares the same support.
        any_thin = False
        for i in range(len(keys)):
            for j in range(len(keys)):
                v = c[i, j]
                thin = not np.isnan(v) and counts[i, j] < _MIN_RELIABLE_N
                text = "–" if np.isnan(v) else f"{v:.2f}"
                if thin:
                    text += f"\n(n={counts[i, j]})"
                    any_thin = True
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", edgecolor="#00000066", linewidth=0))
                color = "white" if (not np.isnan(v) and abs(v) > 0.6) else "#222"
                ax.text(j, i, text, ha="center", va="center", fontsize=4.5, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        n, n_total = len(table), len(self._table)
        subset = f"  ·  {n} of {n_total} scenarios" if n != n_total else ""
        thin_note = f"  ·  hatched = fewer than {_MIN_RELIABLE_N} scenarios support that pair" if any_thin else ""
        ax.set_title(f"Response correlations (Pearson r){subset}{thin_note}", fontsize=8)
        fig.subplots_adjust(top=0.9, bottom=0.24, left=0.2, right=0.98)
        self.corr_canvas.draw_idle()
        self._render_stats()
        self._reset_drilldown()

    def _render_stats(self) -> None:
        table = self._filtered_table()
        scores = sa.outlier_scores(table)
        order = np.argsort(scores)[::-1]
        top = [f"{table[i]['folder']} ({scores[i]:.2f}σ)" for i in order[:3]] if len(table) else []
        stats = sa.study_statistics(table)
        n, n_total = len(table), len(self._table)
        subset = f" (filtered: {n} of {n_total} scenarios)" if n != n_total else ""
        lines = [f"<b>Most unusual scenarios</b>{subset} (standardized distance): "
                 + (", ".join(top) if top else "not enough data"),
                 "<b>Study statistics</b>:"]
        for key in sa.RESPONSE_KEYS:
            st = stats[key]
            if st["n"]:
                lines.append(
                    f"&nbsp;&nbsp;{sa.RESPONSE_LABEL[key]}: mean {st['mean']:.1f}, "
                    f"range {st['min']:.1f}–{st['max']:.1f} {sa.RESPONSE_UNIT[key]} (n={st['n']})")
        self.stats_label.setText("<br>".join(lines))

    # ------------------------------------------------------ drill-down (click)
    def _on_corr_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None or not self._corr_keys:
            return
        n = len(self._corr_keys)
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= col < n and 0 <= row < n):
            return
        self._render_drilldown(self._corr_keys[row], self._corr_keys[col])

    def _reset_drilldown(self) -> None:
        fig = self.corr_drilldown_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Click a cell in the matrix\nto see the scatter plot\n"
                          "(diagonal = distribution)",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        self.corr_drilldown_canvas.draw_idle()

    def _render_drilldown(self, key_y: str, key_x: str) -> None:
        table = self._filtered_table()
        x = sa.column(table, key_x)
        y = sa.column(table, key_y)
        fig = self.corr_drilldown_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        x_label = sa.RESPONSE_LABEL[key_x] + (f" ({sa.RESPONSE_UNIT[key_x]})" if sa.RESPONSE_UNIT[key_x] else "")
        if key_x == key_y:
            vals = x[~np.isnan(x)]
            if vals.size:
                ax.hist(vals, bins=min(10, max(3, vals.size // 2)), color="#E8622C", edgecolor="white")
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel("scenarios", fontsize=8)
            ax.set_title(f"{sa.RESPONSE_LABEL[key_x]} distribution (n={vals.size})", fontsize=9)
        else:
            mask = ~(np.isnan(x) | np.isnan(y))
            xs, ys = x[mask], y[mask]
            ax.scatter(xs, ys, color="#2563EB", s=20, alpha=0.85)
            r = float("nan")
            if xs.size >= 3 and xs.std() > 0:
                slope, intercept = np.polyfit(xs, ys, 1)
                xr = np.array([xs.min(), xs.max()])
                ax.plot(xr, slope * xr + intercept, color="#B91C1C", linewidth=1.2, linestyle="--")
            if xs.size >= 2 and xs.std() > 0 and ys.std() > 0:
                r = float(np.corrcoef(xs, ys)[0, 1])
            r_text = f"r = {r:.2f}" if not np.isnan(r) else "r = n/a"
            caveat = "  -- too few points to trust" if 0 < xs.size < _MIN_RELIABLE_N else ""
            y_label = sa.RESPONSE_LABEL[key_y] + (f" ({sa.RESPONSE_UNIT[key_y]})" if sa.RESPONSE_UNIT[key_y] else "")
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_title(f"{sa.RESPONSE_LABEL[key_y]} vs {sa.RESPONSE_LABEL[key_x]}  "
                        f"({r_text}, n={xs.size}){caveat}", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.88, bottom=0.18, left=0.18, right=0.96)
        self.corr_drilldown_canvas.draw_idle()
