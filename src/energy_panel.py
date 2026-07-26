"""Energy-Budget panel (V2 roadmap M1.2, feature F4).

Full visualization of the per-scenario *_hrr.csv FDS output -- the heat
release rate and its Q_* budget components plus mass loss rates -- of
which the app previously used only two derived numbers (peak HRR, total
energy, M2.5). Static, playback-independent, CSV-driven: reads the CSV
directly via summary_stats.read_hrr_table, never touches ScenarioStore.

Pure metric computations live at module level for unit testing.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from summary_stats import fit_growth_alpha, read_hrr_table
from widgets import MplCanvas

# Budget components plotted against HRR, in FDS's own column names.
BUDGET_COLUMNS = ("Q_RADI", "Q_CONV", "Q_COND", "Q_TOTAL")
MLR_COLUMNS = ("MLR_FUEL", "MLR_TOTAL")


def energy_metrics(table: dict) -> dict:
    """Deterministic summary numbers from a read_hrr_table() dict:
      total_energy_kj   -- integral of HRR
      radiative_fraction -- integral of |Q_RADI| / integral of HRR
                            (Q_RADI is a loss term, negative in FDS output)
      budget_gap_fraction -- |integral(HRR) - integral(|Q_TOTAL|)| /
                             integral(HRR), a closure sanity check
      growth_alpha_kw_s2 -- summary_stats.fit_growth_alpha
    Values are None where the inputs don't support them."""
    times = table.get("Time")
    hrr = table.get("HRR")
    if times is None or hrr is None or times.size < 2:
        return {"total_energy_kj": None, "radiative_fraction": None,
                "budget_gap_fraction": None, "growth_alpha_kw_s2": None}
    total = float(np.trapz(hrr, times))
    metrics = {
        "total_energy_kj": total,
        "radiative_fraction": None,
        "budget_gap_fraction": None,
        "growth_alpha_kw_s2": fit_growth_alpha(times, hrr),
    }
    if total > 0.0:
        q_radi = table.get("Q_RADI")
        if q_radi is not None:
            metrics["radiative_fraction"] = float(np.trapz(np.abs(q_radi), times)) / total
        q_total = table.get("Q_TOTAL")
        if q_total is not None:
            metrics["budget_gap_fraction"] = abs(
                total - float(np.trapz(np.abs(q_total), times))) / total
    return metrics


class EnergyBudgetPanel(QtWidgets.QWidget):
    """Analysis-page tab: scenario combo, two axes (Q_* budget curves,
    MLR curves), and a deterministic metrics line. Lazy: nothing is read
    until ensure_loaded() (Analysis page on_enter), same convention as
    TimeSeriesPanel."""

    def __init__(self, manifest: list, parent=None):
        super().__init__(parent)
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._loaded = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Energy budget")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Energy-budget scenario")
        self.scenario_combo.setToolTip("Scenario whose HRR/energy-budget curves are shown")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Energy budget plot")
        layout.addWidget(self.canvas, 1)

        self.metrics_label = QtWidgets.QLabel("")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setProperty("role", "value")
        layout.addWidget(self.metrics_label)

        self.scenario_combo.currentIndexChanged.connect(self._plot_scenario)

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._plot_scenario(0)

    def _plot_scenario(self, combo_index: int) -> None:
        case_index = self.scenario_combo.itemData(combo_index)
        entry = next((e for e in self._manifest if e.case_index == case_index), None)
        if entry is None:
            return
        table = read_hrr_table(entry.path)
        fig = self.canvas.fig
        fig.clear()
        if table is None:
            ax = fig.add_subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])
            self.metrics_label.setText("")
            self.canvas.draw_idle()
            return

        times = table["Time"]
        budget_ax = fig.add_subplot(121)
        budget_ax.plot(times, table["HRR"], label="HRR", color="#E8622C", linewidth=1.5)
        for name in BUDGET_COLUMNS:
            if name in table:
                budget_ax.plot(times, table[name], label=name, linewidth=1.0)
        budget_ax.set_xlabel("Time (s)", fontsize=8)
        budget_ax.set_ylabel("kW", fontsize=8)
        budget_ax.set_title("Heat release & budget", fontsize=9, fontweight="bold")
        budget_ax.tick_params(labelsize=7)
        budget_ax.legend(fontsize=6)

        mlr_ax = fig.add_subplot(122)
        for name in MLR_COLUMNS:
            if name in table:
                mlr_ax.plot(times, table[name], label=name, linewidth=1.0)
        mlr_ax.set_xlabel("Time (s)", fontsize=8)
        mlr_ax.set_ylabel("kg/s", fontsize=8)
        mlr_ax.set_title("Mass loss rate", fontsize=9, fontweight="bold")
        mlr_ax.tick_params(labelsize=7)
        mlr_ax.legend(fontsize=6)

        fig.subplots_adjust(top=0.90, bottom=0.16, left=0.10, right=0.97, wspace=0.32)
        self.canvas.draw_idle()

        m = energy_metrics(table)
        parts = []
        if m["total_energy_kj"] is not None:
            parts.append(f"Total energy {m['total_energy_kj']:.2f} kJ")
        if m["radiative_fraction"] is not None:
            parts.append(f"radiative fraction {m['radiative_fraction']:.2f}")
        if m["growth_alpha_kw_s2"] is not None:
            parts.append(f"growth fit α = {m['growth_alpha_kw_s2']:.2g} kW/s²")
        if m["budget_gap_fraction"] is not None:
            parts.append(f"budget gap {m['budget_gap_fraction']:.1%}")
        self.metrics_label.setText(" · ".join(parts))
