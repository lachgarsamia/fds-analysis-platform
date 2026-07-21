"""Tests for the Energy-Budget panel (V2 roadmap M1.2)."""

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energy_panel import EnergyBudgetPanel, energy_metrics  # noqa: E402


def _table(times, hrr, q_radi=None, q_total=None):
    t = {"Time": np.asarray(times, dtype=float), "HRR": np.asarray(hrr, dtype=float)}
    if q_radi is not None:
        t["Q_RADI"] = np.asarray(q_radi, dtype=float)
    if q_total is not None:
        t["Q_TOTAL"] = np.asarray(q_total, dtype=float)
    return t


class TestEnergyMetrics:
    def test_total_energy_is_trapezoid_integral(self):
        m = energy_metrics(_table([0, 1, 3], [0, 10, 20]))
        assert m["total_energy_kj"] == 35.0

    def test_radiative_fraction_uses_abs_of_loss_term(self):
        # Q_RADI is negative (a loss) in FDS output; fraction must be positive.
        m = energy_metrics(_table([0, 2], [10, 10], q_radi=[-3, -3]))
        assert m["radiative_fraction"] == pytest.approx(0.3)

    def test_budget_gap_zero_when_totals_match(self):
        m = energy_metrics(_table([0, 2], [10, 10], q_total=[10, 10]))
        assert m["budget_gap_fraction"] == pytest.approx(0.0)

    def test_empty_table_returns_all_none(self):
        m = energy_metrics({})
        assert all(v is None for v in m.values())


class FakeEntry:
    def __init__(self, case_index, folder, path):
        self.case_index = case_index
        self.folder = folder
        self.path = path
        self.candles = self.door = self.vod = self.voc = 0


@pytest.fixture
def scenario_dir(tmp_path):
    folder = tmp_path / "c1_d0_vod0_voc0"
    folder.mkdir()
    with open(folder / "c1_d0_vod0_voc0_hrr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "kW", "kW", "kg/s"])
        writer.writerow(["Time", "HRR", "Q_RADI", "MLR_FUEL"])
        writer.writerows([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, -0.3, 0.01),
                          (2.0, 4.0, -1.2, 0.02)])
    return folder


class TestEnergyBudgetPanel:
    def test_ensure_loaded_plots_and_reports_metrics(self, qapp, scenario_dir):
        panel = EnergyBudgetPanel([FakeEntry(0, scenario_dir.name, str(scenario_dir))])
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == 1
        assert "Total energy" in panel.metrics_label.text()
        assert "radiative fraction" in panel.metrics_label.text()
        panel.deleteLater()

    def test_missing_csv_shows_placeholder_not_crash(self, qapp, tmp_path):
        empty = tmp_path / "empty_case"
        empty.mkdir()
        panel = EnergyBudgetPanel([FakeEntry(0, "empty_case", str(empty))])
        panel.ensure_loaded()
        assert panel.metrics_label.text() == ""
        panel.deleteLater()

    def test_lazy_until_ensure_loaded(self, qapp, scenario_dir):
        panel = EnergyBudgetPanel([FakeEntry(0, scenario_dir.name, str(scenario_dir))])
        assert panel.scenario_combo.count() == 0
        panel.deleteLater()
