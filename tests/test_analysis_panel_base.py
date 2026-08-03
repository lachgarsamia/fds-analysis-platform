"""Unit tests for analysis_panel_base.py's populate_scenario_combo (Compare
& scenario-naming polish): every analysis panel's scenario_combo should
show a human-readable factor-level label, not the raw disk folder name,
with the folder kept as each item's tooltip."""

from PyQt5 import QtCore, QtWidgets

from analysis_panel_base import populate_scenario_combo
from manifest import ScenarioEntry


def _entry(case_index, folder, candles=0, door=0, vod=0, voc=0):
    return ScenarioEntry(case_index=case_index, folder=folder, path=f"/x/{folder}",
                         candles=candles, door=door, vod=vod, voc=voc)


class TestPopulateScenarioCombo:
    def test_display_text_is_the_human_label_userdata_is_case_index(self, qapp):
        combo = QtWidgets.QComboBox()
        populate_scenario_combo(combo, [_entry(7, "c1_d1_vod0_voc0", candles=0, door=1)])
        assert combo.itemText(0) == "1 candle · Wide door · Vent 1 open · Vent 2 open"
        assert combo.itemData(0) == 7

    def test_folder_is_kept_as_the_item_tooltip(self, qapp):
        combo = QtWidgets.QComboBox()
        populate_scenario_combo(combo, [_entry(0, "c1_d0_vod0_voc0")])
        assert combo.itemData(0, QtCore.Qt.ToolTipRole) == "c1_d0_vod0_voc0"

    def test_populates_one_item_per_entry_in_order(self, qapp):
        combo = QtWidgets.QComboBox()
        entries = [_entry(0, "c1_d0_vod0_voc0"), _entry(1, "c1_d0_vod0_voc1")]
        populate_scenario_combo(combo, entries)
        assert combo.count() == 2
        assert [combo.itemData(i) for i in range(2)] == [0, 1]
