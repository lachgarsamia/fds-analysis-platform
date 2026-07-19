"""Tests for multi-study loading (V2 roadmap M2.5)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifest import scan_generic_study, scan_study  # noqa: E402
from data_provider import load_study, DataLoadError  # noqa: E402


def _touch_smv(directory):
    os.makedirs(directory, exist_ok=True)
    open(os.path.join(directory, "case.smv"), "w").close()


class TestScanStudy:
    def test_single_case_directory_is_degenerate_and_first_class(self, tmp_path):
        case = tmp_path / "line_burner"
        _touch_smv(str(case))
        entries, is_factorial = scan_study(str(case))
        assert is_factorial is False
        assert len(entries) == 1
        assert entries[0].case_index == 0
        assert entries[0].folder == "line_burner"
        assert entries[0].candles == entries[0].door == entries[0].vod == entries[0].voc == 0

    def test_generic_multi_scenario_directory(self, tmp_path):
        _touch_smv(str(tmp_path / "study" / "runA"))
        _touch_smv(str(tmp_path / "study" / "runB"))
        entries, is_factorial = scan_study(str(tmp_path / "study"))
        assert is_factorial is False
        assert {e.folder for e in entries} == {"runA", "runB"}
        assert [e.case_index for e in entries] == [0, 1]

    def test_candle_factorial_is_detected(self, tmp_path):
        # scan_scenarios only parses folder names, no .smv needed.
        (tmp_path / "c1_d0_vod0_voc0").mkdir()
        (tmp_path / "c2_d0_vod0_voc0").mkdir()
        entries, is_factorial = scan_study(str(tmp_path))
        assert is_factorial is True
        assert len(entries) == 2

    def test_generic_scan_ignores_subfolders_without_smv(self, tmp_path):
        (tmp_path / "study" / "not_a_case").mkdir(parents=True)
        _touch_smv(str(tmp_path / "study" / "real_case"))
        entries = scan_generic_study(str(tmp_path / "study"))
        assert [e.folder for e in entries] == ["real_case"]


class TestLoadStudy:
    def test_degenerate_study_loads_without_demo_fallback(self, tmp_path):
        case = tmp_path / "single_case"
        _touch_smv(str(case))
        sim_data = load_study(str(case))
        assert sim_data.is_factorial is False
        assert sim_data.is_demo is False
        assert len(sim_data.manifest) == 1
        assert sim_data.data_matrix.shape == (1, 1, 1, 1)

    def test_generic_study_data_matrix_maps_scenarios(self, tmp_path):
        _touch_smv(str(tmp_path / "s" / "a"))
        _touch_smv(str(tmp_path / "s" / "b"))
        sim_data = load_study(str(tmp_path / "s"))
        assert sim_data.data_matrix.shape == (2, 1, 1, 1)
        assert sorted(sim_data.data_matrix.ravel().tolist()) == [0, 1]

    def test_empty_directory_raises_dataloaderror(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(DataLoadError):
            load_study(str(empty))

    def test_nonexistent_directory_raises_dataloaderror(self, tmp_path):
        with pytest.raises(DataLoadError):
            load_study(str(tmp_path / "does_not_exist"))
