"""Tests for session save/restore (V2 roadmap M2.4)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from session import build_session_dict, cell_to_dict, read_session, write_session  # noqa: E402
from slice_key import SliceKey  # noqa: E402


class FakeCell:
    def __init__(self, cell_type, quantity="TEMPERATURE", case_index=0,
                 case_index_a=0, case_index_b=1, ensemble_case_indices=None, ensemble_stat="mean"):
        self.cell_type = cell_type
        self.quantity_key = SliceKey(quantity)
        self.case_index = case_index
        self.case_index_a = case_index_a
        self.case_index_b = case_index_b
        self.ensemble_case_indices = ensemble_case_indices or []
        self.ensemble_stat = ensemble_stat


class TestCellToDict:
    def test_slice_cell(self):
        d = cell_to_dict(FakeCell("slice", case_index=3))
        assert d == {"cell_type": "slice", "quantity": "TEMPERATURE", "case_index": 3,
                    "direction": 1, "offset": 0}

    def test_difference_cell(self):
        d = cell_to_dict(FakeCell("difference", case_index_a=2, case_index_b=5))
        assert d == {"cell_type": "difference", "quantity": "TEMPERATURE",
                     "case_index_a": 2, "case_index_b": 5, "direction": 1, "offset": 0}

    def test_ensemble_cell(self):
        d = cell_to_dict(FakeCell("ensemble", ensemble_case_indices=[1, 2, 3], ensemble_stat="std"))
        assert d == {"cell_type": "ensemble", "quantity": "TEMPERATURE",
                     "ensemble_case_indices": [1, 2, 3], "ensemble_stat": "std",
                     "direction": 1, "offset": 0}


class TestBuildSessionDict:
    def test_shape(self):
        cells = [FakeCell("slice", case_index=0), FakeCell("slice", case_index=1)]
        session = build_session_dict("1x2", cells, active_index=1, time_index=42,
                                      link_clim=True, colormap="viridis", isotherms_enabled=False)
        assert session["version"] == 2  # V4-M2 bumped the schema for the notebook
        assert session["layout"] == "1x2"
        assert len(session["cells"]) == 2
        assert session["active_index"] == 1
        assert session["time_index"] == 42
        assert session["link_clim"] is True
        assert session["colormap"] == "viridis"
        assert session["notebook"] == []  # empty by default

    def test_v1_session_still_loads(self, tmp_path):
        # backward compatibility: a pre-notebook session reads without error
        path = tmp_path / "v1.json"
        path.write_text('{"version": 1, "layout": "1x1", "cells": []}')
        loaded = read_session(str(path))
        assert loaded["version"] == 1 and "notebook" not in loaded


class TestReadWriteRoundTrip:
    def test_round_trip(self, tmp_path):
        cells = [FakeCell("slice", case_index=0)]
        session = build_session_dict("1x1", cells, 0, 10, False, "gist_heat", True)
        path = str(tmp_path / "session.json")
        write_session(path, session)
        loaded = read_session(path)
        assert loaded == session

    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            read_session(str(tmp_path / "nope.json"))

    def test_wrong_version_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"version": 999, "layout": "1x1", "cells": []}')
        with pytest.raises(ValueError):
            read_session(str(path))

    def test_missing_required_field_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"version": 1}')
        with pytest.raises(ValueError):
            read_session(str(path))

    def test_non_json_file_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        with pytest.raises(ValueError):
            read_session(str(path))
