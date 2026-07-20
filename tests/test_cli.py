"""CI-style smoke tests for the headless CLI (V2 roadmap M3.4).

The pure argument/resolution helpers are tested on synthetic data; the
end-to-end subcommands (which need real .sf/.smv/.s3d output) run against
the real dataset and skip when it's absent, matching the app's other
real-data-gated tests. No QApplication is ever created -- that's the
point of a headless CLI.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cli  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402
from slice_key import SOOT_QUANTITY  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present")


class FakeEntry:
    def __init__(self, case_index, folder):
        self.case_index = case_index
        self.folder = folder


class TestHelpers:
    def test_resolve_case_index_by_folder(self):
        m = [FakeEntry(0, "runA"), FakeEntry(1, "runB")]
        assert cli.resolve_case_index(m, "runB") == 1

    def test_resolve_case_index_by_number(self):
        m = [FakeEntry(0, "runA"), FakeEntry(3, "runB")]
        assert cli.resolve_case_index(m, "3") == 3

    def test_resolve_case_index_unknown_returns_none(self):
        m = [FakeEntry(0, "runA")]
        assert cli.resolve_case_index(m, "nope") is None
        assert cli.resolve_case_index(m, "99") is None

    def test_resolve_quantity_key_soot_gets_side_plane(self):
        key = cli.resolve_quantity_key(SOOT_QUANTITY)
        assert key.quantity == SOOT_QUANTITY
        assert key.plane_pos == 0.0

    def test_resolve_quantity_key_sf_is_plain(self):
        key = cli.resolve_quantity_key("TEMPERATURE")
        assert key.quantity == "TEMPERATURE"
        assert key.plane_pos is None


class TestParser:
    def test_requires_subcommand(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_stats_defaults(self):
        args = cli.build_parser().parse_args(["stats", "/study"])
        assert args.command == "stats" and args.format == "csv"

    def test_report_vs_optional(self):
        args = cli.build_parser().parse_args(["report", "/s", "--scenario", "0", "-o", "r.html"])
        assert args.vs is None


@requires_real_dataset
class TestEndToEnd:
    def test_stats_writes_csv(self, tmp_path):
        out = str(tmp_path / "stats.csv")
        assert cli.main(["stats", SIM_ROOT, "--format", "csv", "-o", out]) == 0
        lines = open(out).read().splitlines()
        assert len(lines) == 25  # header + 24 scenarios
        assert "time_to_untenable_s" in lines[0]

    def test_stats_json(self, tmp_path):
        out = str(tmp_path / "stats.json")
        assert cli.main(["stats", SIM_ROOT, "--format", "json", "-o", out]) == 0
        data = json.load(open(out))
        assert len(data) == 24

    def test_export_writes_png(self, tmp_path):
        out = str(tmp_path / "fig.png")
        assert cli.main(["export", SIM_ROOT, "--scenario", "0", "-o", out]) == 0
        assert open(out, "rb").read(4) == b"\x89PNG"

    def test_report_scenario_and_comparison(self, tmp_path):
        r1 = str(tmp_path / "r.html")
        assert cli.main(["report", SIM_ROOT, "--scenario", "0", "-o", r1]) == 0
        assert "data:image/png;base64," in open(r1).read()
        r2 = str(tmp_path / "cmp.html")
        assert cli.main(["report", SIM_ROOT, "--scenario", "0", "--vs", "12", "-o", r2]) == 0
        assert "vs" in open(r2).read()

    def test_session_render(self, tmp_path):
        session = {
            "version": 1, "layout": "1x2", "time_index": 20,
            "cells": [
                {"cell_type": "slice", "quantity": "TEMPERATURE", "case_index": 0},
                {"cell_type": "difference", "quantity": "TEMPERATURE",
                 "case_index_a": 0, "case_index_b": 12},
            ],
            "active_index": 0, "link_clim": False, "colormap": "gist_heat",
            "isotherms_enabled": False,
        }
        sp = tmp_path / "s.json"
        sp.write_text(json.dumps(session))
        outdir = str(tmp_path / "render")
        assert cli.main(["session-render", str(sp), SIM_ROOT, "-o", outdir]) == 0
        files = sorted(os.listdir(outdir))
        assert files == ["cell_0_slice.png", "cell_1_difference.png"]

    def test_unknown_scenario_returns_error_code(self, tmp_path):
        assert cli.main(["export", SIM_ROOT, "--scenario", "nope", "-o", str(tmp_path / "x.png")]) == 2
