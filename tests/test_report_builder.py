"""Tests for the scientific report builder (V2 roadmap M3.3)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report_builder import (  # noqa: E402
    build_scenario_report, build_comparison_report, write_report,
)


class FakeEntry:
    def __init__(self, folder):
        self.folder = folder
        self.path = "/x"
        self.candles = self.door = self.vod = self.voc = 0


class FakeSummary:
    def __init__(self, peak=250.0):
        self.max_temp_c = peak
        self.max_temp_by_frame_c = [20.0, peak]
        self.time_to_100c_s = 1.0
        self.time_to_300c_s = None
        self.time_to_600c_s = None
        self.mean_upper_temp_c = 55.0
        self.peak_hrr_kw = 0.12
        self.total_energy_kj = 9.0
        self.growth_alpha_kw_s2 = 1e-5
        self.layer_min_height_m = 0.3
        self.time_to_untenable_s = 0.5


# A 1x1 transparent PNG.
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


class TestScenarioReport:
    def test_contains_all_sections(self):
        html = build_scenario_report(FakeEntry("c1_d0"), FakeSummary(), "Peak 250C.", _PNG, "prov line")
        assert "<!doctype html>" in html
        assert "c1_d0" in html
        assert "Peak 250C." in html                # prose
        assert "data:image/png;base64," in html    # embedded figure
        assert "Peak temperature" in html          # stats table
        assert "250.0" in html
        assert "prov line" in html                 # provenance
        assert "@media print" in html              # print stylesheet for PDF

    def test_none_stat_renders_as_na(self):
        html = build_scenario_report(FakeEntry("x"), FakeSummary(), "s", _PNG, "p")
        assert "n/a" in html  # time_to_300c_s is None

    def test_html_escapes_folder_name(self):
        html = build_scenario_report(FakeEntry("a<b>&c"), FakeSummary(), "s", _PNG, "p")
        assert "a<b>&c" not in html
        assert "a&lt;b&gt;&amp;c" in html

    def test_missing_figure_omits_image(self):
        html = build_scenario_report(FakeEntry("x"), FakeSummary(), "s", b"", "p")
        assert "data:image/png" not in html


class TestComparisonReport:
    def test_two_column_stats_and_both_prose(self):
        html = build_comparison_report(
            FakeEntry("A"), FakeEntry("B"), FakeSummary(200.0), FakeSummary(400.0),
            "prose A", "prose B", _PNG, "prov A", "prov B")
        assert "A vs B" in html
        assert "prose A" in html and "prose B" in html
        assert "200.0" in html and "400.0" in html   # both columns
        assert "prov A" in html and "prov B" in html
        assert "data:image/png;base64," in html      # difference figure


def test_write_report_round_trip(tmp_path):
    html = build_scenario_report(FakeEntry("x"), FakeSummary(), "s", _PNG, "p")
    path = str(tmp_path / "r.html")
    write_report(path, html)
    assert open(path, encoding="utf-8").read() == html
