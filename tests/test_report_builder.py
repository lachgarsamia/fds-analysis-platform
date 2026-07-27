"""Tests for the scientific report builder (V2 roadmap M3.3)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report_builder import (  # noqa: E402
    build_scenario_report, build_comparison_report, write_report,
    build_session_report, _devices_block, _vector_probes_block,
    _comparisons_block,
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


class TestDevicesBlock:
    def test_empty_renders_nothing(self):
        assert _devices_block([]) == ""

    def test_thermocouple_shows_peak_temperature(self):
        d = {"name": "TC-01", "type": "thermocouple", "position": [1.0, 2.0],
            "results": {"max_temperature_C": 123.4}}
        html = _devices_block([d])
        assert "TC-01" in html and "123" in html and "1.00, 2.00" in html

    def test_detector_shows_activation(self):
        d = {"name": "HD-01", "type": "heat_detector", "position": [0.0, 0.0],
            "results": {"activated": True, "activation_time_s": 42.5}}
        assert "activated at 42.5 s" in _devices_block([d])

    def test_not_yet_computed(self):
        d = {"name": "TC-02", "type": "thermocouple", "position": [0.0, 0.0], "results": None}
        assert "not yet computed" in _devices_block([d])

    def test_non_default_plane_is_annotated(self):
        d = {"name": "TC-03", "type": "thermocouple", "position": [0.0, 0.0],
            "direction": 1, "offset": 15, "results": {"max_temperature_C": 50.0}}
        assert "y=15" in _devices_block([d])

    def test_default_plane_has_no_annotation(self):
        d = {"name": "TC-04", "type": "thermocouple", "position": [0.0, 0.0],
            "direction": 1, "offset": 0, "results": {"max_temperature_C": 50.0}}
        assert "plane" not in _devices_block([d])

    def test_missing_plane_fields_omit_annotation(self):
        """A session saved before V6-M5 has no direction/offset -- must not crash."""
        d = {"name": "TC-05", "type": "thermocouple", "position": [0.0, 0.0],
            "results": {"max_temperature_C": 50.0}}
        assert "plane" not in _devices_block([d])


class TestVectorProbesBlock:
    def test_empty_renders_nothing(self):
        assert _vector_probes_block([]) == ""

    def test_gated_shows_reason(self):
        p = {"name": "VP-01", "position": [0.0, 0.0],
            "results": {"gated": True, "reason": "Requires the M-SIM cluster re-run"}}
        html = _vector_probes_block([p])
        assert "VP-01" in html and "gated" in html and "M-SIM" in html

    def test_computed_shows_peak_speed(self):
        p = {"name": "VP-02", "position": [0.0, 0.0], "results": {"max_speed_m_s": 3.2}}
        assert "3.2 m/s" in _vector_probes_block([p])


class TestSessionReportIncludesDevicesAndProbes:
    def test_devices_and_probes_render_when_present(self):
        session = {"name": "s", "devices": [{"name": "TC-01", "type": "thermocouple",
                                            "position": [0.0, 0.0], "results": None}],
                  "vector_probes": [{"name": "VP-01", "position": [0.0, 0.0],
                                    "results": {"max_speed_m_s": 1.0}}]}
        html = build_session_report(session)
        assert "TC-01" in html and "VP-01" in html

    def test_absent_devices_and_probes_omit_sections(self):
        html = build_session_report({"name": "s"})
        assert "Virtual devices" not in html and "Vector probes" not in html


class TestComparisonsBlock:
    """Analysis-improvement roadmap Phase C: comparisons pinned from
    Compare Axes -- reuses _differences_block's exact rendering."""

    def test_empty_renders_nothing(self):
        assert _comparisons_block([]) == ""

    def test_pinned_comparison_shows_labels_and_differences(self):
        c = {"label_a": "c0_d0", "label_b": "c1_d1", "case_a": 0, "case_b": 1,
            "quantity": "TEMPERATURE", "differences": ["A peaks 40% hotter than B."]}
        html = _comparisons_block([c])
        assert "c0_d0" in html and "c1_d1" in html
        assert "A peaks 40% hotter than B." in html

    def test_no_differences_omits_list_but_keeps_header(self):
        c = {"label_a": "c0_d0", "label_b": "c1_d1", "differences": []}
        html = _comparisons_block([c])
        assert "c0_d0 vs c1_d1" in html
        assert "<ul>" not in html


class TestSessionReportIncludesComparisons:
    def test_comparisons_render_when_present(self):
        session = {"name": "s", "comparisons": [
            {"label_a": "A", "label_b": "B", "differences": ["A is hotter."]}]}
        html = build_session_report(session)
        assert "A is hotter." in html

    def test_absent_comparisons_omits_section(self):
        html = build_session_report({"name": "s"})
        assert "Comparisons" not in html
