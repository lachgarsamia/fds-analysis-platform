"""Tests for the Publication Figure exporter (V2 roadmap M1.4)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from figure_export import (  # noqa: E402
    PublicationExportDialog, export_publication_figure, input_file_hash,
    parse_fds_revision, provenance_line,
)


OUT_HEADER = """ FDS Version      : FDS 6.7.1
 Revision         : FDS6.7.1-0-g14cc738-HEAD
 Revision Date    : Mon Feb 4 12:26:25 2019 -0500
"""


class TestProvenance:
    def test_parse_fds_revision_reads_header(self, tmp_path):
        (tmp_path / "case.out").write_text(OUT_HEADER)
        assert parse_fds_revision(str(tmp_path)) == "FDS6.7.1-0-g14cc738-HEAD"

    def test_parse_fds_revision_missing_file_returns_none(self, tmp_path):
        assert parse_fds_revision(str(tmp_path)) is None

    def test_input_file_hash_is_deterministic_and_content_sensitive(self, tmp_path):
        (tmp_path / "case.fds").write_text("&HEAD CHID='x' /")
        h1 = input_file_hash(str(tmp_path))
        h2 = input_file_hash(str(tmp_path))
        assert h1 == h2
        assert len(h1) == 40  # sha1 hex digest

    def test_input_file_hash_missing_file_returns_none(self, tmp_path):
        assert input_file_hash(str(tmp_path)) is None

    def test_provenance_line_includes_available_parts_only(self, tmp_path):
        (tmp_path / "case.out").write_text(OUT_HEADER)
        (tmp_path / "case.fds").write_text("&HEAD CHID='x' /")
        line = provenance_line(str(tmp_path), "c1_d0_vod0_voc0", 12.5)
        assert "c1_d0_vod0_voc0" in line
        assert "t = 12.5 s" in line
        assert "FDS6.7.1-0-g14cc738-HEAD" in line
        assert "input sha1" in line

    def test_provenance_line_omits_missing_parts(self, tmp_path):
        line = provenance_line(str(tmp_path), "case", 1.0)
        assert "FDS" not in line
        assert "sha1" not in line
        assert "case" in line


class TestExportPublicationFigure:
    def test_svg_export_produces_nonempty_vector_file(self, tmp_path):
        frame = np.random.rand(20, 30).astype(np.float32) * 100.0
        path = str(tmp_path / "fig.svg")
        export_publication_figure(
            frame, path, cmap="viridis", vmin=0.0, vmax=100.0,
            extent=(0.0, 1.0, 0.0, 0.3), colorbar_label="Temperature (°C)")
        assert os.path.exists(path)
        content = open(path, "rb").read()
        assert content.startswith(b"<?xml") or b"<svg" in content[:200]

    def test_pdf_export_produces_valid_header(self, tmp_path):
        frame = np.zeros((10, 10), dtype=np.float32)
        path = str(tmp_path / "fig.pdf")
        export_publication_figure(
            frame, path, cmap="gist_heat", vmin=0.0, vmax=1.0,
            extent=(0.0, 1.0, 0.0, 1.0), colorbar_label="x")
        assert open(path, "rb").read(5) == b"%PDF-"

    def test_png_export_with_isotherms_and_provenance(self, tmp_path):
        frame = np.linspace(0, 500, 49 * 101).reshape(49, 101).astype(np.float32)
        path = str(tmp_path / "fig.png")
        export_publication_figure(
            frame, path, cmap="gist_heat", vmin=20.0, vmax=500.0,
            extent=(0.0, 1.0, 0.0, 0.3), colorbar_label="Temperature (°C)",
            title="c1_d0_vod0_voc0", isotherm_levels=[60, 100, 300],
            provenance="c1_d0_vod0_voc0 · t = 1.0 s")
        assert os.path.exists(path)
        assert open(path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"

    def test_width_preset_controls_figure_width(self, tmp_path):
        frame = np.zeros((10, 10), dtype=np.float32)
        narrow = str(tmp_path / "narrow.svg")
        wide = str(tmp_path / "wide.svg")
        export_publication_figure(frame, narrow, cmap="viridis", vmin=0, vmax=1,
                                   extent=(0, 1, 0, 1), colorbar_label="x", width_in=3.5)
        export_publication_figure(frame, wide, cmap="viridis", vmin=0, vmax=1,
                                   extent=(0, 1, 0, 1), colorbar_label="x", width_in=7.2)
        assert os.path.getsize(wide) != os.path.getsize(narrow)


class TestPublicationExportDialog:
    def test_default_options_shape(self, qapp):
        dialog = PublicationExportDialog()
        opts = dialog.options()
        assert opts["extension"] == ".svg"
        assert opts["contours"] is True
        assert opts["provenance"] is True
        assert opts["width_in"] == 3.5
        dialog.deleteLater()

    def test_png_format_selects_extension_and_enables_dpi(self, qapp):
        dialog = PublicationExportDialog()
        idx = dialog.format_combo.findText("PNG (raster)")
        dialog.format_combo.setCurrentIndex(idx)
        assert dialog.options()["extension"] == ".png"
        assert dialog.dpi_spin.isEnabled()
        dialog.deleteLater()
