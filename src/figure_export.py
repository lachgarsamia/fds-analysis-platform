"""Publication figure export (V2 roadmap M1.4, feature F6).

Vector SVG/PDF (or high-DPI PNG) of a single slice frame at journal
preset sizes, rendered through a dedicated offscreen Agg figure (never
the live canvas -- same isolation as export.py's AnimationExporter),
with proper physical axes, a labeled colorbar, optionally inline-labeled
isotherm contours, and an optional provenance footer built from data
already on disk: the FDS revision string in the scenario's `.out` file
and a content hash of its `.fds` input deck.

Pure helpers (provenance parsing, hashing, the render itself) are
module-level and Qt-free; only the options dialog needs Qt.
"""

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Journal figure-width presets, inches (single/double column are the
# near-universal 90 mm / 190 mm conventions).
WIDTH_PRESETS = {
    "Single column (3.5 in)": 3.5,
    "Double column (7.2 in)": 7.2,
}

FORMATS = ("SVG (vector)", "PDF (vector)", "PNG (raster)")
_FORMAT_EXTENSIONS = {"SVG (vector)": ".svg", "PDF (vector)": ".pdf", "PNG (raster)": ".png"}


def parse_fds_revision(folder: str) -> str | None:
    """The FDS revision string (e.g. 'FDS6.7.1-0-g14cc738-HEAD') from the
    scenario's `.out` file header, or None if no `.out`/no line. Only the
    header is scanned -- the revision block sits in the first lines."""
    paths = glob.glob(os.path.join(folder, "*.out"))
    if not paths:
        return None
    try:
        with open(paths[0], errors="replace") as f:
            for _i, line in zip(range(50), f):
                if ":" in line and line.split(":", 1)[0].strip() == "Revision":
                    value = line.split(":", 1)[1].strip()
                    return value or None
    except OSError:
        return None
    return None


def input_file_hash(folder: str) -> str | None:
    """sha1 of the scenario's `.fds` input deck (first match), or None."""
    paths = sorted(glob.glob(os.path.join(folder, "*.fds")))
    if not paths:
        return None
    h = hashlib.sha1()
    with open(paths[0], "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_line(folder: str, scenario_name: str, time_s: float) -> str:
    """One deterministic footer line: scenario, playback time, FDS
    revision, input-deck hash -- every part read from disk, none
    generated. Missing parts are omitted, not invented."""
    parts = [scenario_name, f"t = {time_s:.1f} s"]
    revision = parse_fds_revision(folder)
    if revision:
        parts.append(revision)
    digest = input_file_hash(folder)
    if digest:
        parts.append(f"input sha1 {digest[:8]}")
    return " · ".join(parts)


def export_publication_figure(frame: np.ndarray, path: str, *, cmap: str,
                               vmin: float, vmax: float, extent: tuple,
                               colorbar_label: str, title: str = "",
                               width_in: float = 3.5, font_pt: float = 8.0,
                               dpi: int = 300, isotherm_levels: list = None,
                               provenance: str = None) -> None:
    """Renders `frame` to `path`; format follows the file extension
    (.svg/.pdf vector, .png raster at `dpi`). Height is derived from the
    physical extent's aspect ratio plus fixed margins, so the data panel
    is never distorted."""
    fig = _build_publication_figure(
        frame, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
        colorbar_label=colorbar_label, title=title, width_in=width_in,
        font_pt=font_pt, dpi=dpi, isotherm_levels=isotherm_levels, provenance=provenance)
    fig.savefig(path, dpi=dpi)


def _build_publication_figure(frame, *, cmap, vmin, vmax, extent, colorbar_label,
                               title="", width_in=3.5, font_pt=8.0, dpi=300,
                               isotherm_levels=None, provenance=None):
    """Shared figure builder for export_publication_figure and
    figure_png_bytes (V2 roadmap M3.3 -- reports embed the same figure)."""
    x0, x1, z0, z1 = extent
    data_aspect = (z1 - z0) / (x1 - x0) if x1 != x0 else 0.5
    # Data panel occupies ~78% of the width (colorbar + labels take the
    # rest); margins below reserve room for axis labels and the footer.
    height_in = max(1.2, width_in * 0.78 * data_aspect + 0.9)

    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    image = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                       aspect="equal", interpolation="nearest")
    ax.set_xlabel("x (m)", fontsize=font_pt)
    ax.set_ylabel("z (m)", fontsize=font_pt)
    ax.tick_params(labelsize=font_pt - 1)
    if title:
        ax.set_title(title, fontsize=font_pt + 1)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=font_pt)
    colorbar.ax.tick_params(labelsize=font_pt - 1)

    if isotherm_levels:
        levels = sorted(set(isotherm_levels))
        n_z, n_x = frame.shape
        xs = np.linspace(x0, x1, n_x)
        zs = np.linspace(z1, z0, n_z)  # row 0 = z1 (top), origin='upper' convention
        contours = ax.contour(xs, zs, frame, levels=levels, colors="white",
                               linewidths=0.6)
        ax.clabel(contours, fmt="%g", fontsize=font_pt - 2)

    bottom_margin = 0.16 if provenance else 0.12
    fig.subplots_adjust(top=0.90, bottom=bottom_margin, left=0.13, right=0.92)
    if provenance:
        fig.text(0.01, 0.01, provenance, fontsize=max(4.0, font_pt - 3),
                  color="#555555", ha="left", va="bottom")
    return fig


def figure_png_bytes(frame, *, cmap, vmin, vmax, extent, colorbar_label,
                      title="", width_in=5.0, font_pt=9.0, dpi=150,
                      isotherm_levels=None, provenance=None) -> bytes:
    """The same publication figure as export_publication_figure, but
    returned as PNG bytes (for embedding in an M3.3 report's self-
    contained HTML as a base64 data URI) instead of written to a path."""
    import io
    fig = _build_publication_figure(
        frame, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
        colorbar_label=colorbar_label, title=title, width_in=width_in,
        font_pt=font_pt, dpi=dpi, isotherm_levels=isotherm_levels, provenance=provenance)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


class PublicationExportDialog(QtWidgets.QDialog):
    """Collects export options (format, width preset, font size, PNG
    DPI, contour labels, provenance footer). The caller reads options()
    after Accepted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Publication Figure")
        form = QtWidgets.QFormLayout(self)

        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.setAccessibleName("Figure format")
        self.format_combo.addItems(FORMATS)
        form.addRow("Format:", self.format_combo)

        self.width_combo = QtWidgets.QComboBox()
        self.width_combo.setAccessibleName("Figure width preset")
        self.width_combo.addItems(list(WIDTH_PRESETS))
        form.addRow("Width preset:", self.width_combo)

        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setAccessibleName("Figure font size")
        self.font_spin.setRange(5, 14)
        self.font_spin.setValue(8)
        self.font_spin.setSuffix(" pt")
        form.addRow("Font size:", self.font_spin)

        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setAccessibleName("Raster resolution")
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        form.addRow("PNG resolution:", self.dpi_spin)
        self.format_combo.currentTextChanged.connect(
            lambda text: self.dpi_spin.setEnabled(text.startswith("PNG")))
        self.dpi_spin.setEnabled(False)

        self.contours_check = QtWidgets.QCheckBox("Labeled isotherm contours")
        self.contours_check.setChecked(True)
        form.addRow(self.contours_check)

        self.provenance_check = QtWidgets.QCheckBox(
            "Provenance footer (scenario · time · FDS revision · input hash)")
        self.provenance_check.setChecked(True)
        form.addRow(self.provenance_check)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def options(self) -> dict:
        return {
            "extension": _FORMAT_EXTENSIONS[self.format_combo.currentText()],
            "width_in": WIDTH_PRESETS[self.width_combo.currentText()],
            "font_pt": float(self.font_spin.value()),
            "dpi": int(self.dpi_spin.value()),
            "contours": self.contours_check.isChecked(),
            "provenance": self.provenance_check.isChecked(),
        }
