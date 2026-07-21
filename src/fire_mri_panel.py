"""Fire MRI panel (V3-M1, Fire Intelligence Layer).

Renders a scenario's temporal signature (signatures.py) as a single
image: instead of scrubbing 400 frames, one map answers where the fire
arrived first, where it peaked, where danger persisted, and where it
cooled. A channel selector switches between the signature maps; the
cursor probe reports the full per-cell signature at a point; an optional
overlay draws first-arrival isochrones (the "fire front" over time).

Static/playback-independent, lazy on first tab show -- the same
Analysis-panel convention as tenability/factor-effects. Signatures are
computed once per (scenario, quantity) and cached in memory, so switching
channels is instant.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from timeseries import phys_to_index
import signatures as sg


def _channel_display(name: str, unit: str, quantity_cmap: str):
    """(human label, colormap) for a signature channel key."""
    if name == "peak":
        return f"Peak ({unit})", quantity_cmap
    if name == "time_of_peak":
        return "Time of peak (s)", "viridis"
    if name == "cooling_rate":
        return f"Cooling rate ({unit}/s)", "cividis"
    if name == "thermal_dose":
        return f"Thermal dose ({unit}·s)", "inferno"
    if name.startswith("first_crossing_"):
        return f"First arrival > {name.split('_')[-1]} {unit} (s)", "viridis"
    if name.startswith("duration_above_"):
        return f"Duration above {name.split('_')[-1]} {unit} (s)", "inferno"
    return name, "viridis"


class FireMRIPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, quantity_options: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        # Signatures are 2D temporal aggregates; restrict to `.sf` slice
        # quantities (a volumetric SOOT signature would decode the whole
        # `.s3d` set). Registry `kind` is the discriminator (M0.2).
        self._quantity_options = [(label, key) for label, key in quantity_options
                                  if get_quantity(key.quantity).kind == "slice2d"]
        self._fps = max(1, fps)
        self._loaded = False
        self._sig_cache: dict = {}   # (case, quantity) -> SignatureSet
        self._sig = None
        self._ax = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Fire MRI")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Fire MRI scenario")
        header.addWidget(self.scenario_combo)
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Fire MRI quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.setAccessibleName("Fire MRI channel")
        self.channel_combo.setToolTip("Which temporal signature to show as a map")
        header.addWidget(self.channel_combo)
        self.isochrone_check = QtWidgets.QCheckBox("Fire-front isochrones")
        self.isochrone_check.setToolTip("Overlay contours of first-arrival time")
        header.addWidget(self.isochrone_check)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "One image summarizing the whole simulation. Hover the map to read a "
            "cell's full thermal history.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Fire MRI map")
        layout.addWidget(self.canvas, 1)

        self.probe_label = QtWidgets.QLabel("Hover the map to inspect a cell.")
        self.probe_label.setWordWrap(True)
        self.probe_label.setProperty("role", "value")
        layout.addWidget(self.probe_label)

        self.scenario_combo.currentIndexChanged.connect(self._reload_signature)
        self.quantity_combo.currentIndexChanged.connect(self._reload_signature)
        self.channel_combo.currentIndexChanged.connect(self._render)
        self.isochrone_check.toggled.connect(self._render)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    @property
    def _current_key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest or not self._quantity_options:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._reload_signature()

    def _reload_signature(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        key = self._current_key
        if case_index is None or key is None:
            return
        cache_key = (case_index, key.quantity)
        if cache_key not in self._sig_cache:
            self._sig_cache[cache_key] = sg.load_signatures(
                self._store, case_index, key, self._fps)
        self._sig = self._sig_cache[cache_key]
        # repopulate channel combo (levels can differ per quantity)
        current = self.channel_combo.currentData()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for name in self._sig.channel_names():
            label, _cmap = _channel_display(name, self._sig.unit, "viridis")
            self.channel_combo.addItem(label, name)
        idx = self.channel_combo.findData(current)
        self.channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.channel_combo.blockSignals(False)
        self._render()

    def _render(self) -> None:
        if self._sig is None:
            return
        name = self.channel_combo.currentData()
        if name is None:
            return
        qcmap = get_quantity(self._current_key.quantity).cmap
        label, cmap_name = _channel_display(name, self._sig.unit, qcmap)
        field = np.array(self._sig.map(name), dtype=float)
        # "never crossed" (inf) shows as a distinct neutral colour.
        display = np.where(np.isfinite(field), field, np.nan)
        cmap = mpl.colormaps[cmap_name].copy()
        cmap.set_bad("#d9d9d9")
        finite = field[np.isfinite(field)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0

        fig = self.canvas.fig
        fig.clear()
        self._ax = fig.add_subplot(111)
        image = self._ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                                 extent=self._sig.extent if self._sig.extent else None)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title(label, fontsize=9, fontweight="bold")
        cbar = fig.colorbar(image, ax=self._ax, fraction=0.046, pad=0.02)
        cbar.set_label(label, fontsize=8)
        if self.isochrone_check.isChecked():
            self._draw_isochrones()
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.canvas.draw_idle()

    def _draw_isochrones(self) -> None:
        """Contours of the earliest first-arrival channel -- the fire front
        advancing over time."""
        first = next((n for n in self._sig.channel_names() if n.startswith("first_crossing_")), None)
        if first is None or self._sig.extent is None:
            return
        field = np.array(self._sig.map(first), dtype=float)
        finite = field[np.isfinite(field)]
        if finite.size < 2 or finite.max() == finite.min():
            return
        x0, x1, z0, z1 = self._sig.extent
        n_z, n_x = field.shape
        xs = np.linspace(x0, x1, n_x)
        zs = np.linspace(z1, z0, n_z)  # row 0 = z1 (top)
        levels = np.linspace(finite.min(), finite.max(), 5)
        masked = np.where(np.isfinite(field), field, np.nan)
        self._ax.contour(xs, zs, masked, levels=levels, colors="white", linewidths=0.7)

    def _on_move(self, event) -> None:
        if self._sig is None or event.inaxes != self._ax or event.xdata is None:
            return
        if self._sig.extent is None:
            return
        shape = next(iter(self._sig.channels.values())).shape
        row, col = phys_to_index(self._sig.extent, shape, event.xdata, event.ydata)
        vals = self._sig.at_cell(row, col)
        unit = self._sig.unit
        parts = [f"x = {event.xdata:.2f} m, z = {event.ydata:.2f} m"]
        peak, tpeak = vals.get("peak"), vals.get("time_of_peak")
        if peak is not None:
            parts.append(f"peak {peak:.0f} {unit} at t = {tpeak:.1f} s")
        dose = vals.get("thermal_dose")
        if dose is not None:
            parts.append(f"thermal dose {dose:.0f} {unit}·s")
        firsts = [(n, v) for n, v in vals.items() if n.startswith("first_crossing_")]
        for n, v in firsts:
            lvl = n.split("_")[-1]
            parts.append(f"reached {lvl} {unit} " + (f"at t = {v:.1f} s" if np.isfinite(v) else "never"))
        self.probe_label.setText("  ·  ".join(parts))
