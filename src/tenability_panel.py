"""Tenability screening panel (V2 roadmap M3.2), an Analysis-page tab.

Shows a time-to-untenable map for one scenario at a configurable
convected-heat threshold: each cell colored by when it first becomes
untenable (red = early, green = late, blank = never). A prominent
disclaimer states this is a PARTIAL, temperature-only screen -- there is
no CO/CO2 output in this dataset, so it is not a full FED analysis.

Static/playback-independent (a single time-to-untenable field per
scenario, not per-frame), same convention as the other Analysis panels.
Lazy: computed on first tab show.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import DEFAULT_SLICE_KEY
import tenability as tn

_DISCLAIMER = (
    "⚠ Partial hazard screen — convected heat (temperature) only. This dataset has "
    "no CO/CO₂ output, so this is NOT a full FED (Fractional Effective Dose) analysis. "
    "Toxic-gas tenability is not assessed."
)


class TenabilityPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Tenability screening")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Tenability scenario")
        header.addWidget(self.scenario_combo)
        header.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setAccessibleName("Tenability temperature threshold")
        self.threshold_spin.setRange(30, 600)
        self.threshold_spin.setSingleStep(10)
        self.threshold_spin.setValue(int(tn.TENABILITY_THRESHOLD_C))
        self.threshold_spin.setSuffix(" °C")
        self.threshold_spin.setToolTip("Air temperature above which exposure is treated as untenable")
        header.addWidget(self.threshold_spin)
        layout.addLayout(header)

        disclaimer = QtWidgets.QLabel(_DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setProperty("role", "caption")
        layout.addWidget(disclaimer)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Time to untenable map")
        layout.addWidget(self.canvas, 1)

        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setProperty("role", "value")
        layout.addWidget(self.stats_label)

        self.scenario_combo.currentIndexChanged.connect(self._refresh)
        self.threshold_spin.valueChanged.connect(self._refresh)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._refresh()

    def _extent(self, case_index):
        try:
            return self._store.get_extent(case_index, DEFAULT_SLICE_KEY)
        except Exception:  # noqa: BLE001 - geometry is a nice-to-have
            return None

    def _refresh(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        threshold = float(self.threshold_spin.value())
        data = np.asarray(self._store.get(case_index, DEFAULT_SLICE_KEY))
        field = tn.time_to_untenable_field(data, threshold, self._fps)
        scalar = tn.time_to_untenable_scalar(data, threshold, self._fps)
        end_frac = tn.untenable_fraction(data, threshold, data.shape[0] - 1)

        display = np.where(np.isfinite(field), field, np.nan)
        finite = field[np.isfinite(field)]
        vmax = float(finite.max()) if finite.size else 1.0
        extent = self._extent(case_index)

        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps["RdYlGn"].copy()
        cmap.set_bad("#e8e8e8")  # cells that never become untenable
        image = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto",
                           extent=extent if extent is not None else None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Time to untenable (>{int(threshold)} °C)", fontsize=9, fontweight="bold")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("First-crossing time (s) — red = early, grey = never", fontsize=8)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.canvas.draw_idle()

        onset = f"{scalar:.1f} s" if scalar is not None else "never reached"
        self.stats_label.setText(
            f"Onset of untenable heat: {onset} · {end_frac:.0%} of the slice is "
            f"untenable at the end of the run.")
