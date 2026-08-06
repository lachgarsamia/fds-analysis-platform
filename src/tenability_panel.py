"""Tenability screening panel (V2 roadmap M3.2; full FED V6-M6), an
Analysis-page tab.

Shows a time-to-untenable map for one scenario. By default (no CO data) this
is convected-heat-only, at a configurable temperature threshold -- each cell
coloured by when it first becomes untenable (red = early, green = late,
blank = never) -- with a prominent disclaimer that this is a PARTIAL screen.

V6-M6: on each scenario load, the panel tries to read 'CARBON MONOXIDE
VOLUME FRACTION' through QuantityProvider. Where CO is available, it shows
the full FED (Fractional Effective Dose: toxic-gas dose + convected-heat
dose, tenability.full_fed) instead, with a disclaimer that states so --
never silently mixing the two. Today's dataset has no CO output, so
QuantityProvider raises GatedQuantityError immediately (the registry's own
gate) and the panel falls back to the partial screen, unchanged from M3.2.

Static/playback-independent (a single time-to-untenable/FED field per
scenario, not per-frame), same convention as the other Analysis panels.
Lazy: computed on first tab show.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import DEFAULT_SLICE_KEY, SliceKey
from quantity_provider import GatedQuantityError
from analysis_panel_base import populate_scenario_combo
import hazard_spaces as hz
import tenability as tn

# Core claim sourced from hazard_spaces.basis_caption() (Analysis roadmap
# B6) -- previously hand-written here independently of hazard_panel.py's
# own (differently-worded) disclaimer for the identical partial-vs-full-
# FED fact. The icon prefix, the explicit "not a full FED analysis"
# contrast, and the tenability-specific second sentence stay panel-
# specific; only the shared core sentence is now one source.
_DISCLAIMER = ("⚠ " + hz.basis_caption() + " -- NOT a full FED (Fractional Effective Dose) "
               "analysis. Toxic-gas tenability is not assessed.")
_FULL_FED_NOTICE = "✓ " + hz.basis_caption(co_based=True) + ". FED ≥ 1.0 marks incapacitation."


class TenabilityPanel(QtWidgets.QWidget):
    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._has_co = False

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

        self.disclaimer = QtWidgets.QLabel(_DISCLAIMER)
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setProperty("role", "caption")
        layout.addWidget(self.disclaimer)

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
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._refresh()

    def _extent(self, case_index):
        try:
            return self._provider.get_extent(case_index, DEFAULT_SLICE_KEY)
        except Exception:  # noqa: BLE001 - geometry is a nice-to-have
            return None

    def _co_field(self, case_index):
        """V6-M6: a real CO ppm field for this scenario, or None if gated.
        GatedQuantityError is the registry's own gate (fires before any
        store access) -- this is never a broad except-and-hope."""
        try:
            return np.asarray(self._provider.get(
                case_index, SliceKey("CARBON MONOXIDE VOLUME FRACTION")))
        except GatedQuantityError:
            return None

    def _refresh(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        threshold = float(self.threshold_spin.value())
        data = np.asarray(self._provider.get(case_index, DEFAULT_SLICE_KEY))
        co = self._co_field(case_index)
        self._has_co = co is not None
        extent = self._extent(case_index)
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps["RdYlGn"].copy()
        cmap.set_bad("#e8e8e8")  # cells that never become untenable/incapacitated

        if self._has_co:
            self.disclaimer.setText(_FULL_FED_NOTICE)
            fed = tn.full_fed(data, co, self._fps)
            field = tn.time_to_fed_field(fed, self._fps)
            scalar = tn.time_to_fed_scalar(fed, self._fps)
            end_frac = float(np.mean(fed[-1] >= tn.FED_INCAPACITATION))
            title = "Time to FED ≥ 1.0 (incapacitation)"
            onset_label = "Onset of incapacitation (full FED)"
            end_label = "incapacitated (FED ≥ 1.0)"
        else:
            self.disclaimer.setText(_DISCLAIMER)
            field = tn.time_to_untenable_field(data, threshold, self._fps)
            scalar = tn.time_to_untenable_scalar(data, threshold, self._fps)
            end_frac = tn.untenable_fraction(data, threshold, data.shape[0] - 1)
            title = f"Time to untenable (>{int(threshold)} °C)"
            onset_label = "Onset of untenable heat"
            end_label = "untenable"

        display = np.where(np.isfinite(field), field, np.nan)
        finite = field[np.isfinite(field)]
        vmax = float(finite.max()) if finite.size else 1.0
        image = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto",
                           extent=extent if extent is not None else None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, fontweight="bold")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("First-crossing time (s) — red = early, grey = never", fontsize=8)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.canvas.draw_idle()

        onset = f"{scalar:.1f} s" if scalar is not None else "never reached"
        self.stats_label.setText(
            f"{onset_label}: {onset} · {end_frac:.0%} of the slice is "
            f"{end_label} at the end of the run.")
