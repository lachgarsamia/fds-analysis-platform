"""Mission-Control Dashboard (V5-M4), an Analysis-page tab.

One synchronized overview of the *selected* scenario at the *selected* time:
current phase, HRR, smoke-layer height, maximum hazard class, critical-cell
fraction + hottest location, door setting, and the current insight. Everything
reads from the SelectionBus (M1) -- change the selection anywhere (a panel, an
Insight, the Live Viewer) and the board updates instantly. A small workspace
preset raises the most relevant analysis tab (Adaptive Workspace hook).

Per-scenario series are computed once and cached; a selection change is a cheap
index. Reuses descriptors + events (phase), layer_height, the HRR CSV reader,
and hazard_spaces. Temperature-only partial hazard screen (the FED gate).
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from slice_key import SliceKey
from registry import AMBIENT_C
from descriptors import compute_descriptors
from events import detect_events
from layer_height import smoke_layer_height_series
from summary_stats import read_hrr_table
from linked_inspection import value_at_time
from summary_stats import fmt_hrr as _fmt_hrr
import hazard_spaces as hz

# Preset names only; main_window owns the tab+quantity focus (it holds the
# panels and the SelectionBus). See MainWindow._WORKSPACE.
WORKSPACE_PRESETS = ("Overview", "Temperature study", "Ventilation study",
                     "Smoke study", "Study analytics")


class DashboardPanel(QtWidgets.QWidget):
    workspace_requested = QtCore.pyqtSignal(str)   # preset name (MainWindow resolves)

    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._by_index = {e.case_index: e for e in self._manifest}
        self._fps = max(1, fps)
        self._models = {}
        self._bus = None
        self._current_ci = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Mission control")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Workspace:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setAccessibleName("Workspace preset")
        for name in WORKSPACE_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.activated.connect(self._on_preset)
        header.addWidget(self.preset_combo)
        layout.addLayout(header)

        # Jump to peak moment (Analysis-improvement roadmap Phase A, folded
        # in from the removed Inspect Moment tab): that panel's whole value
        # was "click a temperature peak, see everything at that instant" --
        # this dashboard already reads the current instant's HRR/layer/
        # hazard, so the only missing piece was a way to jump *to* the peak.
        self.jump_to_peak_button = QtWidgets.QPushButton("Jump to peak moment")
        self.jump_to_peak_button.setAccessibleName("Jump to this scenario's peak-temperature moment")
        self.jump_to_peak_button.clicked.connect(self._on_jump_to_peak)
        layout.addWidget(self.jump_to_peak_button)

        self.caption = QtWidgets.QLabel(
            "Synchronized to the current selection. Hazard is a temperature-only "
            "partial screen (no CO/CO₂).")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        self._cards = {}
        specs = ["Scenario", "Time", "Phase", "Heat release rate", "Smoke layer",
                 "Max hazard", "Critical cells", "Door", "Current insight"]
        for i, name in enumerate(specs):
            frame = QtWidgets.QFrame()
            frame.setProperty("role", "card")
            frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            fl = QtWidgets.QVBoxLayout(frame)
            fl.setContentsMargins(10, 8, 10, 8)
            cap = QtWidgets.QLabel(name)
            cap.setProperty("role", "caption")
            val = QtWidgets.QLabel("—")
            val.setProperty("role", "value")
            val.setWordWrap(True)
            fl.addWidget(cap)
            fl.addWidget(val)
            grid.addWidget(frame, i // 3, i % 3)
            self._cards[name] = val
        layout.addLayout(grid)
        layout.addStretch(1)

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)
        self._update(bus.current)

    def _on_selection(self, selection, _origin) -> None:
        self._update(selection)

    def _on_preset(self, _i) -> None:
        self.workspace_requested.emit(self.preset_combo.currentText())

    def _on_jump_to_peak(self) -> None:
        if self._bus is None or self._current_ci is None:
            return
        m = self._models.get(self._current_ci)
        if m is None:
            return
        peak_frame = int(np.argmax(m["peakT"]))
        self._bus.update(origin=self, time_s=peak_frame / self._fps)

    # --------------------------------------------------------------- model
    def _model(self, case_index):
        if case_index in self._models:
            return self._models[case_index]
        entry = self._by_index.get(case_index)
        if entry is None:
            return None
        data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
        extent = self._store.get_extent(case_index, SliceKey("TEMPERATURE"))
        desc = compute_descriptors(data, extent, self._fps)
        hrr = read_hrr_table(entry.path) if entry.path else None
        classes = hz.classify_series(data, hz.band_thresholds("TEMPERATURE"), self._fps)
        model = {
            "folder": entry.folder, "door": getattr(entry, "door", None),
            "extent": extent, "data": data, "n": data.shape[0],
            "peakT": data.max(axis=(1, 2)),
            "layer": (smoke_layer_height_series(data, extent, AMBIENT_C)
                      if extent is not None else None),
            "worst": hz.worst_class(classes),
            "crit_frac": hz.critical_fraction(classes),
            "classes": classes,
            "hrr_t": hrr.get("Time") if hrr else None,
            "hrr_v": hrr.get("HRR") if hrr else None,
            "events": [(e.primary_time(), e.statement) for e in detect_events(desc)
                       if e.primary_time() is not None],
        }
        self._models[case_index] = model
        return model

    # --------------------------------------------------------------- update
    def _default_scenario(self):
        return self._manifest[0].case_index if self._manifest else None

    def _update(self, selection) -> None:
        ci = selection.scenario if selection.scenario is not None else self._default_scenario()
        m = self._model(ci) if ci is not None else None
        if m is None:
            return
        self._current_ci = ci
        t = selection.time_s if selection.time_s is not None else 0.0
        frame = min(max(int(round(t * self._fps)), 0), m["n"] - 1)

        self._cards["Scenario"].setText(m["folder"])
        self._cards["Time"].setText(f"{t:.1f} s")
        # phase / current insight: most recent detected event at or before t
        past = [(et, st) for et, st in m["events"] if et <= t + 1e-9]
        phase = past[-1][1] if past else "pre-ignition"
        self._cards["Phase"].setText(phase.split(":")[0].split("(")[0].strip())
        self._cards["Current insight"].setText(past[-1][1] if past else "—")
        # HRR at t
        if m["hrr_t"] is not None:
            self._cards["Heat release rate"].setText(_fmt_hrr(value_at_time(m["hrr_t"], m["hrr_v"], t)))
        else:
            self._cards["Heat release rate"].setText("n/a")
        # layer height
        self._cards["Smoke layer"].setText(
            f"{m['layer'][frame]:.2f} m" if m["layer"] is not None else "n/a")
        # hazard
        self._cards["Max hazard"].setText(hz.CLASS_NAMES[int(m["worst"][frame])])
        crit = m["crit_frac"][frame] * 100.0
        hottest = np.unravel_index(int(np.argmax(m["data"][frame])), m["data"][frame].shape)
        loc = self._cell_to_xz(m["extent"], m["data"][frame].shape, hottest)
        self._cards["Critical cells"].setText(
            f"{crit:.0f}% at Critical+ · hottest at ({loc[0]:.2f}, {loc[1]:.2f}) m")
        self._cards["Door"].setText(
            f"width level {m['door']}" if m["door"] is not None else "n/a")

    @staticmethod
    def _cell_to_xz(extent, shape, cell):
        if extent is None:
            return (float(cell[1]), float(cell[0]))
        x0, x1, z0, z1 = extent
        n_z, n_x = shape
        x = x0 + cell[1] / max(n_x - 1, 1) * (x1 - x0)
        z = z1 - cell[0] / max(n_z - 1, 1) * (z1 - z0)   # row 0 = ceiling
        return (x, z)
