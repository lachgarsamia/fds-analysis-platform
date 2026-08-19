"""Device panel (V6-M2 Virtual Device Network), an Analysis-page tab.

Turns "visualizing fields" into "instrumenting a simulation like an
experiment": place a virtual thermocouple, heat detector, or RTI sprinkler
at a physical point and see its measurement evolve. Same interaction
convention as the Measurement Tools panel (measurement_panel.py) -- a
locator heatmap you click on -- so placing a device is the same gesture as
placing a probe, just producing a devices.py Device instead of a
measure.py Measurement.

A device's time series is computed once (Device.compute(), on place/edit)
and cached on `results`; nothing here or in the Live Viewer ever recomputes
it per frame -- playback only ever indexes the cached arrays (see
Device.state_at() and MainWindow._device_markers_for()).

Live playback (Probe & Measure domain-coverage/live-readout pass): the
locator canvas crops to the room's real wall bounds with aspect="equal",
the same treatment B1 (views.py) gave the Live Viewer, inlined directly
off schematic.ROOM_X/ROOM_Z rather than reusing set_room_outline() itself
-- that method is entangled with SliceView-only blit/artist state this
panel doesn't have. set_bus() (matching time_window_panel/spacetime_panel/
dashboard_panel's own precedent over the generic bind_to_bus slider-sync,
which this panel has no frame_slider for) follows Selection.time_s so the
marker color and readout track whatever frame the pinned playback bar is
on, not the placement instant -- gated on isVisible() (same "a hidden
analysis tab must not re-render on every playback tick" rule
bind_to_bus's own _on_selection already applies) so a backgrounded tab
costs nothing per tick.

Reuses QuantityProvider (device readings), the registry (TEMPERATURE
display), the Insight model (activation navigation, via the same
insight_activated wiring every other V3 feature uses), and
timeseries.write_series_csv (CSV export). No parallel measurement
pipeline, no cinematic/store/TimeController changes.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import SliceKey, AXIS_TO_DIRECTION
from registry import get_quantity
from analysis_panel_base import populate_scenario_combo
from schematic import ROOM_X, ROOM_Z
import devices as dv

_PLANE_AXES = ("y", "x", "z")   # y first: the app's default/verified plane

_PREFIX = {"thermocouple": "TC", "heat_detector": "HD", "sprinkler": "SP"}

# Scientific, non-decorative state colors: neutral reading vs. activated.
_COLOR_IDLE = "#3DA5FF"
_COLOR_ACTIVE = "#FF5252"


class DevicePanel(QtWidgets.QWidget):
    devices_changed = QtCore.pyqtSignal()      # placed/edited/deleted -- refresh Live Viewer markers
    device_activated = QtCore.pyqtSignal(object)   # an Insight -- jump-to navigation

    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._data = None      # TEMPERATURE background for the locator canvas
        self._extent = None
        self._devices: list = []
        self._counters = {"thermocouple": 0, "heat_detector": 0, "sprinkler": 0}
        self._loc_ax = None
        self._bus = None
        self._current_index = 0    # live playback frame -- see set_bus()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Virtual devices")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        header = QtWidgets.QHBoxLayout()
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Device scenario")
        self.scenario_combo.setToolTip("Which scenario new devices are placed in")
        header.addWidget(self.scenario_combo)
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.setAccessibleName("Device type")
        self.type_combo.setToolTip("The device type the next map click will place")
        for t in dv.KINDS:
            self.type_combo.addItem(dv.KIND_LABELS[t], t)
        header.addWidget(self.type_combo)
        # V6-M5: which plane new devices read. Y is the app's verified
        # plane (offset 0 or 15, both real); X/Z are offered because the
        # engine supports any plane, but this dataset has no X/Z-normal
        # slices -- placing on one cleanly shows "gated", never fabricated.
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.setAccessibleName("Device plane axis")
        self.direction_combo.setToolTip("Which axis the plane is normal to (Y is verified; X/Z are gated "
                                        "on this dataset -- see docs/msim-preparation.md)")
        for axis in _PLANE_AXES:
            self.direction_combo.addItem(axis.upper(), AXIS_TO_DIRECTION[axis])
        header.addWidget(self.direction_combo)
        self.offset_spin = QtWidgets.QSpinBox()
        self.offset_spin.setAccessibleName("Device plane offset")
        self.offset_spin.setToolTip("The plane's mesh-cell offset along its normal axis")
        self.offset_spin.setRange(0, 999)
        header.addWidget(self.offset_spin)
        header.addStretch(1)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Click the map to place the selected device type at that point. Playback "
            "runs once; a device's history is computed at placement and cached -- "
            "results save with the session. \"Compare\" evaluates the same device "
            "(position, type, parameters) across every scenario.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        # Analysis UX + reliability pass: a heat detector and a sprinkler at
        # the same point can legitimately disagree -- they're independent
        # devices with independently different, both-correct physical
        # models (instant threshold vs. an RTI thermal-lag ODE), not a
        # synchronized pair. Spelled out explicitly rather than silently
        # matching their booleans, since the Live Viewer map now also marks
        # each kind with its own shape (see views.py's _DEVICE_MARKER_SHAPES)
        # so the two are visually distinguishable, not just by color.
        self.model_note = QtWidgets.QLabel(
            "Heat detectors activate instantly at their threshold temperature; "
            "sprinklers respond via a physical thermal-lag model (RTI) and are "
            "expected to activate later than a heat detector at the same point "
            "-- that's modeled behavior, not a fault.")
        self.model_note.setWordWrap(True)
        self.model_note.setProperty("role", "caption")
        layout.addWidget(self.model_note)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        layout.addWidget(self.status)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Device placement canvas")
        layout.addWidget(self.canvas, 1)

        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("role", "value")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        list_row = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Devices list")
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(lambda _i: self._render())
        list_row.addWidget(self.list, 1)
        btns = QtWidgets.QVBoxLayout()
        _TOOLTIPS = {
            "device-rename": "Rename the selected device",
            "device-edit": "Edit RTI/activation-temperature parameters and recompute",
            "device-jump": "Reveal this device's result across the app (Live Viewer, Graph, Context)",
            "device-compare": "Evaluate this device across every scenario",
            "device-export": "Export this device's time series as CSV",
            "device-delete": "Delete the selected device",
        }
        for text, slot, name in (
                ("Rename", self._rename, "device-rename"),
                ("Edit parameters", self._edit_parameters, "device-edit"),
                ("Jump to", self._jump_to, "device-jump"),
                ("Compare", self._compare_across_scenarios, "device-compare"),
                ("Export CSV", self._export, "device-export"),
                ("Delete", self._delete, "device-delete")):
            b = QtWidgets.QPushButton(text)
            b.setAccessibleName(name)
            b.setToolTip(_TOOLTIPS[name])
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        list_row.addLayout(btns)
        layout.addLayout(list_row)

        self.compare_table = QtWidgets.QTableWidget()
        self.compare_table.setAccessibleName("Device comparison table")
        self.compare_table.setMaximumHeight(150)
        self.compare_table.setVisible(False)
        layout.addWidget(self.compare_table)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.canvas.mpl_connect("button_press_event", self._on_press)

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        was_loaded = self._loaded
        self.ensure_loaded()
        if was_loaded:
            # Not the first show (ensure_loaded already rendered that case):
            # catch up on whatever frame playback moved to while this tab
            # was hidden and set_bus()'s isVisible() gate was skipping it.
            self._render()

    def set_bus(self, bus) -> None:
        """Follow the shared playback frame (Selection.time_s), same
        set_bus precedent as time_window_panel/spacetime_panel/
        dashboard_panel -- this panel has no frame_slider for the generic
        bind_to_bus sync to hook. One-way: this panel never publishes a
        selection, only reacts."""
        self._bus = bus
        bus.changed.connect(self._on_selection)
        self._on_selection(bus.current, None)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.time_s is None:
            return
        self._current_index = max(0, int(round(sel.time_s * self._fps)))
        if self._loaded and self.isVisible():
            self._render()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        key = SliceKey("TEMPERATURE")
        self._data = np.asarray(self._provider.get(case_index, key))
        self._extent = self._provider.get_extent(case_index, key)
        self._render()

    # -------------------------------------------------- session hooks (V6-M2)
    def get_devices(self) -> list:
        return [d.to_dict() for d in self._devices]

    def set_devices(self, data: list) -> None:
        """Restores devices with their cached `results` verbatim -- no
        recompute, so a reopened session reproduces identical numbers."""
        self._devices = [dv.Device.from_dict(d) for d in (data or []) if isinstance(d, dict)]
        for d in self._devices:
            prefix = _PREFIX.get(d.type, "DV")
            n = 0
            if d.name.startswith(prefix + "-"):
                try:
                    n = int(d.name.split("-")[-1])
                except ValueError:
                    n = 0
            self._counters[d.type] = max(self._counters.get(d.type, 0), n)
        self._refresh_list()
        if self._loaded:
            self._render()

    # ----------------------------------------------------------- interaction
    def _on_press(self, event) -> None:
        if self._data is None or event.inaxes is not self._loc_ax or event.xdata is None:
            return
        self._place(event.xdata, event.ydata)

    def _place(self, x: float, z: float) -> None:
        device_type = self.type_combo.currentData()
        self._counters[device_type] += 1
        name = f"{_PREFIX[device_type]}-{self._counters[device_type]:02d}"
        case_index = self.scenario_combo.currentData()
        d = dv.Device(id=f"{device_type}-{id(object())}", name=name, type=device_type,
                      scenario=case_index, position=(float(x), float(z)),
                      parameters=dv.default_parameters(device_type),
                      direction=self.direction_combo.currentData(), offset=self.offset_spin.value())
        try:
            d.compute(self._provider, self._fps)
            self.status.setText("")
        except Exception as e:
            # V6-M5: an X/Z-normal (or absent-offset) plane this dataset
            # doesn't have -- the device is still placed (it will "light up"
            # automatically once that plane exists, no re-placement needed),
            # results stay None ("not yet computed"), never fabricated.
            # Caught broadly (not just GatedQuantityError): a plane can be
            # *declared* in the .smv inventory yet still fail to actually
            # load (a plain numpy/IO error deep in the slice reader) -- an
            # uncaught exception here would escape this Qt slot and PyQt5
            # aborts the process on that, so this boundary must never let
            # one through.
            self.status.setText(f"Gated: {e}")
        self._devices.append(d)
        self._refresh_list()
        self.list.setCurrentRow(len(self._devices) - 1)
        self._render()
        self.devices_changed.emit()

    # --------------------------------------------------------------- actions
    def _current(self):
        i = self.list.currentRow()
        return self._devices[i] if 0 <= i < len(self._devices) else None

    def _rename(self) -> None:
        d = self._current()
        if d is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename device", "Name:", text=d.name)
        if ok and name.strip():
            d.name = name.strip()
            self._refresh_list()
            self._render()
            self.devices_changed.emit()

    def _edit_parameters(self) -> None:
        d = self._current()
        if d is None or not d.parameters:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Edit parameters -- {d.name}")
        form = QtWidgets.QFormLayout(dialog)
        spins = {}
        for key, value in d.parameters.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(2)
            spin.setValue(float(value))
            form.addRow(key, spin)
            spins[key] = spin
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            for key, spin in spins.items():
                d.parameters[key] = spin.value()
            try:
                d.compute(self._provider, self._fps)
                self.status.setText("")
            except Exception as e:
                self.status.setText(f"Gated: {e}")
            self._refresh_list()
            self._render()
            self.devices_changed.emit()

    def _delete(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self._devices):
            del self._devices[i]
            self._refresh_list()
            self.compare_table.setVisible(False)
            self._render()
            self.devices_changed.emit()

    def _compare_across_scenarios(self) -> None:
        """Cross-scenario comparison (Analysis-improvement roadmap Phase C):
        the exact pattern Zone Statistics already has -- place a device
        once, then evaluate its type/position/parameters/plane at every
        scenario in one click, streaming one scenario at a time."""
        d = self._current()
        if d is None:
            return
        self.compare_table.clear()
        self.compare_table.setColumnCount(2)
        self.compare_table.setRowCount(len(self._manifest))
        self.compare_table.setHorizontalHeaderLabels(["Scenario", "Result"])
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            pairs = dv.compare_across_scenarios(d, self._provider, self._manifest, self._fps)
            for r, (entry, computed) in enumerate(pairs):
                text = self._headline(computed) if computed is not None else "gated (plane unavailable)"
                self.compare_table.setItem(r, 0, QtWidgets.QTableWidgetItem(entry.folder))
                self.compare_table.setItem(r, 1, QtWidgets.QTableWidgetItem(text))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.compare_table.resizeColumnsToContents()
        self.compare_table.setVisible(True)

    def _jump_to(self) -> None:
        d = self._current()
        if d is None:
            return
        insight = d.summary_insight()
        if insight is not None:
            self.device_activated.emit(insight)

    def _export(self) -> None:
        d = self._current()
        if d is None or d.results is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export device data", f"{d.name}.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        dv.export_csv(d, path)

    # --------------------------------------------------------------- render
    def _live_readout(self, d: dv.Device) -> str:
        """The device's reading at the currently active playback frame
        (Device.state_at() -- an index into the already-computed series,
        never a recompute), so scrubbing/playing the pinned playback bar
        updates this line -- unlike _headline()'s run-aggregate summary."""
        state = d.state_at(self._current_index)
        temp = state.get("temperature_C")
        if temp is None:
            return ""
        t_s = self._current_index / self._fps
        live = f"{temp:.0f} °C at t={t_s:.1f}s"
        if d.type != "thermocouple":
            live += " · active" if state.get("active") else " · idle"
        return live

    def _headline(self, d: dv.Device) -> str:
        r = d.results or {}
        if not r:
            return "not yet computed"
        if d.type == "thermocouple":
            # V6-M6: show full FED when CO is available (max_fed_full is
            # None while CO is gated -- never fabricated), else the heat-
            # only FED, alongside the existing peak/heating-rate readout.
            if r.get("max_fed_full") is not None:
                fed_part = f" · FED {r['max_fed_full']:.2f}"
            else:
                fed_part = f" · heat-FED {r.get('max_fed_heat', 0.0):.2f}"
            return (f"peak {r['max_temperature_C']:.0f} °C · "
                    f"{r['heating_rate_C_per_s']:.1f} °C/s{fed_part}")
        if r.get("activated"):
            return f"activated at {r['activation_time_s']:.1f} s"
        if d.type == "sprinkler" and r.get("link_temperature_C"):
            # A sprinkler's thermal-lag link can trail the actual gas
            # temperature by a lot -- "did not activate" alone reads like a
            # heat detector at the same point should have tripped it too,
            # when in fact its slow-response link simply never caught up
            # to a brief heat pulse before the room cooled again (real
            # sprinkler physics: this is exactly why "quick response"
            # sprinklers with a much lower RTI exist). Showing how close it
            # got is the honest way to answer "why didn't this activate"
            # without pretending it should have.
            peak_link = max(r["link_temperature_C"])
            threshold = d.parameters.get("activation_temp_C", 68.0)
            return f"did not activate (link peaked at {peak_link:.0f} °C, needs {threshold:g} °C)"
        return "did not activate"

    def _refresh_list(self) -> None:
        self.list.clear()
        for d in self._devices:
            item = QtWidgets.QListWidgetItem(
                f"{d.name} ({dv.KIND_LABELS[d.type]}) — {self._headline(d)}")
            item.setToolTip((d.results or {}).get("basis", ""))
            self.list.addItem(item)

    def _render(self) -> None:
        if self._data is None:
            return
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._loc_ax = ax
        frame = self._data[int(self._data.shape[0] * 0.6)]
        q = get_quantity("TEMPERATURE")
        ax.imshow(frame, cmap=q.cmap, vmin=q.vmin, vmax=q.slider_default,
                  aspect="equal", extent=self._extent if self._extent else None)
        # Domain-coverage fix: crop to the room's real wall bounds on both
        # axes, same treatment B1 (views.py) gave the Live Viewer -- inlined
        # off the shared ROOM_X/ROOM_Z constants rather than the full
        # SliceView.set_room_outline() (that method also draws wall/door/
        # vent LineCollection artists and manages blit-cache invalidation,
        # neither of which this from-scratch-redraw canvas has or needs).
        x_left, x_right = min(ROOM_X), max(ROOM_X)
        z_bottom, z_top = min(ROOM_Z), max(ROOM_Z)
        x_margin = (x_right - x_left) * 0.05
        z_margin = (z_top - z_bottom) * 0.05
        ax.set_xlim(x_left - x_margin, x_right + x_margin)
        ax.set_ylim(z_bottom - z_margin, z_top + z_margin)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Click to place a device", fontsize=8)
        case_index = self.scenario_combo.currentData()
        selected = self._current()
        for d in self._devices:
            if d.scenario != case_index:
                continue
            active = (bool(d.state_at(self._current_index).get("active"))
                      if d.type != "thermocouple" else False)
            color = _COLOR_ACTIVE if active else _COLOR_IDLE
            ax.plot(d.position[0], d.position[1], "D", color=color, markersize=8,
                    markeredgecolor="black", markeredgewidth=1.0)
            ax.annotate(d.name, d.position, fontsize=7, xytext=(4, 4),
                       textcoords="offset points", color=color)
            if d is selected:
                ax.plot(d.position[0], d.position[1], "o", color="none",
                        markersize=16, markeredgecolor=color, markeredgewidth=1.5)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.03, right=0.97)
        self.canvas.draw_idle()
        d = self._current()
        if d is not None:
            live = self._live_readout(d)
            prefix = f"{d.name}: {live} · " if live else f"{d.name}: "
            self.readout.setText(f"{prefix}{self._headline(d)}  ·  {(d.results or {}).get('basis', '')}")
        else:
            self.readout.setText("")
