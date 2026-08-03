"""Time-Series Workspace (V2 roadmap M1.1, feature F1).

Point/line/region probes over the cached 2D slice arrays, producing XY
plots over time (point value, region mean) or over distance (line
profile at a chosen frame), with multi-scenario overlay and CSV export.
This is the app's first non-heatmap analysis surface: everything here is
a pure in-memory reduction of arrays ScenarioStore already caches -- no
new data paths, no connection to TimeController (same static-panel
convention as analytics_panel.py/forecasting_panel.py).

Pure computation helpers live at module level so they are unit-testable
without Qt; the panel widget only wires them to clicks and combos.
"""

from __future__ import annotations

import csv
import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates

from widgets import MplCanvas, plot_fg_color
from views import EnsemblePickerDialog
from analysis_panel_base import populate_scenario_combo

logger = logging.getLogger(__name__)

MODES = ("point", "line", "region")
MODE_LABELS = {
    "point": "Point probe (value over time)",
    "line": "Line profile (value along a segment)",
    "region": "Region mean (average over time)",
}
MODE_HINTS = {
    "point": "Click the map to place the probe point.",
    "line": "Click the map twice: segment start, then end.",
    "region": "Click the map twice: opposite corners of the region.",
}

LINE_PROFILE_SAMPLES = 100


def phys_to_index(extent: tuple, shape: tuple, x: float, z: float) -> tuple:
    """Physical (x, z) -> nearest (row, col), clipped into bounds.

    Same convention as SliceView.value_at (row 0 = z1/top, matching
    load_data.py's vertical flip + matplotlib origin='upper'), so a click
    on the locator map lands on the same array cell the Live probe
    reports for that physical position.
    """
    x0, x1, z0, z1 = extent
    n_z, n_x = shape
    col = int(round((x - x0) / (x1 - x0) * (n_x - 1)))
    row = int(round((z1 - z) / (z1 - z0) * (n_z - 1)))
    return (min(max(row, 0), n_z - 1), min(max(col, 0), n_x - 1))


def point_series(data: np.ndarray, row: int, col: int) -> np.ndarray:
    """Value at (row, col) for every frame -- shape (n_times,)."""
    return np.asarray(data[:, row, col], dtype=float)


def region_series(data: np.ndarray, row0: int, col0: int, row1: int, col1: int) -> np.ndarray:
    """Mean over the inclusive rectangle for every frame -- shape (n_times,)."""
    r0, r1 = sorted((row0, row1))
    c0, c1 = sorted((col0, col1))
    return np.asarray(data[:, r0:r1 + 1, c0:c1 + 1].mean(axis=(1, 2)), dtype=float)


def line_profile(data: np.ndarray, index: int, row0: float, col0: float,
                  row1: float, col1: float, n_samples: int = LINE_PROFILE_SAMPLES) -> np.ndarray:
    """Bilinearly-sampled values along the segment at frame `index` --
    shape (n_samples,). Distances are the caller's concern (they depend
    on physical extent, which the array doesn't carry)."""
    rows = np.linspace(row0, row1, n_samples)
    cols = np.linspace(col0, col1, n_samples)
    return map_coordinates(np.asarray(data[index], dtype=float), [rows, cols], order=1)


def write_series_csv(path: str, x_label: str, x_values: np.ndarray,
                      series: list, metadata: dict | None = None) -> None:
    """series: [(column_label, values), ...] -- all same length as x_values.

    `metadata` (V6-M2, virtual devices): optional provenance lines written
    as `# key,value` comments before the table -- device type, coordinates,
    parameters -- so an export is traceable on its own. Absent for every
    pre-existing caller (default None), so their files are byte-for-byte
    unchanged."""
    with open(path, "w", newline="") as f:
        if metadata:
            writer = csv.writer(f)
            for key, value in metadata.items():
                writer.writerow([f"# {key}", value])
        writer = csv.writer(f)
        writer.writerow([x_label] + [label for label, _values in series])
        for i, x in enumerate(x_values):
            writer.writerow([f"{x:.6g}"] + [f"{values[i]:.6g}" for _label, values in series])


class TimeSeriesPanel(QtWidgets.QWidget):
    """Analysis-page workspace: a locator map (time-max composite of the
    chosen scenario/quantity, clickable) driving a linked curve plot.

    Lazy by design (the M3.1 startup-regression lesson): nothing touches
    ScenarioStore until ensure_loaded() -- MainWindow calls it from the
    Analysis page's on_enter, never at construction.
    """

    def __init__(self, store, manifest: list, quantity_options: list, fps: int,
                 field_fn=None, extent_fn=None, parent=None):
        """quantity_options: [(label, SliceKey), ...] -- same shape
        MainWindow._quantity_options() already produces for GridCell.

        field_fn/extent_fn (Live-polish follow-up, "better ways to
        visualize dynamic pressure/temperature rise" pass): optional
        (case_index, key) -> ndarray/extent callables -- MainWindow passes
        its own computed-vs-native-aware routing (self._field/
        self._extent_for) so a computed quantity (DYNAMIC PRESSURE,
        TEMPERATURE RISE) in quantity_options actually works here, not just
        native ones. Default to the store directly (this panel's original,
        native-only behavior) so every pre-existing caller/test needs no
        changes."""
        super().__init__(parent)
        self._store = store
        self._field_fn = field_fn or (lambda case_index, key: store.get(case_index, key))
        self._extent_fn = extent_fn or (lambda case_index, key: store.get_extent(case_index, key))
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._quantity_options = list(quantity_options)
        self._fps = max(1, fps)
        self._loaded = False
        self._mode = "point"
        self._probe = None            # mode-dependent: (x, z) / ((x0,z0),(x1,z1))
        self._pending_start = None    # first click of a two-click gesture
        self._overlay_cases: list = []
        self._locator_ax = None
        self._locator_image = None
        self._marker_artists: list = []
        self._last_curves: list = []  # [(label, values)] of the current plot
        self._last_x = None           # (axis label, values) of the current plot
        self._bus = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Time series")
        title.setProperty("role", 
                          "section-title")
        header.addWidget(title)
        header.addStretch(1)

        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Time-series scenario")
        self.scenario_combo.setToolTip("Scenario shown on the locator map (and first overlay curve)")
        header.addWidget(self.scenario_combo)

        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Time-series quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setAccessibleName("Probe mode")
        for mode in MODES:
            self.mode_combo.addItem(MODE_LABELS[mode])
        header.addWidget(self.mode_combo)

        self.overlay_button = QtWidgets.QPushButton("Overlay scenarios…")
        self.overlay_button.setToolTip(
            "Choose additional scenarios to plot at the same probe (spaghetti overlay)")
        self.overlay_button.clicked.connect(self._open_overlay_picker)
        header.addWidget(self.overlay_button)

        self.export_button = QtWidgets.QPushButton("Export CSV…")
        self.export_button.setToolTip("Save the currently plotted curves as a CSV file")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_csv)
        header.addWidget(self.export_button)
        layout.addLayout(header)

        self.hint_label = QtWidgets.QLabel(MODE_HINTS[self._mode])
        self.hint_label.setProperty("role", "caption")
        layout.addWidget(self.hint_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.locator_canvas = MplCanvas(self)
        self.locator_canvas.setAccessibleName("Probe locator map")
        splitter.addWidget(self.locator_canvas)
        self.plot_canvas = MplCanvas(self)
        self.plot_canvas.setAccessibleName("Time-series plot")
        splitter.addWidget(self.plot_canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("Profile frame: t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Line-profile frame")
        self.frame_slider.setToolTip("Which frame the line profile is sampled at")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        self._frame_row_widget = QtWidgets.QWidget()
        self._frame_row_widget.setLayout(frame_row)
        self._frame_row_widget.setVisible(False)
        layout.addWidget(self._frame_row_widget)

        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
        self.quantity_combo.currentIndexChanged.connect(self._on_quantity_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.locator_canvas.mpl_connect("button_press_event", self._on_locator_click)

    # ------------------------------------------------------------ lazy load
    def ensure_loaded(self) -> None:
        """First-use initialization: populate the scenario combo and draw
        the locator map. Idempotent; safe to call on every page enter."""
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._overlay_cases = [self._manifest[0].case_index]
        self._reload_locator()

    @property
    def current_case_index(self) -> int:
        return self.scenario_combo.currentData()

    @property
    def current_key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    def _data(self, case_index: int) -> np.ndarray:
        return np.asarray(self._field_fn(case_index, self.current_key))

    def _extent(self, case_index: int):
        extent = self._extent_fn(case_index, self.current_key)
        if extent is None:
            # Fall back to pixel-index coordinates -- still functional,
            # same degradation SliceView documents for a missing extent.
            data = self._data(case_index)
            extent = (0.0, float(data.shape[2] - 1), 0.0, float(data.shape[1] - 1))
        return extent

    # ------------------------------------------------------------- locator
    def _reload_locator(self) -> None:
        case_index = self.current_case_index
        if case_index is None:
            return
        data = self._data(case_index)
        composite = data.max(axis=0)
        extent = self._extent(case_index)
        fig = self.locator_canvas.fig
        fig.clear()
        self._marker_artists = []
        self._locator_ax = fig.add_subplot(111)
        # RC polish (visualization policy): the composite uses the selected
        # quantity's registry colormap, so the same quantity looks identical here
        # and in the Live Viewer / Analysis, instead of an ad-hoc gist_heat.
        from registry import get_quantity
        self._locator_image = self._locator_ax.imshow(
            composite, cmap=get_quantity(self.current_key.quantity).cmap,
            aspect="auto", extent=extent)
        self._locator_ax.set_title("Time-max composite — click to probe", fontsize=8)
        self._locator_ax.set_xlabel("x (m)", fontsize=8)
        self._locator_ax.set_ylabel("z (m)", fontsize=8)
        self._locator_ax.tick_params(labelsize=7)
        fig.subplots_adjust(top=0.90, bottom=0.18, left=0.14, right=0.97)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, data.shape[0] - 1)
        self.frame_slider.blockSignals(False)
        self._draw_probe_markers()
        self.locator_canvas.draw_idle()

    def _clear_probe(self) -> None:
        self._probe = None
        self._pending_start = None
        self._draw_probe_markers()
        self.locator_canvas.draw_idle()

    def _draw_probe_markers(self) -> None:
        for artist in self._marker_artists:
            artist.remove()
        self._marker_artists = []
        if self._locator_ax is None:
            return
        if self._pending_start is not None:
            x, z = self._pending_start
            self._marker_artists.append(
                self._locator_ax.plot(x, z, "+", color="#00E5FF", markersize=10)[0])
        if self._probe is None:
            return
        if self._mode == "point":
            x, z = self._probe
            self._marker_artists.append(
                self._locator_ax.plot(x, z, "x", color="#00E5FF", markersize=10)[0])
        elif self._mode == "line":
            (x0, z0), (x1, z1) = self._probe
            self._marker_artists.append(
                self._locator_ax.plot([x0, x1], [z0, z1], "-o", color="#00E5FF",
                                       markersize=4, linewidth=1.5)[0])
        elif self._mode == "region":
            (x0, z0), (x1, z1) = self._probe
            patch = Rectangle((min(x0, x1), min(z0, z1)), abs(x1 - x0), abs(z1 - z0),
                               fill=False, edgecolor="#00E5FF", linewidth=1.5)
            self._locator_ax.add_patch(patch)
            self._marker_artists.append(patch)

    def _on_locator_click(self, event) -> None:
        if event.inaxes != self._locator_ax or event.xdata is None or event.ydata is None:
            return
        self._apply_click(float(event.xdata), float(event.ydata))

    def _apply_click(self, x: float, z: float) -> None:
        """One locator click at physical (x, z). Split from the mpl event
        handler so tests can drive the full gesture without synthesizing
        matplotlib mouse events."""
        if self._mode == "point":
            self._probe = (x, z)
        elif self._pending_start is None:
            self._pending_start = (x, z)
            self._draw_probe_markers()
            self.locator_canvas.draw_idle()
            return
        else:
            self._probe = (self._pending_start, (x, z))
            self._pending_start = None
        self._draw_probe_markers()
        self.locator_canvas.draw_idle()
        self._update_plot()
        self._publish_probe()

    # ------------------------------------------------------------- bus (Phase 2)
    def set_bus(self, bus) -> None:
        """Publishes the current probe as Selection.point/region (previously
        unused by any panel) so another panel that reacts to the shared
        selection -- SpaceTimePanel already does, via its own custom
        set_bus -- can follow this probe without the researcher re-picking
        it. One-way (publish only): this panel's own mode/probe state stays
        local, so accepting an external pick isn't required to get the
        cross-panel value."""
        self._bus = bus

    def _publish_probe(self) -> None:
        if self._bus is None or self._probe is None:
            return
        if self._mode == "point":
            self._bus.update(origin=self, point=self._probe)
        elif self._mode == "region":
            (x0, z0), (x1, z1) = self._probe
            self._bus.update(origin=self, region=(min(x0, x1), max(x0, x1),
                                                   min(z0, z1), max(z0, z1)))
        # line mode has no matching Selection field -- not published.

    # ------------------------------------------------------------- controls
    def _on_scenario_changed(self, _idx: int) -> None:
        # The locator scenario is always the first overlay curve; keep any
        # additional picker-chosen overlays.
        case_index = self.current_case_index
        if case_index is not None and case_index not in self._overlay_cases:
            self._overlay_cases = [case_index] + [
                c for c in self._overlay_cases if c != case_index]
        self._reload_locator()
        self._update_plot()

    def _on_quantity_changed(self, _idx: int) -> None:
        # PyQt5 aborts the whole process (qFatal -> abort()) on an unhandled
        # exception inside a connected slot rather than raising a catchable
        # Python error -- a crashed app is worse than a logged one, so this
        # slot (the one directly wired to the quantity combo) must never let
        # an exception escape into Qt's event loop.
        try:
            self._reload_locator()
            key = self.current_key
            if (key is not None and key.quantity == "TEMPERATURE RISE"
                    and self._probe is None and self._pending_start is None):
                # Live-polish follow-up ("better ways to visualize temperature
                # rise"): DeltaT's default view is a fixed-height probe in the
                # doorway (x=0.25 m; z=0.11 m -- this dataset's room is a
                # 0.22 m-tall scaled model, not a real building, so "head
                # height" doesn't apply here; z=0.11 m is the room's own
                # geometric mid-height, ROOM_Z=(0.0, 0.22) in schematic.py, not
                # a human-scale guess), not the full 2D slice -- there, the
                # signal is a thin plume region dwarfed by the rest of the
                # domain. Only auto-placed when nothing has been probed yet, so
                # it never overrides a deliberate existing pick.
                if self._mode != "point":
                    self._mode = "point"
                    self.mode_combo.blockSignals(True)
                    self.mode_combo.setCurrentIndex(MODES.index("point"))
                    self.mode_combo.blockSignals(False)
                    self.hint_label.setText(MODE_HINTS[self._mode])
                    self._frame_row_widget.setVisible(False)
                self._apply_click(0.25, 0.11)
                return
            self._update_plot()
        except Exception:
            logger.exception("time series: failed to switch to quantity %r",
                              self.quantity_combo.currentText())
            self._show_plot_placeholder("Could not load this quantity here -- see logs.")

    def _on_mode_changed(self, idx: int) -> None:
        self._mode = MODES[idx]
        self.hint_label.setText(MODE_HINTS[self._mode])
        self._frame_row_widget.setVisible(self._mode == "line")
        self._clear_probe()
        self._show_plot_placeholder(MODE_HINTS[self._mode])

    def _on_frame_slider_changed(self, value: int) -> None:
        self.frame_label.setText(f"Profile frame: t = {value / self._fps:.1f} s")
        if self._mode == "line" and self._probe is not None:
            self._update_plot()

    def _open_overlay_picker(self) -> None:
        dialog = EnsemblePickerDialog(self._manifest, self._overlay_cases, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected = dialog.selected_case_indices()
            current = self.current_case_index
            if current is not None and current not in selected:
                selected = [current] + selected
            self._overlay_cases = selected
            self._update_plot()

    # ----------------------------------------------------------------- plot
    def _show_plot_placeholder(self, text: str) -> None:
        fig = self.plot_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, wrap=True, color=plot_fg_color())
        ax.set_xticks([])
        ax.set_yticks([])
        self.plot_canvas.draw_idle()
        self._last_curves = []
        self._last_x = None
        self.export_button.setEnabled(False)

    def _folder_for(self, case_index: int) -> str:
        for entry in self._manifest:
            if entry.case_index == case_index:
                return entry.folder
        return f"scenario {case_index}"

    def _update_plot(self) -> None:
        if self._probe is None or not self._loaded:
            return
        unit_label = self.quantity_combo.currentText()
        curves = []
        x_axis = None
        for case_index in self._overlay_cases:
            data = self._data(case_index)
            extent = self._extent(case_index)
            if self._mode == "point":
                row, col = phys_to_index(extent, data.shape[1:], *self._probe)
                values = point_series(data, row, col)
                x_axis = ("Time (s)", np.arange(data.shape[0]) / self._fps)
            elif self._mode == "region":
                (x0, z0), (x1, z1) = self._probe
                r0, c0 = phys_to_index(extent, data.shape[1:], x0, z0)
                r1, c1 = phys_to_index(extent, data.shape[1:], x1, z1)
                values = region_series(data, r0, c0, r1, c1)
                x_axis = ("Time (s)", np.arange(data.shape[0]) / self._fps)
            else:  # line
                (x0, z0), (x1, z1) = self._probe
                r0, c0 = phys_to_index(extent, data.shape[1:], x0, z0)
                r1, c1 = phys_to_index(extent, data.shape[1:], x1, z1)
                index = min(self.frame_slider.value(), data.shape[0] - 1)
                values = line_profile(data, index, r0, c0, r1, c1)
                length = float(np.hypot(x1 - x0, z1 - z0))
                x_axis = ("Distance along segment (m)",
                           np.linspace(0.0, length, len(values)))
            curves.append((self._folder_for(case_index), values))

        fig = self.plot_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        for label, values in curves:
            ax.plot(x_axis[1], values, label=label, linewidth=1.2)
        ax.set_xlabel(x_axis[0], fontsize=8)
        ax.set_ylabel(unit_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        if self._mode == "point":
            x, z = self._probe
            ax.set_title(f"Point ({x:.2f}, {z:.2f}) m", fontsize=8)
        elif self._mode == "region":
            ax.set_title("Region mean", fontsize=8)
        else:
            ax.set_title(f"Line profile at t = {self.frame_slider.value() / self._fps:.1f} s",
                          fontsize=8)
        fig.subplots_adjust(top=0.90, bottom=0.18, left=0.12, right=0.97)
        self.plot_canvas.draw_idle()

        self._last_curves = curves
        self._last_x = x_axis
        self.export_button.setEnabled(True)

    # ------------------------------------------------------------------ csv
    def _export_csv(self) -> None:
        if not self._last_curves:
            return
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export curves as CSV", "fds_timeseries.csv", "CSV files (*.csv)")
        if not path:
            return
        self.export_csv_to(path)

    def export_csv_to(self, path: str) -> None:
        """Writes the currently plotted curves; split from the dialog so
        it is directly testable (and scriptable)."""
        if not self._last_curves or self._last_x is None:
            raise RuntimeError("nothing plotted yet -- place a probe first")
        write_series_csv(path, self._last_x[0], self._last_x[1], self._last_curves)
