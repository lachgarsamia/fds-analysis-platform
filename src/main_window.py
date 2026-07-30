"""
main_window.py
---------------
The rebuilt main window. Key differences from the original fixed-.ui version:

1. Layout is built entirely in code with QSplitter/QVBoxLayout/QGridLayout
   and real size policies - no fixed-geometry `.ui` file (main_window.ui is
   left on disk but unused by this entry point), no `showFullScreen()`
   forced at startup.
2. The window is genuinely resizable from a defined minimum size up to
   large/multi-monitor displays; splitter position and window geometry are
   restored via QSettings between sessions.
3. Styling comes from theme.py's token system (light/dark, runtime switch),
   not a single hardcoded stylesheet.
4. Every interactive control has an accessible name/description, a logical
   tab order, and a visible focus ring (via QSS `:focus`).
5. View code only calls into SimulationController - it does not manage
   threads or scenario state directly, and it fetches frames through
   ScenarioStore.get() rather than indexing a single preloaded array.
"""

import os
import logging

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from config import DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC, QUANTITY_DISPLAY, ISOTHERM_LEVELS
from theme import THEMES, apply_card_shadow, build_qss
from widgets import ToggleGroup, CollapsibleSection, TimelineWidget
from simulation_controller import SimulationController
from time_controller import TimeController
from data_provider import SimulationData, load_study, DataLoadError
from schematic import SchematicWidget, resolve_room_extent, _VOD_STATES, _VOC_STATES, room_overlay_geometry
from controls.candle_card import CandleCard
from controls.door_widget import DoorWidget
from controls.vent_widget import VentWidget
from inspector import InspectorPanel, InspectorStack
from export import AnimationExporter, ffmpeg_available
from load_data import SIM_ROOT
from slice_key import (SliceInfo, SliceKey, DEFAULT_SLICE_KEY, available_slices,
                        SOOT_QUANTITY, AXIS_TO_DIRECTION)
from views import ViewGrid, DifferenceView, EnsembleView
from summary_stats import build_summary_index, _read_hrr_csv
from descriptors import compute_descriptors
from events import detect_events
from browser import ExperimentBrowserDock, _SummaryTextWorker
from analytics_panel import AnalyticsPanelDock, _AnalyticsFeatureWorker
from auto_summary import export_markdown, generate_summary
from prediction_store import PredictionSource
from forecasting_panel import ForecastingPanel
from timeseries import TimeSeriesPanel
from energy_panel import EnergyBudgetPanel
from factor_effects_panel import FactorEffectsPanel
from tenability_panel import TenabilityPanel
from fire_mri_panel import FireMRIPanel
from figure_export import (PublicationExportDialog, export_publication_figure,
                           provenance_line, figure_png_bytes)
from report_builder import (build_scenario_report, build_comparison_report,
                            build_session_report, write_report)
from semantic_diff import compare as compare_scenarios, difference_statements
from query_panel import QueryPanel
from state_space_panel import StateSpacePanel
from attention_panel import AttentionPanel
from cause_panel import CausePanel
from height_panel import HeightPanel
from zone_panel import ZonePanel
from time_window_panel import TimeWindowPanel
from measurement_panel import MeasurementPanel
from advanced_compare_panel import AdvancedComparePanel
from probe_measure_panel import ProbeMeasurePanel
from spatiotemporal_panel import SpatiotemporalPanel
import manifest as manifest_mod
from quantities_panel import QuantitiesPanel
from assistant_panel import AssistantPanel
from assistant_query_panel import AssistantQueryPanel
import assistant as assistant_mod
from selection import Selection, SelectionBus
from quantity_provider import QuantityProvider
from analysis_panel_base import bind_to_bus
from study_panel import StudyPanel
from parallel_coordinates_panel import ParallelCoordinatesPanel
from compare_discover_panel import CompareDiscoverPanel
from sensitivity_panel import SensitivityPanel
from hazard_panel import HazardPanel
from hazard_tenability_panel import HazardTenabilityPanel
from dashboard_panel import DashboardPanel
from spacetime_panel import SpaceTimePanel
from narrative_panel import NarrativePanel
from ensemble_panel import EnsemblePanel
from graph_panel import GraphPanel
from history import InvestigationHistory
import field_calculator as field_calculator_mod
from device_panel import DevicePanel
from velocity_panel import VelocityPanel
from figure_export import save_figure
from report_builder import build_publication_manifest
import study_analytics as study_analytics_mod
import session_store
from evidence_notebook_panel import EvidenceNotebookDock
from evidence_notebook import EvidenceNotebook
from diff_analysis import DifferenceOverTimeDialog
from session import build_session_dict, read_session, write_session
from nav import NavRail
from pages.live import LivePage
from pages.home import HomePage
from pages.compare import ComparePage
from pages.dataset import DatasetPage
from pages.analysis import AnalysisPage
from pages.export_page import ExportPage
from pages.about import AboutPage
from kiosk import KioskController

logger = logging.getLogger(__name__)

ORG_NAME = "FZJuelich"
APP_NAME = "FDSSLCFVisualizer"

# Compare page story presets (FireLab roadmap Phase 4, pages/compare.py):
# each key resolves to two scenarios differing in exactly one factor
# (others held at config.py's own DEFAULT_* values) and the quantity that
# best shows the effect. "door" uses VELOCITY, not TEMPERATURE, per M2.3's
# verified finding that the door-width effect shows up in airflow, not heat.
_COMPARE_PRESETS = {
    "door": {"factor": "door", "values": (1, 0), "quantity": "VELOCITY"},
    "candles": {"factor": "candles", "values": (0, 1), "quantity": "TEMPERATURE"},
    "ventilation": {"factor": "vod", "values": (0, 1), "quantity": "TEMPERATURE"},
}

# Colormap options. Default (gist_heat) confirmed by the M1.3s validation
# spike (docs/spike-parser-validation.md): already a black-red-orange-yellow-
# white blackbody/flame progression, kept as-is rather than replaced.
COLORMAPS = [
    ("Fire (calibrated, default)", "fds_fire"),
    ("Flow (calibrated, default for air speed)", "fds_flow"),
    ("Heat (gist_heat)", "gist_heat"),
    ("Inferno", "inferno"),
    ("Viridis (colorblind-safe)", "viridis"),
    ("Cividis (colorblind-safe)", "cividis"),
]

# Bilinear is the default (GUI modernization pass, item 7) -- matplotlib's
# imshow default of "nearest" is genuinely blocky at this grid's native
# 49x101 resolution stretched to fill a much larger on-screen cell.
# Bicubic was considered (matplotlib supports it too) but rejected as the
# default: for a scientific tool, cubic interpolation's mild ringing/
# overshoot near sharp gradients can visually suggest values the
# underlying simulation never produced, which bilinear doesn't do.
# "Nearest" stays available for anyone who specifically wants to see raw
# cell boundaries (e.g. to sanity-check against the actual mesh).
INTERPOLATIONS = [
    ("Bilinear", "bilinear"),
    ("Nearest", "nearest"),
]

MIN_WIDTH = 900
MIN_HEIGHT = 600

# M2.5: windows opened via "Open Study…" are kept referenced here for the
# process's lifetime -- a shown top-level QWidget with no Python reference
# can be garbage-collected out from under Qt and crash the app.
_OPEN_STUDY_WINDOWS = []


def _detect_system_theme() -> str:
    """The OS appearance as 'light' or 'dark' (RC polish, "Follow System").
    macOS reports Dark via AppleInterfaceStyle; elsewhere we default to light."""
    import subprocess
    import sys
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                                 capture_output=True, text=True, timeout=1)
            return "dark" if "dark" in out.stdout.lower() else "light"
        except Exception:
            return "light"
    return "light"


# Inspector "Slice" field (scientific-visualization completion pass):
# SliceKey.direction is 0-indexed per slice_key.py's own convention
# (direction=1 is documented there as "normal to y").
_AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


class ExportRangeDialog(QtWidgets.QDialog):
    """Lets the user pick fps and a frame range before exporting (M1.5.1).
    Defaults to the full scenario at the app's own playback fps, so
    clicking OK immediately exports everything -- the dialog exists for
    the "chosen fps/range" spec requirement, not to force a decision."""

    def __init__(self, parent, n_frames: int, default_fps: int):
        super().__init__(parent)
        self.setWindowTitle("Export Animation")

        layout = QtWidgets.QFormLayout(self)

        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(default_fps)
        self.fps_spin.setAccessibleName("Export frame rate")
        layout.addRow("Frames per second:", self.fps_spin)

        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, max(n_frames - 1, 0))
        self.start_spin.setValue(0)
        self.start_spin.setAccessibleName("Export start frame")
        layout.addRow("Start frame:", self.start_spin)

        self.end_spin = QtWidgets.QSpinBox()
        self.end_spin.setRange(1, n_frames)
        self.end_spin.setValue(n_frames)
        self.end_spin.setAccessibleName("Export end frame (exclusive)")
        layout.addRow("End frame:", self.end_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self):
        start = self.start_spin.value()
        end = max(self.end_spin.value(), start + 1)
        return start, end, self.fps_spin.value()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sim_data: SimulationData):
        super().__init__()
        self.settings = QtCore.QSettings(ORG_NAME, APP_NAME)
        self.sim_data = sim_data
        # M2.5: a generic/degenerate guest study (opened via "Open Study…")
        # has no candle/door/vent factor axes, so its scenario-parameter
        # controls, schematic, and Compare/analytics surfaces are hidden.
        self.is_factorial = getattr(sim_data, 'is_factorial', True)

        self.controller = SimulationController(
            sim_data.store, sim_data.data_matrix, sim_data.timesteps_per_second
        )
        # Match the controller's initial parameters to the toggle defaults
        # declared in config.py, so the first frame drawn matches what the
        # control panel shows as selected (candles/door/vod/voc all start
        # at their DEFAULT_* value, not implicitly at 0).
        # A generic guest study (M2.5) has a placeholder (n,1,1,1)
        # data_matrix with no door/vod/voc axes, so the candle-factorial
        # DEFAULT_* params would index out of bounds -- start it at
        # scenario 0 (0,0,0,0) instead. The factor controls are hidden for
        # such a study anyway.
        if self.is_factorial:
            self.controller.params.candles = DEFAULT_CANDLES
            self.controller.params.door = DEFAULT_DOOR
            self.controller.params.vod = DEFAULT_VOD
            self.controller.params.voc = DEFAULT_VOC
        else:
            self.controller.params.candles = 0
            self.controller.params.door = 0
            self.controller.params.vod = 0
            self.controller.params.voc = 0
        self.controller.prefetch_finished.connect(self._on_prefetch_finished)
        self.controller.prefetch_error.connect(self._on_prefetch_error)
        # M2.2.4: a second, independent pair of slots on the same signals,
        # for non-active grid cells' own combo picks -- reuses the exact
        # same prefetch()/worker-list machinery M1.4.4 built rather than
        # any new threading code; see _load_cell/_on_grid_prefetch_finished.
        self.controller.prefetch_finished.connect(self._on_grid_prefetch_finished)
        self.controller.prefetch_error.connect(self._on_grid_prefetch_error)
        self._pending_cell_prefetches = set()  # GridCells awaiting their own background load
        # FireLab roadmap Phase 2.1c: case_index -> (times, hrr_kw) from
        # that scenario's *_hrr.csv, or () if none was found -- read once
        # per scenario, not once per playback tick.
        self._hrr_cache = {}
        # V3-M2: per-scenario detected fire events (events.py), computed once.
        self._fire_events_cache = {}
        # FireLab roadmap Phase 3: (case_index, quantity_key) each visible
        # cell's own Inspector section's sparkline/narration were last built
        # for -- one entry per grid position (V6 polish: extended from a
        # single scalar to a list when the Inspector grew a section per
        # cell), recomputed only when that cell's scenario/quantity
        # actually changes, not once per playback tick.
        self._inspector_series_key = []

        # M3.2.5: trained-model predictions, if ml/train.py + ml/rollout.py
        # have ever been run -- absent/empty otherwise (predictions/ simply
        # doesn't exist), same convention as demo mode's absent experiment
        # browser. Never touches the real store's cache: PredictionSource
        # loads its own small .npy files eagerly and independently.
        # Predictions belong to the candle study (ml/rollout.py exports
        # predictions/*.npy keyed by its case indices); a generic guest
        # study (M2.5) has none, so forecasting is disabled for it.
        self.prediction_store = PredictionSource(
            sim_data.store, enabled=not sim_data.is_demo and self.is_factorial)

        # M1.4: pull-based playback clock. frame_count_fn must never trigger
        # a load itself -- it just reports self._current_n_frames, which is
        # only ever updated once a scenario is confirmed loaded (see
        # _on_scenario_param_changed / _on_prefetch_finished below).
        self._current_n_frames = 0
        self._busy = False               # a cache-miss prefetch is in flight
        self._pending_load_case = None   # which case_index that prefetch is for
        self._pending_load_key = None    # ...and which quantity key (M0.1: the case-vs-key race fix)
        self._was_playing_before_load = False
        self.time_controller = TimeController(
            lambda: self._current_n_frames, sim_data.timesteps_per_second
        )
        self.time_controller.time_changed.connect(self._on_time_changed)
        self.time_controller.playing_changed.connect(self._on_playing_changed)

        self.current_theme_name = self.settings.value("theme", "dark")
        self.ui_scale = float(self.settings.value("ui_scale", 1.0))
        self.current_colormap = self.settings.value("colormap", "gist_heat")
        self.current_interpolation = self.settings.value("interpolation", "bilinear")
        # M2.1: which (quantity, direction, offset) slice the heatmap shows.
        # Set before the control panel/plot are built since both read it.
        self.current_quantity_key = DEFAULT_SLICE_KEY

        self.setWindowTitle("FDS SLCF Fire Visualizer" + (" (demo data)" if sim_data.is_demo else ""))
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self._build_menu()
        # FireLab roadmap Phase 4: experiment_browser/analytics_panel must
        # exist before _build_shell() builds the Dataset/Analysis pages,
        # since those pages embed .widget() -- each dock's own inner
        # content -- rather than duplicating it.
        self._build_experiment_browser()
        self._build_analytics_panel()
        self._build_shell()
        self._build_status_bar()
        self._apply_theme()
        self._restore_window_state()
        self._setup_shortcuts()
        # The very first frame is drawn directly by _init_plot(), not via
        # _on_time_changed (only real ticks/seeks go through that) -- seed
        # the Live Inspector once here so it isn't blank until the user's
        # first interaction.
        self._update_inspector(self.time_controller.index)
        # Populate the Fire story now that the inspector exists (the first
        # marker update during plot-panel build ran before it did, V3-M2).
        self._update_event_markers()
        for i in range(self.inspector_stack.count()):
            self.inspector_stack.section(i).set_story_index(self.time_controller.index)

        # Kiosk / attract mode (FireLab roadmap Phase 5): idle -> Home,
        # any input -> Live. Needs a live QApplication instance, which
        # exists by construction time (main.py creates it before MainWindow).
        self._kiosk = KioskController(
            on_idle=lambda: self._navigate_to("home"),
            on_wake=lambda: self._navigate_to("live"),
            app=QtWidgets.QApplication.instance(),
            cursor_target=self,
            parent=self,
        )
        # Demo-script bookmarks (FireLab roadmap Phase 5): slot -> {page,
        # case_index, time_index}; Ctrl+Shift+<1-9> records, Shift+<1-9> jumps.
        self._demo_bookmarks: dict = {}
        # Guards the Compare -> Home/Live grid-reset bugfix (_navigate_to)
        # from firing on _apply_compare_preset's own internal jump to Live.
        self._applying_compare_preset = False
        # True while the grid is showing a Compare preset's comparison
        # setup -- Compare and the plain Live Viewer are meant to be
        # independent, so _navigate_to resets the grid back to a single
        # view the next time the user actually asks for "live" or "home"
        # (including re-clicking "Live Viewer" while already there).
        self._compare_active = False
        # Esc long-press: "effects off" master switch, tracked via
        # keyPressEvent/keyReleaseEvent below (QShortcut has no notion of
        # hold-duration).
        self._esc_hold_timer = QtCore.QTimer(self)
        self._esc_hold_timer.setSingleShot(True)
        self._esc_hold_timer.timeout.connect(self._toggle_effects_master_switch)

        if sim_data.is_demo:
            self.statusBar().showMessage(
                "Running with generated demo data (fds/sim/ not found).", 8000
            )

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        # V2 roadmap M2.5: open an arbitrary FDS-output directory as a study.
        open_study_action = QtWidgets.QAction("Open Study…", self)
        open_study_action.setToolTip("Open a different FDS output directory (a case or a folder of cases)")
        open_study_action.triggered.connect(self._open_study)
        file_menu.addAction(open_study_action)
        file_menu.addSeparator()

        # V2 roadmap M2.4: session save/restore of the grid workspace.
        save_session_action = QtWidgets.QAction("Save Session…", self)
        save_session_action.setToolTip("Save the current grid layout, scenarios, and playback position")
        save_session_action.triggered.connect(self._save_session)
        file_menu.addAction(save_session_action)
        load_session_action = QtWidgets.QAction("Load Session…", self)
        load_session_action.setToolTip("Restore a previously saved grid workspace")
        load_session_action.triggered.connect(self._load_session)
        file_menu.addAction(load_session_action)

        view_menu = menu_bar.addMenu("&View")

        theme_menu = view_menu.addMenu("Theme")
        self.theme_action_group = QtWidgets.QActionGroup(self)
        for key, label in (("light", "Light"), ("dark", "Dark"),
                           ("system", "Follow System"), ("theatre", "Theatre (demo)")):
            action = QtWidgets.QAction(label, self, checkable=True)
            action.setChecked(key == self.current_theme_name)
            action.triggered.connect(lambda _checked, k=key: self._set_theme(k))
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)

        scale_menu = view_menu.addMenu("UI Scale")
        for scale in (0.85, 1.0, 1.15, 1.3, 1.5):
            action = QtWidgets.QAction(f"{int(scale * 100)}%", self, checkable=True)
            action.setChecked(abs(scale - self.ui_scale) < 1e-6)
            action.triggered.connect(lambda _checked, s=scale: self._set_ui_scale(s))
            scale_menu.addAction(action)
        view_menu.addSeparator()

        colormap_menu = view_menu.addMenu("Colormap")
        self.colormap_action_group = QtWidgets.QActionGroup(self)
        for i, (label, cmap) in enumerate(COLORMAPS):
            if i == 2:  # after the two calibrated defaults, before the stock options
                colormap_menu.addSeparator()
            action = QtWidgets.QAction(label, self, checkable=True)
            action.setChecked(cmap == self.current_colormap)
            action.triggered.connect(lambda _checked, c=cmap: self._set_colormap(c))
            self.colormap_action_group.addAction(action)
            colormap_menu.addAction(action)
        view_menu.addSeparator()

        interpolation_menu = view_menu.addMenu("Interpolation")
        self.interpolation_action_group = QtWidgets.QActionGroup(self)
        for label, interp in INTERPOLATIONS:
            action = QtWidgets.QAction(label, self, checkable=True)
            action.setChecked(interp == self.current_interpolation)
            action.triggered.connect(lambda _checked, i=interp: self._set_interpolation(i))
            self.interpolation_action_group.addAction(action)
            interpolation_menu.addAction(action)

        grid_menu = view_menu.addMenu("Grid Layout")
        self.grid_layout_action_group = QtWidgets.QActionGroup(self)
        for label, layout_name in (("1 view", "1x1"), ("2 views (side by side)", "1x2"),
                                   ("2 views (stacked)", "2x1"),
                                   ("4 views (2x2)", "2x2"), ("9 views (3x3)", "3x3")):
            action = QtWidgets.QAction(label, self, checkable=True)
            action.setChecked(layout_name == "1x1")
            action.triggered.connect(lambda _checked, l=layout_name: self._set_grid_layout(l))
            self.grid_layout_action_group.addAction(action)
            grid_menu.addAction(action)
        view_menu.addSeparator()

        self.link_clim_action = QtWidgets.QAction("Link color scales", self, checkable=True)
        self.link_clim_action.setToolTip(
            "When on, all visible cells share one color scale (the maximum "
            "across them) instead of each keeping its own."
        )
        self.link_clim_action.triggered.connect(self._set_link_clim)
        view_menu.addAction(self.link_clim_action)

        self.isotherms_action = QtWidgets.QAction("Contour overlay", self, checkable=True)
        self.isotherms_action.setToolTip(
            "Draw contour lines at fixed reference levels on every visible "
            "cell -- hazard-band thresholds (60/100/300 °C) for Temperature, "
            "speed bands (1/2/3 m/s) for Air speed. Redraws each frame "
            "instead of blitting while on, so playback is slightly heavier "
            "with this enabled."
        )
        self.isotherms_action.triggered.connect(self._set_isotherms_enabled)
        view_menu.addAction(self.isotherms_action)

        self.velocity_overlay_action = QtWidgets.QAction("Show velocity overlay", self, checkable=True)
        self.velocity_overlay_action.setToolTip(
            "Draw VELOCITY speed-band contours (dashed, blue) on top of any "
            "cell currently showing Temperature -- confirmed VELOCITY shares "
            "the exact same physical plane, so the overlay always lines up. "
            "Off by default; a plain temperature view stays available."
        )
        self.velocity_overlay_action.triggered.connect(self._set_velocity_overlay_enabled)
        view_menu.addAction(self.velocity_overlay_action)
        view_menu.addSeparator()

        self.cinematic_action = QtWidgets.QAction("Cinematic fire view", self, checkable=True)
        self.cinematic_action.setToolTip(
            "Render Temperature cells through FireLab's cinematic pipeline "
            "(black-body glow with alpha, filmic tone mapping, auto-exposure) "
            "over a dark backdrop instead of the plain scientific colormap. "
            "Other quantities/cell types are unaffected."
        )
        self.cinematic_action.triggered.connect(self._set_cinematic_enabled)
        view_menu.addAction(self.cinematic_action)
        view_menu.addSeparator()

        # Evidence Notebook (V4-M2): dock is created after the menu, so the
        # action just toggles it and stays in sync via visibilityChanged.
        self.evidence_notebook_action = QtWidgets.QAction("Evidence Notebook", self, checkable=True)
        self.evidence_notebook_action.setToolTip(
            "Show the Evidence Notebook: the saved, annotatable measurements "
            "for this study. Right-click any finding and choose \"Save to "
            "Evidence Notebook\" to collect it here.")
        self.evidence_notebook_action.triggered.connect(
            lambda checked: self.evidence_dock.setVisible(checked))
        view_menu.addAction(self.evidence_notebook_action)
        view_menu.addSeparator()

        fullscreen_action = QtWidgets.QAction("Toggle Fullscreen\tF11", self)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        export_menu = menu_bar.addMenu("&Export")
        export_animation_action = QtWidgets.QAction("Animation (MP4/GIF)…", self)
        export_animation_action.setToolTip("Export the current scenario's playback as a video or GIF")
        export_animation_action.triggered.connect(self._export_animation)
        export_menu.addAction(export_animation_action)

        export_figure_action = QtWidgets.QAction("Publication figure (SVG/PDF/PNG)…", self)
        export_figure_action.setToolTip(
            "Export the active cell's current frame as a vector or high-DPI figure")
        export_figure_action.triggered.connect(self._export_publication_figure)
        export_menu.addAction(export_figure_action)

        bundle_action = QtWidgets.QAction("Prepare Publication Bundle…", self)
        bundle_action.setToolTip(
            "Export journal-styled figures from the loaded analysis panels plus a "
            "manifest of captions, metadata, and the Evidence Notebook's findings")
        bundle_action.triggered.connect(self._prepare_publication_bundle)
        export_menu.addAction(bundle_action)

    # Panels whose primary plot is worth including in a publication bundle
    # (attr, canvas attr, figure name, caption).
    _BUNDLE_FIGURES = [
        ("height_panel", "plot_canvas", "height_profile", "Vertical temperature profile and layer/plume/ceiling over time."),
        ("zone_panel", "plot_canvas", "zone_stats", "Named-zone temperature and thermal-dose over time."),
        ("hazard_panel", "timeline_canvas", "hazard_timeline", "Hazard-class fractions over time (temperature-only partial screen)."),
        ("study_panel", "parallel_canvas", "study_parallel", "Parameter-vs-response parallel coordinates across the factorial."),
        ("sensitivity_panel", "surface_canvas", "sensitivity_surface", "Estimated response surface (interpolated from existing scenarios)."),
        ("ensemble_panel", "canvas", "ensemble_spread", "Parametric ensemble spread across the existing scenarios."),
    ]

    def _prepare_publication_bundle(self) -> None:
        """One-click publication bundle (V5-M5): journal-styled figures from
        the loaded analysis panels + a manifest of captions, metadata, and the
        Evidence Notebook's findings."""
        if not self.sim_data.manifest:
            QtWidgets.QMessageBox.information(self, "Publication bundle",
                                              "No experiment data available (demo mode).")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose an output folder for the publication bundle")
        if not directory:
            return
        figures = []
        for attr, canvas_attr, name, caption in self._BUNDLE_FIGURES:
            panel = getattr(self, attr, None)
            canvas = getattr(panel, canvas_attr, None) if panel is not None else None
            if canvas is None or not getattr(panel, "_loaded", False):
                continue
            path = os.path.join(directory, name + ".png")
            try:
                save_figure(canvas.fig, path, "Journal — single column")
                figures.append((name + ".png", caption))
            except OSError:
                continue
        sel = self.selection_bus.current if getattr(self, "selection_bus", None) else None
        metadata = {
            "Data run": session_store.data_fingerprint(self.sim_data.manifest),
            "Scenarios": str(len(self.sim_data.manifest)),
            "Prepared": session_store.now_iso(),
            "Selected scenario": (str(sel.scenario) if sel and sel.scenario is not None else ""),
        }
        manifest = build_publication_manifest(
            figures, self.evidence_dock.notebook.to_list(), metadata)
        try:
            write_report(os.path.join(directory, "manifest.html"), manifest)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Publication bundle", f"Could not write: {e}")
            return
        self.statusBar().showMessage(
            f"Publication bundle: {len(figures)} figure(s) + manifest in {directory}", 5000)

    def _build_experiment_browser(self):
        """M2.5: docked sortable/filterable index of all real scenarios.

        Demo mode has no manifest/HRR CSVs, so the browser is simply absent
        there rather than showing synthetic rows that cannot satisfy the
        milestone's "all 24 scenarios" requirement.

        Auto-summary text (M3.1.3, auto_summary.generate_all_summaries())
        is NOT computed here anymore. It used to be, synchronously -- a
        second, independent instance of the same "eager full-ensemble
        store.get() in MainWindow.__init__ before show()" mistake as
        3fd87cf's analytics panel (see _build_analytics_panel's docstring),
        introduced later by M3.1.3's auto-summary wiring: generate_all_
        summaries() calls store.get() per scenario with no cache of its
        own (unlike build_summary_index just above, which hits
        summaries.json and skips the store entirely when fresh). Nothing
        visible at startup actually needs it -- the table is built from
        `summaries`, not summary_texts, and ExperimentBrowserDock already
        treats a missing summary_texts as a legitimate "not computed yet"
        state (only the per-row summary label and the export button's
        enabled state depend on it, neither shown before a user acts).

        So this is fully deferred, like the analytics panel, rather than
        started eagerly here on a background thread: ExperimentBrowserDock
        emits summary_texts_needed the first time a row is actually
        selected (see _on_analytics-equivalent handler below), and that's
        what starts _SummaryTextWorker. Starting it unconditionally at
        construction -- even backgrounded -- was tried first and rejected:
        with a warm disk cache (scenario .npy files already written by
        ScenarioStore's own cache_dir, e.g. from a prior session), the
        background thread can finish before the GUI thread even reaches
        show(), reproducing the exact same "touched every scenario before
        anyone asked" problem on a different thread. Deferring to first
        selection sidesteps that regardless of disk-cache state.
        """
        if not self.sim_data.manifest:
            self.experiment_browser = None
            return
        cache_path = os.path.join(SIM_ROOT, ".cache", "summaries.json")
        summaries = build_summary_index(
            self.sim_data.manifest,
            self.controller.store,
            self.sim_data.timesteps_per_second,
            cache_path,
        )
        self._scenario_summaries = summaries  # reused by export_summaries_requested below
        self.experiment_browser = ExperimentBrowserDock(
            summaries, None, self, has_predictions=self.prediction_store.is_available,
        )
        self.experiment_browser.scenario_activated.connect(self._open_browser_scenario)
        self.experiment_browser.open_grid_requested.connect(self._open_browser_grid)
        self.experiment_browser.open_ensemble_requested.connect(self._open_browser_ensemble)
        self.experiment_browser.export_summaries_requested.connect(self._export_summaries_markdown)
        self.experiment_browser.export_report_requested.connect(self._export_report)
        if self.experiment_browser.open_model_eval_button is not None:
            self.experiment_browser.open_model_eval_requested.connect(self._open_browser_model_eval)
        # FireLab roadmap Phase 4: re-hosted as the Dataset page's content
        # (see pages/dataset.py) instead of a QDockWidget -- never added to
        # a dock area. The dock class itself is unchanged (still a
        # QDockWidget internally; only .widget() -- its own inner content
        # -- gets mounted elsewhere), which is why every signal/attribute
        # wired below still works unmodified. Row-selection is what
        # triggers the deferred summary-text load (see summary_texts_needed
        # below), not dock visibility, so no extra on-enter wiring is
        # needed here the way the analytics panel below needs.
        self._summary_texts_loaded = False
        self._summary_text_workers: list = []  # kept alive until each worker's own finished signal fires, same reasoning as SimulationController._prefetch_workers
        self.experiment_browser.summary_texts_needed.connect(self._on_summary_texts_needed)

    def _on_summary_texts_needed(self):
        """One-shot: ExperimentBrowserDock.summary_texts_needed fires the
        first time a user selects a row while summary_texts is still
        empty. Starts the background load; later selections before it
        finishes are no-ops (the dock's own _summary_texts_requested
        guard already prevents re-emitting)."""
        if self._summary_texts_loaded:
            return
        self._summary_texts_loaded = True
        worker = _SummaryTextWorker(
            self.sim_data.manifest, self._scenario_summaries, self.controller.store,
            self.sim_data.timesteps_per_second,
        )
        self._summary_text_workers.append(worker)
        worker.finished_ok.connect(self._on_summary_texts_ready)
        worker.error.connect(self._on_summary_texts_error)
        worker.finished.connect(lambda w=worker: self._cleanup_summary_text_worker(w))
        worker.start()

    def _cleanup_summary_text_worker(self, worker):
        if worker in self._summary_text_workers:
            self._summary_text_workers.remove(worker)

    def _on_summary_texts_ready(self, summary_texts):
        if self.experiment_browser is not None:
            self.experiment_browser.set_summary_texts(summary_texts)

    def _on_summary_texts_error(self, message):
        logger.error(message)

    def _build_analytics_panel(self):
        """M3.1.2: docked PCA scatter + clustering over the ensemble's
        feature vectors. Same demo-mode absence as the experiment browser
        (no manifest, nothing to analyze across 24 real scenarios) and
        tabbed with it rather than stacked -- both are "pick a scenario or
        two to study" tools competing for the same side-panel space, and
        tabbing keeps either one a click away without permanently eating
        screen real estate for both at once.

        The feature index itself is NOT computed here. It used to be
        (3fd87cf), synchronously, which meant every one of the 24 real
        scenarios got pulled through ScenarioStore.get() -- and cached --
        before the window was even shown: a startup-latency regression,
        and (found via git bisect) the reason
        test_stale_prefetch_error_does_not_discard_newer_success and
        test_simultaneous_scenario_and_quantity_switch_cache_miss_race
        started failing, since both depend on a specific scenario starting
        uncached. Instead: the dock is built with an empty placeholder, and
        the real feature index is computed on a background thread
        (_AnalyticsFeatureWorker) only once the panel is actually shown
        (its visibilityChanged(True) fires) -- so a user who never opens
        this tab never touches the store for it at all, and a user who
        does open it doesn't block the GUI thread while all 24 scenarios
        load."""
        # M2.5: PCA/clustering is over the candle factorial's feature
        # space (it aligns clusters with candle count etc.) and needs many
        # scenarios; a generic guest study has neither, so no analytics
        # panel is built for it -- the Analysis page then shows only its
        # time-series/energy/forecasting tabs.
        if not self.sim_data.manifest or not self.is_factorial:
            self.analytics_panel = None
            return
        self.analytics_panel = AnalyticsPanelDock(parent=self)
        self.analytics_panel.scenario_activated.connect(self._open_browser_scenario)
        # FireLab roadmap Phase 4: re-hosted as the Analysis page's content
        # (see pages/analysis.py) instead of a QDockWidget/tab -- never
        # added to a dock area, never tabified. The one-shot feature-index
        # load used to be triggered by the dock's own visibilityChanged
        # (tab raised); that trigger doesn't exist for a plain page, so
        # AnalysisPage.on_enter() calls _on_analytics_panel_visibility_changed(True)
        # directly instead -- same guarded, one-shot method, new caller.
        self._analytics_features_loaded = False
        self._analytics_workers: list = []  # kept alive until each worker's own finished signal fires, same reasoning as SimulationController._prefetch_workers

    def _on_analytics_panel_visibility_changed(self, visible: bool):
        """One-shot trigger, called from AnalysisPage.on_enter() the first
        time the Analysis page is actually shown (not at construction).
        Later calls (revisiting the page) are no-ops once loaded. Also a
        no-op in demo mode (self.analytics_panel is None, and the
        _analytics_features_loaded/_analytics_workers attrs below are
        never set in that branch of _build_analytics_panel)."""
        if self.analytics_panel is None or not visible or self._analytics_features_loaded:
            return
        self._analytics_features_loaded = True
        worker = _AnalyticsFeatureWorker(
            self.sim_data.manifest, self.controller.store, self.sim_data.timesteps_per_second,
        )
        self._analytics_workers.append(worker)
        worker.finished_ok.connect(self._on_analytics_features_ready)
        worker.error.connect(self._on_analytics_features_error)
        worker.finished.connect(lambda w=worker: self._cleanup_analytics_worker(w))
        worker.start()

    def _cleanup_analytics_worker(self, worker):
        if worker in self._analytics_workers:
            self._analytics_workers.remove(worker)

    def _on_analytics_features_ready(self, features):
        if self.analytics_panel is not None:
            self.analytics_panel.load_features(features)

    def _on_analytics_features_error(self, message):
        logger.error(message)
        if self.analytics_panel is not None:
            self.analytics_panel.status_label.setText(message)

    def _build_shell(self):
        """Nav rail + page host (FireLab roadmap Phase 1, pages given real
        content in Phase 4): replaces the old single setCentralWidget(central)
        call with a NavRail alongside a QStackedWidget of pages. LivePage
        wraps _build_central_widget()'s existing content unchanged -- built
        eagerly, right here, at the exact same point in __init__ as before
        this shell existed, so every attribute it sets (self.view_grid,
        self.timeline, self.temp_slider, ...) is available immediately, not
        deferred behind a page switch. Dataset/Analysis embed the
        experiment_browser/analytics_panel docks' own inner content
        (already built -- see __init__'s ordering comment); the Simulation
        Viewer (LivePage) is the page shown first, per the UI/UX
        modernization spec -- Home remains reachable from the nav rail."""
        live_content = self._build_central_widget()

        # Time-Series Workspace (V2 roadmap M1.1): real-data only (needs a
        # manifest for scenario identity), lazy-loaded via the Analysis
        # page's on_enter -- never touches the store at construction.
        if self.sim_data.manifest:
            # QuantityProvider (V5-M1 Layer 1) — created here so the Field
            # Calculator panel (V6-M1) can share it; _build_selection reuses it.
            self.quantity_provider = QuantityProvider(
                self.controller.store, fps=self.sim_data.timesteps_per_second)
            # V6-M1: restore any calculated fields before the panels read the
            # registry (a fresh window starts with none). CalculatorPanel
            # (the only UI to author a *new* calculated field) was removed
            # (Analysis UX + reliability pass) -- field_calculator.py itself
            # stays: quantity_provider.py calls it directly for every
            # calculated-field read, and session save/restore round-trips
            # CalculatedField here independent of any panel.
            field_calculator_mod.clear()
            # Virtual Device Network (V6-M2): thermocouples/detectors/
            # sprinklers placed at a point, computed once and cached.
            self.device_panel = DevicePanel(
                self.quantity_provider, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            # True velocity (V6-M3): gated on U/W-VELOCITY -- the panel
            # itself shows the gate reason plainly until the M-SIM re-run
            # provides them; see velocity.py/velocity_panel.py.
            self.velocity_panel = VelocityPanel(
                self.quantity_provider, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            self.timeseries_panel = TimeSeriesPanel(
                self.controller.store, self.sim_data.manifest,
                self._quantity_options(), self.sim_data.timesteps_per_second)
            self.energy_panel = EnergyBudgetPanel(self.sim_data.manifest)
            # Tenability screening (M3.2; full FED V6-M6): works for any
            # study with a manifest, factorial or not (it's per-scenario, no
            # factor axes). Takes the provider (not the raw store) so a CO
            # read cleanly gates instead of touching the store.
            self.tenability_panel = TenabilityPanel(
                self.quantity_provider, self.sim_data.manifest, self.sim_data.timesteps_per_second)
            # Fire MRI (V3-M1): per-scenario temporal signature maps.
            self.fire_mri_panel = FireMRIPanel(
                self.controller.store, self.sim_data.manifest,
                self._quantity_options(), self.sim_data.timesteps_per_second)
            # Physics query engine (V3-M4).
            self.query_panel = QueryPanel(
                self.controller.store, self.sim_data.manifest, self.sim_data.timesteps_per_second)
            # State space + Fire Genome (V3-M5). Summaries (already computed
            # for the browser) supply the genome's energy trait.
            self.state_space_panel = StateSpacePanel(
                self.controller.store, self.sim_data.manifest, self.sim_data.timesteps_per_second,
                summaries=getattr(self, "_scenario_summaries", None))
            # Physics attention map (V3-M6): heuristic saliency, per frame.
            self.attention_panel = AttentionPanel(
                self.controller.store, self.sim_data.manifest, self.sim_data.timesteps_per_second)
            # Cause explorer (V3-M7, gated): why-is-it-hot gradient tracing.
            self.cause_panel = CausePanel(
                self.controller.store, self.sim_data.manifest, self.sim_data.timesteps_per_second)
            # Height-aware analysis workspace (V4-M1): vertical profiles,
            # smoke layer, plume, ceiling jet.
            self.height_panel = HeightPanel(
                self.controller.store, self.sim_data.manifest,
                self._quantity_options(), self.sim_data.timesteps_per_second)
            # Named region / zone statistics (V4-M4): persistent zones with
            # a full stats bundle, compared across scenarios, session-saved.
            self.zone_panel = ZonePanel(
                self.controller.store, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            # Time-window / interval analysis (V4-M5): time as a selectable
            # dimension -- interval stats, before/after, detected phases.
            self.time_window_panel = TimeWindowPanel(
                self.controller.store, self.sim_data.manifest,
                self._quantity_options(), self.sim_data.timesteps_per_second)
            # Spatiotemporal Analysis (Analysis section consolidation Phase
            # 6): a thin QTabWidget wrapper over three pre-existing "how does
            # a quantity evolve across time and/or space?" tools -- vertical
            # profile, point/region/line probe, and whole-field/interval.
            # Every child's own construction/store access/lazy-load/bus
            # wiring is unchanged; only the tab-level presentation is
            # consolidated. Space-time is deliberately excluded (structurally
            # different mechanism, see spatiotemporal_panel.py's docstring).
            self.spatiotemporal_panel = SpatiotemporalPanel(
                height=self.height_panel, timeseries=self.timeseries_panel,
                time_window=self.time_window_panel)
            # Measurement tools (V4-M7; UX consolidation pass reduced the
            # tool set to rectangle/probe): disposable, un-named area/point
            # reads on the field; measurements persist with the session.
            self.measurement_panel = MeasurementPanel(
                self.controller.store, self.sim_data.manifest,
                self._quantity_options(), self.sim_data.timesteps_per_second)
            # Probe & Measure (Analysis section consolidation Phase 4): a
            # thin QTabWidget wrapper over four pre-existing "what happens
            # at this location/region?" tools -- devices, zones, velocity,
            # and quick (disposable) measurement. Every child's own
            # construction/store access/lazy-load/bus wiring is unchanged;
            # only the tab-level presentation is consolidated.
            self.probe_measure_panel = ProbeMeasurePanel(
                devices=self.device_panel, zones=self.zone_panel,
                velocity=self.velocity_panel, measure=self.measurement_panel)
            # Advanced comparison (V4-M8): temporal / spatial / physics axes.
            # Needs two scenarios to compare (like the semantic diff).
            self.advanced_compare_panel = (
                AdvancedComparePanel(
                    self.controller.store, self.sim_data.manifest,
                    self._quantity_options(), self.sim_data.timesteps_per_second,
                    summaries=getattr(self, "_scenario_summaries", None))
                if len(self.sim_data.manifest) >= 2 else None)
            # Quantity reference/breadth (V4-M11): available / derived / gated.
            self.quantities_panel = QuantitiesPanel(
                self.controller.store, self.sim_data.manifest)
            # Safe Assistant (V4-M12): bounded, deterministic organization of
            # computed evidence; main_window supplies context and runs it.
            self.assistant_panel = AssistantPanel()
            # Ask (Analysis-improvement roadmap Phase B): merged into the
            # Assistant tab as a second mode -- both panels unchanged, only
            # the tab-level presentation is consolidated (same pattern as
            # Hazard & Tenability).
            self.assistant_query_panel = AssistantQueryPanel(self.assistant_panel, self.query_panel)
            # Research workspace (V5-M4): hazard spaces + mission-control
            # dashboard. Per-scenario, so any study (not just the factorial).
            # V6-M6: takes the provider (not the raw store) so a CO read
            # (full FED) cleanly gates instead of touching the store.
            self.hazard_panel = HazardPanel(
                self.quantity_provider, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            # Hazard & Tenability (Analysis-improvement roadmap Phase B): a
            # mode-toggle wrapper, not a rewrite -- both panels keep their
            # own full functionality/bus wiring/disclaimers unchanged, this
            # only merges two overlapping top-level tabs into one.
            self.hazard_tenability_panel = HazardTenabilityPanel(
                self.hazard_panel, self.tenability_panel)
            self.dashboard_panel = DashboardPanel(
                self.controller.store, self.sim_data.manifest,
                self.sim_data.timesteps_per_second, energy_content=self.energy_panel)
            # Space-time cube (V5-M4/P4) + Fire Narrative++ (V5-M5); V6-M5
            # multi-plane cross-sections -- takes the provider (not the raw
            # store) so an unavailable X/Z-normal plane raises
            # GatedQuantityError instead of a raw disk read.
            self.spacetime_panel = SpaceTimePanel(
                self.quantity_provider, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            self.narrative_panel = NarrativePanel(
                self.controller.store, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            # Ensemble spread (V5-M5): min/mean/max envelopes across scenarios.
            self.ensemble_panel = EnsemblePanel(
                self.controller.store, self.sim_data.manifest,
                self.sim_data.timesteps_per_second)
            # Research Knowledge Graph (V5 Phase 6): laboratory memory over
            # every existing artifact. `app=self` lets it gather live artifacts.
            self.graph_panel = GraphPanel(
                self.controller.store, self.sim_data.manifest,
                self.sim_data.timesteps_per_second, app=self)
            # Factor-effect maps (M3.1) need the candle factorial's factor
            # axes; a generic guest study has none, so it's factorial-only.
            self.factor_effects_panel = (
                FactorEffectsPanel(self.controller.store, self.sim_data.manifest,
                                   self._quantity_options(), self.sim_data.timesteps_per_second)
                if self.is_factorial else None)
            # Sensitivity explorer (V5-M3): interpolate responses across the
            # existing factorial ("what-if"); estimates only, never a new run.
            # Constructed before StudyPanel (Analysis section consolidation
            # Phase 5) since it's now folded in as one of its sub-tabs, the
            # same "thin slot, not a rewrite" pattern already used for
            # Factor effects -- this panel's own sliders/tabs/bus wiring are
            # completely unchanged.
            self.sensitivity_panel = (
                SensitivityPanel(getattr(self, "_scenario_summaries", None) or [],
                                 self.sim_data.manifest)
                if self.is_factorial else None)
            # Study-level analytics (V5-M2): the factorial as a designed
            # experiment. Needs the factor axes + computed summaries.
            self.study_panel = (
                StudyPanel(getattr(self, "_scenario_summaries", None) or [],
                           self.sim_data.manifest,
                           factor_effects_content=self.factor_effects_panel,
                           sensitivity_content=self.sensitivity_panel)
                if self.is_factorial else None)
            # Parallel coordinates (Analysis section consolidation Phase 3):
            # extracted from StudyPanel -- it conceptually belongs with the
            # other cross-scenario discovery tools (Compare & Discover), not
            # Study's factor/response tabs. Needs the same factor axes +
            # computed summaries StudyPanel does.
            self.parallel_coordinates_panel = (
                ParallelCoordinatesPanel(getattr(self, "_scenario_summaries", None) or [],
                                         self.sim_data.manifest)
                if self.is_factorial else None)
            # Compare & Discover (Analysis section consolidation Phase 3): a
            # thin QTabWidget wrapper over four pre-existing "how do
            # scenarios compare?" tools -- pairwise, parallel coordinates,
            # ensemble envelope, and PCA/clustering. Every child's own
            # construction/store access/lazy-load/bus wiring is unchanged;
            # only the tab-level presentation is consolidated (same pattern
            # as HazardTenabilityPanel/AssistantQueryPanel). Any child may be
            # absent (e.g. fewer than 2 scenarios means no pairwise panel);
            # the wrapper only adds a tab for what's supplied.
            self.compare_discover_panel = CompareDiscoverPanel(
                pairwise=self.advanced_compare_panel,
                parallel=self.parallel_coordinates_panel,
                ensemble=self.ensemble_panel,
                clustering=(self.analytics_panel.widget() if self.analytics_panel is not None else None))
        else:
            self.quantity_provider = None
            self.device_panel = None
            self.velocity_panel = None
            self.timeseries_panel = None
            self.energy_panel = None
            self.factor_effects_panel = None
            self.study_panel = None
            self.parallel_coordinates_panel = None
            self.compare_discover_panel = None
            self.sensitivity_panel = None
            self.tenability_panel = None
            self.fire_mri_panel = None
            self.query_panel = None
            self.state_space_panel = None
            self.attention_panel = None
            self.cause_panel = None
            self.height_panel = None
            self.zone_panel = None
            self.time_window_panel = None
            self.measurement_panel = None
            self.probe_measure_panel = None
            self.spatiotemporal_panel = None
            self.advanced_compare_panel = None
            self.quantities_panel = None
            self.assistant_panel = None
            self.assistant_query_panel = None
            self.hazard_panel = None
            self.hazard_tenability_panel = None
            self.dashboard_panel = None
            self.spacetime_panel = None
            self.narrative_panel = None
            self.ensemble_panel = None
            self.graph_panel = None

        dataset_content = self.experiment_browser.widget() if self.experiment_browser is not None else None
        # The dock objects themselves are now empty shells (their content
        # was just reparented onto a page) -- never added to a dock area,
        # but still MainWindow-parented QWidgets, so hide them explicitly
        # rather than leave them as stray unlaid-out children.
        if self.experiment_browser is not None:
            self.experiment_browser.hide()
        if self.analytics_panel is not None:
            self.analytics_panel.hide()

        self.pages = {
            "home": HomePage(on_start=lambda: self._navigate_to("live")),
            "live": LivePage(live_content, self.time_controller, settings=self.settings),
            "compare": ComparePage(on_preset=self._apply_compare_preset),
            "dataset": DatasetPage(dataset_content),
            "analysis": AnalysisPage(
                on_shown=self._on_analysis_page_shown,
                playback_bar=self._build_analysis_playback_bar(),
                forecasting_content=ForecastingPanel(
                    self.prediction_store, self.controller.store, self.sim_data.manifest),
                fire_mri_content=self.fire_mri_panel,
                state_space_content=self.state_space_panel,
                attention_content=self.attention_panel,
                cause_content=self.cause_panel,
                spatiotemporal_content=self.spatiotemporal_panel,
                probe_measure_content=self.probe_measure_panel,
                compare_discover_content=self.compare_discover_panel,
                study_content=self.study_panel,
                hazard_tenability_content=self.hazard_tenability_panel,
                dashboard_content=self.dashboard_panel,
                spacetime_content=self.spacetime_panel,
                narrative_content=self.narrative_panel,
                graph_content=self.graph_panel,
                quantities_content=self.quantities_panel,
                assistant_content=self.assistant_query_panel),
            "export": ExportPage(
                on_export_animation=self._export_animation, on_export_postcard=self._export_postcard),
            "about": AboutPage(),
        }
        has_manifest = bool(self.sim_data.manifest)
        self.pages["home"].set_stats(
            len(self.sim_data.manifest or []), self._current_n_frames, len(self.quantity_infos))
        self.pages["compare"].set_available(has_manifest)

        nav_entries = [
            ("home", "Home"), ("live", "Live Viewer"), ("compare", "Compare"),
            ("dataset", "Dataset Explorer"), ("analysis", "Analysis"), ("export", "Export"),
            ("about", "About"),
        ]
        # M2.5: the Compare page's presets are candle-factor comparisons
        # ("door open vs closed", etc.); a generic guest study has no such
        # factors, so drop Compare from its navigation.
        if not self.is_factorial:
            nav_entries = [(k, label) for k, label in nav_entries if k != "compare"]

        self.page_stack = QtWidgets.QStackedWidget()
        for key, _label in nav_entries:
            self.page_stack.addWidget(self.pages[key])

        self.nav_rail = NavRail(nav_entries)
        self.nav_rail.page_selected.connect(self._navigate_to)

        shell = QtWidgets.QWidget()
        shell.setObjectName("shellWidget")
        shell_layout = QtWidgets.QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self.nav_rail)
        shell_layout.addWidget(self.page_stack, 1)
        self.setCentralWidget(shell)

        self._build_evidence_notebook()
        self._build_sessions()
        self._build_selection()

        self._active_page_key = None
        self._navigate_to("live")

    def _build_selection(self) -> None:
        """Shared Selection Model (V5-M1): create the SelectionBus (Layer 2)
        and the QuantityProvider (Layer 1), then bind every analysis panel's
        scenario/frame controls to the bus so a change in one syncs the rest.
        The Live Viewer is wired last and minimally (it reacts to a selected
        time by seeking, and publishes its own seeks) to protect playback and
        the cinematic pipeline. Deeper per-panel point/region sync rides along
        as panels are individually touched (M2-M6)."""
        self.selection_bus = SelectionBus()
        if getattr(self, "quantity_provider", None) is None:
            self.quantity_provider = QuantityProvider(
                self.controller.store, fps=self.time_controller.timesteps_per_second)
        self._seek_from_bus = False
        fps = self.time_controller.timesteps_per_second
        for attr in ("height_panel", "zone_panel", "time_window_panel",
                     "measurement_panel", "fire_mri_panel",
                     "query_panel", "state_space_panel", "attention_panel", "cause_panel",
                     "factor_effects_panel", "tenability_panel", "timeseries_panel",
                     "energy_panel", "forecasting_panel", "quantities_panel",
                     "advanced_compare_panel", "study_panel", "parallel_coordinates_panel",
                     "hazard_panel", "spacetime_panel", "narrative_panel", "ensemble_panel",
                     "device_panel", "velocity_panel", "dashboard_panel"):
            panel = getattr(self, attr, None)
            if panel is not None:
                bind_to_bus(panel, self.selection_bus, fps)
        # V5-M3: the Sensitivity panel drives the bus by publishing the nearest
        # existing run for the current slider setting, and reacts to a scenario
        # selection by snapping its sliders -- it has its own set_bus.
        if self.sensitivity_panel is not None:
            self.sensitivity_panel.set_bus(self.selection_bus)
        # V5-M4: the dashboard reads the whole selection (scenario + time) and
        # its workspace preset raises the most relevant analysis tab.
        if self.dashboard_panel is not None:
            self.dashboard_panel.set_bus(self.selection_bus)
            self.dashboard_panel.workspace_requested.connect(self._on_workspace_preset)
        # V5-M4: the space-time panel publishes/reacts to the selected point.
        if self.spacetime_panel is not None:
            self.spacetime_panel.set_bus(self.selection_bus)
        # Consolidation Phase 2: the A/B pair publishes/reacts to
        # Selection.comparison (previously defined but unused by any panel).
        if self.advanced_compare_panel is not None:
            self.advanced_compare_panel.set_bus(self.selection_bus)
        # Consolidation Phase 2: the point/region probe publishes to the bus
        # (previously local-only) so other panels (e.g. SpaceTimePanel) can follow it.
        if self.timeseries_panel is not None:
            self.timeseries_panel.set_bus(self.selection_bus)
        # Consolidation Phase 2: the selected window publishes Selection.interval.
        if self.time_window_panel is not None:
            self.time_window_panel.set_bus(self.selection_bus)
        # V5-M5: a narrative step publishes its Insight's selection (seek + sync).
        if self.narrative_panel is not None:
            self.narrative_panel.event_activated.connect(self._on_insight_activated)
        # V5 Phase 6: the knowledge graph publishes a node's Selection and
        # rebuilds its selected-scenario events when the selection changes.
        if self.graph_panel is not None:
            self.graph_panel.set_bus(self.selection_bus)
        # V6-M2: a placed/edited/deleted device refreshes the Live Viewer's
        # markers; clicking a device's result (an Insight) seeks/highlights
        # through the same shared navigation every other V3 feature uses.
        if self.device_panel is not None:
            self.device_panel.devices_changed.connect(self._refresh_device_markers)
            self.device_panel.device_activated.connect(self._on_insight_activated)
        # V6-M3: a placed/renamed/deleted vector probe refreshes the Live
        # Viewer's quiver/streamline overlay; jump-to reuses the same shared
        # navigation as devices/insights.
        if self.velocity_panel is not None:
            self.velocity_panel.probes_changed.connect(self._refresh_vector_field)
            self.velocity_panel.probe_activated.connect(self._on_insight_activated)
        # V6-M4 Investigation History: records every *meaningful* selection
        # (skips its own back/forward replay via the `self.history` sentinel
        # origin, and MainWindow's own playback-tick echo via `self` -- time_s
        # alone changes on every tick and must not flood the log).
        self.history = InvestigationHistory()
        self.selection_bus.changed.connect(self._on_history_changed)
        self.selection_bus.changed.connect(self._on_bus_changed)
        # RC polish: switching analysis tabs re-syncs the newly-shown panel to
        # the current selection/time. Phase D: AnalysisPage.tab_shown covers
        # every level a panel can go from hidden to visible at (the outer
        # group tabs, a group's own inner tabs, or expanding Experimental).
        analysis_page = self.pages.get("analysis")
        if analysis_page is not None and hasattr(analysis_page, "tab_shown"):
            analysis_page.tab_shown.connect(self.selection_bus.resend)

    # Adaptive Workspace presets (V5-M4/P4): each focuses the app on a task by
    # raising its primary analysis tab and publishing the quantity that task
    # cares about, so every quantity-aware panel follows (via the M1 binder).
    _WORKSPACE = {
        "Overview": ("dashboard_panel", None),
        "Temperature study": ("hazard_tenability_panel", "TEMPERATURE"),
        "Ventilation study": ("sensitivity_panel", "VELOCITY"),
        "Smoke study": ("height_panel", "TEMPERATURE"),
        "Study analytics": ("study_panel", None),
    }

    def _on_workspace_preset(self, preset: str) -> None:
        panel_attr, quantity = self._WORKSPACE.get(preset, (None, None))
        if quantity is not None and getattr(self, "selection_bus", None) is not None:
            self.selection_bus.update(origin=None, quantity=quantity)
        panel = getattr(self, panel_attr, None) if panel_attr else None
        if panel is not None:
            self._reveal(panel)

    def _reveal(self, panel: QtWidgets.QWidget) -> None:
        """Raise the Analysis page and show `panel`'s tab (V6-M4): the one
        cross-navigation primitive every "reveal in Analysis" interaction
        should share, instead of each feature re-deriving the same two
        lines (previously duplicated in _on_workspace_preset and the
        now-removed Experiments panel's comparison hand-off)."""
        self._navigate_to("analysis")
        self.pages["analysis"].show_tab(panel)

    def _on_history_changed(self, selection, origin) -> None:
        """V6-M4 Investigation History: record every selection except a
        replay of the history itself (`origin is self.history`) or
        MainWindow's own playback-tick echo (`origin is self` -- time_s
        alone changes every tick and must not flood the log)."""
        if origin is self.history or origin is self:
            return
        self.history.record(selection)

    def _history_back(self) -> None:
        history = getattr(self, "history", None)
        if history is None or getattr(self, "selection_bus", None) is None:
            return
        sel = history.back()
        if sel is not None:
            self.selection_bus.set(sel, origin=history)

    def _history_forward(self) -> None:
        history = getattr(self, "history", None)
        if history is None or getattr(self, "selection_bus", None) is None:
            return
        sel = history.forward()
        if sel is not None:
            self.selection_bus.set(sel, origin=history)

    def _on_bus_changed(self, selection, origin) -> None:
        """React to the shared selection in the Live Viewer: seek playback to
        the selected time. Skips our own seek echo (origin is self)."""
        if origin is self or selection.time_s is None or self._current_n_frames <= 0:
            return
        fps = self.time_controller.timesteps_per_second
        fi = min(max(int(round(selection.time_s * fps)), 0), self._current_n_frames - 1)
        self._seek_from_bus = True
        try:
            self._on_seek_requested(fi)
        finally:
            self._seek_from_bus = False

    def _publish_insight_selection(self, insight) -> None:
        """Route an activated Insight through the bus (V5-M1): publish the
        fields it carries (time/point/region/quantity), preserving the current
        scenario, so every panel and the Live Viewer follow together."""
        fields = {}
        t = insight.primary_time()
        if t is not None:
            fields["time_s"] = t
        if getattr(insight, "location", None):
            fields["point"] = insight.location
        if getattr(insight, "region", None):
            fields["region"] = insight.region
        if getattr(insight, "quantity", None):
            fields["quantity"] = insight.quantity
        if fields:
            self.selection_bus.update(origin=None, **fields)

    def _build_sessions(self) -> None:
        """Named analysis sessions (V4-M6): _sessions_dir is still needed by
        the autosave-on-close draft (closeEvent, below) even though the
        Sessions tab's own UI (which used to wire save/load/delete/export
        signals here) was removed (Analysis UX + reliability pass) --
        session_store.py itself, and every function still reachable through
        it (_refresh_sessions/_on_session_*/_apply_analysis_session), stay
        as reusable session-management logic, just not wired to a visible
        tab today."""
        self._sessions_dir = session_store.default_sessions_dir()
        # V4-M12: the Safe Assistant runs on computed context supplied here.
        if self.assistant_panel is not None:
            self.assistant_panel.action_requested.connect(self._on_assistant_action)
            self.assistant_panel.query_submitted.connect(self._on_assistant_query)
            self.assistant_panel.save_requested.connect(self._on_assistant_save)

    def _run_assistant(self, action: str) -> str:
        """Gather computed context and run one deterministic assistant
        action. No physics is inferred -- every result is a template filled
        from already-computed values."""
        if action == "summarize_session":
            return assistant_mod.summarize_session(self._collect_session_dict())
        if action == "list_key_findings":
            return assistant_mod.list_key_findings(self.evidence_dock.notebook.to_list())
        if action == "report_outline":
            return assistant_mod.report_outline(self.evidence_dock.notebook.to_list())
        if action == "compare_intervals":
            return self._assistant_compare_intervals()
        if action == "figure_caption":
            return self._assistant_figure_caption()
        return assistant_mod.REFUSAL

    def _assistant_compare_intervals(self) -> str:
        import time_window as tw
        p = self.time_window_panel
        if p is None or getattr(p, "_series", None) is None:
            return "Open the Intervals tab and select a window first."
        s = p._series
        if p._mode == "split" and p._split is not None:
            a, b = tw.before_after_split(s["mean"], s["max"], s["times"], p._split)
            la, lb = f"before {p._split:.0f} s", f"after {p._split:.0f} s"
        else:
            a = tw.interval_stats(s["mean"], s["max"], s["times"], p._t0, p._t1)
            b = tw.interval_stats(s["mean"], s["max"], s["times"],
                                  float(s["times"][0]), float(s["times"][-1]))
            la, lb = "selected window", "whole run"
        return assistant_mod.compare_intervals(a, b, la, lb, s["unit"])

    def _assistant_figure_caption(self) -> str:
        import numpy as np
        from registry import get_quantity
        cell = self.view_grid.active_cell()
        if cell is None or cell.cell_type != "slice":
            return "Select a single-scenario cell in the Live viewer for a caption."
        data = np.asarray(self.controller.store.get(cell.case_index, cell.quantity_key))
        idx = min(self.time_controller.index, data.shape[0] - 1)
        peak = float(data[idx].max())
        scenario = next((e.folder for e in self.sim_data.manifest
                         if e.case_index == cell.case_index), str(cell.case_index))
        q = get_quantity(cell.quantity_key.quantity)
        fps = self.time_controller.timesteps_per_second
        return assistant_mod.figure_caption(scenario, q.label, q.unit, idx / fps, peak)

    def _on_assistant_action(self, action: str) -> None:
        self.assistant_panel.show_result(self._run_assistant(action))

    def _on_assistant_query(self, text: str) -> None:
        action = assistant_mod.interpret_request(text)
        if action == "refuse":
            self.assistant_panel.show_result(assistant_mod.REFUSAL, savable=False)
        elif action == "search_scenarios":
            # Assistant++ (V5-M5): experiment search over the study table.
            summaries = getattr(self, "_scenario_summaries", None)
            if not summaries:
                self.assistant_panel.show_result(
                    "Scenario search needs the factorial's computed summaries.",
                    savable=False)
                return
            table = study_analytics_mod.build_table(summaries)
            self.assistant_panel.show_result(assistant_mod.search_scenarios(table, text))
        else:
            self.assistant_panel.show_result(self._run_assistant(action))

    def _on_assistant_save(self, text: str) -> None:
        if not text:
            return
        from insight import Insight
        first_line = text.strip().splitlines()[0] if text.strip() else "Assistant note"
        self.evidence_dock.add_insight(Insight(
            first_line[:200], category="query",
            basis="assistant: organized from computed evidence (no cause inferred)"))
        self.statusBar().showMessage("Saved assistant output to the Evidence Notebook", 4000)

    def _build_evidence_notebook(self) -> None:
        """Evidence Notebook (V4-M2): a dockable, session-backed collection
        of saved measurements. Every panel's InsightList already exposes a
        "Save to Evidence Notebook" action (insight.py) and an
        insight_saved signal, so wiring is one connection per panel; the
        dock's own click reuses the shared navigation. Starts hidden (View
        menu toggles it); the first saved insight reveals it."""
        self.evidence_dock = EvidenceNotebookDock(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.evidence_dock)
        self.evidence_dock.insight_activated.connect(self._on_insight_activated)
        self.evidence_dock.visibilityChanged.connect(self.evidence_notebook_action.setChecked)
        self.evidence_dock.hide()
        for panel_attr, list_attr in (
            ("inspector", "story_list"), ("height_panel", "insights"),
            ("zone_panel", "insights"),
            ("time_window_panel", "insights"),
            ("advanced_compare_panel", "temporal_list"),
            ("advanced_compare_panel", "spatial_list"),
            ("advanced_compare_panel", "physics_list"),
            ("advanced_compare_panel", "semantic_diff_list"),
            ("query_panel", "results"),
            ("cause_panel", "chain"),
        ):
            panel = getattr(self, panel_attr, None)
            insight_list = getattr(panel, list_attr, None) if panel is not None else None
            if insight_list is not None:
                insight_list.insight_saved.connect(self.evidence_dock.add_insight)

    def _on_analysis_page_shown(self) -> None:
        """Analysis page on_enter: kick the analytics panel's one-shot
        background feature load (pre-existing behavior) and the
        time-series workspace's lazy first load (V2 M1.1)."""
        self._on_analytics_panel_visibility_changed(True)
        if self.timeseries_panel is not None:
            self.timeseries_panel.ensure_loaded()
        if self.energy_panel is not None:
            self.energy_panel.ensure_loaded()

    def _navigate_to(self, key: str) -> None:
        page = self.pages.get(key)
        if page is None:
            return
        # Compare and the plain Live Viewer are meant to be independent: a
        # preset's comparison grid must not stick around as "the" Live
        # Viewer. Checked before the same-page early-return below so
        # re-clicking "Live Viewer" while its comparison grid is still
        # showing there also resets it, not just navigating in from
        # elsewhere -- _navigate_to's only other caller with this guard
        # already up is _apply_compare_preset's own internal jump.
        if (key in ("home", "live") and getattr(self, "_compare_active", False)
                and not getattr(self, "_applying_compare_preset", False)):
            self._reset_grid_after_compare()
        if key == self._active_page_key:
            return
        old_page = self.pages.get(self._active_page_key)
        if old_page is not None:
            old_page.on_leave()
        self._active_page_key = key
        self.page_stack.setCurrentWidget(page)
        self.nav_rail.set_active(key)
        page.on_enter()
        # RC polish: a panel that becomes visible catches up to the current
        # selection/time (hidden panels skip live updates for performance).
        if getattr(self, "selection_bus", None) is not None:
            self.selection_bus.resend()

    def _reset_grid_after_compare(self) -> None:
        """Bugfix: leaving a Compare preset's comparison grid (2x1, two
        plain-slice scenarios -- see _apply_compare_preset) back to Home
        or Live must not leave that setup behind -- reset to a plain 1x1
        view of whichever scenario was the comparison's first (A)."""
        self._compare_active = False
        cells = self.view_grid.visible_cells()
        if not cells:
            return
        cell = cells[0]
        case_index = cell.case_index if cell.cell_type == "slice" else cell.case_index_a
        self._set_grid_layout("1x1")
        cell = self.view_grid.visible_cells()[0]
        if cell.cell_type != "slice":
            cell.set_cell_type("slice")
        self._select_scenario_in_cell(cell, case_index)

    def _build_central_widget(self) -> QtWidgets.QWidget:
        """Builds the Live Viewer's content (control panel + plot grid),
        unchanged from before the nav-rail shell existed -- FireLab
        roadmap Phase 1 wraps this in a LivePage instead of setting it
        directly as MainWindow's central widget (see _build_shell()), but
        every widget built here (self.splitter, self.view_grid,
        self.timeline, self.temp_slider, ...) is constructed exactly as
        before, at the same point in __init__, so it's available as a
        MainWindow attribute immediately -- not lazily -- for every
        existing call site and test."""
        central = QtWidgets.QWidget()
        central.setObjectName("centralWidget")
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(True)  # user can fully collapse the panel
        root_layout.addWidget(self.splitter)

        self.splitter.addWidget(self._build_control_panel())
        self.splitter.addWidget(self._build_plot_panel())
        self.splitter.addWidget(self._build_inspector_panel())
        # Control panel and inspector get fixed-ish starting shares; plot
        # gets the rest and does the growing when the window resizes.
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([380, 700, 280])
        return central

    def _build_inspector_panel(self) -> QtWidgets.QWidget:
        """Right-hand Live Inspector (FireLab roadmap Phase 3): probe
        readout, peak-temperature sparkline, HRR gauge, live narration --
        see _update_inspector() for how it's kept in sync with playback.
        One section per visible grid cell (V6 polish: a 2+ cell comparison
        grid used to only ever show the active cell's stats); self.inspector
        stays a direct alias to section 0 for any 1x1-layout code that only
        ever cared about "the" inspector."""
        self.inspector_stack = InspectorStack()
        self.inspector_stack.ensure_count(1)
        self.inspector_stack.section_added.connect(self._wire_inspector_section)
        self.inspector = self.inspector_stack.section(0)
        self._wire_inspector_section(0, self.inspector)
        return self.inspector_stack

    def _wire_inspector_section(self, index: int, panel) -> None:
        """Connects one InspectorStack section's own widgets the first time
        it's created (section 0 at construction, any later one lazily via
        InspectorStack.section_added). The diff-plot button needs to know
        *which* cell it belongs to (looked up by grid position at click
        time, never a stale captured reference); the fire-story list
        doesn't -- an activated Insight already carries its own scenario."""
        panel.diff_plot_button.clicked.connect(
            lambda _checked=False, i=index: self._show_difference_over_time(self._cell_at_section_index(i)))
        panel.story_list.insight_activated.connect(self._on_insight_activated)

    def _cell_at_section_index(self, index: int):
        cells = self.view_grid.visible_cells()
        return cells[index] if index < len(cells) else None

    def _inspector_section_index_for_cell(self, cell):
        try:
            return self.view_grid.visible_cells().index(cell)
        except ValueError:
            return None

    def _on_insight_activated(self, insight) -> None:
        """Shared navigation, now via the SelectionBus (V5-M1): publishing the
        Insight's selection seeks playback (through _on_bus_changed) and syncs
        every subscribed panel. Falls back to a direct seek if the bus is not
        yet built (defensive)."""
        if getattr(self, "selection_bus", None) is not None:
            self._publish_insight_selection(insight)
            return
        fi = insight.frame_index(self.time_controller.timesteps_per_second)
        if fi is not None and self._current_n_frames > 0:
            self._on_seek_requested(min(max(fi, 0), self._current_n_frames - 1))

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("controlPanel")
        panel.setMinimumWidth(220)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(18)

        title = QtWidgets.QLabel("FDS SLCF Visualizer")
        title.setProperty("role", "title")
        title.setWordWrap(True)
        outer.addWidget(title)

        # --- Room diagram ----------------------------------------------------
        # A live schematic for non-specialist users: shows the room, door,
        # vents, and candle(s) matching the toggles below, proportioned from
        # the scenario's real parsed .smv mesh extent (see schematic.py).
        schematic_section = CollapsibleSection("Room diagram")
        schematic_section.setToolTip(
            "A simplified top-down diagram of the room, proportioned to match "
            "the real physical layout. It updates automatically as you change "
            "the candles, vents, and door below."
        )
        self.schematic = SchematicWidget()
        schematic_section.add_row(self.schematic)
        outer.addWidget(schematic_section)
        room_extent = resolve_room_extent(self.controller.store, self.controller.current_case_index())
        self.schematic.set_room_extent(room_extent)
        self.schematic.update_state(DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC)

        # --- Simulation transport controls (UI/UX modernization Phase 4:
        # wrapped in a card like every other control-panel section below,
        # instead of sitting bare between two manual dividers) ---------------
        playback_section = CollapsibleSection("Playback")
        transport_row = QtWidgets.QHBoxLayout()
        transport_row.setSpacing(8)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setAccessibleName("Start simulation")
        self.start_button.setToolTip("Start the fire simulation animation (Space)")
        self.start_button.clicked.connect(self._start_simulation)

        self.stop_button = QtWidgets.QPushButton("Pause")
        self.stop_button.setAccessibleName("Pause simulation")
        self.stop_button.setToolTip("Pause the simulation (Space)")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_simulation)

        self.restart_button = QtWidgets.QPushButton("Restart")
        self.restart_button.setAccessibleName("Restart simulation from the beginning")
        self.restart_button.setToolTip("Restart the simulation from t=0 (Ctrl+R)")
        self.restart_button.clicked.connect(self._restart_simulation)

        for b in (self.start_button, self.stop_button, self.restart_button):
            transport_row.addWidget(b)
        transport_container = QtWidgets.QWidget()
        transport_container.setLayout(transport_row)
        playback_section.add_row(transport_container)

        # M1.4: interactive scrubber (play/pause + seek slider + time label +
        # loop toggle), replacing the old read-only QProgressBar.
        self.timeline = TimelineWidget()
        self.timeline.play_pause_clicked.connect(self._toggle_play_pause)
        self.timeline.seek_requested.connect(self._on_seek_requested)
        self.timeline.loop_toggled.connect(self.time_controller.set_loop)
        playback_section.add_row(self.timeline)
        outer.addWidget(playback_section)

        # --- Scenario sections ----------------------------------------------
        speed_section = CollapsibleSection("Playback speed")
        self.speed_toggle = ToggleGroup(
            [("1x", 1), ("2x", 2), ("3x", 3)], default_index=0,
            accessible_name="Playback speed",
        )
        self.speed_toggle.setToolTip(
            "Controls how fast the simulation plays back -- 2x and 3x speed "
            "up the animation without changing the underlying simulation."
        )
        self.speed_toggle.value_changed.connect(self.time_controller.set_speed)
        speed_section.add_row(self.speed_toggle)
        outer.addWidget(speed_section)

        # Ventilation first (user feedback): the vents are the primary
        # thing people compare, so they sit at the top of the scenario
        # controls with short "Vent 1"/"Vent 2" labels (the VOD/VOC codes
        # stay in the accessible names and tooltips for traceability).
        vod_section = CollapsibleSection("Vent 1")
        self.vod_toggle = VentWidget(
            [("Open", 0), ("Closed", 1), ("HVAC", 2)], state_labels=_VOD_STATES, default_index=DEFAULT_VOD,
            accessible_name="Air vent 1 (VOD) state",
        )
        self.vod_toggle.setToolTip(
            "Vent 1 (VOD): opens, closes, or connects an air vent in the room "
            "to a fan (HVAC). Open lets smoke and hot air escape and fresh air "
            "in; closed traps heat and smoke inside."
        )
        self.vod_toggle.value_changed.connect(self._on_vod_changed)
        vod_section.add_row(self.vod_toggle)
        outer.addWidget(vod_section)

        voc_section = CollapsibleSection("Vent 2")
        self.voc_toggle = VentWidget(
            [("Open", 0), ("Closed", 1)], state_labels=_VOC_STATES, default_index=DEFAULT_VOC,
            accessible_name="Air vent 2 (VOC) state",
        )
        self.voc_toggle.setToolTip(
            "Vent 2 (VOC): opens or closes a second air vent in the room. "
            "Works the same way as Vent 1 -- open lets air move through, "
            "closed seals the room."
        )
        self.voc_toggle.value_changed.connect(self._on_voc_changed)
        voc_section.add_row(self.voc_toggle)
        outer.addWidget(voc_section)

        candle_section = CollapsibleSection("Number of candles")
        self.candle_toggle = CandleCard(
            [("1 candle", 0), ("2 candles", 1)], default_index=DEFAULT_CANDLES,
            accessible_name="Number of candles",
        )
        self.candle_toggle.setToolTip(
            "Sets how many lit candles are burning in the room. More candles "
            "create a bigger, hotter fire source and change how quickly the "
            "room heats up."
        )
        self.candle_toggle.value_changed.connect(self._on_candle_changed)
        candle_section.add_row(self.candle_toggle)
        outer.addWidget(candle_section)

        door_section = CollapsibleSection("Door opening width")
        # Options list is [("Wide open", 1), ("Narrow", 0)]; DEFAULT_DOOR=1 is
        # at position 0, so default_index=0 correctly preselects "Wide open".
        self.door_toggle = DoorWidget(
            [("Wide open", 1), ("Narrow", 0)], default_index=0,
            accessible_name="Door state",
        )
        self.door_toggle.setToolTip(
            "Sets how wide the door opening is. A wider opening lets more "
            "air, smoke, and heat move in and out of the room; 'Narrow' "
            "restricts the flow."
        )
        self.door_toggle.value_changed.connect(self._on_door_changed)
        door_section.add_row(self.door_toggle)
        outer.addWidget(door_section)

        # M2.5: these sections describe the candle factorial's scenario
        # parameters; a generic guest study has no such axes, so hide them
        # (their handlers/widgets still exist, just operate on hidden
        # widgets harmlessly if ever driven). The grid, timeline, quantity
        # selector, display scale, and analysis panels below stay active.
        if not self.is_factorial:
            for section in (schematic_section, candle_section, vod_section,
                            voc_section, door_section):
                section.setVisible(False)

        outer.addWidget(self._divider())

        # --- Quantity selector (M2.1) ----------------------------------------
        quantity_section = CollapsibleSection("Data shown")
        self.quantity_infos = self._discover_quantities()
        # V6-M1.5: the Live combo shows native quantities + computed
        # (derived/calculated) fields; quantity_infos stays native so the
        # analysis panels and _quantity_options are unchanged.
        self._combo_quantity_list = list(self.quantity_infos) + self._computed_quantity_infos()
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Quantity shown in the heatmap")
        multiple_available = len(self._combo_quantity_list) > 1
        tooltip = "Choose what the color map shows (including calculated fields)."
        if not multiple_available:
            tooltip += " (Only temperature is available in demo-data mode.)"
        self.quantity_combo.setToolTip(tooltip)
        for i, info in enumerate(self._combo_quantity_list):
            self.quantity_combo.addItem(self._quantity_label(info))
            item_tip = self._quantity_tooltip(info)
            if item_tip:
                self.quantity_combo.setItemData(i, item_tip, QtCore.Qt.ToolTipRole)
        self.quantity_combo.setEnabled(multiple_available)
        self.quantity_combo.currentIndexChanged.connect(self._on_quantity_changed)
        quantity_section.add_row(self.quantity_combo)
        # RC polish (§3): the quantity selector sits directly below the Playback
        # section so changing what's shown is immediately accessible, instead of
        # being buried below the scenario controls.
        outer.insertWidget(outer.indexOf(playback_section) + 1, quantity_section)

        # --- Display controls -------------------------------------------------
        # Range/default/label come from QUANTITY_DISPLAY for the starting
        # quantity (current_quantity_key, set in __init__); switching
        # quantity later re-applies these via _apply_quantity_display_defaults.
        initial_display = self._display_for(self.current_quantity_key.quantity)
        temp_section = CollapsibleSection("Display scale (max)")
        temp_row = QtWidgets.QHBoxLayout()
        self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.temp_slider.setRange(initial_display['slider_min'], initial_display['slider_max'])
        self.temp_slider.setValue(initial_display['slider_default'])
        self.temp_slider.setAccessibleName(
            f"Maximum {initial_display['label'].lower()} scale, {initial_display['unit']}")
        self.temp_slider.setToolTip(
            f"Adjust the maximum {initial_display['label'].lower()} shown on the color scale")
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        self.temp_label = QtWidgets.QLabel(f"{initial_display['slider_default']} {initial_display['unit']}")
        self.temp_label.setProperty("role", "value")
        self.temp_label.setMinimumWidth(60)
        temp_row.addWidget(self.temp_slider, 1)
        temp_row.addWidget(self.temp_label)
        temp_section.add_row(self._wrap(temp_row))
        outer.addWidget(temp_section)

        outer.addStretch(1)

        quit_button = QtWidgets.QPushButton("Quit")
        quit_button.setAccessibleName("Quit application")
        quit_button.setToolTip("Close the application (Ctrl+Q)")
        quit_button.clicked.connect(self.close)
        outer.addWidget(quit_button)

        return scroll

    def _scenario_options(self) -> list:
        """[(label, case_index), ...] for a grid cell's per-cell scenario
        combo (M2.2.2), sourced from the manifest (M2.1). Empty in demo
        mode -- there's no real manifest to pick scenarios from, same
        convention as the quantity combo's demo-mode fallback. label is a
        human-readable factor-level summary (manifest.scenario_label),
        not the raw "c1_d0_vod0_voc0" disk folder name."""
        if not self.sim_data.manifest:
            return []
        return [(self._scenario_label(e.case_index), e.case_index)
                for e in sorted(self.sim_data.manifest, key=lambda e: e.case_index)]

    def _quantity_options(self) -> list:
        """[(label, SliceKey), ...] for a grid cell's per-cell quantity
        combo -- same entries/labels as the control panel's own quantity
        combo (self.quantity_infos, computed in _build_control_panel).

        Native quantities only -- calculated/derived fields are computed via the
        QuantityProvider and the analysis panels still read the store directly,
        so this list (which drives their combos) stays native (V6-M1.5)."""
        return [(self._quantity_label(info), info.key) for info in self.quantity_infos]

    # --- V6-M1.5: computed quantities as first-class visual quantities ----
    @staticmethod
    def _is_computed(quantity: str) -> bool:
        """A calculated (Field Calculator) or legacy derived quantity -- these go
        through the QuantityProvider, not the store."""
        from registry import get_quantity
        q = get_quantity(quantity)
        return getattr(q, "calculated", False) or q.kind == "derived"

    def _field(self, store, case_index: int, quantity_key):
        """Field data for (case, quantity): computed quantities via the provider
        (the compute layer, memoized); native quantities via the given store
        (override/cache aware). One routing point -- no second pipeline."""
        if self._is_computed(quantity_key.quantity):
            return self.quantity_provider.get(case_index, quantity_key)
        return store.get(case_index, quantity_key)

    def _display_for(self, quantity: str) -> dict:
        """Display config (label/unit/cmap/scale) for a quantity. Native ones
        come from QUANTITY_DISPLAY unchanged; computed ones are built from their
        registry entry (which carries the same fields). Behaviour-preserving for
        native quantities."""
        if quantity in QUANTITY_DISPLAY:
            return QUANTITY_DISPLAY[quantity]
        from registry import get_quantity
        q = get_quantity(quantity)
        return {"label": q.label, "unit": q.unit, "cmap": q.cmap, "vmin": q.vmin,
                "slider_min": q.slider_min, "slider_max": q.slider_max,
                "slider_default": q.slider_default}

    def _computed_quantity_infos(self) -> list:
        """SliceInfo entries for calculated/derived quantities whose inputs are
        available, so they appear in the Live Viewer combo alongside native
        quantities (V6-M1.5)."""
        # Demo mode has no store/provider to compute against.
        if not self.sim_data.manifest:
            return []
        from registry import QUANTITY_REGISTRY, get_quantity
        import field_calculator as fc
        from derived_quantities import source_quantity
        native = {i.key.quantity for i in self.quantity_infos}
        out = []
        for name, q in QUANTITY_REGISTRY.items():
            if q.gated:
                continue
            if getattr(q, "calculated", False):
                field = fc.get_field(name)
                deps = field.dependencies if field else ()
            elif q.kind == "derived":
                src = source_quantity(name)
                deps = (src,) if src else ()
            else:
                continue
            resolvable = deps and all(
                d in native or self._is_computed(d) for d in deps)
            if resolvable:
                out.append(SliceInfo(SliceKey(name), q.label, q.unit))
        return out

    def _refresh_quantity_list(self) -> None:
        """Rebuild the Live Viewer quantity combo to include current computed
        fields, preserving the selection (V6-M1.5). The provider's memo is
        invalidated so redefined fields recompute."""
        if not hasattr(self, "quantity_combo"):
            return
        if getattr(self, "quantity_provider", None) is not None:
            self.quantity_provider.invalidate()
        current = (self._combo_quantity_list[self.quantity_combo.currentIndex()].key.quantity
                   if self._combo_quantity_list else None)
        self._combo_quantity_list = list(self.quantity_infos) + self._computed_quantity_infos()
        self.quantity_combo.blockSignals(True)
        self.quantity_combo.clear()
        for i, info in enumerate(self._combo_quantity_list):
            self.quantity_combo.addItem(self._quantity_label(info))
            item_tip = self._quantity_tooltip(info)
            if item_tip:
                self.quantity_combo.setItemData(i, item_tip, QtCore.Qt.ToolTipRole)
        idx = next((i for i, info in enumerate(self._combo_quantity_list)
                    if info.key.quantity == current), 0)
        self.quantity_combo.setCurrentIndex(idx)
        self.quantity_combo.setEnabled(len(self._combo_quantity_list) > 1)
        self.quantity_combo.blockSignals(False)

    def _build_plot_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # M2.2.2: single-view mode is just the grid at its 1x1 default --
        # not a separately-maintained code path, so it's pixel-equivalent
        # to pre-refactor by construction.
        self.view_grid = ViewGrid(self._scenario_options(), self._quantity_options(),
                                   self.sim_data.manifest, panel)
        self.view_grid.cell_created.connect(self._on_cell_created)
        self.view_grid.active_cell_changed.connect(self._on_active_cell_changed)
        self.view_grid.cell_scenario_selected.connect(self._on_cell_scenario_selected)
        self.view_grid.cell_quantity_selected.connect(self._on_cell_quantity_selected)
        # M2.3.3: cell-type switching (slice/difference/ensemble).
        self.view_grid.cell_type_changed.connect(self._on_cell_type_changed)
        self.view_grid.cell_difference_scenarios_changed.connect(self._on_cell_difference_scenarios_changed)
        self.view_grid.cell_ensemble_changed.connect(self._on_cell_ensemble_changed)

        # The toolbar (pan/zoom/save) stays bound to the first cell's canvas
        # for its whole lifetime -- matplotlib's NavigationToolbar2QT isn't
        # designed to be rebound to a different canvas at runtime, and
        # pan/zoom is ambiguous across an ensemble grid anyway (you want
        # every cell showing the same framing, not independently panned
        # views). Shown only in 1x1 mode; see _set_grid_layout.
        self.toolbar = NavigationToolbar(self.view_grid.active_view().widget(), panel)
        self.toolbar.setAccessibleName("Plot navigation toolbar: pan, zoom, save")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.view_grid, 1)  # grid gets all extra vertical space

        self._init_plot()
        return panel

    def _build_status_bar(self):
        self.setStatusBar(QtWidgets.QStatusBar())
        self.statusBar().showMessage("Ready.")

    @staticmethod
    def _divider() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QtWidgets.QFrame.HLine)
        return line

    @staticmethod
    def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    # ------------------------------------------------------------ plotting
    # Thin delegating properties (M2.2): the actual matplotlib objects now
    # live on the *active* grid cell's SliceView (views.ViewGrid), not
    # MainWindow directly -- kept here so existing call sites/tests that
    # read window.heatmap etc. (observable rendering state, legitimate to
    # assert on) don't need to know about the ViewGrid indirection. In the
    # default 1x1 layout the active cell is the only cell, so this is
    # exactly the pre-M2.2 single-view behavior.
    @property
    def heatmap(self):
        return self.view_grid.active_view().heatmap

    @property
    def colorbar(self):
        return self.view_grid.active_view().colorbar

    @property
    def canvas(self):
        return self.view_grid.active_view().canvas

    @property
    def ax(self):
        return self.view_grid.active_view().ax

    def _discover_quantities(self) -> list:
        """Which (quantity, direction, offset) slices the current dataset
        actually offers, restricted to the slice plane the app has always
        rendered (DEFAULT_SLICE_KEY's direction/offset) so the combo box
        only ever changes *what* is shown, not *where* -- plus, when
        volumetric `.s3d` SOOT DENSITY data is present (M2.2), one or more
        smoke planes (a side view and a vertical doorway slice), which
        *do* change where by design. Demo mode has no real .smv to
        inspect, so it falls back to a single TEMPERATURE entry."""
        if self.sim_data.manifest:
            try:
                path = self.sim_data.manifest[0].path
                infos = available_slices(path)
                matching = [i for i in infos
                            if i.key.direction == DEFAULT_SLICE_KEY.direction
                            and i.key.offset == DEFAULT_SLICE_KEY.offset]
                if matching:
                    matching.sort(key=lambda i: i.key.quantity != DEFAULT_SLICE_KEY.quantity)
                    return matching + self._discover_soot_planes(path)
            except (FileNotFoundError, OSError) as e:
                logger.warning("could not discover available quantities (%s); "
                                "falling back to TEMPERATURE only", e)
        return [SliceInfo(DEFAULT_SLICE_KEY, 'temp', QUANTITY_DISPLAY[DEFAULT_SLICE_KEY.quantity]['unit'])]

    # SOOT planes surfaced in the quantity combo when `.s3d` data exists
    # (M2.2 any-plane slicing): a side view on the app's usual y=0 plane,
    # and a vertical slice through the doorway (x=0.25 m mesh boundary,
    # the roadmap's named demo case). Each is a distinct SliceKey carrying
    # its physical plane_pos, so it flows through the store/extent/probe/
    # isotherm machinery exactly like any other quantity.
    _SOOT_PLANES = (
        (SliceKey(SOOT_QUANTITY, AXIS_TO_DIRECTION['y'], 0, 0.0), 'Smoke — side view (y = 0)'),
        (SliceKey(SOOT_QUANTITY, AXIS_TO_DIRECTION['x'], 0, 0.25), 'Smoke — doorway (x = 0.25 m)'),
    )

    def _discover_soot_planes(self, scenario_path: str) -> list:
        """SliceInfo entries for the SOOT planes, but only if the scenario
        actually ships `.s3d` files (they aren't in demo data or a trimmed
        fixture)."""
        import glob
        if not glob.glob(os.path.join(scenario_path, '*.s3d')):
            return []
        unit = QUANTITY_DISPLAY[SOOT_QUANTITY]['unit']
        return [SliceInfo(key, label, unit) for key, label in self._SOOT_PLANES]

    @staticmethod
    def _quantity_label(info) -> str:
        """Combo label for a quantity entry. SOOT planes carry their own
        plane-specific label on the SliceInfo (two entries share the
        'SOOT DENSITY' quantity, so the QUANTITY_DISPLAY label alone
        wouldn't distinguish them); every other quantity uses its
        QUANTITY_DISPLAY label."""
        if info.key.quantity == SOOT_QUANTITY:
            return info.label
        # Computed fields aren't in QUANTITY_DISPLAY -- use their own label.
        return QUANTITY_DISPLAY.get(info.key.quantity, {}).get('label', info.label)

    @staticmethod
    def _quantity_tooltip(info) -> str:
        """Per-item dropdown tooltip: the registry's own plain-language
        "interpretation" (the same text quantities_panel.py already shows),
        reused here so a technical label like "Dynamic pressure" or a
        coordinate-heavy one like "Smoke — doorway (x = 0.25 m)" doesn't
        require visiting that separate panel just to know what it means."""
        from registry import get_quantity
        q = get_quantity(info.key.quantity)
        if not q.interpretation:
            return ""
        if info.key.quantity == SOOT_QUANTITY:
            return f"{info.label} — {q.interpretation}"
        return q.interpretation

    def _apply_quantity_display_defaults(self, quantity: str):
        """Re-applies the colormap/clim/label defaults for `quantity`
        (M2.1). Called right after self.current_quantity_key changes, so
        the temp_slider.setValue() below -- via _on_temp_changed, which
        reads self.current_quantity_key -- already computes the correct
        vmin/unit for the new quantity."""
        display = self._display_for(quantity)
        self.current_colormap = display['cmap']
        self.settings.setValue("colormap", display['cmap'])
        for action, (_, cmap) in zip(self.colormap_action_group.actions(), COLORMAPS):
            action.setChecked(cmap == display['cmap'])
        self.view_grid.active_view().set_cmap(display['cmap'])

        self.temp_slider.setRange(display['slider_min'], display['slider_max'])
        self.temp_slider.setAccessibleName(
            f"Maximum {display['label'].lower()} scale, {display['unit']}")
        self.temp_slider.setToolTip(
            f"Adjust the maximum {display['label'].lower()} shown on the color scale")
        self.temp_slider.setValue(display['slider_default'])

        self.view_grid.active_view().set_colorbar_label(f"{display['label']} ({display['unit']})")
        self._apply_contour_overlay_state(self.view_grid.active_cell())  # levels are quantity-specific
        self._apply_link_clim()

    def _extent_for(self, case_index: int, quantity_key) -> tuple:
        """Physical (x0, x1, z0, z1) for a (scenario, quantity), or None if
        unavailable (e.g. a demo-mode store with no real .smv, or a
        genuine parse failure) -- init_plot()/imshow() both accept None
        and fall back to pixel-index axes, so this never has to be a hard
        error for the probe/isotherm features to at least not crash.
        Fixed at cell-view-creation time (M2.6): a slice-type cell's later
        scenario changes don't re-fetch/rebind extent, since matplotlib
        has no in-place way to change an already-plotted image's extent
        without recreating the artist, and every scenario in this dataset
        shares one fixed room footprint (verified in M2.1/M2.3's work) --
        documented simplification, not an oversight, for the case a future
        dataset has per-scenario geometry."""
        try:
            # V6-M1.5: computed quantities share their source's extent (via the
            # provider); native quantities read the store directly.
            if self._is_computed(quantity_key.quantity):
                extent = self.quantity_provider.get_extent(case_index, quantity_key)
            else:
                extent = self.controller.store.get_extent(case_index, quantity_key)
        except Exception as e:  # noqa: BLE001 - geometry is a nice-to-have, never fatal
            logger.warning("could not fetch extent for case=%s key=%s: %s", case_index, quantity_key, e)
            return None
        return tuple(extent) if extent is not None else None

    def _sync_cell_extent(self, cell):
        """Update a cell's plotted extent if its current (case, quantity)
        lives on a different physical plane than what the view was last
        drawn with (M2.2 -- SOOT's doorway slice differs from the standard
        side view). A no-op for every same-plane change (the extents
        compare equal), so scenario switches and TEMPERATURE/VELOCITY/
        SOOT-side toggles cost nothing here."""
        new_extent = self._extent_for(cell.case_index, cell.quantity_key)
        if new_extent != getattr(cell.view, "_extent", None):
            cell.view.set_extent(new_extent)
        cell.view.set_room_outline(self._room_outline_for(cell))

    def _room_outline_for(self, cell):
        """The enclosed room's physical geometry (schematic.py's
        room_overlay_geometry -- walls, door gap, vent states, the sidebar
        diagram's own single source of truth) for a "slice" cell on the
        y-normal plane, so the boundary/door/vents read clearly on the
        heatmap itself rather than relying on however a given quantity's
        colormap happens to render cells near the wall. None for any other
        plane or cell type (those constants aren't validated there, and a
        difference/ensemble cell has no single door/vent state to show --
        same gating principle as everywhere else: never draw a guessed
        outline)."""
        key = cell.quantity_key
        if cell.cell_type != "slice" or key is None or key.direction != 1:
            return None
        entry = next((e for e in (self.sim_data.manifest or []) if e.case_index == cell.case_index), None)
        if entry is None:
            return None
        return room_overlay_geometry(entry.door, entry.vod, entry.voc)

    def _setup_cell_probe_and_isotherms(self, cell):
        """Wire the cursor probe and sync isotherm state (M2.6) -- call
        once per view *instance*, right after its one-time init_plot(),
        not on every scenario/quantity redraw (the callback/levels don't
        need re-registering just because the displayed data changed)."""
        cell.view.enable_probe(lambda x, z, v, c=cell: self._on_cell_probe(c, x, z, v))
        self._apply_contour_overlay_state(cell)
        self._apply_cinematic_state(cell)

    def _on_cell_probe(self, cell, x, z, value):
        """Cursor probe callback (M2.6.1): x/z are physical meters, value
        is the displayed quantity's reading at that point -- shown in the
        status bar rather than a floating tooltip, consistent with how
        _on_time_changed already reports "t = …s" there. Also mirrored,
        large-type, into that cell's own Inspector section (FireLab roadmap
        Phase 3, extended for multi-cell comparison in V6 polish) -- each
        visible cell has its own section, so they no longer fight over one
        shared inspector panel."""
        section_index = self._inspector_section_index_for_cell(cell)
        section = self.inspector_stack.section(section_index) if section_index is not None else None
        if x is None:
            if not self.time_controller.is_playing():
                self.statusBar().showMessage("Ready.")
            if section is not None:
                section.set_probe(None, None, None)
            return
        display = QUANTITY_DISPLAY.get(cell.quantity_key.quantity if cell.quantity_key else None)
        unit = display['unit'] if display else ""
        value_text = f"{value:.1f}{unit}" if value is not None else "—"
        self.statusBar().showMessage(f"x = {x:.3f} m, z = {z:.3f} m, value = {value_text}")
        if section is not None:
            section.set_probe(x, z, value, unit)

    def _apply_contour_overlay_state(self, cell):
        """Sync one cell's contour overlays to their global View-menu
        toggles and quantity-specific levels (config.ISOTHERM_LEVELS) --
        called for every cell whenever either toggle changes, and once for
        each cell right after its view is first initialized. Two
        independent overlays: the cell's own quantity's isotherms/speed-
        bands (drawn on itself), and -- only for a "slice" cell currently
        showing TEMPERATURE -- the opt-in VELOCITY overlay (GUI
        modernization pass, item 6)."""
        quantity = cell.quantity_key.quantity if cell.quantity_key else None
        levels = ISOTHERM_LEVELS.get(quantity, [])
        cell.view.set_isotherm_levels(levels)
        cell.view.set_isotherms_enabled(getattr(self, "_isotherms_enabled", False))

        velocity_overlay_on = getattr(self, "_velocity_overlay_enabled", False)
        applies_here = velocity_overlay_on and cell.cell_type == "slice" and quantity == "TEMPERATURE"
        cell.view.set_velocity_overlay_levels(ISOTHERM_LEVELS.get("VELOCITY", []))
        cell.view.set_velocity_overlay_enabled(applies_here)
        if applies_here and cell.view.heatmap is not None:
            velocity_frame = self._velocity_overlay_frame_for_cell(cell, self.time_controller.index)
            if velocity_frame is not None:
                cell.view.show_frame(cell.view.heatmap.get_array(), velocity_frame=velocity_frame)

    def _velocity_overlay_frame_for_cell(self, cell, index: int):
        """The VELOCITY frame to overlay on `cell` at timeline `index` --
        only ever called for a "slice" cell showing TEMPERATURE. VELOCITY
        shares the exact same plane/extent as TEMPERATURE for every real
        scenario (confirmed directly against the dataset, not assumed:
        identical (n_times, 49, 101) shape and physical extent for both
        quantities at every offset/direction combination), so this always
        aligns pixel-for-pixel with whatever frame the cell's own heatmap
        is already showing. No prefetch orchestration here -- same
        accepted tradeoff as _render_difference_cell's two-scenario fetch:
        a one-time synchronous cost the first time the overlay is toggled
        on for a given cell, not a per-frame cost once the store has it
        cached."""
        store = self._store_for_cell(cell)
        velocity_key = SliceKey("VELOCITY", cell.quantity_key.direction, cell.quantity_key.offset)
        try:
            data = store.get(cell.case_index, velocity_key)
        except Exception as e:  # noqa: BLE001 - overlay is a nice-to-have, must not blank the cell
            logger.warning("velocity overlay: failed to fetch VELOCITY for case %s: %s", cell.case_index, e)
            return None
        idx = min(index, data.shape[0] - 1)
        return data[idx]

    def _init_cell_view(self, cell) -> int:
        """Fetch data for `cell`'s current (case_index, quantity_key) and
        run its SliceView's one-time init_plot() (M2.2.2) -- used both for
        the very first cell (_init_plot below) and for cells created later
        when the grid grows (_on_cell_created). The active cell picks up
        the app's current display settings (colormap/vmax); any other
        cell -- including the very first cell before it's ever been made
        active by anything other than being cell 0 -- defaults from the
        quantity's own QUANTITY_DISPLAY entry, so growing the grid doesn't
        silently copy the active cell's possibly-customized clim onto a
        brand-new cell showing a different scenario."""
        data = self._field(self.controller.store, cell.case_index, cell.quantity_key)
        display = self._display_for(cell.quantity_key.quantity)
        is_active = cell is self.view_grid.active_cell()
        cmap = self.current_colormap if is_active else display['cmap']
        vmax = self.temp_slider.value() if is_active else display['slider_default']
        index = min(self.time_controller.index, data.shape[0] - 1)
        cell.view.init_plot(
            data[index],
            cmap=cmap,
            interpolation=self.current_interpolation,
            vmin=display['vmin'],
            vmax=vmax,
            colorbar_label=f"{display['label']} ({display['unit']})",
            extent=self._extent_for(cell.case_index, cell.quantity_key),
        )
        cell.view.set_room_outline(self._room_outline_for(cell))
        self._setup_cell_probe_and_isotherms(cell)
        return data.shape[0]

    def _init_plot(self):
        # Use the controller's actual default parameters (candles/door/vod/voc
        # as set in __init__) so the first frame shown matches what the
        # control-panel toggles display as selected. ViewGrid always starts
        # with exactly one (active) cell; sync it to the controller's
        # defaults before initializing its plot (M2.2.2's cell_created
        # signal isn't connected yet this early, so this cell is handled
        # directly rather than through _on_cell_created).
        cell = self.view_grid.active_cell()
        cell.set_scenario_silently(self.controller.current_case_index())
        cell.set_quantity_silently(self.current_quantity_key)
        self._current_n_frames = self._init_cell_view(cell)
        self.timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        if getattr(self, "analysis_timeline", None) is not None:
            self.analysis_timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        self.timeline.set_index(0)
        self._update_event_markers()

    def _fire_events_for_case(self, case_index: int) -> list:
        """The scenario's detected fire events (V3-M2, events.py) as
        insight.Insight objects, computed once per scenario from its
        TEMPERATURE field and cached. The fire story is about the fire, so
        it is always temperature-based regardless of the displayed
        quantity."""
        cache = self._fire_events_cache
        if case_index not in cache:
            key = DEFAULT_SLICE_KEY
            fps = self.time_controller.timesteps_per_second
            try:
                data = self.controller.store.get(case_index, key)
                extent = self.controller.store.get_extent(case_index, key)
                table = compute_descriptors(data, extent, fps)
                cache[case_index] = detect_events(table, key.quantity)
            except Exception as e:  # noqa: BLE001 - the story is a nice-to-have, never fatal
                logger.warning("could not compute fire events for case %s: %s", case_index, e)
                return []
        return cache[case_index]

    def _update_event_markers(self) -> None:
        """Event Timeline (V2 M1.3, enriched in V3-M2): auto-detected
        markers on the scrubber for the *active* cell's scenario (the
        scrubber is one shared widget, so it can only ever track one
        cell), now from the events engine (ignition, hazard crossings,
        fastest heating, peak, layer descent, stabilization) rather than
        the summary stats alone. Also feeds every visible cell's own
        Inspector section Fire story list (V6 polish: a 2+ cell comparison
        grid used to only ever populate the active cell's story).
        _fire_events_for_case always reads TEMPERATURE itself (see its own
        docstring) and is cached per case_index, independent of what
        quantity a cell currently displays -- so looping every visible cell
        here costs nothing extra for cells sharing a scenario already seen."""
        active_cell = self.view_grid.active_cell()
        fps = self.time_controller.timesteps_per_second
        has_inspector = getattr(self, "inspector_stack", None) is not None
        if has_inspector:
            self.inspector_stack.ensure_count(len(self.view_grid.visible_cells()))
        for i, cell in enumerate(self.view_grid.visible_cells()):
            show = bool(self.sim_data.manifest and cell.cell_type == "slice"
                        and cell.quantity_key is not None)
            events = self._fire_events_for_case(cell.case_index) if show else []
            if cell is active_cell:
                markers = [(ev.frame_index(fps), ev.statement) for ev in events
                           if ev.frame_index(fps) is not None]
                self.timeline.set_event_markers(markers)
            # The inspector is built after the plot panel (which triggers
            # the first marker update), so guard its first call.
            if has_inspector:
                self.inspector_stack.section(i).set_story(events, fps)

    def _on_cell_created(self, cell):
        """A new grid cell was instantiated because the grid grew (M2.2.2).
        Give it real data before it's ever shown -- SliceView.show_frame()
        assumes init_plot() already ran."""
        self._init_cell_view(cell)

    def _redraw(self, frame):
        """Per-frame playback path for the *active* cell specifically:
        SliceView.show_frame() blits instead of a full draw_idle() so only
        the image artist's pixels are re-rendered (M1.3.3). Anything that
        changes what's underneath the image (clim/colormap/theme/
        interpolation/resize) goes through the dedicated setters below,
        which do a full draw + recapture instead of calling this. Other
        grid cells are redrawn directly in _on_time_changed, which loops
        every visible cell rather than just the active one."""
        self.view_grid.active_view().show_frame(frame)

    # -------------------------------------------------------- signal slots
    def _store_for_cell(self, cell):
        """M3.2.5: a cell normally reads the one real ScenarioStore;
        store_override (set only by _open_browser_model_eval) routes a
        "slice" cell's data through self.prediction_store instead, so the
        model-evaluation grid reuses the exact same rendering path as
        every other cell rather than a parallel display system."""
        return cell.store_override if cell.store_override is not None else self.controller.store

    def _next_frame_for_slice_cell(self, cell, index: int):
        """Frame at index+1 for a "slice" cell, or None past the end of
        the series -- only used for cinematic mode's sub-frame
        interpolation (FireLab roadmap Phase 2.1d), which needs a
        lookahead endpoint to blend toward. Free: store.get() already
        returns the whole cached array, this is just one more index into
        it, not an extra load."""
        store = self._store_for_cell(cell)
        data = store.get(cell.case_index, cell.quantity_key)
        nxt = index + 1
        return data[nxt] if nxt < data.shape[0] else None

    def _hrr_intensity_for_cell(self, cell, index: int) -> float:
        """cell's current HRR(t) normalized to that scenario's own peak
        (FireLab roadmap Phase 2.1c) -- 1.0 (neutral) if there's no
        manifest entry, no *_hrr.csv, or a zero peak, so demo-data mode
        and any scenario missing the CSV still renders bloom/flicker, just
        without the extra data-driven modulation."""
        if not self.sim_data.manifest:
            return 1.0
        cached = self._hrr_cache.get(cell.case_index)
        if cached is None:
            entry = next((e for e in self.sim_data.manifest if e.case_index == cell.case_index), None)
            hrr_data = _read_hrr_csv(entry.path) if entry else None
            cached = hrr_data if hrr_data is not None else ()
            self._hrr_cache[cell.case_index] = cached
        if cached == ():
            return 1.0
        times, hrr_kw = cached
        peak = float(np.max(hrr_kw)) if len(hrr_kw) else 0.0
        if peak <= 0:
            return 1.0
        t_now = index / self.time_controller.timesteps_per_second
        current = float(np.interp(t_now, times, hrr_kw))
        return max(0.15, min(1.5, current / peak))

    def _frame_for_cell(self, cell, index: int):
        """The frame `cell` should show at timeline `index`, dispatched by
        cell_type (M2.3.3): a plain slice, an A-B difference, or an
        ensemble composite. Returns None for an ensemble cell with no
        scenarios selected yet (nothing to show)."""
        if cell.cell_type == "slice":
            store = self._store_for_cell(cell)
            return self._field(store, cell.case_index, cell.quantity_key)[index]
        if cell.cell_type == "difference":
            store_b = cell.store_override_b if cell.store_override_b is not None else self.controller.store
            data_a = self.controller.store.get(cell.case_index_a, cell.quantity_key)
            data_b = store_b.get(cell.case_index_b, cell.quantity_key)
            idx = min(index, data_a.shape[0] - 1, data_b.shape[0] - 1)
            return DifferenceView.compute_diff(data_a, data_b, idx)
        if cell.cell_type == "ensemble":
            if not cell.ensemble_case_indices:
                return None
            arrays = [self.controller.store.get(ci, cell.quantity_key) for ci in cell.ensemble_case_indices]
            idx = min(index, min(a.shape[0] for a in arrays) - 1)
            return EnsembleView.compute_composite(arrays, idx, cell.ensemble_stat)
        return None

    def _device_markers_for(self, case_index: int, index: int) -> list:
        """(x, z, color) for every V6-M2 device placed on `case_index`, at
        already-cached frame `index` -- Device.state_at() is a plain index
        into results computed once at placement/edit, never a recompute."""
        panel = getattr(self, "device_panel", None)
        if panel is None:
            return []
        markers = []
        for d in panel._devices:
            if d.scenario != case_index or d.results is None:
                continue
            if d.type == "thermocouple":
                color = "#3DA5FF"
            else:
                color = "#FF5252" if d.state_at(index).get("active") else "#3DA5FF"
            markers.append((d.position[0], d.position[1], color))
        return markers

    def _refresh_device_markers(self) -> None:
        """Re-draw every visible slice cell's device markers at the current
        time -- called when a device is placed/edited/deleted, so the Live
        Viewer reflects it immediately without waiting for the next tick."""
        index = self.time_controller.index
        for cell in self.view_grid.visible_cells():
            if cell.cell_type == "slice":
                cell.view.set_device_markers(self._device_markers_for(cell.case_index, index))
                cell.view.redraw_overlays_now()

    def _vector_field_for(self, case_index: int, index: int):
        """(quiver, streamlines, colors) for `case_index` at already-cached
        frame `index` -- VectorField.quiver_at()/streamline_at() are O(1)/
        memoized (see velocity.py), never a recompute here. Empty/None when
        no VectorField has been successfully computed for this scenario
        (gated, or no panel/probes yet)."""
        panel = getattr(self, "velocity_panel", None)
        if panel is None:
            return None, [], []
        field = panel._fields.get(case_index)
        if field is None:
            return None, [], []
        quiver = None
        if panel.mode in ("quiver", "both"):
            quiver = field.quiver_at(index, density=panel.density_spin.value())
        streamlines, colors = [], []
        if panel.mode in ("streamlines", "both"):
            for p in panel._probes:
                if p.scenario != case_index or p.gated or not p.results:
                    continue
                pts = field.streamline_at(p.position, index, max_length=panel.length_spin.value())
                if len(pts) < 2:
                    continue
                streamlines.append(pts)
                # "activation-state-like" coloring: current speed vs. this
                # probe's own historical mean -- a real, derived state, not
                # a fabricated threshold.
                speed_now = p.state_at(index).get("speed_m_s")
                mean_speed = float(np.mean(p.results["speed_m_s"]))
                fast = speed_now is not None and speed_now > mean_speed
                colors.append("#FF5252" if fast else "#3DA5FF")
        return quiver, streamlines, colors

    def _refresh_vector_field(self) -> None:
        """Re-draw every visible slice cell's quiver/streamlines at the
        current time -- called when a probe is placed/renamed/deleted, so
        the Live Viewer reflects it immediately without waiting for the
        next tick."""
        index = self.time_controller.index
        for cell in self.view_grid.visible_cells():
            if cell.cell_type == "slice":
                quiver, streamlines, colors = self._vector_field_for(cell.case_index, index)
                cell.view.set_streamline_colors(colors)
                cell.view.set_vector_field(quiver=quiver, streamlines=streamlines)
                cell.view.redraw_overlays_now()

    def _on_time_changed(self, index: int):
        """TimeController's tick/seek signal (M1.4.1): pull the frame for
        *every visible grid cell* at `index` and redraw each (M2.2.3 --
        the pull model makes this a plain loop, no per-cell timers needed).
        Relies on M1.2's disk cache making an already-warm store.get()
        ~1-6ms -- cheap enough to call directly here without stalling the
        GUI thread on every tick, for however many cells are visible."""
        frames = {}
        for cell in self.view_grid.visible_cells():
            try:
                frame = self._frame_for_cell(cell, index)
            except Exception as e:  # noqa: BLE001 - one bad cell must not blank the rest
                logger.warning("failed to fetch frame for grid cell (type=%s): %s", cell.cell_type, e)
                continue
            frames[id(cell)] = frame
            if frame is not None:
                cinematic = cell.cell_type == "slice" and getattr(cell.view, "cinematic_enabled", False)
                extra = {}
                if cinematic:
                    extra["next_frame"] = self._next_frame_for_slice_cell(cell, index)
                    extra["bloom_intensity"] = self._hrr_intensity_for_cell(cell, index)
                # Cinematic mode's smoke layer (Tier 2, FireLab roadmap
                # Phase 2.1f) wants VELOCITY data every tick regardless of
                # whether the separate contour-overlay checkbox is on.
                if cell.cell_type == "slice":
                    # markers/vectors must be set *before* show_frame's blit
                    # paints this tick's animated artists (V6-M2/V6-M3) --
                    # otherwise the new state wouldn't appear until the next
                    # tick (see the V6-M2 ordering bug fixed for markers).
                    cell.view.set_device_markers(self._device_markers_for(cell.case_index, index))
                    quiver, streamlines, colors = self._vector_field_for(cell.case_index, index)
                    cell.view.set_streamline_colors(colors)
                    cell.view.set_vector_field(quiver=quiver, streamlines=streamlines)
                if cell.view.velocity_overlay_enabled or cinematic:
                    velocity_frame = self._velocity_overlay_frame_for_cell(cell, index)
                    cell.view.show_frame(frame, velocity_frame=velocity_frame, **extra)
                else:
                    cell.view.show_frame(frame, **extra)
        self.timeline.set_index(index)
        if getattr(self, "analysis_timeline", None) is not None:
            self.analysis_timeline.set_index(index)
        current_time = index / self.time_controller.timesteps_per_second
        self.statusBar().showMessage(f"t = {current_time:.1f} s", 2000)
        self._update_inspector(index, frames)
        # RC polish: broadcast the playback time so every bus-bound analysis
        # panel evolves in lockstep with the Live Viewer (both seeks and
        # playback ticks pass through here). Guarded against the echo of a
        # seek that itself originated from the bus.
        if getattr(self, "selection_bus", None) is not None and not self._seek_from_bus:
            self.selection_bus.update(origin=self, time_s=current_time)

    def _scenario_label(self, case_index: int) -> str:
        """A human-readable scenario identity (manifest.scenario_label),
        e.g. for the grid cell's own scenario combo, the Inspector's
        static-info line, and the difference-over-time dialog's legend --
        not the raw "c1_d0_vod0_voc0" disk folder name."""
        for entry in (self.sim_data.manifest or []):
            if entry.case_index == case_index:
                return manifest_mod.scenario_label(entry)
        return f"scenario {case_index}"

    def _slice_location_label(self, slice_key) -> str:
        axis = _AXIS_NAMES.get(slice_key.direction, f"axis {slice_key.direction}")
        return f"{axis}-normal, offset {slice_key.offset}"

    def _update_inspector(self, index: int, frames: dict = None) -> None:
        """Keeps every visible cell's own Inspector section in sync (V6
        polish: a 2+ cell comparison grid used to only ever show the
        active cell's stats -- now each cell gets its own full section,
        stacked in the InspectorStack). Static metadata (scenario/quantity/
        grid size/slice/duration/frames) is only rebuilt inside each
        section's own `key != self._inspector_series_key[i]` guard below --
        i.e. on a scenario/quantity/cell-type change for THAT cell, never
        per tick. `frames` is `{id(cell): frame}`, the same arrays
        _on_time_changed's own loop already computed via _frame_for_cell
        for this exact index -- reused here for the dynamic min/max readout
        instead of re-fetching. The peak-temperature sparkline/HRR gauge
        are always about TEMPERATURE and the scenario's own *_hrr.csv
        respectively -- neither depends on which quantity a cell currently
        displays, so both stay populated for any "slice" cell regardless of
        its displayed quantity (re-reading TEMPERATURE separately only when
        it isn't already the displayed quantity). Difference/ensemble cells
        leave those two neutral -- there's no single scenario to narrate."""
        frames = frames or {}
        cells = self.view_grid.visible_cells()
        self.inspector_stack.ensure_count(len(cells))
        while len(self._inspector_series_key) < len(cells):
            self._inspector_series_key.append(None)

        for i, cell in enumerate(cells):
            section = self.inspector_stack.section(i)
            frame = frames.get(id(cell))
            if cell is None or cell.cell_type not in ("slice", "difference") or not cell.quantity_key:
                section.clear()
                section.clear_difference_stats()
                if self._inspector_series_key[i] is not None:
                    section.set_static_info("—", "—", "—", "—", 0, 1.0)
                section.set_time(index)
                self._inspector_series_key[i] = None
                continue

            quantity = cell.quantity_key.quantity
            display = self._display_for(quantity)
            if cell.cell_type == "slice":
                key = ("slice", cell.case_index, cell.quantity_key)
                scenario_label = self._scenario_label(cell.case_index)
            else:
                key = ("difference", cell.case_index_a, cell.case_index_b, cell.quantity_key)
                scenario_label = (f"{self._scenario_label(cell.case_index_a)} − "
                                 f"{self._scenario_label(cell.case_index_b)}")

            if key != self._inspector_series_key[i]:
                self._inspector_series_key[i] = key
                if cell.cell_type == "slice":
                    ref_data = self._field(self._store_for_cell(cell), cell.case_index, cell.quantity_key)
                else:
                    ref_data = self._field(self.controller.store, cell.case_index_a, cell.quantity_key)
                section.set_static_info(
                    scenario_label, display.get('label', quantity.title()),
                    f"{ref_data.shape[1]} × {ref_data.shape[2]}",
                    self._slice_location_label(cell.quantity_key),
                    ref_data.shape[0], self.time_controller.timesteps_per_second,
                )
                if cell.cell_type == "slice":
                    if quantity == "TEMPERATURE":
                        temp_data = ref_data
                    else:
                        try:
                            temp_data = self._field(self._store_for_cell(cell), cell.case_index, DEFAULT_SLICE_KEY)
                        except Exception as e:  # noqa: BLE001 - sparkline context is a nice-to-have, never fatal
                            logger.warning("could not read TEMPERATURE for inspector context (case %s): %s",
                                           cell.case_index, e)
                            temp_data = None
                    if temp_data is not None:
                        peak_by_frame = temp_data.reshape(temp_data.shape[0], -1).max(axis=1).tolist()
                        door_wide_open = self.controller.params.door == 1
                        section.set_scenario(peak_by_frame, QUANTITY_DISPLAY["TEMPERATURE"]["vmin"], door_wide_open)
                    else:
                        section.clear()
                else:
                    section.clear()

            hrr_fraction = None
            if cell.cell_type == "slice":
                intensity = self._hrr_intensity_for_cell(cell, index)
                cached_hrr = self._hrr_cache.get(cell.case_index)
                hrr_fraction = min(1.0, intensity) if cached_hrr else None

            frame_min = frame_max = None
            if frame is not None:
                frame_min, frame_max = float(np.min(frame)), float(np.max(frame))
            section.set_time(index, hrr_fraction, frame_min, frame_max, display.get('unit', ''))

            if cell.cell_type == "difference" and frame is not None:
                mean_v = float(np.mean(frame))
                rms_v = float(np.sqrt(np.mean(np.square(frame))))
                section.set_difference_stats(frame_min, frame_max, mean_v, rms_v, display.get('unit', ''))
            else:
                section.clear_difference_stats()

    def _on_playing_changed(self, playing: bool):
        self.timeline.set_playing(playing)
        if getattr(self, "analysis_timeline", None) is not None:
            self.analysis_timeline.set_playing(playing)
        self.start_button.setEnabled(not playing)
        self.stop_button.setEnabled(playing)

    def _on_seek_requested(self, index: int):
        # The time is broadcast to the bus from _on_time_changed (the single
        # per-frame hook), so seeks and playback ticks are handled uniformly.
        self.time_controller.seek(index)

    def _on_sim_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Simulation error", message)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    def _on_prefetch_finished(self, case_idx: int):
        """A background scenario load completed (M1.4.4). If the user has
        since switched to a different combination, this is stale -- ignore
        it silently, the busy state stays active for whichever request is
        still pending (or was already cleared by a cache hit).

        M0.1: prefetch_finished carries only case_idx, but two prefetches
        for the *same* case at different quantity keys can be in flight at
        once (a scenario toggle then a quantity switch that land on the
        same case). Gate the busy-end on the pending *key* actually being
        cached now, so an earlier key's completion can't end the busy
        state while the user is still waiting on the current key -- rather
        than widening this signal to carry the key (a
        simulation_controller threading change, deliberately avoided)."""
        if case_idx != self._pending_load_case:
            return
        if not self.controller.is_cached(case_idx, self._pending_load_key):
            return  # a different key on this case finished; keep waiting for the pending one
        self._pending_load_case = None
        self._pending_load_key = None
        self._end_busy_state()
        self._sync_current_scenario(case_idx)
        if self._was_playing_before_load:
            self.time_controller.play()

    def _on_prefetch_error(self, case_idx: int, message: str):
        """A background scenario load failed (M1.4.4). Same staleness guard
        as _on_prefetch_finished: if a newer request has since superseded
        this one, an older/now-irrelevant failure must not clear the busy
        state or _pending_load_case out from under the still-in-flight
        newer request -- doing so would (a) prematurely show the UI as
        "ready" while a real load is still running, and (b) cause that
        newer request's eventual success to be silently discarded, since
        _on_prefetch_finished's own guard would then compare against a
        _pending_load_case that was cleared by this unrelated failure."""
        if case_idx != self._pending_load_case:
            return
        # M0.1: if the pending key is already cached, the current request
        # actually succeeded and this is a late error for a superseded key
        # on the same case -- ignore it rather than blanking a good load.
        if self.controller.is_cached(case_idx, self._pending_load_key):
            return
        self._pending_load_case = None
        self._pending_load_key = None
        self._end_busy_state()
        self._on_sim_error(message)

    def _on_candle_changed(self, value):
        self.controller.set_candles(value)
        self._on_scenario_param_changed()
        self._refresh_schematic()

    def _on_vod_changed(self, value):
        self.controller.set_vod(value)
        self._on_scenario_param_changed()
        self._refresh_schematic()

    def _on_voc_changed(self, value):
        self.controller.set_voc(value)
        self._on_scenario_param_changed()
        self._refresh_schematic()

    def _on_door_changed(self, value):
        self.controller.set_door(value)
        self._on_scenario_param_changed()
        self._refresh_schematic()

    def _refresh_schematic(self):
        """The schematic mirrors the controller's own params -- no scenario
        state is duplicated here, this just repaints from the same source
        of truth the heatmap already reads."""
        p = self.controller.params
        self.schematic.update_state(p.candles, p.door, p.vod, p.voc)

    def _on_scenario_param_changed(self):
        """A scenario-defining toggle changed (M1.4.4). Ensures the new
        scenario is displayed without ever blocking the GUI thread on a
        cold parse, whether paused or mid-playback: a cache hit updates
        immediately; a cache miss pauses the timeline, shows a busy cursor
        and status message, and prefetches on a background thread, resuming
        playback (if it was playing) once the prefetch completes.

        `_pending_load_case` always tracks the *latest requested* scenario,
        so rapid re-toggling while a prefetch is in flight is safe: only the
        prefetch matching the current selection is allowed to end the busy
        state (see _on_prefetch_finished) -- an earlier, now-superseded
        prefetch simply finishes in the background and is ignored.

        Fixed in M0.1: `_pending_load_case` is now paired with
        `_pending_load_key`, and _on_prefetch_finished only ends the busy
        state once *that key* is cached -- so a scenario toggle and a
        quantity switch landing on the same case no longer race.
        """
        case_idx = self.controller.current_case_index()
        self.view_grid.active_cell().set_scenario_silently(case_idx)
        if self.controller.is_cached(case_idx, self.current_quantity_key):
            self._pending_load_case = None
            self._pending_load_key = None
            if self._busy:
                self._end_busy_state()
            self._sync_current_scenario(case_idx)
            return

        self._pending_load_case = case_idx
        self._pending_load_key = self.current_quantity_key
        if not self._busy:
            self._begin_busy_state()
        self.controller.prefetch(case_idx, self.current_quantity_key)

    def _begin_busy_state(self):
        self._busy = True
        self._was_playing_before_load = self.time_controller.is_playing()
        self.time_controller.pause()
        self.timeline.slider.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage("Loading scenario…")

    def _end_busy_state(self):
        self._busy = False
        QtWidgets.QApplication.restoreOverrideCursor()
        self.timeline.slider.setEnabled(True)
        self.statusBar().showMessage("Ready.", 2000)

    def _sync_current_scenario(self, case_idx: int):
        """Refresh frame-count/timeline range for the now-confirmed-loaded
        scenario at case_idx, and redraw the currently displayed frame
        index against it. Only redraws (doesn't touch play state) -- the
        caller decides whether to resume playback."""
        self._current_n_frames = self._field(
            self.controller.store, case_idx, self.current_quantity_key).shape[0]
        self.timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        if getattr(self, "analysis_timeline", None) is not None:
            self.analysis_timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        self._update_event_markers()
        active = self.view_grid.active_cell()
        if active.cell_type == "slice":
            self._sync_cell_extent(active)  # M2.2: a SOOT-plane switch may change the extent
        if not self.time_controller.is_playing():
            self._on_time_changed(self.time_controller.index)

    def _on_quantity_changed(self, combo_index: int):
        """Quantity combo box changed (M2.1). Mirrors
        _on_scenario_param_changed's cache-hit/cache-miss handling, but for
        the quantity axis instead of a candles/door/vod/voc toggle -- same
        busy-cursor/prefetch machinery, now keyed by SliceKey too."""
        info = self._combo_quantity_list[combo_index]
        if info.key == self.current_quantity_key:
            return
        self.current_quantity_key = info.key
        self.view_grid.active_cell().set_quantity_silently(info.key)
        self._apply_quantity_display_defaults(info.key.quantity)

        case_idx = self.controller.current_case_index()
        # V6-M1.5: a computed field is not in the store's cache and has no
        # prefetch -- its source is already cached and the expression is cheap,
        # so render it directly, bypassing the store prefetch/busy machinery.
        if self._is_computed(info.key.quantity):
            self._sync_current_scenario(case_idx)
            return
        if self.controller.is_cached(case_idx, self.current_quantity_key):
            self._pending_load_case = None
            self._pending_load_key = None
            if self._busy:
                self._end_busy_state()
            self._sync_current_scenario(case_idx)
            return

        self._pending_load_case = case_idx
        self._pending_load_key = self.current_quantity_key
        if not self._busy:
            self._begin_busy_state()
        self.controller.prefetch(case_idx, self.current_quantity_key)

    # ------------------------------------------------- grid cells (M2.2.2)
    def _set_grid_layout(self, layout_name: str):
        """View -> Grid Layout menu. Growing the grid creates new cells
        (each initialized via _on_cell_created); shrinking just hides
        cells, preserving their state for when the layout grows again."""
        self.view_grid.set_layout(layout_name)
        # Pan/zoom is ambiguous across multiple cells and
        # NavigationToolbar2QT isn't meant to be rebound at runtime -- see
        # _build_plot_panel's comment. Only meaningful (and only shown) in
        # single-view mode.
        self.toolbar.setVisible(layout_name == "1x1")
        if layout_name == "2x1" and not getattr(self, "_link_clim", False):
            # The stacked layout exists specifically to compare two
            # scenarios directly, one above the other -- unlinked color
            # scales would let the same value read as two different colors
            # between them, defeating that. Auto-enable "Link color
            # scales" (still a normal, user-toggleable checkbox afterward).
            self._link_clim = True
            if hasattr(self, "link_clim_action"):
                self.link_clim_action.setChecked(True)
        self._apply_link_clim()
        # A newly-grown cell's Inspector section starts with an empty Fire
        # story (_on_cell_created only runs _init_cell_view) -- this is the
        # one place that reliably fires whenever visible cell count changes,
        # so it's where every section's story gets (re)populated.
        self._update_event_markers()
        if not self.time_controller.is_playing():
            self._on_time_changed(self.time_controller.index)

    # ------------------------------------------------- Compare page presets
    def _find_scenario(self, factor: str, value: int):
        """case_index of the manifest entry matching `factor=value` with
        every other factor held at its config.py DEFAULT_*, or None if no
        such scenario exists."""
        target = {"candles": DEFAULT_CANDLES, "door": DEFAULT_DOOR, "vod": DEFAULT_VOD, "voc": DEFAULT_VOC}
        target[factor] = value
        for entry in self.sim_data.manifest or []:
            if all(getattr(entry, f) == v for f, v in target.items()):
                return entry.case_index
        return None

    def _find_quantity_key(self, quantity_name: str, direction: int = None, offset: int = None):
        """The SliceKey for `quantity_name`, preferring an exact
        (direction, offset) match (V6-M5: a quantity can now appear on more
        than one plane, e.g. a second Y-offset) and falling back to the
        first name-only match -- so older session files (no plane fields)
        keep restoring exactly as before."""
        options = self._quantity_options()
        if direction is not None and offset is not None:
            exact = next((key for _label, key in options if key.quantity == quantity_name
                         and key.direction == direction and key.offset == offset), None)
            if exact is not None:
                return exact
        return next((key for _label, key in options if key.quantity == quantity_name), None)

    def _select_scenario_in_cell(self, cell, case_index: int) -> None:
        options = self._scenario_options()
        idx = next((i for i, (_label, ci) in enumerate(options) if ci == case_index), None)
        combo = getattr(cell, "scenario_combo", None)
        if idx is not None and combo is not None:
            combo.setCurrentIndex(idx)

    def _select_difference_scenarios_in_cell(self, cell, case_a: int, case_b: int) -> None:
        options = self._scenario_options()
        idx_a = next((i for i, (_l, ci) in enumerate(options) if ci == case_a), None)
        idx_b = next((i for i, (_l, ci) in enumerate(options) if ci == case_b), None)
        if idx_a is not None:
            cell.scenario_combo_a.setCurrentIndex(idx_a)
        if idx_b is not None:
            cell.scenario_combo_b.setCurrentIndex(idx_b)

    def _select_quantity_in_cell(self, cell, quantity_key) -> None:
        options = self._quantity_options()
        idx = next((i for i, (_label, key) in enumerate(options) if key == quantity_key), None)
        combo = getattr(cell, "quantity_combo", None)
        if idx is not None and combo is not None:
            combo.setCurrentIndex(idx)

    def _apply_compare_preset(self, key: str) -> None:
        """Compare page: jumps into the Live page's own grid, stacked
        (2x1) with scenario A above scenario B, both shown as plain slices
        of the preset's quantity -- switching to "2x1" auto-enables "Link
        color scales" (see _set_grid_layout), so equal values render as
        the same color between the two, directly comparable at a glance.
        Reuses the exact combo-driven signal chain a user's own clicks
        already go through, not a second rendering path. (A computed A-B
        difference is still available by right-clicking a cell -- this
        preset is about direct visual comparison, not a delta.)"""
        preset = _COMPARE_PRESETS.get(key)
        if preset is None or not self.sim_data.manifest:
            return
        quantity_key = self._find_quantity_key(preset["quantity"])
        case_a = self._find_scenario(preset["factor"], preset["values"][0])
        case_b = self._find_scenario(preset["factor"], preset["values"][1])
        if quantity_key is None or case_a is None or case_b is None:
            return

        # Guards _navigate_to()'s Compare -> Home/Live grid reset (bugfix
        # below) from firing on *this* internal jump to Live -- that reset
        # is for a user manually leaving Compare later, not for the preset
        # setting up its own comparison grid.
        self._applying_compare_preset = True
        try:
            self._navigate_to("live")
            self._set_grid_layout("2x1")
            cells = self.view_grid.visible_cells()
            if len(cells) < 2:
                return
            cell_a, cell_b = cells[0], cells[1]

            if cell_a.cell_type != "slice":
                cell_a.set_cell_type("slice")
            self._select_scenario_in_cell(cell_a, case_a)
            self._select_quantity_in_cell(cell_a, quantity_key)

            if cell_b.cell_type != "slice":
                cell_b.set_cell_type("slice")
            self._select_scenario_in_cell(cell_b, case_b)
            self._select_quantity_in_cell(cell_b, quantity_key)
            self._apply_link_clim()
            self._compare_active = True
        finally:
            self._applying_compare_preset = False

    def _set_link_clim(self, checked: bool):
        """View -> Link color scales toggle (M2.2.3)."""
        self._link_clim = checked
        self._apply_link_clim()

    def _set_isotherms_enabled(self, checked: bool):
        """View -> Isotherm overlay toggle (M2.6.2). Applies to every
        visible cell, not just the active one -- unlike clim/colormap
        (M2.2's "active cell only" design decision), an isotherm overlay
        is a read-aid a user comparing several cells would want
        consistently applied across all of them, not a per-cell data-scale
        choice tied to one cell's own range."""
        self._isotherms_enabled = checked
        for cell in self.view_grid.visible_cells():
            self._apply_contour_overlay_state(cell)
        if not self.time_controller.is_playing():
            self._on_time_changed(self.time_controller.index)

    def _set_velocity_overlay_enabled(self, checked: bool):
        """View -> Show velocity overlay toggle (GUI modernization pass,
        item 6). Same "every visible cell, not just the active one" reach
        as the isotherm toggle, for the same reason -- but only actually
        applies to cells currently showing TEMPERATURE as a plain "slice"
        (see _apply_contour_overlay_state); a cell already showing
        VELOCITY, or a difference/ensemble cell, is a no-op, not an error."""
        self._velocity_overlay_enabled = checked
        for cell in self.view_grid.visible_cells():
            self._apply_contour_overlay_state(cell)
        if not self.time_controller.is_playing():
            self._on_time_changed(self.time_controller.index)

    def _set_cinematic_enabled(self, checked: bool):
        """View -> Cinematic fire view toggle (FireLab roadmap Phase 2.1).
        Grid-wide like the isotherm/velocity-overlay toggles, but only
        actually applies to "slice" cells currently showing TEMPERATURE --
        the FireLUT is calibrated to that quantity's hazard-band window
        (see cinema/luts.py); other cells are left in science mode."""
        self._cinematic_enabled = checked
        for cell in self.view_grid.visible_cells():
            self._apply_cinematic_state(cell)
        if not self.time_controller.is_playing():
            self._on_time_changed(self.time_controller.index)

    def _apply_cinematic_state(self, cell):
        """Sync one cell's cinematic-mode flag to the global toggle --
        called for every cell whenever the toggle changes, and once for
        each cell right after its view is first initialized (mirrors
        _apply_contour_overlay_state's pattern)."""
        if not hasattr(cell.view, "set_cinematic_mode"):
            return  # DifferenceView/EnsembleView: cinematic mode is slice-only for now
        quantity = cell.quantity_key.quantity if cell.quantity_key else None
        applies_here = (
            getattr(self, "_cinematic_enabled", False)
            and cell.cell_type == "slice"
            and quantity == "TEMPERATURE"
        )
        if applies_here and not cell.view.cinematic_enabled:
            display = QUANTITY_DISPLAY["TEMPERATURE"]
            is_active = cell is self.view_grid.active_cell()
            vmax_init = self.temp_slider.value() if is_active else display["slider_default"]
            cell.view.set_cinematic_mode(True, vmin=display["vmin"], vmax_init=vmax_init)
        elif not applies_here and cell.view.cinematic_enabled:
            cell.view.set_cinematic_mode(False)

    def _apply_link_clim(self):
        """When linked, every visible *slice-type* cell showing the same
        quantity as at least one other visible slice-type cell shares that
        quantity's data max as a common vmax (mixing vmax across different
        quantities wouldn't be physically meaningful, since they're
        different units) -- computed fresh each time this is called, not
        cached, since it only runs on discrete structural changes (layout/
        scenario/quantity changes), not every playback tick. A no-op if
        linking is off or nothing's visible.

        Difference/ensemble cells (M2.3.3) are deliberately excluded: their
        own clim conventions (symmetric-around-zero for a diff, sigma-scale
        for an ensemble std) don't share "biggest absolute value across
        cells" as a meaningful notion the way two slice-type cells of the
        same quantity do -- linking was designed and asked for in the
        context of comparing scenarios of one quantity, not comparing a
        diff/ensemble cell's fundamentally different scale to a plain
        slice's."""
        if not getattr(self, "_link_clim", False):
            return
        cells = [c for c in self.view_grid.visible_cells() if c.cell_type == "slice"]
        if not cells:
            return
        by_quantity = {}
        for cell in cells:
            by_quantity.setdefault(cell.quantity_key.quantity, []).append(cell)
        for quantity, group in by_quantity.items():
            vmax = max(float(np.max(self.controller.store.get(c.case_index, c.quantity_key))) for c in group)
            vmin = QUANTITY_DISPLAY[quantity]['vmin']
            for c in group:
                c.view.set_clim(vmin, vmax)

    def _on_active_cell_changed(self, cell):
        """A different grid cell became active (user clicked it). The
        control panel now edits *this* cell's scenario/quantity/display
        settings, so pull them from the cell instead of pushing
        MainWindow's previous state onto it."""
        entry = next((e for e in (self.sim_data.manifest or []) if e.case_index == cell.case_index), None)
        if entry is not None:
            self.controller.set_candles(entry.candles)
            self.controller.set_door(entry.door)
            self.controller.set_vod(entry.vod)
            self.controller.set_voc(entry.voc)
            self.candle_toggle.set_value(entry.candles)
            self.door_toggle.set_value(entry.door)
            self.vod_toggle.set_value(entry.vod)
            self.voc_toggle.set_value(entry.voc)
            self._refresh_schematic()

        self.current_quantity_key = cell.quantity_key
        quantity_idx = next((i for i, info in enumerate(self.quantity_infos) if info.key == cell.quantity_key), None)
        if quantity_idx is not None:
            self.quantity_combo.blockSignals(True)
            self.quantity_combo.setCurrentIndex(quantity_idx)
            self.quantity_combo.blockSignals(False)

        # Pull this cell's own last-set colormap/clim into the shared
        # controls rather than pushing MainWindow's previous state onto
        # it -- matches "other cells keep their own last-set clim/colormap
        # independently" when unlinked (M2.2 design decision). A freshly
        # type-switched ensemble cell with no scenarios picked yet (stays
        # blank until the picker dialog runs -- see _on_cell_type_changed)
        # has never rendered a frame, so its heatmap is still None; there's
        # nothing to pull in that case, so the shared controls are simply
        # left as they are rather than crashing on a bare cell that has no
        # colormap/clim of its own yet.
        if cell.view.heatmap is not None:
            cmap_name = cell.view.heatmap.get_cmap().name
            self.current_colormap = cmap_name
            self.settings.setValue("colormap", cmap_name)
            for action, (_, cmap) in zip(self.colormap_action_group.actions(), COLORMAPS):
                action.setChecked(cmap == cmap_name)

            display = self._display_for(cell.quantity_key.quantity)
            _vmin, vmax = cell.view.heatmap.get_clim()
            self.temp_slider.blockSignals(True)
            self.temp_slider.setRange(display['slider_min'], display['slider_max'])
            self.temp_slider.setValue(int(vmax))
            self.temp_slider.blockSignals(False)
            self.temp_slider.setAccessibleName(f"Maximum {display['label'].lower()} scale, {display['unit']}")
            self.temp_slider.setToolTip(f"Adjust the maximum {display['label'].lower()} shown on the color scale")
            self.temp_label.setText(f"{int(vmax)} {display['unit']}")

        self._current_n_frames = self._field(self.controller.store, cell.case_index, cell.quantity_key).shape[0]
        self.timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        if getattr(self, "analysis_timeline", None) is not None:
            self.analysis_timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        self._update_event_markers()

    def _apply_manifest_case_to_controller(self, case_index: int):
        """The active cell's own scenario combo picked a different
        scenario -- equivalent to using the control-panel toggles, so route
        through the same controller state + existing cache-hit/miss path
        (_on_scenario_param_changed) rather than duplicating it."""
        entry = next((e for e in (self.sim_data.manifest or []) if e.case_index == case_index), None)
        if entry is None:
            return
        self.controller.set_candles(entry.candles)
        self.controller.set_door(entry.door)
        self.controller.set_vod(entry.vod)
        self.controller.set_voc(entry.voc)
        self.candle_toggle.set_value(entry.candles)
        self.door_toggle.set_value(entry.door)
        self.vod_toggle.set_value(entry.vod)
        self.voc_toggle.set_value(entry.voc)
        self._on_scenario_param_changed()
        self._refresh_schematic()

    def _open_browser_scenario(self, case_index: int):
        """Double-click in the experiment browser, or a PCA-scatter click in
        Ensemble analytics (Analysis-improvement roadmap Phase A -- that
        panel had zero SelectionBus wiring, a dead-end scatter plot): load
        into the active cell AND publish to the bus, so every other panel
        follows too, not just the Live Viewer."""
        cell = self.view_grid.active_cell()
        if cell.cell_type != "slice":
            cell.set_cell_type("slice")
        self._apply_manifest_case_to_controller(case_index)
        if getattr(self, "selection_bus", None) is not None:
            self.selection_bus.update(origin=self, scenario=case_index)

    def _open_browser_grid(self, case_indices: list):
        """Open up to nine browser-selected scenarios in the grid."""
        if not case_indices:
            return
        selected = case_indices[:9]
        if len(selected) == 1:
            layout_name = "1x1"
        elif len(selected) == 2:
            layout_name = "1x2"
        elif len(selected) <= 4:
            layout_name = "2x2"
        else:
            layout_name = "3x3"
        self.view_grid.set_layout(layout_name)
        self.toolbar.setVisible(layout_name == "1x1")
        for cell, case_index in zip(self.view_grid.visible_cells(), selected):
            if cell.cell_type != "slice":
                cell.set_cell_type("slice")
            cell.set_scenario_silently(case_index)
            if cell is self.view_grid.active_cell():
                self._apply_manifest_case_to_controller(case_index)
            else:
                self._load_cell(cell, case_index, cell.quantity_key)
        self._apply_link_clim()

    def _open_browser_ensemble(self, case_indices: list):
        """Open browser-selected scenarios as an ensemble in the active cell."""
        if not case_indices:
            return
        cell = self.view_grid.active_cell()
        if cell.cell_type != "ensemble":
            cell.set_cell_type("ensemble")
        cell.ensemble_case_indices = sorted(case_indices)
        cell.ensemble_stat = "mean"
        if hasattr(cell, "ensemble_select_button"):
            cell.ensemble_select_button.setText(cell._ensemble_button_text())
        if hasattr(cell, "stat_combo"):
            cell.stat_combo.blockSignals(True)
            cell.stat_combo.setCurrentIndex(0)
            cell.stat_combo.blockSignals(False)
        self._render_ensemble_cell(cell)
        self._apply_link_clim()

    def _open_browser_model_eval(self, case_indices: list):
        """M3.2.5: "View model prediction" button -- ground truth, the
        trained model's prediction, and their difference, side by side,
        for a single (test-set) scenario. Reuses the grid/GridCell
        machinery every other view uses (a 1x3 layout, "slice" and
        "difference" cell types) rather than a parallel display system --
        only the DATA SOURCE differs for the prediction/difference-B cells
        (store_override(_b), see _store_for_cell), not the rendering
        path."""
        if not self.prediction_store.is_available:
            return
        available = set(self.prediction_store.case_indices)
        case_index = next((ci for ci in case_indices if ci in available), None)
        if case_index is None:
            case_index = self.prediction_store.case_indices[0]
            self.statusBar().showMessage(
                "No prediction for the selected scenario -- showing a test-set scenario instead.", 5000,
            )

        self.view_grid.set_layout("1x3")
        self.toolbar.setVisible(False)
        ground_truth_cell, prediction_cell, difference_cell = self.view_grid.visible_cells()

        for cell in (ground_truth_cell, prediction_cell):
            if cell.cell_type != "slice":
                cell.set_cell_type("slice")
        ground_truth_cell.store_override = None
        prediction_cell.store_override = self.prediction_store
        ground_truth_cell.set_scenario_silently(case_index)
        prediction_cell.set_scenario_silently(case_index)
        self._load_cell(ground_truth_cell, case_index, DEFAULT_SLICE_KEY)
        self._load_cell(prediction_cell, case_index, DEFAULT_SLICE_KEY)

        if difference_cell.cell_type != "difference":
            difference_cell.set_cell_type("difference")
        difference_cell.case_index_a = case_index
        difference_cell.case_index_b = case_index
        difference_cell.store_override_b = self.prediction_store
        difference_cell.quantity_key = DEFAULT_SLICE_KEY
        self._render_difference_cell(difference_cell)

        self._apply_link_clim()

    def _export_summaries_markdown(self):
        """Experiment browser's "Export summaries (Markdown)…" button
        (M3.1.3). Reuses self._scenario_summaries (already computed for
        the browser table) rather than recomputing -- the exported text
        is generated the same way, from the same data, as what the
        browser already showed on screen."""
        default_name = "fds_scenario_summaries.md"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Summaries", default_name, "Markdown (*.md)")
        if not path:
            return
        if not path.lower().endswith(".md"):
            path += ".md"
        try:
            export_markdown(
                self.sim_data.manifest, self._scenario_summaries,
                self.controller.store, self.sim_data.timesteps_per_second, path,
            )
        except OSError as e:
            self._on_sim_error(f"Could not write summaries to {path}: {e}")
            return
        self.statusBar().showMessage(f"Exported scenario summaries to {path}", 5000)

    # --------------------------------------------------------- report (M3.3)
    def _entry_for_case(self, case_index):
        return next((e for e in (self.sim_data.manifest or []) if e.case_index == case_index), None)

    def _summary_for_case(self, case_index):
        return next((s for s in getattr(self, "_scenario_summaries", []) if s.case_index == case_index), None)

    def _peak_frame_index(self, summary) -> int:
        return int(np.argmax(summary.max_temp_by_frame_c)) if summary.max_temp_by_frame_c else 0

    def _export_report(self, case_indices: list):
        """Experiment browser's "Generate report…" (M3.3): one selected
        scenario -> a per-scenario HTML report; two -> an A-vs-B
        comparison report. Assembles the F6 publication figure, the
        already-computed summary stats, the deterministic auto-summary
        prose, and a provenance block into one self-contained file."""
        if not case_indices or not getattr(self, "_scenario_summaries", None):
            return
        summaries = self._scenario_summaries
        fps = self.sim_data.timesteps_per_second
        key = DEFAULT_SLICE_KEY
        display = self._display_for(key.quantity)
        try:
            if len(case_indices) == 1:
                html_text, default_name = self._build_scenario_report(case_indices[0], summaries, fps, key, display)
            else:
                html_text, default_name = self._build_comparison_report(
                    case_indices[0], case_indices[1], summaries, fps, key, display)
        except Exception as e:  # noqa: BLE001 - report a failure, never crash
            QtWidgets.QMessageBox.warning(self, "Generate Report", f"Could not build report: {e}")
            return
        if html_text is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Report", default_name, "HTML report (*.html)")
        if not path:
            return
        if not path.lower().endswith(".html"):
            path += ".html"
        try:
            write_report(path, html_text)
        except OSError as e:
            self._on_sim_error(f"Could not write report to {path}: {e}")
            return
        self.statusBar().showMessage(f"Saved report to {path}", 5000)

    def _build_scenario_report(self, case_index, summaries, fps, key, display):
        entry = self._entry_for_case(case_index)
        summary = self._summary_for_case(case_index)
        if entry is None or summary is None:
            return None, None
        data = self.controller.store.get(case_index, key)
        peak = min(self._peak_frame_index(summary), data.shape[0] - 1)
        provenance = provenance_line(entry.path, entry.folder, peak / fps)
        figure_png = figure_png_bytes(
            np.asarray(data[peak]), cmap=self.current_colormap, vmin=display['vmin'],
            vmax=display['slider_default'], extent=self._extent_for(case_index, key),
            colorbar_label=f"{display['label']} ({display['unit']})", title=entry.folder,
            isotherm_levels=ISOTHERM_LEVELS.get(key.quantity))
        summary_text = generate_summary(entry, summary, summaries, self.controller.store, fps, key)
        html_text = build_scenario_report(entry, summary, summary_text, figure_png, provenance)
        return html_text, f"report_{entry.folder}.html"

    def _build_comparison_report(self, case_a, case_b, summaries, fps, key, display):
        entry_a, entry_b = self._entry_for_case(case_a), self._entry_for_case(case_b)
        summary_a, summary_b = self._summary_for_case(case_a), self._summary_for_case(case_b)
        if None in (entry_a, entry_b, summary_a, summary_b):
            return None, None
        data_a = self.controller.store.get(case_a, key)
        data_b = self.controller.store.get(case_b, key)
        peak = min(self._peak_frame_index(summary_a), data_a.shape[0] - 1, data_b.shape[0] - 1)
        diff = np.asarray(data_a[peak]) - np.asarray(data_b[peak])
        vmax = float(np.max(np.abs(diff))) or 1.0
        provenance_a = provenance_line(entry_a.path, entry_a.folder, peak / fps)
        provenance_b = provenance_line(entry_b.path, entry_b.folder, peak / fps)
        diff_png = figure_png_bytes(
            diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=self._extent_for(case_a, key),
            colorbar_label=f"Δ{display['label']} ({display['unit']})",
            title=f"{entry_a.folder} − {entry_b.folder}")
        text_a = generate_summary(entry_a, summary_a, summaries, self.controller.store, fps, key)
        text_b = generate_summary(entry_b, summary_b, summaries, self.controller.store, fps, key)
        # V3-M3: the semantic-diff engine's ranked differences, folded into
        # the report as a "Key differences" list.
        differences = difference_statements(compare_scenarios(
            data_a, data_b, self._extent_for(case_a, key), fps, key.quantity,
            entry_a.folder, entry_b.folder, summary_a, summary_b))
        html_text = build_comparison_report(entry_a, entry_b, summary_a, summary_b,
                                            text_a, text_b, diff_png, provenance_a, provenance_b,
                                            differences=differences)
        return html_text, f"report_{entry_a.folder}_vs_{entry_b.folder}.html"

    def _load_cell(self, cell, case_index: int, quantity_key):
        """A *non-active* cell's own combo picked a new (case, key)
        (M2.2.2/2.2.4). A cache hit redraws immediately; a cache miss
        prefetches in the background (reusing SimulationController's
        existing worker-list-tracked machinery, see _on_grid_prefetch_finished)
        instead of blocking the GUI thread -- unlike the *first* time a
        brand-new grid cell is created (_on_cell_created/_init_cell_view),
        which still does a synchronous init_plot(); making that path
        prefetch-aware too would need a "loading" placeholder frame state,
        deliberately out of scope here (D:Easy T:2h estimate for this
        task) -- that cold-parse hitch stays the same bounded/self-
        correcting shape already characterized in ROADMAP.md's M2.1
        section, just for a cell's *first* ever view, not its combo
        changes thereafter."""
        cell.case_index = case_index
        cell.quantity_key = quantity_key
        store = self._store_for_cell(cell)
        if store.is_cached(case_index, quantity_key):
            self._redraw_cell_now(cell)
            self._apply_link_clim()
            self._refresh_inspector_now()
            return
        # A store_override (M3.2.5's PredictionSource) is always fully
        # loaded at construction, so is_cached() above is always True for
        # it -- this prefetch path only ever actually runs for the real
        # ScenarioStore.
        self._pending_cell_prefetches.add(cell)
        self.controller.prefetch(case_index, quantity_key)

    def _refresh_inspector_now(self) -> None:
        """Refresh every visible cell's Inspector section immediately (V6
        polish) instead of waiting for the next playback tick/seek -- used
        after a *non-active* cell's own scenario/quantity combo changes
        (_load_cell/_on_grid_prefetch_finished), so a comparison grid's
        Inspector sections never show a stale scenario (or fire story) after
        a switch."""
        index = self.time_controller.index
        frames = {}
        for cell in self.view_grid.visible_cells():
            try:
                frames[id(cell)] = self._frame_for_cell(cell, index)
            except Exception:  # noqa: BLE001 - inspector refresh is a nice-to-have, never fatal
                pass
        self._update_inspector(index, frames)
        self._update_event_markers()

    def _redraw_cell_now(self, cell):
        """Assumes cell.case_index/quantity_key are already cached."""
        store = self._store_for_cell(cell)
        display = self._display_for(cell.quantity_key.quantity)
        data = store.get(cell.case_index, cell.quantity_key)
        cell.view.set_cmap(display['cmap'])
        cell.view.set_clim(display['vmin'], display['slider_default'])
        cell.view.set_colorbar_label(f"{display['label']} ({display['unit']})")
        self._sync_cell_extent(cell)  # M2.2: a SOOT-plane switch may change the extent
        self._apply_contour_overlay_state(cell)  # levels are quantity-specific; quantity may have just changed
        self._apply_cinematic_state(cell)  # cinematic mode is TEMPERATURE-only; quantity may have just changed
        index = min(self.time_controller.index, data.shape[0] - 1)
        if cell.view.velocity_overlay_enabled:
            cell.view.show_frame(data[index], velocity_frame=self._velocity_overlay_frame_for_cell(cell, index))
        else:
            cell.view.show_frame(data[index])

    def _on_grid_prefetch_finished(self, case_idx: int):
        """A background prefetch completed -- check every grid cell
        waiting on exactly this case_idx and redraw any that are now
        actually cached (M2.2.4). Purely a "go check what's newly
        available" notification: idempotent, and independent of
        _on_prefetch_finished's own active-cell busy-state bookkeeping,
        which this doesn't touch."""
        ready = [cell for cell in list(self._pending_cell_prefetches)
                 if cell.case_index == case_idx
                 and self.controller.is_cached(cell.case_index, cell.quantity_key)]
        for cell in ready:
            self._pending_cell_prefetches.discard(cell)
            self._redraw_cell_now(cell)
        if ready:
            self._apply_link_clim()
            self._refresh_inspector_now()

    def _on_grid_prefetch_error(self, case_idx: int, message: str):
        """A background prefetch failed. No per-cell error UI (out of
        scope for M2.2.4) -- just stop waiting on it so
        _pending_cell_prefetches doesn't grow stale entries forever. If
        this same failure was also the active cell's load, the existing
        _on_prefetch_error (connected to the same signal) still shows its
        own modal error dialog independently."""
        stale = [cell for cell in list(self._pending_cell_prefetches) if cell.case_index == case_idx]
        for cell in stale:
            self._pending_cell_prefetches.discard(cell)

    def _on_cell_scenario_selected(self, cell, case_index: int):
        if cell is self.view_grid.active_cell():
            self._apply_manifest_case_to_controller(case_index)
        else:
            self._load_cell(cell, case_index, cell.quantity_key)

    def _on_cell_quantity_selected(self, cell, key):
        """A cell's own quantity combo changed. For a slice-type *active*
        cell this still routes through the control panel's own quantity
        combo (unchanged M2.1 behavior). For difference/ensemble cells,
        or any non-active slice cell, the cell's own combo directly
        controls its own quantity_key regardless of active status -- the
        control panel's quantity combo isn't a meaningful "the" quantity
        for a two-scenario diff or N-scenario ensemble, so it's left
        decoupled from those cell types rather than overloading it."""
        if cell.cell_type == "slice":
            if cell is self.view_grid.active_cell():
                idx = next((i for i, info in enumerate(self.quantity_infos) if info.key == key), None)
                if idx is not None:
                    self.quantity_combo.setCurrentIndex(idx)  # triggers _on_quantity_changed
                return
            self._load_cell(cell, cell.case_index, key)
        elif cell.cell_type == "difference":
            cell.quantity_key = key
            self._render_difference_cell(cell)
            self._apply_link_clim()
        elif cell.cell_type == "ensemble":
            cell.quantity_key = key
            self._render_ensemble_cell(cell)
            self._apply_link_clim()

    # ---------------------------------------------- difference/ensemble (M2.3)
    def _on_cell_type_changed(self, cell, new_type: str):
        """A grid cell's type changed via its right-click context menu
        (M2.3.3). Renders the cell's new view immediately from whatever
        state it already carries (GridCell.__init__ defaults
        case_index_a/case_index_b to sensible scenarios; a fresh ensemble
        cell starts with an empty selection and stays blank until the user
        picks scenarios via the picker dialog -- see _on_cell_ensemble_changed)."""
        if new_type == "slice":
            self._init_cell_view(cell)
        elif new_type == "difference":
            self._render_difference_cell(cell)
        elif new_type == "ensemble" and cell.ensemble_case_indices:
            self._render_ensemble_cell(cell)
        self._apply_link_clim()
        if cell is self.view_grid.active_cell():
            self._update_event_markers()  # markers are slice-cell-only (M1.3)

    def _render_difference_cell(self, cell):
        """(Re)renders a difference-type cell for its current
        case_index_a/case_index_b/quantity_key. Synchronous -- same
        deliberate scope decision as _load_cell's non-active-cell combo
        changes (no prefetch orchestration for the two-scenario case; see
        _load_cell's docstring, which applies equally here)."""
        key = cell.quantity_key
        store_b = cell.store_override_b if cell.store_override_b is not None else self.controller.store
        data_a = self.controller.store.get(cell.case_index_a, key)
        data_b = store_b.get(cell.case_index_b, key)
        vmin, vmax = cell.view.symmetric_clim(
            data_a, data_b, cache_key=(cell.case_index_a, cell.case_index_b, key))
        display = self._display_for(key.quantity)
        colorbar_label = f"Δ{display['label']} ({display['unit']})"
        index = min(self.time_controller.index, data_a.shape[0] - 1, data_b.shape[0] - 1)
        frame = DifferenceView.compute_diff(data_a, data_b, index)
        if cell.view.heatmap is None:
            cell.view.init_plot(frame, interpolation=self.current_interpolation,
                                 vmin=vmin, vmax=vmax, colorbar_label=colorbar_label,
                                 extent=self._extent_for(cell.case_index_a, key))
            self._setup_cell_probe_and_isotherms(cell)
        else:
            cell.view.set_clim(vmin, vmax)
            cell.view.set_colorbar_label(colorbar_label)
            new_extent = self._extent_for(cell.case_index_a, key)  # M2.2: SOOT plane may differ
            if new_extent != getattr(cell.view, "_extent", None):
                cell.view.set_extent(new_extent)
            self._apply_contour_overlay_state(cell)  # quantity may have just changed
            cell.view.show_frame(frame)

    def _render_ensemble_cell(self, cell):
        """(Re)renders an ensemble-type cell for its current
        ensemble_case_indices/ensemble_stat/quantity_key. No-op if nothing
        is selected yet."""
        if not cell.ensemble_case_indices:
            return
        key = cell.quantity_key
        arrays = [self.controller.store.get(ci, key) for ci in cell.ensemble_case_indices]
        display = self._display_for(key.quantity)
        stat = cell.ensemble_stat
        cmap = EnsembleView.cmap_for(stat, display['cmap'])
        colorbar_label = EnsembleView.label_for(stat, display['label'], display['unit'])
        if stat == "std":
            vmin = 0.0
            vmax = cell.view.std_vmax(arrays, cache_key=(tuple(cell.ensemble_case_indices), key))
        else:
            vmin, vmax = display['vmin'], display['slider_default']
        index = min(self.time_controller.index, min(a.shape[0] for a in arrays) - 1)
        frame = EnsembleView.compute_composite(arrays, index, stat)
        if cell.view.heatmap is None:
            cell.view.init_plot(frame, cmap=cmap, interpolation=self.current_interpolation,
                                 vmin=vmin, vmax=vmax, colorbar_label=colorbar_label,
                                 extent=self._extent_for(cell.ensemble_case_indices[0], key))
            self._setup_cell_probe_and_isotherms(cell)
        else:
            cell.view.set_cmap(cmap)
            cell.view.set_clim(vmin, vmax)
            cell.view.set_colorbar_label(colorbar_label)
            new_extent = self._extent_for(cell.ensemble_case_indices[0], key)  # M2.2: SOOT plane may differ
            if new_extent != getattr(cell.view, "_extent", None):
                cell.view.set_extent(new_extent)
            self._apply_contour_overlay_state(cell)  # quantity may have just changed
            cell.view.show_frame(frame)

    def _show_difference_over_time(self, cell=None):
        """Inspector's "Plot difference over time…" button (V2 roadmap
        M1.5): opens DifferenceOverTimeDialog for `cell`'s current A/B pair
        and quantity -- the section that owns the clicked button passes its
        own current cell (looked up by grid position at click time, see
        _wire_inspector_section); falls back to the active cell for any
        caller that doesn't know which cell it means. Only meaningful for a
        difference cell -- the button is hidden otherwise (see
        InspectorPanel.clear_difference_stats), so this is a defensive
        guard, not the primary gate."""
        cell = cell or self.view_grid.active_cell()
        if cell is None or cell.cell_type != "difference":
            return
        key = cell.quantity_key
        data_a = self.controller.store.get(cell.case_index_a, key)
        data_b = self.controller.store.get(cell.case_index_b, key)
        display = self._display_for(key.quantity)
        dialog = DifferenceOverTimeDialog(
            data_a, data_b, self.time_controller.timesteps_per_second,
            self._scenario_label(cell.case_index_a), self._scenario_label(cell.case_index_b),
            unit=display.get('unit', ''), parent=self)
        dialog.exec_()

    def _on_cell_difference_scenarios_changed(self, cell, case_a: int, case_b: int):
        self._render_difference_cell(cell)
        self._apply_link_clim()

    def _on_cell_ensemble_changed(self, cell, case_indices: list, stat: str):
        self._render_ensemble_cell(cell)
        self._apply_link_clim()

    def _on_temp_changed(self, value):
        display = QUANTITY_DISPLAY.get(self.current_quantity_key.quantity, QUANTITY_DISPLAY[DEFAULT_SLICE_KEY.quantity])
        self.temp_label.setText(f"{value} {display['unit']}")
        # vmin stays pinned at the quantity's physical floor; only vmax
        # moves with the slider. This -- like the colormap menu -- edits
        # the active cell only (M2.2 design decision: other visible cells
        # keep their own last-set clim/colormap independently unless
        # "Link color scales" is on). SliceView.set_clim does the full
        # redraw (not blit) + recapture itself: the colorbar's tick range
        # depends on clim and needs to actually repaint.
        self.view_grid.active_view().set_clim(display['vmin'], value)
        self._apply_link_clim()

    # -------------------------------------------------------------- export
    def _export_animation(self):
        """Export → Animation (MP4/GIF)… (M1.5). Pauses playback for the
        duration of the export (the exporter renders through its own
        offscreen figure, not the live canvas, but sharing the same
        underlying frame data while it's being read frame-by-frame on a
        background thread is simpler to reason about than racing a live
        QTimer against it) and resumes afterward if it was running."""
        # Defense in depth against the same class of QThread-lifecycle bug
        # found in M1.4.4's prefetch worker: the progress dialog below is
        # window-modal (blocks the menu bar while open), which should
        # already prevent re-entry, but guard explicitly rather than rely
        # on that alone -- only one export is meant to run at a time, so
        # refusing a second start is correct here (unlike prefetch, where
        # concurrent workers are legitimate and get list-tracked instead).
        if getattr(self, "_exporter", None) is not None and self._exporter.isRunning():
            self.statusBar().showMessage("An export is already in progress.", 3000)
            return

        was_playing = self.time_controller.is_playing()
        self.time_controller.pause()

        has_ffmpeg = ffmpeg_available()
        default_ext = "mp4" if has_ffmpeg else "gif"
        filters = (
            "MP4 Video (*.mp4);;GIF Animation (*.gif)" if has_ffmpeg
            else "GIF Animation (*.gif)"
        )
        default_name = (
            f"scenario_{self.controller.current_case_index()}_"
            f"{self.current_quantity_key.quantity.lower()}.{default_ext}"
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Animation", default_name, filters)
        if not path:
            if was_playing:
                self.time_controller.play()
            return

        if not (path.lower().endswith(".mp4") or path.lower().endswith(".gif")):
            path += f".{default_ext}"
        if path.lower().endswith(".mp4") and not has_ffmpeg:
            # Shouldn't happen (mp4 isn't offered in the filter without
            # ffmpeg), but guard anyway rather than fail deep in the thread.
            path = os.path.splitext(path)[0] + ".gif"

        case_idx = self.controller.current_case_index()
        # V6-M1.5: export a computed field via the provider, native via the store.
        data = self._field(self.controller.store, case_idx, self.current_quantity_key)
        n_frames = data.shape[0]

        range_dialog = ExportRangeDialog(self, n_frames, self.time_controller.timesteps_per_second)
        if range_dialog.exec_() != QtWidgets.QDialog.Accepted:
            if was_playing:
                self.time_controller.play()
            return
        start, end, fps = range_dialog.values()

        export_vmin = self._display_for(self.current_quantity_key.quantity)['vmin']
        self._exporter = AnimationExporter(
            data, path, fps, self.current_colormap, export_vmin, self.temp_slider.value(),
            self.current_interpolation, start, end,
        )
        self._export_progress = QtWidgets.QProgressDialog(
            "Exporting animation…", "Cancel", 0, end - start, self
        )
        self._export_progress.setWindowModality(QtCore.Qt.WindowModal)
        self._export_progress.setMinimumDuration(0)
        self._export_progress.setValue(0)

        self._exporter.progress.connect(self._on_export_progress)
        self._exporter.finished_ok.connect(lambda p: self._on_export_finished(p, was_playing))
        self._exporter.error.connect(lambda msg: self._on_export_error(msg, was_playing))
        self._exporter.cancelled.connect(lambda: self._on_export_cancelled(was_playing))
        self._export_progress.canceled.connect(self._exporter.request_cancel)

        self._exporter.start()

    def _on_export_progress(self, done: int, total: int):
        self._export_progress.setValue(done)

    def _on_export_finished(self, output_path: str, was_playing: bool):
        self._export_progress.setValue(self._export_progress.maximum())
        self.statusBar().showMessage(f"Exported animation to {output_path}", 5000)
        if was_playing:
            self.time_controller.play()

    def _on_export_error(self, message: str, was_playing: bool):
        self._export_progress.close()
        self._on_sim_error(message)
        if was_playing:
            self.time_controller.play()

    def _on_export_cancelled(self, was_playing: bool):
        self._export_progress.close()
        self.statusBar().showMessage("Export cancelled.", 3000)
        if was_playing:
            self.time_controller.play()

    def _export_postcard(self):
        """Export page's "demo postcard" (FireLab roadmap Phase 4): a
        one-click PNG of the active cell's current frame with a simple
        FireLab title-card overlay -- a QPainter grab of the live canvas,
        not a re-render, so it always matches exactly what's on screen."""
        cell = self.view_grid.active_cell()
        if cell is None:
            return
        default_name = f"firelab_{self.controller.current_case_index()}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Demo Postcard", default_name, "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        pixmap = cell.view.widget().grab()
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        band_height = max(36, pixmap.height() // 12)
        band_rect = QtCore.QRect(0, pixmap.height() - band_height, pixmap.width(), band_height)
        painter.fillRect(band_rect, QtGui.QColor(11, 13, 18, 200))
        painter.setPen(QtGui.QColor("#FF6B35"))
        font = painter.font()
        font.setPointSizeF(max(font.pointSizeF() * 1.3, 12))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(band_rect.adjusted(12, 0, -12, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                          "FireLab Digital Twin")
        painter.end()
        pixmap.save(path, "PNG")
        self.statusBar().showMessage(f"Saved {path}", 4000)

    # --------------------------------------------------------- multi-study (M2.5)
    def _open_study(self):
        """File -> Open Study… (M2.5): load a different FDS-output
        directory (a single case, a folder of cases, or another candle
        factorial) into a fresh window. Opening in a new MainWindow rather
        than live-swapping the data layer avoids any partial-state
        inconsistency across the grid/controller/browser/panels -- the new
        window is built cleanly from the new study, and the old one closes.
        """
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Open FDS Study")
        if not root:
            return
        try:
            sim_data = load_study(root)
            sim_data.store.get(0)  # warm the first scenario so the new window opens instantly
        except DataLoadError as e:
            QtWidgets.QMessageBox.warning(self, "Open Study", f"{e.message}\n\n{e.technical_detail}")
            return
        except Exception as e:  # noqa: BLE001 - a bad directory must not crash the app
            QtWidgets.QMessageBox.warning(self, "Open Study", f"Could not open study: {e}")
            return

        new_window = MainWindow(sim_data)
        new_window.setWindowIcon(self.windowIcon())
        # A shown top-level QWidget with no Python reference can be garbage
        # collected out from under Qt; keep every opened window referenced
        # for the process's lifetime.
        _OPEN_STUDY_WINDOWS.append(new_window)
        new_window.show()
        self.close()

    # ------------------------------------------------------------- session (M2.4)
    def _collect_session_dict(self, name: str = "", intent: str = "") -> dict:
        """The full analysis-session snapshot (V4-M6): grid + view + the
        Evidence Notebook + zones + the interval selection + browser
        filters + metadata (author, timestamps, data fingerprint). Shared
        by the file-based Save Session and the named Sessions panel."""
        visible_cells = self.view_grid.visible_cells()
        active_index = visible_cells.index(self.view_grid.active_cell())
        metadata = session_store.make_metadata(
            data_version=session_store.data_fingerprint(self.sim_data.manifest))
        return build_session_dict(
            self.view_grid.layout_name, visible_cells,
            active_index, self.time_controller.index,
            getattr(self, "_link_clim", False), self.current_colormap,
            self.isotherms_action.isChecked(),
            notebook=self.evidence_dock.notebook.to_list(),
            zones=self.zone_panel.get_zones() if self.zone_panel is not None else [],
            name=name, intent=intent, metadata=metadata,
            time_window=(self.time_window_panel.get_state()
                         if self.time_window_panel is not None else {}),
            filters=(self.experiment_browser.get_filter_state()
                     if self.experiment_browser is not None else {}),
            measurements=(self.measurement_panel.get_measurements()
                          if self.measurement_panel is not None else []),
            selection=(self.selection_bus.current.to_dict()
                       if getattr(self, "selection_bus", None) is not None else {}),
            calculated_fields=[f.to_dict() for f in field_calculator_mod.all_fields()],
            devices=(self.device_panel.get_devices()
                     if self.device_panel is not None else []),
            vector_probes=(self.velocity_panel.get_probes()
                          if self.velocity_panel is not None else []),
            comparisons=(self.advanced_compare_panel.get_comparisons()
                        if self.advanced_compare_panel is not None else []))

    # --------------------------------------------------- named sessions (M6)
    def _refresh_sessions(self) -> None:
        """No-op today (the Sessions tab that displayed this list was
        removed, Analysis UX + reliability pass) -- kept as a safe no-op
        rather than deleted, since _on_session_save/_on_session_delete
        below still call it."""
        panel = getattr(self, "sessions_panel", None)
        if panel is not None:
            panel.set_sessions(session_store.list_sessions(self._sessions_dir))

    def _on_session_save(self, name: str, intent: str) -> None:
        session = self._collect_session_dict(name, intent)
        try:
            session_store.save_session(self._sessions_dir, session)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Save session", f"Could not save: {e}")
            return
        self._refresh_sessions()
        self.statusBar().showMessage(f"Saved session '{name}'", 4000)

    def _on_session_load(self, path: str) -> None:
        try:
            session = session_store.load_session(path)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Load session", str(e))
            return
        # Version pinning: warn (do not block) if the session was made
        # against a different data run than the one currently loaded.
        saved_fp = (session.get("metadata") or {}).get("data_version", "")
        current_fp = session_store.data_fingerprint(self.sim_data.manifest)
        if saved_fp and current_fp and saved_fp != current_fp:
            QtWidgets.QMessageBox.warning(
                self, "Different data run",
                "This session was saved against a different data run. The view "
                "and annotations are restored, but computed values may differ.")
        self._apply_analysis_session(session)
        self.statusBar().showMessage(f"Loaded session '{session.get('name', '')}'", 4000)

    def _on_session_delete(self, path: str) -> None:
        session_store.delete_session(path)
        self._refresh_sessions()

    def _on_session_export(self, path: str) -> None:
        try:
            session = session_store.load_session(path)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Export report", str(e))
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export session report", "session_report.html", "HTML (*.html)")
        if not out:
            return
        if not out.lower().endswith(".html"):
            out += ".html"
        try:
            write_report(out, build_session_report(session))
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Export report", f"Could not write: {e}")
            return
        self.statusBar().showMessage(f"Exported report {out}", 4000)

    def _apply_analysis_session(self, session: dict) -> None:
        """Restore a full named session: grid/view/notebook/zones via
        _apply_session, plus the interval selection and browser filters."""
        self._apply_session(session)
        if self.time_window_panel is not None:
            self.time_window_panel.set_state(session.get("time_window", {}))
        if self.experiment_browser is not None:
            self.experiment_browser.set_filter_state(session.get("filters", {}))
        if self.measurement_panel is not None:
            self.measurement_panel.set_measurements(session.get("measurements", []))
        # V6-M1: restore the Field Calculator definitions (absent -> none). Clear
        # first so a load replaces rather than accumulates.
        field_calculator_mod.clear()
        for d in session.get("calculated_fields", []) or []:
            try:
                field_calculator_mod.register(field_calculator_mod.CalculatedField.from_dict(d))
            except Exception:  # noqa: BLE001 - a bad definition must not block the load
                continue
        self._refresh_quantity_list()   # V6-M1.5: reflect restored fields in the Live combo
        # V6-M2: restore devices with their cached results verbatim (no recompute).
        if self.device_panel is not None:
            self.device_panel.set_devices(session.get("devices", []))
            self._refresh_device_markers()
        # V6-M3: restore vector probes with their cached results verbatim (no recompute).
        if self.velocity_panel is not None:
            self.velocity_panel.set_probes(session.get("vector_probes", []))
            self._refresh_vector_field()
        # Analysis-improvement roadmap Phase C: restore pinned Compare Axes results.
        if self.advanced_compare_panel is not None:
            self.advanced_compare_panel.set_comparisons(session.get("comparisons", []))
        # V5-M1: restore the shared selection (absent in older sessions -> empty).
        if getattr(self, "selection_bus", None) is not None:
            self.selection_bus.set(Selection.from_dict(session.get("selection", {})))

    def _save_session(self):
        if not self.sim_data.manifest:
            QtWidgets.QMessageBox.information(self, "Save Session", "No experiment data available (demo mode).")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Session", "fds_session.json", "Session files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        session = self._collect_session_dict()
        try:
            write_session(path, session)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Save Session", f"Could not save: {e}")
            return
        self.statusBar().showMessage(f"Saved {path}", 4000)

    def _load_session(self):
        if not self.sim_data.manifest:
            QtWidgets.QMessageBox.information(self, "Load Session", "No experiment data available (demo mode).")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Session", "", "Session files (*.json)")
        if not path:
            return
        try:
            session = read_session(path)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Load Session", str(e))
            return
        self._apply_session(session)
        self.statusBar().showMessage(f"Loaded {path}", 4000)

    def _apply_session(self, session: dict) -> None:
        """Restores a session dict built by session.build_session_dict,
        driving the same combo/signal chain a user's own clicks go
        through (same pattern as _apply_compare_preset) rather than
        touching cell/store internals directly."""
        self._set_grid_layout(session["layout"])
        cells = self.view_grid.visible_cells()
        for cell, cell_state in zip(cells, session.get("cells", [])):
            quantity_key = self._find_quantity_key(
                cell_state.get("quantity"), cell_state.get("direction"), cell_state.get("offset"))
            cell_type = cell_state.get("cell_type", "slice")
            if cell.cell_type != cell_type:
                cell.set_cell_type(cell_type)
            if cell_type == "slice":
                if "case_index" in cell_state:
                    self._select_scenario_in_cell(cell, cell_state["case_index"])
            elif cell_type == "difference":
                if "case_index_a" in cell_state and "case_index_b" in cell_state:
                    self._select_difference_scenarios_in_cell(
                        cell, cell_state["case_index_a"], cell_state["case_index_b"])
            elif cell_type == "ensemble":
                cell.ensemble_case_indices = list(cell_state.get("ensemble_case_indices", []))
                cell.ensemble_stat = cell_state.get("ensemble_stat", "mean")
                if hasattr(cell, "ensemble_select_button"):
                    cell.ensemble_select_button.setText(cell._ensemble_button_text())
                if hasattr(cell, "stat_combo"):
                    idx = EnsembleView.STATS.index(cell.ensemble_stat)
                    cell.stat_combo.blockSignals(True)
                    cell.stat_combo.setCurrentIndex(idx)
                    cell.stat_combo.blockSignals(False)
                self._on_cell_ensemble_changed(cell, cell.ensemble_case_indices, cell.ensemble_stat)
            if quantity_key is not None and cell.quantity_key != quantity_key:
                self._select_quantity_in_cell(cell, quantity_key)

        active_idx = session.get("active_index", 0)
        if 0 <= active_idx < len(cells):
            cells[active_idx].activated.emit(cells[active_idx])

        if session.get("colormap"):
            self._set_colormap(session["colormap"])
        self.link_clim_action.setChecked(bool(session.get("link_clim", False)))
        self._set_link_clim(bool(session.get("link_clim", False)))
        self.isotherms_action.setChecked(bool(session.get("isotherms_enabled", False)))
        self._set_isotherms_enabled(bool(session.get("isotherms_enabled", False)))

        # V4-M2: restore the Evidence Notebook (absent in v1 sessions -> empty).
        self.evidence_dock.load_notebook(EvidenceNotebook.from_list(session.get("notebook")))
        if not self.evidence_dock.notebook.is_empty():
            self.evidence_dock.show()

        # V4-M4: restore named zones (absent in older sessions -> none).
        if self.zone_panel is not None:
            self.zone_panel.set_zones(session.get("zones", []))

        time_index = session.get("time_index", 0)
        self.time_controller.seek(min(max(time_index, 0), self._current_n_frames - 1))

    def _export_publication_figure(self):
        """Export -> Publication figure… (V2 roadmap M1.4): a real
        re-render (not a screen grab, unlike _export_postcard) through
        figure_export.export_publication_figure, at a chosen journal
        width/format, with proper physical axes, labeled isotherms, and
        an optional provenance footer parsed from the scenario's `.out`/
        `.fds` files. "Slice"-type cells only -- a difference/ensemble
        cell's frame data doesn't map to one scenario's provenance."""
        cell = self.view_grid.active_cell()
        if cell is None or cell.cell_type != "slice":
            QtWidgets.QMessageBox.information(
                self, "Publication Figure",
                "Publication export is only available for a plain slice cell "
                "(not a difference or ensemble view).")
            return

        dialog = PublicationExportDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        options = dialog.options()

        default_name = f"figure_{cell.case_index}{options['extension']}"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Publication Figure", default_name,
            f"Figure ({'*' + options['extension']})")
        if not path:
            return
        if not path.lower().endswith(options["extension"]):
            path += options["extension"]

        data = self.controller.store.get(cell.case_index, cell.quantity_key)
        index = min(self.time_controller.index, data.shape[0] - 1)
        frame = np.asarray(data[index])
        display = self._display_for(cell.quantity_key.quantity)
        vmin, vmax = cell.view.heatmap.get_clim()
        extent = self._extent_for(cell.case_index, cell.quantity_key)
        entry = next((e for e in (self.sim_data.manifest or []) if e.case_index == cell.case_index), None)

        provenance = None
        if options["provenance"] and entry is not None:
            time_s = index / self.time_controller.timesteps_per_second
            provenance = provenance_line(entry.path, entry.folder, time_s)

        isotherm_levels = ISOTHERM_LEVELS.get(cell.quantity_key.quantity, []) if options["contours"] else None

        try:
            export_publication_figure(
                frame, path, cmap=self.current_colormap, vmin=vmin, vmax=vmax,
                extent=extent, colorbar_label=f"{display['label']} ({display['unit']})",
                title=entry.folder if entry is not None else "",
                width_in=options["width_in"], font_pt=options["font_pt"],
                dpi=options["dpi"], isotherm_levels=isotherm_levels, provenance=provenance)
        except Exception as e:  # noqa: BLE001 - report, don't crash the app
            QtWidgets.QMessageBox.warning(self, "Publication Figure", f"Export failed: {e}")
            return
        self.statusBar().showMessage(f"Saved {path}", 4000)

    def _start_simulation(self):
        self.time_controller.play()

    def _stop_simulation(self):
        self.time_controller.pause()

    def _restart_simulation(self):
        self.time_controller.restart()

    # ------------------------------------------------------------- theming
    def _set_theme(self, name: str):
        self.current_theme_name = name
        self.settings.setValue("theme", name)
        self._apply_theme()

    def _set_ui_scale(self, scale: float):
        self.ui_scale = scale
        self.settings.setValue("ui_scale", scale)
        self._apply_theme()

    def _set_colormap(self, cmap: str):
        # Active cell only, same M2.2 design decision as _on_temp_changed.
        self.current_colormap = cmap
        self.settings.setValue("colormap", cmap)
        self.view_grid.active_view().set_cmap(cmap)

    def _set_interpolation(self, interpolation: str):
        # Interpolation is treated as a global "look" setting (applies to
        # every cell's initial display), but like colormap/clim, the View
        # menu only edits the active cell live -- other cells keep
        # whatever interpolation they were created with until re-activated
        # or reloaded.
        self.current_interpolation = interpolation
        self.settings.setValue("interpolation", interpolation)
        self.view_grid.active_view().set_interpolation(interpolation)

    def _apply_theme(self):
        palette = THEMES[self._resolve_theme(self.current_theme_name)]
        self.setStyleSheet(build_qss(palette, self.ui_scale))
        self.schematic.apply_palette(palette)
        self.schematic.set_ui_scale(self.ui_scale)
        # Nothing inside a matplotlib canvas (colorbar ticks/label, the
        # room overlay's line widths, ...) responded to UI Scale at all
        # before this -- only the Qt-side chrome (QSS fonts/padding) did,
        # so the plot itself looked completely unaffected by the setting.
        # SliceView.set_ui_scale -> MplCanvas.set_dpi_scale scales the
        # figure's DPI, which scales every point-sized artist together.
        for cell in self.view_grid.visible_cells():
            cell.view.set_ui_scale(self.ui_scale)
        self._refresh_toggle_icons(palette)
        # RC polish: every matplotlib plot follows the theme's chrome colors.
        from widgets import set_plot_theme
        set_plot_theme(palette)

    @staticmethod
    def _resolve_theme(name: str) -> str:
        """Map the chosen theme to a concrete palette key. 'system' follows the
        OS appearance (macOS); anything unknown falls back to light."""
        if name == "system":
            return _detect_system_theme()
        return name if name in THEMES else "light"
        self.view_grid.apply_accent(palette.accent)
        # Card shadows are Python-side (QGraphicsDropShadowEffect, not QSS --
        # see theme.apply_card_shadow), so a theme switch has to reapply
        # them explicitly: light/dark need different shadow opacity.
        for section in self.findChildren(CollapsibleSection):
            apply_card_shadow(section.card, palette)
        # The QSS restyle doesn't touch the matplotlib canvases themselves,
        # but the cached blit backgrounds are invalidated defensively per
        # M1.3.3's spec (resize/theme/colormap changes all recapture) in
        # case the canvas ever becomes theme-aware later -- every visible
        # cell, not just the active one.
        for cell in self.view_grid.visible_cells():
            cell.view.capture_background()

    def _refresh_toggle_icons(self, palette):
        """Icon color is redrawn per theme so it stays legible against both
        the light and dark control-panel background (ROADMAP §4 M1.6.4).
        FireLab roadmap Phase 3: these are now animated physical-mirror
        widgets (CandleCard/DoorWidget/VentWidget), each owning its own
        icon regeneration -- set_palette() just tells them which colors
        to use next time they redraw, same intent as the old set_icon()
        calls this replaces."""
        self.candle_toggle.set_palette(palette)
        self.door_toggle.set_palette(palette)
        self.vod_toggle.set_palette(palette)
        self.voc_toggle.set_palette(palette)
        self.inspector_stack.set_palette(palette)

    # -------------------------------------------------------- misc/window
    def _setup_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_play_pause)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self._restart_simulation)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, activated=self.close)
        # Investigation History (V6-M4): back/forward through recorded
        # selections -- previously only reachable via the removed Context
        # Panel's buttons (UX consolidation pass); browser-style shortcuts
        # keep the feature reachable without dedicating screen space to it.
        QtWidgets.QShortcut(QtGui.QKeySequence("Alt+Left"), self, activated=self._history_back)
        QtWidgets.QShortcut(QtGui.QKeySequence("Alt+Right"), self, activated=self._history_forward)
        # M1.4.3: frame/second stepping while paused or playing.
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self, activated=lambda: self.time_controller.step(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self, activated=lambda: self.time_controller.step(1))
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Left"), self,
            activated=lambda: self.time_controller.step(-self.time_controller.timesteps_per_second),
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Right"), self,
            activated=lambda: self.time_controller.step(self.time_controller.timesteps_per_second),
        )
        # FireLab roadmap Phase 1: 1-7 jump straight to a nav-rail page, in
        # the same display order as the rail itself.
        for i, key in enumerate(("home", "live", "compare", "dataset", "analysis", "export", "about"), start=1):
            QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), self, activated=lambda k=key: self._navigate_to(k))
        # Demo-script bookmarks (FireLab roadmap Phase 5): Ctrl+Shift+<n>
        # records the current (page, scenario, time) into slot n;
        # Shift+<n> jumps back to it -- lets a presenter move beat-to-beat
        # without touching a mouse.
        for slot in range(1, 10):
            QtWidgets.QShortcut(
                QtGui.QKeySequence(f"Ctrl+Shift+{slot}"), self,
                activated=lambda s=slot: self._record_bookmark(s),
            )
            QtWidgets.QShortcut(
                QtGui.QKeySequence(f"Shift+{slot}"), self,
                activated=lambda s=slot: self._jump_to_bookmark(s),
            )

    def _record_bookmark(self, slot: int) -> None:
        cell = self.view_grid.active_cell()
        self._demo_bookmarks[slot] = {
            "page": self._active_page_key,
            "case_index": cell.case_index if cell is not None and cell.cell_type == "slice" else None,
            "time_index": self.time_controller.index,
        }
        self.statusBar().showMessage(f"Bookmark {slot} recorded.", 3000)

    def _jump_to_bookmark(self, slot: int) -> None:
        bookmark = self._demo_bookmarks.get(slot)
        if bookmark is None:
            self.statusBar().showMessage(f"Bookmark {slot} is empty.", 3000)
            return
        self._navigate_to(bookmark["page"])
        if bookmark["page"] == "live" and bookmark["case_index"] is not None:
            cell = self.view_grid.active_cell()
            if cell is not None and cell.cell_type == "slice":
                self._select_scenario_in_cell(cell, bookmark["case_index"])
        self.time_controller.seek(bookmark["time_index"])

    def _toggle_effects_master_switch(self) -> None:
        """Esc long-press (FireLab roadmap Phase 5): an emergency "effects
        off" switch -- toggles the same View -> Cinematic fire view
        control every cinematic cell already listens to, so this is a
        thin trigger on existing, tested machinery, not new per-cell
        bookkeeping."""
        self.cinematic_action.setChecked(not self.cinematic_action.isChecked())
        self._set_cinematic_enabled(self.cinematic_action.isChecked())
        state = "enabled" if self.cinematic_action.isChecked() else "disabled"
        self.statusBar().showMessage(f"Cinematic effects {state}.", 3000)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and not event.isAutoRepeat():
            if not self._esc_hold_timer.isActive():
                self._esc_hold_timer.start(600)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and not event.isAutoRepeat():
            self._esc_hold_timer.stop()
        super().keyReleaseEvent(event)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_play_pause(self):
        if self.time_controller.is_playing():
            self._stop_simulation()
        else:
            self._start_simulation()

    # --- Analysis-page playback transport (RC polish) --------------------
    def _build_analysis_playback_bar(self) -> QtWidgets.QWidget:
        """A compact transport for the Analysis page, driving the *same*
        TimeController as the Live Viewer -- so the temporal analysis panels
        play/pause/step/loop in lockstep. Reuses TimelineWidget + the existing
        seek/play/loop/speed handlers; no new playback engine."""
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        def _tool(text, tip, slot):
            btn = QtWidgets.QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setAccessibleName(tip)
            btn.clicked.connect(slot)
            return btn

        row.addWidget(_tool("⏮", "Step back one frame",
                            lambda: self._on_seek_requested(max(0, self.time_controller.index - 1))))
        self.analysis_timeline = TimelineWidget()
        self.analysis_timeline.play_pause_clicked.connect(self._toggle_play_pause)
        self.analysis_timeline.seek_requested.connect(self._on_seek_requested)
        self.analysis_timeline.loop_toggled.connect(self.time_controller.set_loop)
        row.addWidget(self.analysis_timeline, 1)
        row.addWidget(_tool("⏭", "Step forward one frame",
                            lambda: self._on_seek_requested(
                                min(self._current_n_frames - 1, self.time_controller.index + 1))))
        row.addWidget(_tool("⏹", "Stop and return to the start", self._analysis_stop))
        self.analysis_speed = ToggleGroup(
            [("1x", 1), ("2x", 2), ("3x", 3)], default_index=0,
            accessible_name="Analysis playback speed")
        self.analysis_speed.value_changed.connect(self.time_controller.set_speed)
        row.addWidget(self.analysis_speed)
        return bar

    def _analysis_stop(self) -> None:
        if self.time_controller.is_playing():
            self._toggle_play_pause()
        self._on_seek_requested(0)

    def mouseDoubleClickEvent(self, event):
        # Kept as an explicit opt-in gesture, but no longer forced at startup.
        self._toggle_fullscreen()

    def _restore_window_state(self):
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1280, 820)

        splitter_state = self.settings.value("splitter_state")
        if splitter_state is not None:
            self.splitter.restoreState(splitter_state)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter_state", self.splitter.saveState())
        # V4-M6: autosave a draft session so an unsaved investigation is
        # recoverable next launch (best-effort; never blocks close). Gated
        # only on having a manifest -- not on the (now-removed, Analysis UX
        # + reliability pass) Sessions tab existing, since this must keep
        # firing for everyone regardless of that UI.
        if self.sim_data.manifest:
            try:
                session_store.save_draft(self._sessions_dir, self._collect_session_dict())
            except OSError:
                pass
        self.time_controller.pause()
        # A cache-miss prefetch left in flight when the window closes
        # (SimulationController._prefetch_workers is fire-and-forget by
        # design -- see that module's own docstring -- so closing doesn't
        # cancel it) must not leave QApplication's override-cursor stack
        # holding a WaitCursor push with no corresponding pop: that stack
        # is process-global, outliving this one window, so an unresolved
        # busy state here would otherwise show a stuck wait cursor (or, in
        # tests, a stale cursor bleeding into whatever runs next).
        if self._busy:
            self._end_busy_state()
        # Same reasoning as the busy-cursor cleanup above: the kiosk
        # controller's event filter lives on the process-global
        # QApplication, not this window, so it must be explicitly removed
        # here rather than left to Python GC timing (see kiosk.py's
        # shutdown() docstring for the measured cost of skipping this).
        self._kiosk.shutdown()
        super().closeEvent(event)
