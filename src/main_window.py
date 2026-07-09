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

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import cm

from config import DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC, AMBIENT_C
from theme import THEMES, build_qss
from widgets import MplCanvas, ToggleGroup, CollapsibleSection, TimelineWidget
from simulation_controller import SimulationController
from time_controller import TimeController
from data_provider import SimulationData
from schematic import SchematicWidget, flame_icon, door_icon, vent_icon, resolve_room_extent

ORG_NAME = "FZJuelich"
APP_NAME = "FDSSLCFVisualizer"

# Colormap options. Default (gist_heat) confirmed by the M1.3s validation
# spike (docs/spike-parser-validation.md): already a black-red-orange-yellow-
# white blackbody/flame progression, kept as-is rather than replaced.
COLORMAPS = [
    ("Heat (gist_heat)", "gist_heat"),
    ("Inferno", "inferno"),
    ("Viridis (colorblind-safe)", "viridis"),
    ("Cividis (colorblind-safe)", "cividis"),
]

INTERPOLATIONS = [
    ("Nearest", "nearest"),
    ("Bilinear", "bilinear"),
]

MIN_WIDTH = 900
MIN_HEIGHT = 600


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sim_data: SimulationData):
        super().__init__()
        self.settings = QtCore.QSettings(ORG_NAME, APP_NAME)
        self.sim_data = sim_data

        self.controller = SimulationController(
            sim_data.store, sim_data.data_matrix, sim_data.timesteps_per_second
        )
        # Match the controller's initial parameters to the toggle defaults
        # declared in config.py, so the first frame drawn matches what the
        # control panel shows as selected (candles/door/vod/voc all start
        # at their DEFAULT_* value, not implicitly at 0).
        self.controller.params.candles = DEFAULT_CANDLES
        self.controller.params.door = DEFAULT_DOOR
        self.controller.params.vod = DEFAULT_VOD
        self.controller.params.voc = DEFAULT_VOC
        self.controller.prefetch_finished.connect(self._on_prefetch_finished)
        self.controller.prefetch_error.connect(self._on_prefetch_error)

        # M1.4: pull-based playback clock. frame_count_fn must never trigger
        # a load itself -- it just reports self._current_n_frames, which is
        # only ever updated once a scenario is confirmed loaded (see
        # _on_scenario_param_changed / _on_prefetch_finished below).
        self._current_n_frames = 0
        self._busy = False               # a cache-miss prefetch is in flight
        self._pending_load_case = None   # which case_index that prefetch is for
        self._was_playing_before_load = False
        self.time_controller = TimeController(
            lambda: self._current_n_frames, sim_data.timesteps_per_second
        )
        self.time_controller.time_changed.connect(self._on_time_changed)
        self.time_controller.playing_changed.connect(self._on_playing_changed)

        self.current_theme_name = self.settings.value("theme", "dark")
        self.ui_scale = float(self.settings.value("ui_scale", 1.0))
        self.current_colormap = self.settings.value("colormap", "gist_heat")
        self.current_interpolation = self.settings.value("interpolation", "nearest")

        self.setWindowTitle("FDS SLCF Fire Visualizer" + (" (demo data)" if sim_data.is_demo else ""))
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self._build_menu()
        self._build_central_widget()
        self._build_status_bar()
        self._apply_theme()
        self._restore_window_state()
        self._setup_shortcuts()

        if sim_data.is_demo:
            self.statusBar().showMessage(
                "Running with generated demo data (fds/sim/ not found).", 8000
            )

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menu_bar = self.menuBar()

        view_menu = menu_bar.addMenu("&View")

        theme_menu = view_menu.addMenu("Theme")
        self.theme_action_group = QtWidgets.QActionGroup(self)
        for key, label in (("light", "Light"), ("dark", "Dark")):
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

        colormap_menu = view_menu.addMenu("Colormap")
        self.colormap_action_group = QtWidgets.QActionGroup(self)
        for label, cmap in COLORMAPS:
            action = QtWidgets.QAction(label, self, checkable=True)
            action.setChecked(cmap == self.current_colormap)
            action.triggered.connect(lambda _checked, c=cmap: self._set_colormap(c))
            self.colormap_action_group.addAction(action)
            colormap_menu.addAction(action)

        interpolation_menu = view_menu.addMenu("Interpolation")
        self.interpolation_action_group = QtWidgets.QActionGroup(self)
        for label, interp in INTERPOLATIONS:
            action = QtWidgets.QAction(label, self, checkable=True)
            action.setChecked(interp == self.current_interpolation)
            action.triggered.connect(lambda _checked, i=interp: self._set_interpolation(i))
            self.interpolation_action_group.addAction(action)
            interpolation_menu.addAction(action)

        fullscreen_action = QtWidgets.QAction("Toggle Fullscreen\tF11", self)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

    def _build_central_widget(self):
        central = QtWidgets.QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(True)  # user can fully collapse the panel
        root_layout.addWidget(self.splitter)

        self.splitter.addWidget(self._build_control_panel())
        self.splitter.addWidget(self._build_plot_panel())
        # Control panel gets a fixed-ish starting share; plot gets the rest
        # and does the growing when the window resizes (stretch factors).
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([320, 880])

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

        outer.addWidget(self._divider())

        # --- Simulation transport controls ---------------------------------
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
        outer.addLayout(transport_row)

        # M1.4: interactive scrubber (play/pause + seek slider + time label +
        # loop toggle), replacing the old read-only QProgressBar.
        self.timeline = TimelineWidget()
        self.timeline.play_pause_clicked.connect(self._toggle_play_pause)
        self.timeline.seek_requested.connect(self._on_seek_requested)
        self.timeline.loop_toggled.connect(self.time_controller.set_loop)
        outer.addWidget(self.timeline)

        outer.addWidget(self._divider())

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

        candle_section = CollapsibleSection("Number of candles")
        self.candle_toggle = ToggleGroup(
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

        # Plain-language primary label with the technical code (VOD) kept as
        # secondary text, per the 2026-07-09c non-specialist label pass --
        # raw variable names must never appear unexplained (ROADMAP §4 M1.6.4).
        vod_section = CollapsibleSection("Air vent 1 (VOD)")
        self.vod_toggle = ToggleGroup(
            [("Open", 0), ("Closed", 1), ("HVAC", 2)], default_index=DEFAULT_VOD,
            accessible_name="Air vent 1 (VOD) state",
        )
        self.vod_toggle.setToolTip(
            "Opens, closes, or connects an air vent in the room to a fan "
            "(HVAC). Open lets smoke and hot air escape and fresh air in; "
            "closed traps heat and smoke inside."
        )
        self.vod_toggle.value_changed.connect(self._on_vod_changed)
        vod_section.add_row(self.vod_toggle)
        outer.addWidget(vod_section)

        voc_section = CollapsibleSection("Air vent 2 (VOC)")
        self.voc_toggle = ToggleGroup(
            [("Open", 0), ("Closed", 1)], default_index=DEFAULT_VOC,
            accessible_name="Air vent 2 (VOC) state",
        )
        self.voc_toggle.setToolTip(
            "Opens or closes a second air vent in the room. Works the same "
            "way as Air vent 1 -- open lets air move through, closed seals "
            "the room."
        )
        self.voc_toggle.value_changed.connect(self._on_voc_changed)
        voc_section.add_row(self.voc_toggle)
        outer.addWidget(voc_section)

        door_section = CollapsibleSection("Door opening width")
        # Options list is [("Wide open", 1), ("Narrow", 0)]; DEFAULT_DOOR=1 is
        # at position 0, so default_index=0 correctly preselects "Wide open".
        self.door_toggle = ToggleGroup(
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

        outer.addWidget(self._divider())

        # --- Display controls -------------------------------------------------
        temp_section = CollapsibleSection("Temperature scale (max)")
        temp_row = QtWidgets.QHBoxLayout()
        self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.temp_slider.setRange(50, 1000)
        self.temp_slider.setValue(300)
        self.temp_slider.setAccessibleName("Maximum temperature scale, degrees Celsius")
        self.temp_slider.setToolTip("Adjust the maximum temperature shown on the color scale")
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        self.temp_label = QtWidgets.QLabel("300 °C")
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

    def _build_plot_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = MplCanvas(panel)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        self.toolbar.setAccessibleName("Plot navigation toolbar: pan, zoom, save")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)  # canvas gets all extra vertical space

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
    def _init_plot(self):
        self.ax = self.canvas.fig.add_subplot(111)
        # Use the controller's actual default parameters (candles/door/vod/voc
        # as set in __init__) so the first frame shown matches what the
        # control-panel toggles display as selected.
        first_scenario = self.controller.store.get(self.controller.current_case_index())
        self._current_n_frames = first_scenario.shape[0]
        first_frame = first_scenario[0, :, :]
        self.heatmap = self.ax.imshow(
            first_frame,
            cmap=self.current_colormap,
            interpolation=self.current_interpolation,
            aspect="auto",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.fig.subplots_adjust(top=0.97, bottom=0.03, left=0.02, right=0.95)
        self.colorbar = self.canvas.fig.colorbar(self.heatmap, fraction=0.04, pad=0.02)
        self.colorbar.set_label("Temperature (°C)")
        # Explicit vmin so the color scale's lower bound is always ambient
        # temperature, not whatever frame 0 happened to show (M1.3.2).
        self.heatmap.set_clim(vmin=AMBIENT_C, vmax=self.temp_slider.value())
        self.canvas.capture_background()
        self.timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        self.timeline.set_index(0)

    def _redraw(self, frame):
        """Per-frame playback path: blit instead of a full draw_idle() so
        only the image artist's pixels are re-rendered (M1.3.3). Anything
        that changes what's underneath the image (clim/colormap/theme/
        interpolation/resize) goes through the dedicated setters below,
        which do a full draw + recapture instead of calling this."""
        self.heatmap.set_data(frame)
        self.canvas.blit_update(self.heatmap)

    # -------------------------------------------------------- signal slots
    def _on_time_changed(self, index: int):
        """TimeController's tick/seek signal (M1.4.1): pull the frame for
        the *current* scenario at `index` and redraw. Relies on M1.2's disk
        cache making an already-warm store.get() ~1-6ms -- cheap enough to
        call directly here without stalling the GUI thread on every tick."""
        case_idx = self.controller.current_case_index()
        frame = self.controller.store.get(case_idx)[index]
        self._redraw(frame)
        self.timeline.set_index(index)
        current_time = index / self.time_controller.timesteps_per_second
        self.statusBar().showMessage(f"t = {current_time:.1f} s", 2000)

    def _on_playing_changed(self, playing: bool):
        self.timeline.set_playing(playing)
        self.start_button.setEnabled(not playing)
        self.stop_button.setEnabled(playing)

    def _on_seek_requested(self, index: int):
        self.time_controller.seek(index)

    def _on_sim_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Simulation error", message)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    def _on_prefetch_finished(self, case_idx: int):
        """A background scenario load completed (M1.4.4). If the user has
        since switched to a different combination, this is stale -- ignore
        it silently, the busy state stays active for whichever request is
        still pending (or was already cleared by a cache hit)."""
        if case_idx != self._pending_load_case:
            return
        self._pending_load_case = None
        self._end_busy_state()
        self._sync_current_scenario(case_idx)
        if self._was_playing_before_load:
            self.time_controller.play()

    def _on_prefetch_error(self, message: str):
        self._pending_load_case = None
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
        """
        case_idx = self.controller.current_case_index()
        if self.controller.is_cached(case_idx):
            self._pending_load_case = None
            if self._busy:
                self._end_busy_state()
            self._sync_current_scenario(case_idx)
            return

        self._pending_load_case = case_idx
        if not self._busy:
            self._begin_busy_state()
        self.controller.prefetch(case_idx)

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
        self._current_n_frames = self.controller.store.get(case_idx).shape[0]
        self.timeline.set_range(self._current_n_frames, self.time_controller.timesteps_per_second)
        if not self.time_controller.is_playing():
            self._on_time_changed(self.time_controller.index)

    def _on_temp_changed(self, value):
        self.temp_label.setText(f"{value} °C")
        # vmin stays pinned at AMBIENT_C; only vmax moves with the slider.
        self.heatmap.set_clim(vmin=AMBIENT_C, vmax=value)
        # Full redraw (not blit): the colorbar's tick range depends on clim
        # and needs to actually repaint, and the cached blit background must
        # be recaptured so subsequent playback frames blit against it.
        self.canvas.capture_background()

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
        self.current_colormap = cmap
        self.settings.setValue("colormap", cmap)
        self.heatmap.set_cmap(cm.get_cmap(cmap))
        self.canvas.capture_background()

    def _set_interpolation(self, interpolation: str):
        self.current_interpolation = interpolation
        self.settings.setValue("interpolation", interpolation)
        self.heatmap.set_interpolation(interpolation)
        self.canvas.capture_background()

    def _apply_theme(self):
        palette = THEMES[self.current_theme_name]
        self.setStyleSheet(build_qss(palette, self.ui_scale))
        self.schematic.apply_palette(palette)
        self._refresh_toggle_icons(palette)
        # The QSS restyle doesn't touch the matplotlib canvas itself, but the
        # cached blit background is invalidated defensively per M1.3.3's spec
        # (resize/theme/colormap changes all recapture) in case the canvas
        # ever becomes theme-aware later.
        self.canvas.capture_background()

    def _refresh_toggle_icons(self, palette):
        """Icon color is redrawn per theme so it stays legible against both
        the light and dark control-panel background (ROADMAP §4 M1.6.4)."""
        color = palette.text_secondary
        self.candle_toggle.set_icon(flame_icon(color))
        self.door_toggle.set_icon(door_icon(color))
        self.vod_toggle.set_icon(vent_icon(color))
        self.voc_toggle.set_icon(vent_icon(color))

    # -------------------------------------------------------- misc/window
    def _setup_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_play_pause)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self._restart_simulation)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, activated=self.close)
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
        self.time_controller.pause()
        super().closeEvent(event)
