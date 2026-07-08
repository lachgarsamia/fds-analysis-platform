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

from config import DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC
from theme import THEMES, build_qss
from widgets import MplCanvas, ToggleGroup, CollapsibleSection
from simulation_controller import SimulationController
from data_provider import SimulationData

ORG_NAME = "FZJuelich"
APP_NAME = "FDSSLCFVisualizer"

# Colormap options: keep the original gist_heat, add colorblind-safe alternatives.
COLORMAPS = [
    ("Heat (gist_heat)", "gist_heat"),
    ("Viridis (colorblind-safe)", "viridis"),
    ("Cividis (colorblind-safe)", "cividis"),
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
        self.controller.frame_ready.connect(self._on_frame)
        self.controller.error.connect(self._on_sim_error)

        self.current_theme_name = self.settings.value("theme", "dark")
        self.ui_scale = float(self.settings.value("ui_scale", 1.0))
        self.current_colormap = self.settings.value("colormap", "gist_heat")

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
        self.restart_button.setEnabled(False)
        self.restart_button.clicked.connect(self._restart_simulation)

        for b in (self.start_button, self.stop_button, self.restart_button):
            transport_row.addWidget(b)
        outer.addLayout(transport_row)

        self.time_progress = QtWidgets.QProgressBar()
        self.time_progress.setAccessibleName("Simulation time progress")
        self.time_progress.setFormat("t = %v s")
        self.time_progress.setRange(0, 100)
        outer.addWidget(self.time_progress)

        outer.addWidget(self._divider())

        # --- Scenario sections ----------------------------------------------
        speed_section = CollapsibleSection("Playback speed")
        self.speed_toggle = ToggleGroup(
            [("1x", 1), ("2x", 2), ("3x", 3)], default_index=0,
            accessible_name="Playback speed",
        )
        self.speed_toggle.value_changed.connect(self.controller.set_speed)
        speed_section.add_row(self.speed_toggle)
        outer.addWidget(speed_section)

        candle_section = CollapsibleSection("Candles")
        self.candle_toggle = ToggleGroup(
            [("1 candle", 0), ("2 candles", 1)], default_index=DEFAULT_CANDLES,
            accessible_name="Number of candles",
        )
        self.candle_toggle.value_changed.connect(self._on_candle_changed)
        candle_section.add_row(self.candle_toggle)
        outer.addWidget(candle_section)

        vod_section = CollapsibleSection("Ventilation opening damper (VOD)")
        self.vod_toggle = ToggleGroup(
            [("Open", 0), ("Closed", 1), ("HVAC", 2)], default_index=DEFAULT_VOD,
            accessible_name="VOD state",
        )
        self.vod_toggle.value_changed.connect(self._on_vod_changed)
        vod_section.add_row(self.vod_toggle)
        outer.addWidget(vod_section)

        voc_section = CollapsibleSection("Ventilation opening cover (VOC)")
        self.voc_toggle = ToggleGroup(
            [("Open", 0), ("Closed", 1)], default_index=DEFAULT_VOC,
            accessible_name="VOC state",
        )
        self.voc_toggle.value_changed.connect(self._on_voc_changed)
        voc_section.add_row(self.voc_toggle)
        outer.addWidget(voc_section)

        door_section = CollapsibleSection("Door")
        # Options list is [("Wide open", 1), ("Narrow", 0)]; DEFAULT_DOOR=1 is
        # at position 0, so default_index=0 correctly preselects "Wide open".
        self.door_toggle = ToggleGroup(
            [("Wide open", 1), ("Narrow", 0)], default_index=0,
            accessible_name="Door state",
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
        first_frame = self.controller.store.get(self.controller.current_case_index())[0, :, :]
        self.heatmap = self.ax.imshow(
            first_frame,
            cmap=self.current_colormap,
            aspect="auto",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.fig.subplots_adjust(top=0.97, bottom=0.03, left=0.02, right=0.95)
        self.colorbar = self.canvas.fig.colorbar(self.heatmap, fraction=0.04, pad=0.02)
        self.colorbar.set_label("Temperature (°C)")
        self.heatmap.set_clim(vmax=self.temp_slider.value())
        self.canvas.draw_idle()

    def _redraw(self, frame):
        self.heatmap.set_data(frame)
        self.heatmap.set_clim(vmax=self.temp_slider.value())
        self.canvas.draw_idle()

    # -------------------------------------------------------- signal slots
    def _on_frame(self, frame, current_time, index):
        self._redraw(frame)
        self.time_progress.setValue(current_time)
        self.statusBar().showMessage(f"Simulating - t = {current_time}s", 2000)

    def _on_sim_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Simulation error", message)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    def _on_candle_changed(self, value):
        self.controller.set_candles(value)
        self._refresh_paused_frame()

    def _on_vod_changed(self, value):
        self.controller.set_vod(value)
        self._refresh_paused_frame()

    def _on_voc_changed(self, value):
        self.controller.set_voc(value)
        self._refresh_paused_frame()

    def _on_door_changed(self, value):
        self.controller.set_door(value)
        self._refresh_paused_frame()

    def _refresh_paused_frame(self):
        """When paused, changing a scenario toggle should still update the
        plot immediately instead of waiting for the next Start press.

        Note: if the newly selected scenario isn't cached, this synchronously
        parses it on the GUI thread (~1-1.5s pause) -- same documented
        trade-off as the original app's update_figure_once(). While playing,
        the controller's worker thread loads scenarios on its own thread instead.
        """
        if not self.controller.is_running():
            self._redraw(self.controller.current_frame())

    def _on_temp_changed(self, value):
        self.temp_label.setText(f"{value} °C")
        self.heatmap.set_clim(vmax=value)
        self.canvas.draw_idle()

    def _start_simulation(self):
        self.controller.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.restart_button.setEnabled(True)

    def _stop_simulation(self):
        self.controller.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _restart_simulation(self):
        self.controller.restart()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.time_progress.setValue(0)

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
        self.canvas.draw_idle()

    def _apply_theme(self):
        palette = THEMES[self.current_theme_name]
        self.setStyleSheet(build_qss(palette, self.ui_scale))

    # -------------------------------------------------------- misc/window
    def _setup_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_play_pause)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self._restart_simulation)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, activated=self.close)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_play_pause(self):
        if self.start_button.isEnabled():
            self._start_simulation()
        elif self.stop_button.isEnabled():
            self._stop_simulation()

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
        self.controller.stop()
        super().closeEvent(event)
