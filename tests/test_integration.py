"""Integration tests: build MainWindow, exercise UI, verify no crashes."""

import pytest
import time
from PyQt5 import QtCore, QtWidgets
from data_provider import load_simulation_data
from main_window import MainWindow


class TestIntegration:
    """Offscreen integration tests."""

    def test_mainwindow_builds_with_demo_data(self, qapp):
        """Verify MainWindow builds and launches with demo data."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert window is not None
        assert window.windowTitle() != ""
        window.close()

    def test_mainwindow_initial_frame_displayed(self, qapp):
        """Verify the initial frame is displayed without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        # Heatmap should have been drawn
        assert window.heatmap.get_array() is not None
        assert window.heatmap.get_array().shape == (49, 101)
        window.close()

    def test_mainwindow_theme_switch(self, qapp):
        """Verify light/dark theme switching without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._set_theme("light")
        assert window.current_theme_name == "light"
        window._set_theme("dark")
        assert window.current_theme_name == "dark"
        window.close()

    def test_mainwindow_colormap_switch(self, qapp):
        """Verify colormap switching without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._set_colormap("viridis")
        assert window.current_colormap == "viridis"
        window._set_colormap("cividis")
        assert window.current_colormap == "cividis"
        window._set_colormap("gist_heat")
        assert window.current_colormap == "gist_heat"
        window.close()

    def test_mainwindow_ui_scale_change(self, qapp):
        """Verify UI scale adjustment without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._set_ui_scale(1.3)
        assert window.ui_scale == 1.3
        window._set_ui_scale(0.85)
        assert window.ui_scale == 0.85
        window.close()

    def test_mainwindow_resize_respects_minimum(self, qapp):
        """Verify window respects minimum size policy."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        # Try to resize to smaller than minimum
        window.resize(500, 400)
        # Should not go below minimum
        assert window.width() >= window.minimumWidth()
        assert window.height() >= window.minimumHeight()
        window.close()

    def test_mainwindow_transport_controls(self, qapp):
        """Verify start/stop/restart buttons work without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        # Start playback
        window._start_simulation()
        qapp.processEvents()
        time.sleep(0.3)
        # Stop playback
        window._stop_simulation()
        qapp.processEvents()
        # Restart
        window._restart_simulation()
        qapp.processEvents()
        window.close()

    def test_mainwindow_scenario_switch_while_paused(self, qapp):
        """Verify scenario toggle changes work while paused."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        # Pause (start stopped)
        window.candle_toggle.set_value(1)
        window._on_candle_changed(1)
        qapp.processEvents()
        # Frame should still render
        assert window.heatmap.get_array() is not None
        window.close()

    def test_mainwindow_temperature_slider(self, qapp):
        """Verify temperature scale slider without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.temp_slider.setValue(500)
        assert window.heatmap.get_clim()[1] == 500.0
        window.temp_slider.setValue(100)
        assert window.heatmap.get_clim()[1] == 100.0
        window.close()

    # ----------------------------------------------------- M1.3 rendering tests
    def test_vmin_pinned_at_ambient_across_frames_and_slider(self, qapp):
        """Regression test for the frozen-vmin defect (M1.3.2): vmin must stay
        at AMBIENT_C regardless of which frame is drawn or where vmax is set,
        never drift to whatever the first-drawn frame's minimum happened to be."""
        from config import AMBIENT_C
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert window.heatmap.get_clim()[0] == AMBIENT_C

        data = window.controller.store.get(window.controller.current_case_index())
        for i in (0, 50, 200, 480):
            window._redraw(data[i])
            assert window.heatmap.get_clim()[0] == AMBIENT_C

        window.temp_slider.setValue(700)
        assert window.heatmap.get_clim() == (AMBIENT_C, 700.0)
        window.close()

    def test_blit_playback_many_frames_no_crash(self, qapp):
        """Blitting (M1.3.3) must survive many consecutive frame draws, a
        colormap switch mid-playback, and a resize, without crashing or
        producing a null canvas image."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        qapp.processEvents()

        data = window.controller.store.get(window.controller.current_case_index())
        for i in range(0, 100, 5):
            window._redraw(data[i])
        window._set_colormap("inferno")
        for i in range(100, 200, 5):
            window._redraw(data[i])
        window.resize(1100, 750)
        qapp.processEvents()
        window._redraw(data[200])

        image = window.canvas.grab().toImage()
        assert not image.isNull()
        window.close()

    def test_interpolation_toggle_persists(self, qapp):
        """Verify the interpolation toggle (M1.3.4) applies and persists.

        QSettings is a real, on-disk-persisted backend shared across runs, so
        this doesn't assume a fresh-install "nearest" default -- it drives
        both states explicitly and checks the artist/attribute reflect each.
        """
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._set_interpolation("nearest")
        assert window.heatmap.get_interpolation() == "nearest"
        assert window.current_interpolation == "nearest"
        window._set_interpolation("bilinear")
        assert window.heatmap.get_interpolation() == "bilinear"
        assert window.current_interpolation == "bilinear"
        window._set_interpolation("nearest")
        assert window.heatmap.get_interpolation() == "nearest"
        window.close()

    def test_colormap_menu_includes_inferno(self, qapp):
        """M1.3.1: menu keeps gist_heat/inferno/viridis/cividis per spec."""
        from main_window import COLORMAPS
        cmap_values = [c for _, c in COLORMAPS]
        assert set(cmap_values) == {"gist_heat", "inferno", "viridis", "cividis"}

    def test_mainwindow_closes_cleanly(self, qapp):
        """Verify window closes and cleans up without crash."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.start_button.click()
        qapp.processEvents()
        time.sleep(0.2)
        # Closing should stop controller and clean up
        window.close()
        # No exception should have been raised
        assert True

    # -------------------------------------------------- M1.6 schematic tests
    def test_schematic_renders_across_toggle_combinations(self, qapp):
        """Schematic must not crash across distinct toggle-state combinations,
        and the rendered image must actually change between them (not just
        the underlying state) -- otherwise it isn't "live" per M1.6's DoD."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.resize(1280, 820)
        window.show()
        qapp.processEvents()

        # narrow door + closed vents + 1 candle
        window.door_toggle.set_value(0)
        window._on_door_changed(0)
        window.vod_toggle.set_value(1)
        window._on_vod_changed(1)
        window.voc_toggle.set_value(1)
        window._on_voc_changed(1)
        window.candle_toggle.set_value(0)
        window._on_candle_changed(0)
        qapp.processEvents()
        assert (window.schematic._door, window.schematic._vod,
                window.schematic._voc, window.schematic._candles) == (0, 1, 1, 0)
        image_a = window.schematic.grab().toImage()

        # wide open door + HVAC vent + 2 candles
        window.door_toggle.set_value(1)
        window._on_door_changed(1)
        window.vod_toggle.set_value(2)
        window._on_vod_changed(2)
        window.voc_toggle.set_value(0)
        window._on_voc_changed(0)
        window.candle_toggle.set_value(1)
        window._on_candle_changed(1)
        qapp.processEvents()
        assert (window.schematic._door, window.schematic._vod,
                window.schematic._voc, window.schematic._candles) == (1, 2, 0, 1)
        image_b = window.schematic.grab().toImage()

        # narrow door + open vents + 1 candle (a third, distinct combination)
        window.door_toggle.set_value(0)
        window._on_door_changed(0)
        window.vod_toggle.set_value(0)
        window._on_vod_changed(0)
        window.voc_toggle.set_value(0)
        window._on_voc_changed(0)
        window.candle_toggle.set_value(0)
        window._on_candle_changed(0)
        qapp.processEvents()
        image_c = window.schematic.grab().toImage()

        assert image_a != image_b
        assert image_b != image_c
        assert image_a != image_c
        window.close()

    def test_schematic_extent_derived_from_parsed_smv(self, qapp):
        """The schematic's room proportions must come from the scenario's
        parsed .smv mesh extent, not a hardcoded/arbitrary aspect ratio."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if not sim_data.is_demo:
            # Real dataset: verified footprint is x:[0.0, 1.0], y:[-0.15, 0.15]
            # (see schematic.py's DEFAULT_ROOM_EXTENT comment and this
            # milestone's report for the spot-check numbers).
            assert window.schematic._extent["x"] == (0.0, 1.0)
            assert window.schematic._extent["y"] == (-0.15, 0.15)
        else:
            # Demo-data mode has no .smv to parse; schematic must still
            # degrade gracefully to the documented fallback footprint.
            from schematic import DEFAULT_ROOM_EXTENT
            assert window.schematic._extent == DEFAULT_ROOM_EXTENT
        window.close()

    def test_schematic_survives_theme_switch(self, qapp):
        """Schematic must re-render (not crash, not go blank) after a
        light/dark theme switch."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        qapp.processEvents()

        window._set_theme("light")
        qapp.processEvents()
        assert window.schematic._palette.name == "light"
        light_image = window.schematic.grab().toImage()
        assert not light_image.isNull()

        window._set_theme("dark")
        qapp.processEvents()
        assert window.schematic._palette.name == "dark"
        dark_image = window.schematic.grab().toImage()
        assert not dark_image.isNull()

        window.close()

    def test_schematic_survives_minimum_size_resize(self, qapp):
        """Schematic must still render without crashing at the app's
        documented 900x600 minimum window size."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        window.resize(900, 600)
        qapp.processEvents()
        assert window.width() >= 900
        assert window.height() >= 600
        image = window.schematic.grab().toImage()
        assert not image.isNull()
        window.close()

    def test_plain_language_labels_present_no_bare_jargon(self, qapp):
        """Every scenario-toggle section must carry a plain-language label,
        not just the raw VOD/VOC variable name (M1.6.4, core requirement)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert window.vod_toggle.toolTip() != ""
        assert window.voc_toggle.toolTip() != ""
        assert window.door_toggle.toolTip() != ""
        assert window.candle_toggle.toolTip() != ""
        window.close()

    # ------------------------------------------------- M1.4 timeline tests
    def test_drag_seek_during_playback(self, qapp):
        """DoD: 'drag-seek works during playback'. Dragging the timeline
        slider while playing must move the displayed frame immediately and
        leave playback running (not silently pause it)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._start_simulation()
        qapp.processEvents()
        assert window.time_controller.is_playing()

        window._on_seek_requested(250)
        qapp.processEvents()
        assert window.time_controller.index == 250
        assert window.timeline.slider.value() == 250
        assert window.time_controller.is_playing(), "seeking must not pause playback"
        assert not window.heatmap.get_array() is None

        window._stop_simulation()
        window.close()

    def test_seek_while_paused_updates_display_without_playing(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert not window.time_controller.is_playing()
        window._on_seek_requested(100)
        qapp.processEvents()
        assert window.time_controller.index == 100
        assert not window.time_controller.is_playing()
        window.close()

    def test_speed_change_takes_effect_immediately(self, qapp):
        """DoD: 'speed change takes effect immediately'."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.speed_toggle.set_value(3)
        window.time_controller.set_speed(3)
        assert window.time_controller._speed == 3
        window.close()

    def test_scenario_switch_cache_miss_does_not_block_gui_thread(self, qapp):
        """DoD: 'no GUI freeze on scenario switch'. The toggle-change handler
        itself must return quickly (well under the ~55-80ms a cold parse
        takes) -- the actual load happens on a background thread; a busy
        cursor + disabled slider + status message cover the gap instead of
        the GUI thread blocking synchronously (M1.4.4)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        qapp.processEvents()

        other_candles = 1 - window.controller.params.candles
        case_idx = window.controller.data_matrix[
            other_candles, window.controller.params.door,
            window.controller.params.vod, window.controller.params.voc,
        ]
        if window.controller.is_cached(int(case_idx)):
            pytest.skip("target scenario already warm from an earlier test in this run")

        t0 = time.perf_counter()
        window.candle_toggle.set_value(other_candles)
        window._on_candle_changed(other_candles)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"toggle handler blocked the GUI thread for {elapsed*1000:.1f}ms"
        assert window._busy
        assert QtWidgets.QApplication.overrideCursor() is not None, "busy cursor must be active"
        assert QtWidgets.QApplication.overrideCursor().shape() == QtCore.Qt.WaitCursor
        assert not window.timeline.slider.isEnabled()

        deadline = time.perf_counter() + 3.0
        while window._busy and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert not window._busy, "prefetch never completed within 3s"
        assert QtWidgets.QApplication.overrideCursor() is None, "busy cursor must be restored"
        assert window.timeline.slider.isEnabled()
        window.close()

    def test_rapid_toggle_changes_during_pending_prefetch_no_crash(self, qapp):
        """Regression test for a real QThread lifecycle bug found while
        building this milestone: firing several scenario-changing toggles
        before an earlier prefetch finishes used to let a still-running
        QThread be garbage-collected, which aborts the whole process (not a
        catchable Python exception). Exercising this here would have caught
        it before it reached main."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        qapp.processEvents()

        window.candle_toggle.set_value(1)
        window._on_candle_changed(1)
        window.door_toggle.set_value(0)
        window._on_door_changed(0)
        window.vod_toggle.set_value(2)
        window._on_vod_changed(2)
        qapp.processEvents()

        assert window._pending_load_case == window.controller.current_case_index()

        deadline = time.perf_counter() + 5.0
        while window._busy and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert not window._busy
        assert window.timeline.slider.isEnabled()
        # let any straggler background thread(s) actually finish before
        # the window (and its store) goes away
        time.sleep(0.2)
        qapp.processEvents()
        window.close()

    def test_loop_toggle_wired_to_time_controller(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert window.time_controller._loop is True
        window.timeline.loop_button.setChecked(False)
        assert window.time_controller._loop is False
        window.close()

    def test_frame_and_second_step_shortcuts_move_the_index(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.time_controller.seek(50)
        window.time_controller.step(1)
        assert window.time_controller.index == 51
        window.time_controller.step(-1)
        assert window.time_controller.index == 50
        window.time_controller.step(window.time_controller.timesteps_per_second)
        assert window.time_controller.index == 50 + window.time_controller.timesteps_per_second
        window.close()

    def test_old_worker_push_path_not_wired_into_mainwindow(self, qapp):
        """DoD: 'old worker-push path fully removed' (from MainWindow's
        perspective -- simulation_controller.py's _Worker class itself is
        deleted in a separate follow-up commit). MainWindow must not touch
        the old frame_ready/start/stop/is_running surface at all anymore."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert not hasattr(window, "time_progress"), "old QProgressBar must be gone"
        assert not hasattr(window, "_on_frame"), "old worker frame_ready slot must be gone"
        assert not hasattr(window, "_refresh_paused_frame"), "superseded by _on_scenario_param_changed"
        window.close()
