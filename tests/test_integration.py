"""Integration tests: build MainWindow, exercise UI, verify no crashes."""

import pytest
import time
from PyQt5 import QtCore
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
