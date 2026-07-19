"""Integration tests: build MainWindow, exercise UI, verify no crashes."""

import pytest
import time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from data_provider import load_simulation_data
from main_window import MainWindow
from slice_key import DEFAULT_SLICE_KEY


def _drain_workers(qapp, workers: list, timeout: float = 5.0):
    """Pumps the event loop until a QThread worker list drains to empty
    (or `timeout` seconds pass) -- shared by every "fire-and-forget
    background load" path (prefetch, analytics feature index, auto-summary
    text) so tests that need the real result, not just the fact that a
    worker was started, can wait for it deterministically."""
    deadline = time.perf_counter() + timeout
    while workers and time.perf_counter() < deadline:
        qapp.processEvents()
        time.sleep(0.005)


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

    def test_bilinear_is_the_fresh_install_interpolation_default(self, qapp, monkeypatch):
        """GUI modernization pass, item 7: a never-configured install
        (nothing in QSettings yet) must default to bilinear, not the
        blocky "nearest" default matplotlib itself would use. Forces a
        clean QSettings.value() -- always returns the fallback -- rather
        than assuming this machine's real, persisted QSettings happens to
        be unconfigured (it may well already have a saved preference from
        another test run)."""
        from PyQt5 import QtCore
        monkeypatch.setattr(QtCore.QSettings, "value", lambda self, key, default=None, **kw: default)
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert window.current_interpolation == "bilinear"
        assert window.heatmap.get_interpolation() == "bilinear"
        window.close()

    def test_nearest_still_available_in_interpolation_menu(self, qapp):
        from main_window import INTERPOLATIONS
        values = [v for _label, v in INTERPOLATIONS]
        assert "nearest" in values
        assert "bilinear" in values

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
        """M1.3.1's stock options (gist_heat/inferno/viridis/cividis) stay
        available alongside the calibrated fds_fire/fds_flow defaults
        added by the GUI modernization pass."""
        from main_window import COLORMAPS
        cmap_values = [c for _, c in COLORMAPS]
        assert {"gist_heat", "inferno", "viridis", "cividis"}.issubset(set(cmap_values))
        assert {"fds_fire", "fds_flow"}.issubset(set(cmap_values))

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

    # ------------------------------------------------------ M2.5 browser tests
    def test_experiment_browser_lists_all_real_scenarios(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert window.experiment_browser is None
            window.close()
            return

        browser = window.experiment_browser
        assert browser is not None
        assert browser.model.rowCount() == len(sim_data.manifest) == 24
        headers = [
            browser.model.headerData(i, QtCore.Qt.Horizontal)
            for i in range(browser.model.columnCount())
        ]
        assert "Peak HRR (kW)" in headers
        assert "Energy (kJ)" in headers
        window.close()

    def test_experiment_browser_filter_sort_and_double_click_loads(self, qapp):
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("experiment browser is real-dataset only")
        window = MainWindow(sim_data)
        browser = window.experiment_browser

        browser.search_edit.setText("c2_")
        qapp.processEvents()
        assert browser.proxy.rowCount() == 12

        peak_col = next(
            i for i, (key, _label) in enumerate(browser.model.COLUMNS)
            if key == "peak_hrr_kw"
        )
        browser.table.sortByColumn(peak_col, QtCore.Qt.DescendingOrder)
        qapp.processEvents()
        assert browser.proxy.rowCount() == 12

        proxy_index = browser.proxy.index(0, 0)
        source_index = browser.proxy.mapToSource(proxy_index)
        summary = browser.model.data(source_index, QtCore.Qt.UserRole)
        browser._on_double_clicked(proxy_index)
        qapp.processEvents()

        deadline = time.perf_counter() + 3.0
        while window._busy and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        deadline = time.perf_counter() + 3.0
        while window.controller._prefetch_workers and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)

        assert window.view_grid.active_cell().case_index == summary.case_index
        assert window.controller.current_case_index() == summary.case_index
        window.close()

    def test_experiment_browser_open_grid_and_ensemble_actions(self, qapp):
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("experiment browser is real-dataset only")
        window = MainWindow(sim_data)

        window._open_browser_grid([0, 1, 2, 3])
        qapp.processEvents()
        deadline = time.perf_counter() + 3.0
        while (window._busy or window.controller._prefetch_workers) and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert window.view_grid.layout_name == "2x2"
        assert [cell.case_index for cell in window.view_grid.visible_cells()] == [0, 1, 2, 3]

        window._open_browser_ensemble([0, 1, 2])
        qapp.processEvents()
        cell = window.view_grid.active_cell()
        assert cell.cell_type == "ensemble"
        assert cell.ensemble_case_indices == [0, 1, 2]
        assert cell.view.heatmap is not None
        window.close()

    # ------------------------------------------------- M3.2.5 model evaluation
    def test_model_eval_button_matches_prediction_availability(self, qapp):
        """Button presence must track prediction_store.is_available, not
        one hard-coded state -- whether predictions/ exists on a given
        machine depends on whether ml/rollout.py has ever been run there,
        which varies by dev environment (this repo's own included)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        if window.prediction_store.is_available:
            assert window.experiment_browser.open_model_eval_button is not None
        else:
            assert window.experiment_browser.open_model_eval_button is None
        window.close()

    def test_model_eval_grid_populates_ground_truth_prediction_and_difference(
        self, qapp, monkeypatch, tmp_path,
    ):
        """Fabricates a tiny "predictions" export for one real scenario --
        shaped like ml/rollout.py's real output, but without needing an
        actual trained model -- to exercise the in-app wiring (button,
        1x3 layout, each cell pulling from the correct data source). The
        model's own correctness is ml/tests/'s job, not this test's."""
        import functools
        import json

        import prediction_store

        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        case_index = 3
        real_data = sim_data.store.get(case_index)
        fake_pred = real_data + 5.0
        np.save(tmp_path / f"{case_index}.npy", fake_pred)
        manifest = {"cases": {str(case_index): {"folder": "x", "n_frames": int(real_data.shape[0])}}}
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        patched_source = functools.partial(prediction_store.PredictionSource, predictions_dir=str(tmp_path))
        monkeypatch.setattr("main_window.PredictionSource", patched_source)

        window = MainWindow(sim_data)
        assert window.prediction_store.is_available
        assert window.experiment_browser.open_model_eval_button is not None

        window._open_browser_model_eval([case_index])
        qapp.processEvents()

        assert window.view_grid.layout_name == "1x3"
        ground_truth_cell, prediction_cell, difference_cell = window.view_grid.visible_cells()
        assert ground_truth_cell.cell_type == "slice"
        assert ground_truth_cell.store_override is None
        assert prediction_cell.cell_type == "slice"
        assert prediction_cell.store_override is window.prediction_store
        assert difference_cell.cell_type == "difference"
        assert difference_cell.store_override_b is window.prediction_store

        index = window.time_controller.index
        assert np.allclose(ground_truth_cell.view.heatmap.get_array(), real_data[index], atol=1e-3)
        assert np.allclose(prediction_cell.view.heatmap.get_array(), fake_pred[index], atol=1e-3)
        diff_array = difference_cell.view.heatmap.get_array()
        assert np.allclose(diff_array, -5.0, atol=1e-3)

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

        # _busy only tracks the *latest* requested scenario -- the earlier,
        # superseded toggles (candles=1, door=0) each started their own
        # worker too, and those are still in flight in the background even
        # though the UI has already moved on. Wait for *all* of them, then
        # confirm SimulationController._prefetch_workers actually drains
        # back to empty -- not just that the burst didn't crash, but that
        # every worker's cleanup fired and nothing was left referenced
        # forever (a leak that wouldn't crash a short-lived test process,
        # but would accumulate QThread objects over a long-running session).
        deadline = time.perf_counter() + 5.0
        while window.controller._prefetch_workers and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert window.controller._prefetch_workers == [], (
            "prefetch worker list must drain to empty once every in-flight "
            "load (including superseded ones) has actually finished"
        )
        window.close()

    def test_stale_prefetch_error_does_not_discard_newer_success(self, qapp):
        """Regression test for a real bug found while verifying the leak
        fix above: _on_prefetch_error didn't check case_idx against
        _pending_load_case the way _on_prefetch_finished already did. If an
        *older*, superseded load failed while a *newer* one was still in
        flight, the error handler would clear _pending_load_case out from
        under the newer request -- so when the newer request then
        succeeded, _on_prefetch_finished's own staleness guard would
        compare its case_idx against a _pending_load_case that had already
        been wrongly cleared to None, and silently discard a load that
        actually worked. This drives that exact interleaving: A fails,
        B (requested after A, before A's failure arrives) must still
        complete normally."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        qapp.processEvents()

        other_candles = 1 - window.controller.params.candles
        other_door = 1 - window.controller.params.door
        case_a = int(window.controller.data_matrix[
            other_candles, window.controller.params.door,
            window.controller.params.vod, window.controller.params.voc,
        ])
        # case_b as it will actually resolve once BOTH toggles below have
        # applied (params.candles is already other_candles by the time the
        # door toggle fires -- the two changes compound, not independent).
        case_b = int(window.controller.data_matrix[
            other_candles, other_door,
            window.controller.params.vod, window.controller.params.voc,
        ])
        assert case_a != case_b
        assert not window.controller.is_cached(case_a)
        assert not window.controller.is_cached(case_b)

        class FlakyStoreWrapper:
            """Wraps the real store, forcing exactly one case_index to
            raise, so the older-fails/newer-succeeds interleaving can be
            driven deterministically without corrupting the shared fixture
            data other tests in this run also read."""

            def __init__(self, inner, fail_on):
                self._inner = inner
                self._fail_on = fail_on

            def get(self, case_index, key=DEFAULT_SLICE_KEY):
                if case_index == self._fail_on:
                    raise RuntimeError("simulated load failure")
                return self._inner.get(case_index, key)

            def is_cached(self, case_index, key=DEFAULT_SLICE_KEY):
                return self._inner.is_cached(case_index, key)

        window.controller.store = FlakyStoreWrapper(window.controller.store, fail_on=case_a)

        # Toggle to A (will fail) then immediately to B (will succeed),
        # mirroring a user changing their mind before the first load lands.
        window.candle_toggle.set_value(other_candles)
        window._on_candle_changed(other_candles)
        assert window._pending_load_case == case_a

        window.door_toggle.set_value(other_door)
        window._on_door_changed(other_door)
        assert window._pending_load_case == case_b, "B must now be the tracked request, not A"

        deadline = time.perf_counter() + 3.0
        while window._busy and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)

        # B's success -- not A's stale failure -- must be what ended the
        # busy state, and B's frame data must actually be displayed.
        assert not window._busy, "B's completion must have ended the busy state"
        assert window.controller.current_case_index() == case_b
        assert window._current_n_frames > 0, "B's frame count must have been synced, not discarded"
        assert QtWidgets.QApplication.overrideCursor() is None
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

    # -------------------------------------------------------- M1.5 export
    def test_export_finished_updates_status_and_resumes_playback(self, qapp):
        """Drives the completion callback directly (bypassing the native
        QFileDialog/ExportRangeDialog, which can't be driven headlessly) to
        verify the UI reacts correctly: status message shown, and playback
        that was running before the export resumes afterward (DoD: UI stays
        responsive/usable around an export, not just during it)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._start_simulation()
        qapp.processEvents()
        assert window.time_controller.is_playing()

        window.time_controller.pause()  # what _export_animation does before exporting
        window._export_progress = QtWidgets.QProgressDialog("x", "Cancel", 0, 1, window)
        window._on_export_finished("/tmp/fake_output.gif", was_playing=True)
        assert window.time_controller.is_playing(), "must resume playback that was running before export"
        assert "fake_output.gif" in window.statusBar().currentMessage()
        window._stop_simulation()
        window.close()

    def test_export_cancelled_does_not_resume_if_was_not_playing(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert not window.time_controller.is_playing()
        window._export_progress = QtWidgets.QProgressDialog("x", "Cancel", 0, 1, window)
        window._on_export_cancelled(was_playing=False)
        assert not window.time_controller.is_playing()
        assert "cancel" in window.statusBar().currentMessage().lower()
        window.close()

    def test_export_error_shows_message_and_does_not_crash(self, qapp, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window._export_progress = QtWidgets.QProgressDialog("x", "Cancel", 0, 1, window)
        # avoid a real blocking QMessageBox in the test run
        monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)
        window._on_export_error("simulated failure", was_playing=False)
        window.close()

    def test_export_menu_action_present_and_wired(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        export_menu = next(
            (m for m in window.menuBar().findChildren(QtWidgets.QMenu) if m.title() == "&Export"),
            None,
        )
        assert export_menu is not None, "Export menu must exist"
        actions = [a.text() for a in export_menu.actions()]
        assert any("Animation" in t for t in actions)
        window.close()

    def test_second_export_while_one_in_progress_is_refused(self, qapp, tmp_path):
        """Defense-in-depth guard against the same QThread-lifecycle bug
        class found in M1.4: a second _export_animation() call while one is
        still running must not overwrite (and thus orphan) self._exporter."""
        from export import AnimationExporter
        import numpy as np

        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        data = np.random.uniform(20, 300, size=(200, 49, 101)).astype(np.float32)
        window._exporter = AnimationExporter(
            data, str(tmp_path / "slow.gif"), fps=4, cmap="gist_heat",
            vmin=20.0, vmax=300.0, interpolation="nearest", start=0, end=200,
        )
        window._exporter.start()
        qapp.processEvents()
        assert window._exporter.isRunning()

        first_exporter = window._exporter
        window._export_animation()  # must refuse, not replace _exporter
        assert window._exporter is first_exporter, "a running export must not be replaced"

        window._exporter.request_cancel()
        window._exporter.wait(10000)
        qapp.processEvents()
        window.close()

    # -------------------------------------------------- M2.1 quantity switch
    def test_quantity_combo_lists_temperature_and_velocity_for_real_data(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present; demo mode only exposes TEMPERATURE")
        labels = {window.quantity_combo.itemText(i) for i in range(window.quantity_combo.count())}
        # M2.2 adds SOOT any-plane entries when .s3d data is present; the
        # original .sf quantities remain.
        assert {"Temperature", "Air speed"} <= labels
        assert window.quantity_combo.isEnabled()
        window.close()

    def test_switching_quantity_updates_heatmap_and_colorbar(self, qapp):
        from slice_key import SliceKey

        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present; nothing real to switch to")

        assert window.current_quantity_key.quantity == "TEMPERATURE"
        temp_frame = window.heatmap.get_array().copy()

        window.quantity_combo.setCurrentIndex(1)
        deadline = time.perf_counter() + 3.0
        while window._busy and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)

        assert window.current_quantity_key == SliceKey("VELOCITY", 1, 0)
        assert "Air speed" in window.colorbar.ax.get_ylabel()
        assert window.heatmap.get_clim()[0] == 0.0
        assert not (window.heatmap.get_array() == temp_frame).all(), \
            "heatmap must actually show different data after switching quantity"
        window.close()

    def test_quantity_combo_disabled_in_demo_mode(self, qapp, monkeypatch):
        """Demo mode has no real .smv to discover quantities from -- the
        combo must degrade to a single disabled TEMPERATURE entry rather
        than crash or silently offer a non-functional VELOCITY option."""
        monkeypatch.setattr("data_provider.list_scenario_folders", lambda *a, **kw: [])
        sim_data = load_simulation_data()
        assert sim_data.is_demo
        window = MainWindow(sim_data)
        assert window.quantity_combo.count() == 1
        assert window.quantity_combo.itemText(0) == "Temperature"
        assert not window.quantity_combo.isEnabled()
        window.close()

    def test_demo_mode_scenario_toggle_does_not_crash(self, qapp, monkeypatch):
        """Regression test: DemoScenarioStore never implemented is_cached(),
        so any scenario-param toggle in demo mode raised AttributeError
        before this fix (found opportunistically while making the store
        interface key-aware for M2.1)."""
        monkeypatch.setattr("data_provider.list_scenario_folders", lambda *a, **kw: [])
        sim_data = load_simulation_data()
        assert sim_data.is_demo
        window = MainWindow(sim_data)
        window.candle_toggle.set_value(1)
        window._on_candle_changed(1)  # must not raise
        qapp.processEvents()
        assert window.heatmap.get_array() is not None
        window.close()

    def test_simultaneous_scenario_and_quantity_switch_cache_miss_race(self, qapp):
        """Characterizes (does not fix) the race documented in ROADMAP.md's
        M2.1 section: `_pending_load_case` tracks case_idx only, not
        (case_idx, key), so a scenario toggle and a quantity switch that
        both land on the same case_idx while both are cache misses can have
        their background prefetches finish out of order.

        Forces the specific interleaving the roadmap note is worried about
        -- the STALE (pre-switch) key's prefetch finishing first -- via a
        store wrapper that deliberately delays the NEW key's load. Confirms
        two things: (1) the end state is fully correct (right quantity,
        right scenario, right frame data, busy state settles, cursor
        restored) -- not just "doesn't crash"; and (2) the mechanism named
        in the roadmap note is real, not assumed: the cursor is provably
        already restored (busy state already ended) at the moment the GUI
        thread starts its own blocking synchronous fetch for the new key.
        """
        from slice_key import SliceKey

        class OrderedRaceStoreWrapper:
            """Wraps the real store; get() for `slow_key` sleeps before
            delegating, but only while the underlying store doesn't have it
            cached yet -- mirrors real ScenarioStore semantics (first load
            slow, later calls for the same (case, key) are cache hits) so
            _sync_current_scenario's *second* internal call for the new key
            (via _on_time_changed, right after the first) isn't artificially
            slowed too, which would misrepresent the real hitch's shape.
            Records, per call, which thread called, whether it was actually
            a fresh load, and whether the busy cursor had already been
            restored by the time this call started -- the direct evidence
            for where the "lie" happens."""

            def __init__(self, inner, slow_key, delay=0.15):
                self._inner = inner
                self._slow_key = slow_key
                self._delay = delay
                self.completion_log = []  # (case_index, key, is_gui_thread, cursor_was_none_at_start, was_fresh_load)

            def get(self, case_index, key=DEFAULT_SLICE_KEY):
                is_gui_thread = QtCore.QThread.currentThread() is QtWidgets.QApplication.instance().thread()
                cursor_was_none = QtWidgets.QApplication.overrideCursor() is None
                was_fresh_load = not self._inner.is_cached(case_index, key)
                if key == self._slow_key and was_fresh_load:
                    time.sleep(self._delay)
                result = self._inner.get(case_index, key)
                self.completion_log.append((case_index, key, is_gui_thread, cursor_was_none, was_fresh_load))
                return result

            def is_cached(self, case_index, key=DEFAULT_SLICE_KEY):
                return self._inner.is_cached(case_index, key)

        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window.show()
        qapp.processEvents()

        other_candles = 1 - window.controller.params.candles
        target_case = int(window.controller.data_matrix[
            other_candles, window.controller.params.door,
            window.controller.params.vod, window.controller.params.voc,
        ])
        velocity_key = SliceKey("VELOCITY", 1, 0)
        assert not window.controller.is_cached(target_case, DEFAULT_SLICE_KEY)
        assert not window.controller.is_cached(target_case, velocity_key)

        # VELOCITY (the key we're about to switch TO) is made the slow one,
        # so TEMPERATURE (the key active at the moment of the scenario
        # toggle, about to become stale) finishes first -- exactly the
        # interleaving the roadmap note names as the risk.
        wrapper = OrderedRaceStoreWrapper(window.controller.store, slow_key=velocity_key, delay=0.15)
        window.controller.store = wrapper

        window.candle_toggle.set_value(other_candles)
        window._on_candle_changed(other_candles)   # prefetch(target_case, TEMPERATURE) starts
        assert window._pending_load_case == target_case
        window.quantity_combo.setCurrentIndex(1)     # prefetch(target_case, VELOCITY) starts
        assert window.current_quantity_key == velocity_key
        assert window._pending_load_case == target_case, "both requests target the same case_idx"

        deadline = time.perf_counter() + 3.0
        while window._busy and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)

        # -- 1. No crash, no hang, busy state actually settles. --
        assert not window._busy, "busy state must eventually clear"
        assert QtWidgets.QApplication.overrideCursor() is None

        # -- 2. The intended interleaving actually happened: TEMPERATURE's
        #    load completed before VELOCITY's first (real) load. --
        completed_keys_in_order = [k for (_, k, _, _, _) in wrapper.completion_log]
        assert completed_keys_in_order[0] == DEFAULT_SLICE_KEY
        assert velocity_key in completed_keys_in_order

        # -- 3. The mechanism itself: find the GUI-thread call(s) for the
        #    new key that actually triggered a fresh (non-cached) load --
        #    that's _sync_current_scenario's own blocking store.get(),
        #    triggered from inside _on_prefetch_finished -- and confirm the
        #    cursor had ALREADY been restored (busy state already ended)
        #    before it started. This is the literal "busy indicator lies
        #    for the hitch's duration" behavior, not an assumption. --
        gui_thread_fresh_velocity_calls = [
            entry for entry in wrapper.completion_log
            if entry[1] == velocity_key and entry[2] and entry[4]
        ]
        assert len(gui_thread_fresh_velocity_calls) >= 1, (
            "expected at least one GUI-thread fresh-load fetch for the new "
            "key (_sync_current_scenario's synchronous get racing the "
            "still-in-flight background prefetch)"
        )
        assert all(entry[3] for entry in gui_thread_fresh_velocity_calls), (
            "cursor must already have been restored (busy state already "
            "ended) before this synchronous, GUI-thread-blocking fetch "
            "began -- confirming the busy indicator was inaccurate for "
            "its duration, not just eventually consistent"
        )

        # -- 4. Despite the race, the FINAL state is fully correct: right
        #    scenario, right quantity, right frame data, no silent loss. --
        assert window.controller.current_case_index() == target_case
        assert window.current_quantity_key == velocity_key
        assert window._current_n_frames > 0
        expected = wrapper._inner.get(target_case, velocity_key)
        shown_index = window.time_controller.index
        assert (window.heatmap.get_array() == expected[shown_index]).all(), \
            "displayed frame must be the new quantity's real data, not stale/wrong data"

        window.close()

    # --------------------------------------------------------- M2.2 grid
    def test_grid_layout_switch_changes_visible_cell_count(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert len(window.view_grid.visible_cells()) == 1

        window._set_grid_layout("2x2")
        assert len(window.view_grid.visible_cells()) == 4
        for cell in window.view_grid.visible_cells():
            assert cell.view.heatmap.get_array().shape == (49, 101)

        window._set_grid_layout("1x2")
        assert len(window.view_grid.visible_cells()) == 2

        window._set_grid_layout("1x1")
        assert len(window.view_grid.visible_cells()) == 1
        window.close()

    def test_grid_layout_switch_works_in_demo_mode(self, qapp, monkeypatch):
        """Demo mode has no manifest -- grid cells default to case_index 0
        with disabled scenario combos, but the grid itself must still work."""
        monkeypatch.setattr("data_provider.list_scenario_folders", lambda *a, **kw: [])
        sim_data = load_simulation_data()
        assert sim_data.is_demo
        window = MainWindow(sim_data)
        window._set_grid_layout("2x2")
        assert len(window.view_grid.visible_cells()) == 4
        for cell in window.view_grid.visible_cells():
            assert not cell.scenario_combo.isEnabled()
        window.close()

    def test_grid_toolbar_visible_only_in_1x1(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        window._navigate_to("live")  # FireLab nav shell: toolbar lives on the Live page, not shown by default (Home is)
        qapp.processEvents()
        assert window.toolbar.isVisible()
        window._set_grid_layout("2x2")
        assert not window.toolbar.isVisible()
        window._set_grid_layout("1x1")
        assert window.toolbar.isVisible()
        window.close()

    def test_grid_control_panel_toggle_drives_active_cell_only(self, qapp):
        """Toggling a scenario control while in a multi-cell grid must only
        move the *active* cell -- other visible cells must be unaffected,
        confirming the M2.2 design decision (control panel edits the
        active cell only) actually holds, not just for clim/colormap."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()
        other_case_indices_before = [c.case_index for c in cells[1:]]

        other_candles = 1 - window.controller.params.candles
        window.candle_toggle.set_value(other_candles)
        window._on_candle_changed(other_candles)

        assert cells[0] is window.view_grid.active_cell()
        assert cells[0].case_index == window.controller.current_case_index()
        assert [c.case_index for c in cells[1:]] == other_case_indices_before, \
            "non-active cells must not move when the control panel changes"
        window.close()

    def test_grid_clicking_a_cell_makes_it_active_and_syncs_control_panel(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()
        target = cells[2]
        target.scenario_combo.setCurrentIndex(10)  # give it a distinct scenario first
        qapp.processEvents()

        target.activated.emit(target)

        assert window.view_grid.active_cell() is target
        entry = next(e for e in sim_data.manifest if e.case_index == target.case_index)
        assert window.candle_toggle.value == entry.candles
        assert window.door_toggle.value == entry.door
        assert window.vod_toggle.value == entry.vod
        assert window.voc_toggle.value == entry.voc
        assert window.controller.current_case_index() == target.case_index
        window.close()

    def test_grid_link_clim_shares_max_within_quantity_group(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()
        # Give each cell a different scenario so their data maxima differ.
        for i, cell in enumerate(cells):
            cell.scenario_combo.setCurrentIndex(i * 5)
            qapp.processEvents()

        expected_vmax = max(
            float(window.controller.store.get(c.case_index, c.quantity_key).max()) for c in cells
        )

        window.link_clim_action.setChecked(True)
        window._set_link_clim(True)

        clims = [c.view.heatmap.get_clim() for c in cells]
        assert all(vmax == pytest.approx(expected_vmax) for _vmin, vmax in clims), \
            f"all cells should share the same vmax when linked: {clims}"
        assert len({vmin for vmin, _vmax in clims}) == 1, "vmin should also match (same quantity, same floor)"
        window.close()

    def test_grid_unlinked_cells_keep_independent_clim(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cells = window.view_grid.visible_cells()
        assert not window.link_clim_action.isChecked()

        cells[1].scenario_combo.setCurrentIndex(15)
        qapp.processEvents()
        window.temp_slider.setValue(777)  # active cell (cells[0]) only

        assert cells[0].view.heatmap.get_clim()[1] == 777.0
        assert cells[1].view.heatmap.get_clim()[1] != 777.0, \
            "non-active cell's clim must not follow the active cell's slider when unlinked"
        window.close()

    def test_grid_playback_tick_updates_every_visible_cell(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()
        for i, cell in enumerate(cells[1:], start=1):
            cell.scenario_combo.setCurrentIndex(i * 3)
            qapp.processEvents()

        before = [c.view.heatmap.get_array().copy() for c in cells]
        window.time_controller.seek(100)
        qapp.processEvents()
        after = [c.view.heatmap.get_array() for c in cells]

        for b, a in zip(before, after):
            assert not (b == a).all(), "every visible cell must redraw on a timeline seek, not just the active one"
        window.close()

    def test_grid_non_active_cell_scenario_change_does_not_block_and_ends_up_correct(self, qapp):
        """M2.2.4: a non-active cell picking an uncached scenario must not
        freeze the GUI thread -- it prefetches in the background, same
        machinery as the active-cell cache-miss path, and ends up showing
        the right data once ready."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        target = window.view_grid.visible_cells()[1]
        assert not window.controller.is_cached(18, target.quantity_key)

        target.scenario_combo.setCurrentIndex(18)
        assert target in window._pending_cell_prefetches, "must be prefetching, not blocking synchronously"

        deadline = time.perf_counter() + 3.0
        while window._pending_cell_prefetches and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.005)

        assert target not in window._pending_cell_prefetches
        assert target.case_index == 18
        expected = window.controller.store.get(18, target.quantity_key)
        idx = window.time_controller.index
        assert (target.view.heatmap.get_array() == expected[idx]).all()
        window.close()

    def test_grid_shrink_then_regrow_preserves_non_active_cell_state(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()
        cells[3].scenario_combo.setCurrentIndex(7)
        qapp.processEvents()

        window._set_grid_layout("1x1")
        window._set_grid_layout("2x2")

        assert window.view_grid.visible_cells()[3].case_index == 7
        window.close()

    # ------------------------------------------ M2.3 difference/ensemble
    def test_cell_switched_to_difference_renders_immediately(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cell = window.view_grid.visible_cells()[1]

        cell._set_cell_type("difference")

        from views import DifferenceView
        assert isinstance(cell.view, DifferenceView)
        assert cell.view.heatmap is not None
        vmin, vmax = cell.view.heatmap.get_clim()
        assert vmin == -vmax, "difference cell must render with a symmetric clim immediately"
        window.close()

    def test_difference_cell_scenario_change_recomputes_diff(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cell = window.view_grid.visible_cells()[1]
        cell._set_cell_type("difference")
        window.time_controller.seek(150)  # away from t=0, where scenarios start near-identical
        qapp.processEvents()

        before = cell.view.heatmap.get_array().copy()
        cell.scenario_combo_a.setCurrentIndex(2)
        after = cell.view.heatmap.get_array()

        assert not (before == after).all(), "changing scenario A must recompute and redraw the diff"
        expected = window.controller.store.get(cell.case_index_a, cell.quantity_key)[150] - \
            window.controller.store.get(cell.case_index_b, cell.quantity_key)[150]
        assert (after == expected).all()
        window.close()

    def test_cell_switched_to_ensemble_stays_blank_until_scenarios_picked(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cell = window.view_grid.visible_cells()[1]

        cell._set_cell_type("ensemble")

        from views import EnsembleView
        assert isinstance(cell.view, EnsembleView)
        assert cell.view.heatmap is None, "an ensemble cell with nothing selected must not render yet"
        window.close()

    def test_ensemble_selection_renders_composite(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cell = window.view_grid.visible_cells()[1]
        cell._set_cell_type("ensemble")

        cell.ensemble_case_indices = [0, 6, 12, 18]
        cell.ensemble_changed.emit(cell, cell.ensemble_case_indices, cell.ensemble_stat)

        assert cell.view.heatmap is not None
        vmin, vmax = cell.view.heatmap.get_clim()
        assert vmin == 20.0, "mean-stat ensemble should keep TEMPERATURE's own vmin (ambient)"
        window.close()

    def test_ensemble_std_stat_uses_zero_floor_and_sigma_label(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cell = window.view_grid.visible_cells()[1]
        cell._set_cell_type("ensemble")
        cell.ensemble_case_indices = [0, 6, 12, 18]
        cell.ensemble_changed.emit(cell, cell.ensemble_case_indices, "mean")

        cell.ensemble_stat = "std"
        cell.stat_combo.setCurrentIndex(cell.stat_combo.findText("Std"))

        vmin, vmax = cell.view.heatmap.get_clim()
        assert vmin == 0.0
        assert vmax > 0.0
        assert "σ" in cell.view.colorbar.ax.get_ylabel()
        window.close()

    def test_playback_tick_updates_difference_and_ensemble_cells(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()
        cells[1]._set_cell_type("difference")
        cells[2]._set_cell_type("ensemble")
        cells[2].ensemble_case_indices = [0, 6, 12, 18]
        cells[2].ensemble_changed.emit(cells[2], cells[2].ensemble_case_indices, "mean")

        before = [c.view.heatmap.get_array().copy() for c in cells]
        window.time_controller.seek(200)
        qapp.processEvents()
        after = [c.view.heatmap.get_array() for c in cells]

        for i, (b, a) in enumerate(zip(before, after)):
            assert not (b == a).all(), f"cell {i} (type={cells[i].cell_type}) must redraw on seek"
        window.close()

    def test_link_clim_ignores_difference_and_ensemble_cells(self, qapp):
        """Linking is defined for slice-type cells sharing a quantity;
        difference/ensemble cells have their own clim conventions (see
        _apply_link_clim's docstring) and must be left alone."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        cells = window.view_grid.visible_cells()
        cells[1]._set_cell_type("difference")
        diff_clim_before = cells[1].view.heatmap.get_clim()

        window.link_clim_action.setChecked(True)
        window._set_link_clim(True)

        assert cells[1].view.heatmap.get_clim() == diff_clim_before, \
            "linking must not touch a difference cell's own symmetric clim"
        window.close()

    # ------------------------------------------------ M2.6 probe/isotherms
    def test_active_cell_gets_a_real_extent(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert window.view_grid.active_view()._extent == (0.0, 1.0, 0.0, 0.48)
        window.close()

    def test_demo_mode_still_gets_a_stable_extent(self, qapp, monkeypatch):
        monkeypatch.setattr("data_provider.list_scenario_folders", lambda *a, **kw: [])
        sim_data = load_simulation_data()
        assert sim_data.is_demo
        window = MainWindow(sim_data)
        assert window.view_grid.active_view()._extent is not None
        window.close()

    def test_probe_reports_physical_coordinates_and_value_in_status_bar(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        cell = window.view_grid.active_cell()

        class FakeEvent:
            inaxes = cell.view.ax
            xdata = 0.5
            ydata = 0.24

        cell.view._on_mouse_move(FakeEvent())
        message = window.statusBar().currentMessage()
        assert "x = 0.500 m" in message
        assert "z = 0.240 m" in message
        assert "°C" in message
        window.close()

    def test_probe_resets_status_bar_when_mouse_leaves(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        cell = window.view_grid.active_cell()

        class FakeEventOutside:
            inaxes = None
            xdata = None
            ydata = None

        cell.view._on_mouse_move(FakeEventOutside())
        assert window.statusBar().currentMessage() == "Ready."
        window.close()

    def test_probe_wired_for_every_visible_cell_including_newly_grown_ones(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        for cell in window.view_grid.visible_cells():
            assert cell.view._motion_cid is not None
            assert cell.view._extent == (0.0, 1.0, 0.0, 0.48)
        window.close()

    def test_isotherm_toggle_applies_to_every_visible_cell(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cells = window.view_grid.visible_cells()

        window.isotherms_action.setChecked(True)
        window._set_isotherms_enabled(True)

        for cell in cells:
            assert cell.view.isotherms_enabled
        window.close()

    def test_isotherm_toggle_off_clears_every_cell(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("1x2")
        window.isotherms_action.setChecked(True)
        window._set_isotherms_enabled(True)

        window.isotherms_action.setChecked(False)
        window._set_isotherms_enabled(False)

        for cell in window.view_grid.visible_cells():
            assert not cell.view.isotherms_enabled
            assert cell.view._contour_artist is None
        window.close()

    def test_isotherm_redraws_on_playback_tick(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window.isotherms_action.setChecked(True)
        window._set_isotherms_enabled(True)
        cell = window.view_grid.active_cell()
        first_artist = cell.view._contour_artist

        window.time_controller.seek(200)
        qapp.processEvents()

        assert cell.view._contour_artist is not None
        assert cell.view._contour_artist is not first_artist
        window.close()

    def test_isotherm_off_state_does_not_touch_contour_artist(self, qapp):
        """DoD: off-state performance/behavior unchanged -- confirms no
        contour work happens at all while the toggle is off, not just that
        it's invisible."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert not window.isotherms_action.isChecked()
        cell = window.view_grid.active_cell()

        window.time_controller.seek(150)
        qapp.processEvents()

        assert cell.view._contour_artist is None
        window.close()

    def test_switching_quantity_updates_isotherm_levels(self, qapp):
        """TEMPERATURE has default hazard-band levels; VELOCITY has speed-
        band levels (config.ISOTHERM_LEVELS) -- switching quantity on an
        isotherm-enabled active cell must pick up the new quantity's own
        levels, not keep the stale ones."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window.isotherms_action.setChecked(True)
        window._set_isotherms_enabled(True)
        cell = window.view_grid.active_cell()
        assert cell.view._isotherm_levels == [60, 100, 300]

        window.quantity_combo.setCurrentIndex(1)  # switch to Air speed (VELOCITY)

        assert cell.view._isotherm_levels == [1.0, 2.0, 3.0]
        window.close()

    # ------------------------------------------- GUI modernization item 6
    def test_velocity_overlay_off_by_default(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert not window.velocity_overlay_action.isChecked()
        cell = window.view_grid.active_cell()
        assert not cell.view.velocity_overlay_enabled
        window.close()

    def test_velocity_overlay_applies_only_to_temperature_slice_cells(self, qapp):
        """Opt-in, grid-wide toggle (View -> Show velocity overlay) --
        applies to a "slice" cell showing TEMPERATURE, is a no-op for a
        cell already showing VELOCITY (overlaying velocity on itself makes
        no sense) or a difference/ensemble cell."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        window._open_browser_grid([0, 1])
        qapp.processEvents()
        temp_cell, velocity_cell = window.view_grid.visible_cells()
        idx = next(i for i, info in enumerate(window.quantity_infos) if info.key.quantity == "VELOCITY")
        velocity_cell.set_quantity_silently(velocity_cell._quantity_options[idx][1])
        window._load_cell(velocity_cell, velocity_cell.case_index, velocity_cell._quantity_options[idx][1])
        qapp.processEvents()

        window.velocity_overlay_action.setChecked(True)
        window._set_velocity_overlay_enabled(True)
        qapp.processEvents()

        assert temp_cell.view.velocity_overlay_enabled
        assert not velocity_cell.view.velocity_overlay_enabled
        window.close()

    def test_velocity_overlay_contour_reflects_real_aligned_data(self, qapp):
        """Real-data verification (not assumed): TEMPERATURE and VELOCITY
        share the exact same physical plane/extent for a real scenario
        (confirmed directly against fds/sim/ before building this), so the
        overlay's contour must be drawn from the SAME scenario's real
        VELOCITY array at the SAME frame the temperature heatmap is
        showing, not placeholder/zero data."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        window.velocity_overlay_action.setChecked(True)
        window._set_velocity_overlay_enabled(True)
        qapp.processEvents()

        cell = window.view_grid.active_cell()
        assert cell.view.velocity_overlay_enabled
        assert cell.view._velocity_frame is not None

        from slice_key import SliceKey
        expected = window.controller.store.get(
            cell.case_index, SliceKey("VELOCITY", cell.quantity_key.direction, cell.quantity_key.offset),
        )[window.time_controller.index]
        assert np.array_equal(cell.view._velocity_frame, expected)
        window.close()

    # ------------------------------------------------ M3.1 ensemble analytics
    def test_analytics_panel_present_for_real_data(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert window.analytics_panel is not None
        # Feature index is lazy-loaded on first show -- FireLab roadmap
        # Phase 4 re-hosted this as the Analysis page's content, so
        # navigating there (AnalysisPage.on_enter()) is what now drives
        # that trigger, replacing the old dock-tab-raise mechanism.
        window.show()
        window._navigate_to("analysis")
        qapp.processEvents()
        _drain_workers(qapp, window._analytics_workers)
        assert len(window.analytics_panel._features) == 24
        window.close()

    def test_analytics_panel_absent_in_demo_mode(self, qapp, monkeypatch):
        monkeypatch.setattr("data_provider.list_scenario_folders", lambda *a, **kw: [])
        sim_data = load_simulation_data()
        assert sim_data.is_demo
        window = MainWindow(sim_data)
        assert window.analytics_panel is None
        window.close()

    def test_clicking_analytics_point_loads_scenario_into_active_cell(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        panel = window.analytics_panel
        # Feature index is lazy-loaded on first show -- see the comment in
        # test_analytics_panel_present_for_real_data above.
        window.show()
        window._navigate_to("analysis")
        qapp.processEvents()
        _drain_workers(qapp, window._analytics_workers)
        target_case = panel._case_indices[5]

        class FakeEvent:
            inaxes = panel.ax
            xdata = panel._coords[5, 0]
            ydata = panel._coords[5, 1]

        panel._on_click(FakeEvent())

        assert window.controller.current_case_index() == target_case
        window.close()

    def test_browser_selection_shows_auto_summary(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        # Selecting a row triggers summary_texts_needed, which starts the
        # background auto-summary load (see _build_experiment_browser) --
        # drain it before checking the label it fills in.
        window.experiment_browser.table.selectRow(0)
        _drain_workers(qapp, window._summary_text_workers)
        text = window.experiment_browser.summary_label.text()
        assert text.startswith("Peak ")
        assert "°C at t=" in text
        window.close()

    def test_export_summaries_writes_all_24_and_updates_status_bar(self, qapp, monkeypatch, tmp_path):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        out_path = str(tmp_path / "summaries.md")
        monkeypatch.setattr(
            "PyQt5.QtWidgets.QFileDialog.getSaveFileName", lambda *a, **kw: (out_path, "")
        )

        window._export_summaries_markdown()

        content = open(out_path).read()
        assert content.count("## ") == 24
        assert "Exported scenario summaries" in window.statusBar().currentMessage()
        window.close()

    def test_export_summaries_cancelled_dialog_does_not_write_or_crash(self, qapp, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        monkeypatch.setattr("PyQt5.QtWidgets.QFileDialog.getSaveFileName", lambda *a, **kw: ("", ""))

        window._export_summaries_markdown()  # must not raise
        window.close()

    def test_analytics_panel_present_does_not_change_playback_fps_path(self, qapp):
        """DoD: panel doesn't degrade playback. Direct mechanism check: the
        panel widget has none of PlotView's per-frame methods (show_frame)
        and isn't a TimeController slot target, so its mere presence can't
        add per-tick cost to _on_time_changed -- confirmed structurally,
        not just by a timing measurement (which would be flaky in CI)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert window.analytics_panel is not None
        assert not hasattr(window.analytics_panel, "show_frame")
        assert not hasattr(window.analytics_panel, "_on_time_changed")
        window.close()


class TestEventTimeline:
    """V2 roadmap M1.3: auto-detected event markers on the scrubber."""

    def test_real_scenario_gets_markers_including_peak(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert window.timeline.marker_bar.markers == []
            window.close()
            return
        markers = window.timeline.marker_bar.markers
        assert markers, "real data must produce at least the peak-temperature marker"
        labels = [label for _f, label in markers]
        assert any("Peak temperature" in label for label in labels)
        n = window._current_n_frames
        assert all(0 <= frame < n for frame, _l in markers)
        window.close()

    def test_velocity_quantity_clears_markers(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        velocity_idx = next((i for i, info in enumerate(window.quantity_infos)
                             if info.key.quantity == "VELOCITY"), None)
        if velocity_idx is None:
            window.close()
            return
        window.quantity_combo.setCurrentIndex(velocity_idx)
        _drain_workers(qapp, window.controller._prefetch_workers)
        qapp.processEvents()
        assert window.timeline.marker_bar.markers == []
        window.close()

    def test_marker_click_seeks_playback(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        markers = window.timeline.marker_bar.markers
        assert markers
        target_frame = markers[-1][0]
        window.timeline.marker_bar.marker_clicked.emit(target_frame)
        assert window.time_controller.index == target_frame
        window.close()


class TestPublicationFigureExport:
    """V2 roadmap M1.4: Export -> Publication figure… menu action."""

    def test_export_writes_svg_for_active_slice_cell(self, qapp, tmp_path, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return

        from figure_export import PublicationExportDialog
        out_path = str(tmp_path / "fig.svg")
        monkeypatch.setattr(
            PublicationExportDialog, "exec_", lambda self: QtWidgets.QDialog.Accepted)
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (out_path, ""))

        window._export_publication_figure()
        assert (tmp_path / "fig.svg").exists()
        window.close()

    def test_export_on_difference_cell_shows_message_not_crash(self, qapp, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        cell = window.view_grid.active_cell()
        cell.set_cell_type("difference")
        window._on_cell_type_changed(cell, "difference")

        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
        window._export_publication_figure()  # must not raise
        window.close()


class TestDifferenceOverTimeButton:
    """V2 roadmap M1.5: Inspector's "Plot difference over time…" button."""

    def test_button_visible_only_for_difference_cell(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        window._navigate_to("live")
        qapp.processEvents()
        assert not window.inspector.diff_plot_button.isVisible()
        cell = window.view_grid.active_cell()
        cell.set_cell_type("difference")
        window._on_cell_type_changed(cell, "difference")
        window._on_time_changed(window.time_controller.index)
        assert window.inspector.diff_plot_button.isVisible()
        window.close()

    def test_button_click_opens_dialog_without_crash(self, qapp, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        cell = window.view_grid.active_cell()
        cell.set_cell_type("difference")
        window._on_cell_type_changed(cell, "difference")

        from diff_analysis import DifferenceOverTimeDialog
        monkeypatch.setattr(DifferenceOverTimeDialog, "exec_", lambda self: None)
        window._show_difference_over_time()  # must not raise
        window.close()

    def test_no_op_when_active_cell_is_a_plain_slice(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        assert window.view_grid.active_cell().cell_type == "slice"
        window._show_difference_over_time()  # must not raise
        window.close()


class TestSessionSaveLoad:
    """V2 roadmap M2.4: File -> Save/Load Session."""

    def test_save_then_load_restores_grid_state(self, qapp, tmp_path, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return

        window._set_grid_layout("1x2")
        cells = window.view_grid.visible_cells()
        options = window._scenario_options()
        case_a, case_b = options[0][1], options[-1][1]
        window._select_scenario_in_cell(cells[0], case_a)
        cells[1].set_cell_type("difference")
        window._on_cell_type_changed(cells[1], "difference")
        window._select_difference_scenarios_in_cell(cells[1], case_a, case_b)
        target_frame = min(3, window._current_n_frames - 1)
        window.time_controller.seek(target_frame)

        path = str(tmp_path / "session.json")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
        window._save_session()
        assert (tmp_path / "session.json").exists()

        # Reset to a different state before loading, so restoration is
        # actually exercised rather than trivially already-true.
        window._set_grid_layout("1x1")

        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))
        window._load_session()

        assert window.view_grid.layout_name == "1x2"
        restored = window.view_grid.visible_cells()
        assert restored[0].case_index == case_a
        assert restored[1].cell_type == "difference"
        assert restored[1].case_index_a == case_a
        assert restored[1].case_index_b == case_b
        assert window.time_controller.index == target_frame
        window.close()

    def test_load_rejects_malformed_file_without_crash(self, qapp, tmp_path, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        path = tmp_path / "bad.json"
        path.write_text("not json")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
        window._load_session()  # must not raise
        window.close()


class TestSootAnyPlaneUI:
    """V2 roadmap M2.2: SOOT DENSITY any-plane slicing surfaced in the UI."""

    def test_soot_planes_absent_in_demo_mode(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if not sim_data.is_demo:
            window.close()
            return
        quantities = {info.key.quantity for info in window.quantity_infos}
        assert "SOOT DENSITY" not in quantities
        window.close()

    def test_soot_planes_appear_with_real_s3d_data(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        soot = [i for i in window.quantity_infos if i.key.quantity == "SOOT DENSITY"]
        assert len(soot) == 2  # side view + doorway
        positions = {i.key.plane_pos for i in soot}
        assert positions == {0.0, 0.25}
        # Labels are plane-distinct (not two identical "Smoke (soot)" entries).
        labels = [window._quantity_label(i) for i in soot]
        assert len(set(labels)) == 2
        window.close()

    def test_switching_to_doorway_plane_changes_extent_and_shape(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        door_idx = next((i for i, info in enumerate(window.quantity_infos)
                         if info.key.plane_pos == 0.25), None)
        assert door_idx is not None
        before_shape = window.heatmap.get_array().shape
        window.quantity_combo.setCurrentIndex(door_idx)
        _drain_workers(qapp, window.controller._prefetch_workers)
        qapp.processEvents()
        after_shape = window.heatmap.get_array().shape
        # Doorway (y x z) plane differs in width from the side (x x z) plane.
        assert after_shape != before_shape
        assert window.view_grid.active_view()._extent[0] < 0  # y spans negative
        window.close()


class TestMultiStudyGuestStudy:
    """V2 roadmap M2.5: a generic guest study opened via load_study --
    validated against a real single candle-scenario folder as a
    standalone degenerate study (the line-burner has no computed output
    to open)."""

    def _guest_case_dir(self):
        import os
        from load_data import SIM_ROOT
        return os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")

    def test_degenerate_study_builds_with_candle_ui_hidden(self, qapp):
        import os
        from data_provider import load_study
        case_dir = self._guest_case_dir()
        if not os.path.isdir(case_dir):
            pytest.skip("real dataset not present")
        window = MainWindow(load_study(case_dir))
        assert window.is_factorial is False
        assert not window.candle_toggle.isVisibleTo(window)
        assert window.analytics_panel is None
        in_stack = any(window.page_stack.widget(i) is window.pages["compare"]
                       for i in range(window.page_stack.count()))
        assert in_stack is False
        window.close()

    def test_degenerate_study_renders_and_switches_quantity(self, qapp):
        import os
        from data_provider import load_study
        case_dir = self._guest_case_dir()
        if not os.path.isdir(case_dir):
            pytest.skip("real dataset not present")
        window = MainWindow(load_study(case_dir))
        assert window.heatmap.get_array().shape == (49, 101)
        door_idx = next((i for i, info in enumerate(window.quantity_infos)
                        if info.key.plane_pos == 0.25), None)
        assert door_idx is not None
        window.quantity_combo.setCurrentIndex(door_idx)
        _drain_workers(qapp, window.controller._prefetch_workers)
        qapp.processEvents()
        assert window.heatmap.get_array().shape == (49, 31)
        window.close()
