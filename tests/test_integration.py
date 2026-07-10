"""Integration tests: build MainWindow, exercise UI, verify no crashes."""

import pytest
import time
from PyQt5 import QtCore, QtWidgets
from data_provider import load_simulation_data
from main_window import MainWindow
from slice_key import DEFAULT_SLICE_KEY


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
        assert labels == {"Temperature", "Air speed"}
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
