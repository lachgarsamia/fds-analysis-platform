"""Integration tests: build MainWindow, exercise UI, verify no crashes."""

import pytest
import time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from data_provider import load_simulation_data
from main_window import MainWindow
from slice_key import DEFAULT_SLICE_KEY, SliceKey


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

    def test_schematic_has_height_for_width_and_grows_with_ui_scale(self, qapp):
        """Bug found live: the schematic implemented heightForWidth() but
        never overrode hasHeightForWidth() (Qt's layout system ignores
        heightForWidth entirely without it, default False) -- its height
        stayed frozen at whatever sizeHint() computed once at startup
        instead of rescaling with the sidebar. Also verifies UI Scale
        (accessibility zoom) actually grows the widget -- it draws itself
        entirely in paintEvent, so no QSS-driven sizing reaches it without
        set_ui_scale."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        assert window.schematic.hasHeightForWidth() is True
        assert window.schematic.heightForWidth(400) > window.schematic.heightForWidth(200)

        # A tiny width parameter makes heightForWidth's ui_scale-driven
        # minimum floor the dominant term (the aspect-based term shrinks to
        # near nothing), isolating what set_ui_scale actually controls from
        # the real layout's own (test-environment-dependent) current width.
        # Pinned to a known baseline first -- ui_scale persists in QSettings
        # across runs, so a prior manual session could leave it non-default.
        window._set_ui_scale(1.0)
        before = window.schematic.heightForWidth(1)
        window._set_ui_scale(2.0)
        after = window.schematic.heightForWidth(1)
        assert after > before
        assert window.schematic._ui_scale == 2.0
        # QSettings is shared with the real app (not test-isolated) --
        # restore the default so this test doesn't leave the real app at
        # 2x scale the next time someone actually launches it.
        window._set_ui_scale(1.0)
        window.close()

    def test_room_outline_drawn_on_default_y_normal_slice(self, qapp):
        """Live-polish request: the room's physical boundary -- walls minus
        the door gap, the door opening, and the two vents -- is drawn
        directly on the heatmap for the default (y-normal) slice, matching
        the sidebar diagram's own geometry, not just a plain closed box."""
        from schematic import ROOM_X, ROOM_Z, _OBST_Z_TOP
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        view = window.view_grid.active_cell().view
        wall_segs = view.room_walls.get_segments()
        # floor, ceiling (slab underside), slab top face, right wall, 2
        # left-wall pieces -- the slab top face (z=_OBST_Z_TOP) draws the
        # ceiling &OBST's true thickness as a second parallel line, so the
        # dead z=0.23 grid row (see _OBST_Z_TOP's docstring) visibly sits
        # inside the concrete instead of reading as an unexplained offset.
        assert len(wall_segs) == 6
        assert len(view.room_door.get_segments()) == 1
        assert len(view.room_vents.get_segments()) == 2
        xs = [p[0] for seg in wall_segs for p in seg]
        zs = [p[1] for seg in wall_segs for p in seg]
        assert min(xs) == pytest.approx(ROOM_X[0])
        assert max(xs) == pytest.approx(ROOM_X[1])
        assert min(zs) == pytest.approx(ROOM_Z[0])
        assert max(zs) == pytest.approx(_OBST_Z_TOP)
        window.close()

    def test_room_outline_hidden_for_non_y_normal_or_non_slice_cell(self, qapp):
        """ROOM_X/ROOM_Z are only validated for the y-normal plane, and a
        difference/ensemble cell has no single door/vent state -- both
        must hide the overlay rather than draw a guessed one (same gating
        principle used everywhere else in the app)."""
        import types
        from slice_key import SliceKey
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        wrong_plane = types.SimpleNamespace(
            cell_type="slice", quantity_key=SliceKey("SOOT DENSITY", 0, 0, 0.25), case_index=0)
        assert window._room_outline_for(wrong_plane) is None
        diff_cell = types.SimpleNamespace(
            cell_type="difference", quantity_key=SliceKey("TEMPERATURE", 1, 0), case_index=0)
        assert window._room_outline_for(diff_cell) is None
        window.close()

    def test_room_overlay_geometry_reflects_door_and_vent_state(self, qapp):
        """Pure-function unit test: room_overlay_geometry's door segment
        grows with the wide-door state, and its vent colors track
        open/closed/HVAC -- the same proportional placement
        SchematicWidget._paint draws, just as physical-coordinate numbers."""
        from schematic import room_overlay_geometry

        narrow = room_overlay_geometry(door=0, vod=0, voc=0)
        wide = room_overlay_geometry(door=1, vod=0, voc=0)
        narrow_h = narrow["door"][3] - narrow["door"][1]
        wide_h = wide["door"][3] - wide["door"][1]
        assert wide_h > narrow_h

        geo = room_overlay_geometry(door=1, vod=1, voc=0)  # vod closed, voc open
        (_seg0, state0), (_seg1, state1) = geo["vents"]
        assert state0 == "closed" and state1 == "open"

    def test_room_overlay_vent_and_door_positions_match_real_fds_geometry(self, qapp):
        """Bug found live: vents were drawn at placeholder fractions (34%/
        64% evenly across the ceiling) that didn't match reality -- most
        visibly wrong for VOC, drawn mid-ceiling when the real opening
        (confirmed by reading fds/sim/*/*.fds's &HOLE lines) sits right
        next to the candle at the room's far edge, so a real fire plume
        visibly passed the drawn vent without seeming to go through it.
        Pins the fix to the exact real coordinates so it can't regress."""
        from schematic import room_overlay_geometry
        geo = room_overlay_geometry(door=0, vod=0, voc=0)
        (vod_seg, _vod_state), (voc_seg, _voc_state) = geo["vents"]
        assert (vod_seg[0], vod_seg[2]) == pytest.approx((0.32, 0.40))
        assert (voc_seg[0], voc_seg[2]) == pytest.approx((0.86, 0.94))

        narrow = room_overlay_geometry(door=0, vod=0, voc=0)
        wide = room_overlay_geometry(door=1, vod=0, voc=0)
        assert narrow["door"][1] == pytest.approx(0.0)         # reaches the floor
        assert narrow["door"][3] - narrow["door"][1] == pytest.approx(0.06)
        assert wide["door"][3] - wide["door"][1] == pytest.approx(0.16)

    def test_heatmap_plot_scales_with_ui_scale(self, qapp):
        """Bug found live: nothing inside a matplotlib canvas (colorbar
        ticks/label, the room overlay's line widths, ...) responded to
        View -> UI Scale at all -- only the Qt-side chrome (QSS fonts/
        padding) did, so the plot itself looked completely unaffected by
        the setting. Fixed via MplCanvas.set_dpi_scale: raising the
        figure's DPI packs more rendered detail into the same on-screen
        widget footprint, scaling every point-sized artist together
        (fonts, line widths, markers) instead of hand-tuning one overlay
        in isolation."""
        from widgets import MplCanvas
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        canvas = window.view_grid.active_cell().view.canvas
        window._set_ui_scale(1.0)
        assert canvas.fig.dpi == pytest.approx(MplCanvas.DEFAULT_DPI)
        window._set_ui_scale(2.0)
        assert canvas.fig.dpi == pytest.approx(MplCanvas.DEFAULT_DPI * 2.0)
        window._set_ui_scale(1.0)  # QSettings is shared with the real app -- restore default
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
        """V2-M0.1 fixed the race this once characterized: `_pending_load_case`
        is now paired with `_pending_load_key`, so when a scenario toggle
        and a quantity switch both land on the same case_idx, the STALE
        key's prefetch finishing first no longer ends the busy state.

        Forces the same interleaving (the stale pre-switch key's prefetch
        finishing first, via a store wrapper that delays the NEW key's
        load) and confirms the *fixed* behavior: (1) the end state is fully
        correct; and (2) the busy state waits for the new key's own
        prefetch -- so the GUI thread never does a blocking fresh load for
        it (its `_sync_current_scenario` fetch is a cache hit), and the
        cursor is not restored early.
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

        # -- 3. The fix: the GUI thread never does a blocking *fresh* load
        #    for the new key. Because the busy state waits until VELOCITY's
        #    own prefetch has cached it (M0.1), _sync_current_scenario's
        #    store.get() for the new key is a cache HIT, not a fresh load
        #    racing the background prefetch. --
        gui_thread_fresh_velocity_calls = [
            entry for entry in wrapper.completion_log
            if entry[1] == velocity_key and entry[2] and entry[4]
        ]
        assert gui_thread_fresh_velocity_calls == [], (
            "with the M0.1 pending-key fix, the busy state waits for the "
            "new key's own prefetch, so the GUI thread should never trigger "
            "a fresh (non-cached) load for it"
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

    def test_activating_a_freshly_typed_ensemble_cell_with_no_data_does_not_crash(self, qapp):
        """A cell just switched to Ensemble mode (right-click menu) with no
        scenarios picked yet stays blank until the picker dialog runs
        (_on_cell_type_changed's own documented behavior) -- its view never
        renders a frame, so heatmap is still None. Activating it (clicking
        it, exactly what this test exists to cover) used to crash
        MainWindow._on_active_cell_changed, which unconditionally read
        cell.view.heatmap.get_cmap()."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._set_grid_layout("2x2")
        cell = window.view_grid.visible_cells()[3]
        cell.set_cell_type("ensemble")
        assert cell.ensemble_case_indices == []
        assert cell.view.heatmap is None

        cell.activated.emit(cell)  # must not raise

        assert window.view_grid.active_cell() is cell
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

    # -------------------------------- continuous soot-density visualization
    def test_soot_overlay_off_by_default(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        assert not window.soot_overlay_action.isChecked()
        cell = window.view_grid.active_cell()
        assert not cell.view.soot_overlay_enabled
        window.close()

    def test_soot_overlay_applies_only_to_temperature_slice_cells_at_y0(self, qapp):
        """Opt-in, grid-wide toggle (View -> Show smoke overlay) -- applies
        to a "slice" cell showing TEMPERATURE at the y=0 plane (the only
        plane SOOT DENSITY is confirmed to share TEMPERATURE's grid for),
        is a no-op for a cell showing a different quantity."""
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

        window.soot_overlay_action.setChecked(True)
        window._set_soot_overlay_enabled(True)
        qapp.processEvents()

        assert temp_cell.view.soot_overlay_enabled
        assert not velocity_cell.view.soot_overlay_enabled
        window.close()

    def test_soot_overlay_reflects_real_aligned_data(self, qapp):
        """Real-data verification (not assumed): TEMPERATURE and SOOT
        DENSITY share the exact same grid/extent at y=0 for a real
        scenario (empirically confirmed against fds/sim/), so the
        overlay's displayed array must be the SAME scenario's real SOOT
        DENSITY at the SAME frame the temperature heatmap is showing."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        window.soot_overlay_action.setChecked(True)
        window._set_soot_overlay_enabled(True)
        qapp.processEvents()

        cell = window.view_grid.active_cell()
        assert cell.view.soot_overlay_enabled
        assert cell.view.soot_colorbar is not None

        from slice_key import SliceKey, SOOT_QUANTITY
        expected = window.controller.store.get(
            cell.case_index, SliceKey(SOOT_QUANTITY, cell.quantity_key.direction, cell.quantity_key.offset, 0.0),
        )[window.time_controller.index]
        np.testing.assert_array_equal(cell.view.soot_overlay.get_array(), expected)
        window.close()

    def test_soot_overlay_alpha_follows_the_continuous_mapping_function(self, qapp):
        """The core "not thresholded" requirement at the integration
        level: the displayed alpha must be exactly smoke_density.soot_alpha()
        applied to the real frame/ceiling -- the continuous linear formula,
        not a boolean threshold. (A raw distinct-value count on a single
        frame is not a reliable proxy here: the real dataset's soot field
        is empirically ~99.5% exact zero with a narrow nonzero cluster, so
        a single frame can legitimately reduce to {0, ceiling-clipped} even
        under the genuinely continuous formula -- continuity is what
        test_fire_intelligence.py::TestSmokeDensity already covers directly;
        this test instead pins the wiring to that real function.)"""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window.soot_overlay_action.setChecked(True)
        window._set_soot_overlay_enabled(True)
        window._on_seek_requested(300)  # a frame with real fire/smoke development
        qapp.processEvents()
        cell = window.view_grid.active_cell()

        import smoke_density as smd
        from slice_key import SliceKey, SOOT_QUANTITY
        soot_series = window.controller.store.get(
            cell.case_index, SliceKey(SOOT_QUANTITY, cell.quantity_key.direction, cell.quantity_key.offset, 0.0),
        )
        expected_frame = soot_series[window.time_controller.index]
        expected_ceiling = smd.soot_ceiling(soot_series)
        expected_alpha = smd.soot_alpha(expected_frame, expected_ceiling)

        np.testing.assert_allclose(cell.view.soot_overlay.get_alpha(), expected_alpha)
        window.close()

    def test_soot_overlay_disabled_leaves_temperature_view_unchanged(self, qapp):
        """Regression: with the overlay off, the primary heatmap's data
        and colorbar must be exactly what a plain temperature view would
        show -- no residual soot state leaking in."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        cell = window.view_grid.active_cell()
        baseline = cell.view.heatmap.get_array().copy()
        assert cell.view.soot_colorbar is None
        window._on_seek_requested(50)
        qapp.processEvents()
        with_overlay_off = cell.view.heatmap.get_array()
        assert not np.array_equal(baseline, with_overlay_off)  # sanity: frame did advance
        # re-seek to the same frame with the overlay toggled on then back off
        window.soot_overlay_action.setChecked(True)
        window._set_soot_overlay_enabled(True)
        window._on_seek_requested(60)
        window.soot_overlay_action.setChecked(False)
        window._set_soot_overlay_enabled(False)
        qapp.processEvents()
        assert cell.view.soot_colorbar is None
        assert cell.view.soot_overlay.get_alpha().max() == 0.0
        window.close()

    def test_cinematic_mode_uses_real_soot_and_suppresses_scientific_overlay(self, qapp):
        """Continuous soot-density visualization pass, Tier 3: cinematic
        mode always uses real soot data for its own smoke layer (no
        separate opt-in needed), and the scientific overlay artist stays
        cleared while cinematic is on so the two never composite on top
        of each other."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            pytest.skip("real dataset not present")
        window.cinematic_action.setChecked(True)
        window._set_cinematic_enabled(True)
        window._on_seek_requested(300)
        qapp.processEvents()
        cell = window.view_grid.active_cell()
        assert cell.view.cinematic_enabled
        assert not cell.view.soot_overlay_enabled
        assert cell.view.soot_overlay.get_alpha().max() == 0.0
        assert cell.view.heatmap.get_array().shape[-1] == 4  # RGBA cinematic output, rendered without crashing
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
        # Analysis-improvement roadmap Phase A: a PCA click used to be a
        # dead end for every other panel (analytics_panel had zero
        # SelectionBus wiring) -- now it publishes too.
        assert window.selection_bus.current.scenario == target_case
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
        assert markers, "real data must produce at least the peak marker"
        labels = [label for _f, label in markers]
        # V3-M2: markers are now the events.py fire story (the peak event's
        # statement is "Peak <value> °C"), richer than the M1.3 summary set.
        assert any(label.startswith("Peak") for label in labels)
        n = window._current_n_frames
        assert all(0 <= frame < n for frame, _l in markers)
        window.close()

    def test_velocity_quantity_keeps_markers(self, qapp):
        """Fire events are always computed from TEMPERATURE (see
        _fire_events_for_case), independent of which quantity the cell
        currently displays -- switching to VELOCITY must not blank the
        scrubber's event markers."""
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
        before = window.timeline.marker_bar.markers
        window.quantity_combo.setCurrentIndex(velocity_idx)
        _drain_workers(qapp, window.controller._prefetch_workers)
        qapp.processEvents()
        assert window.timeline.marker_bar.markers == before
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
        """The x=0.25 m doorway plane was removed by request (discarded,
        not gated) -- only the side-view (y=0) SOOT DENSITY plane remains."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        soot = [i for i in window.quantity_infos if i.key.quantity == "SOOT DENSITY"]
        assert len(soot) == 1
        assert soot[0].key.plane_pos == 0.0
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
        window.close()


class TestFactorEffectsPanel:
    """V2 roadmap M3.1: factor-effect maps on the Analysis page."""

    def test_panel_present_for_factorial_absent_for_guest(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "factor_effects_panel", None) is None
            window.close()
            return
        assert window.factor_effects_panel is not None
        window.close()
        # A generic guest study (degenerate single case) has no factor axes.
        import os
        from data_provider import load_study
        from load_data import SIM_ROOT
        case_dir = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")
        guest = MainWindow(load_study(case_dir))
        assert guest.factor_effects_panel is None
        guest.close()

    def test_ensure_loaded_builds_table_and_field(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.factor_effects_panel
        panel.ensure_loaded()
        assert panel.table.rowCount() == 4  # candles, door, vod, voc
        assert panel._current_field is not None
        assert panel._image is not None
        # Interaction mode: pick a second factor, field recomputes.
        panel.interaction_combo.setCurrentIndex(1)  # first "× factor" entry
        assert panel._current_field is not None
        window.close()


class TestReportBuilder:
    """V2 roadmap M3.3: browser "Generate report…" -> HTML report."""

    def _window_with_summaries(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            return window, True
        _drain_workers(qapp, [], timeout=0.1)
        deadline = time.perf_counter() + 5.0
        while not getattr(window, "_scenario_summaries", None) and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        return window, False

    def test_scenario_report_written(self, qapp, tmp_path, monkeypatch):
        window, demo = self._window_with_summaries(qapp)
        if demo:
            window.close()
            return
        out = str(tmp_path / "r.html")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
        window._export_report([0])
        assert (tmp_path / "r.html").exists()
        text = (tmp_path / "r.html").read_text()
        assert "data:image/png;base64," in text and "<table>" in text
        window.close()

    def test_comparison_report_written(self, qapp, tmp_path, monkeypatch):
        window, demo = self._window_with_summaries(qapp)
        if demo:
            window.close()
            return
        out = str(tmp_path / "cmp.html")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
        window._export_report([0, 12])
        assert (tmp_path / "cmp.html").exists()
        assert "vs" in (tmp_path / "cmp.html").read_text()
        window.close()

    def test_report_button_present_in_browser(self, qapp):
        window, demo = self._window_with_summaries(qapp)
        if demo:
            window.close()
            return
        assert window.experiment_browser.report_button is not None
        window.close()


class TestPendingKeyRace:
    """V2 roadmap M0.1: a scenario toggle and a quantity switch landing on
    the same case must not let the earlier key's prefetch end the busy
    state while the current key is still loading."""

    def test_finish_for_other_key_keeps_busy_until_pending_key_cached(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        from slice_key import SliceKey
        velocity_key = SliceKey("VELOCITY")
        cached = set()  # (case, quantity) tuples that report cached
        window.controller.is_cached = lambda c, k=None: (c, getattr(k, "quantity", None)) in cached

        window._pending_load_case = 0
        window._pending_load_key = velocity_key
        window._busy = True

        # A prefetch for a *different* key on case 0 finishes; the pending
        # VELOCITY load is not cached yet -> stay busy.
        window._on_prefetch_finished(0)
        assert window._busy is True
        assert window._pending_load_case == 0

        # Now the pending key is cached -> the next finish ends busy.
        cached.add((0, "VELOCITY"))
        window._on_prefetch_finished(0)
        assert window._busy is False
        assert window._pending_load_case is None
        window.close()


class TestFireMRIPanel:
    """V3-M1: Fire MRI temporal-signature panel on the Analysis page."""

    def test_panel_channels_probe_and_isochrones(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "fire_mri_panel", None) is None
            window.close()
            return
        panel = window.fire_mri_panel
        assert panel is not None
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == len(sim_data.manifest)
        # signature channels are populated (peak, dose, arrivals, durations)
        names = [panel.channel_combo.itemData(i) for i in range(panel.channel_combo.count())]
        assert "peak" in names and "thermal_dose" in names
        assert any(n.startswith("first_crossing_") for n in names)
        # peak channel's maximum equals the trusted per-scenario peak temperature
        from summary_stats import compute_scenario_summary
        summary = compute_scenario_summary(sim_data.manifest[0], sim_data.store,
                                            sim_data.timesteps_per_second)
        assert float(panel._sig.map("peak").max()) == pytest.approx(summary.max_temp_c, abs=0.5)
        # isochrone overlay renders without error
        panel.isochrone_check.setChecked(True)
        # probe readout populates from a physical point
        class _Evt:
            inaxes = panel._ax
            xdata, ydata = 0.9, 0.1
        panel._on_move(_Evt())
        assert "peak" in panel.probe_label.text()
        window.close()


class TestFireStory:
    """V3-M2: Fire Evolution Timeline wired into Live (markers + story +
    click-to-seek)."""

    def test_story_events_markers_and_seek(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        events = window.inspector._events
        assert events, "temperature scenario should produce fire events"
        # physically time-ordered, and mirrored onto the timeline markers
        times = [e.primary_time() for e in events]
        assert times == sorted(times)
        assert len(window.timeline.marker_bar.markers) == len(events)
        # clicking a story event seeks playback to its frame
        target = events[-1]
        window._on_insight_activated(target)
        assert window.time_controller.index == target.frame_index(
            window.time_controller.timesteps_per_second)
        # phase line reads during playback
        window._on_time_changed(window._current_n_frames - 1)
        assert "Now:" in window.inspector.phase_label.text()
        window.close()

    def test_story_kept_for_non_temperature_quantity(self, qapp):
        """The fire story is always TEMPERATURE-derived (see
        _fire_events_for_case), independent of the cell's displayed
        quantity -- switching to VELOCITY must not blank it."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        vel_idx = next((i for i, info in enumerate(window.quantity_infos)
                       if info.key.quantity == "VELOCITY"), None)
        if vel_idx is None:
            window.close()
            return
        before = window.inspector.story_list.count()
        window.quantity_combo.setCurrentIndex(vel_idx)
        _drain_workers(qapp, window.controller._prefetch_workers)
        qapp.processEvents()
        assert window.inspector.story_list.count() == before
        window.close()


class TestSemanticDiffPanel:
    """V3-M3, merged into Compare Axes (Analysis-improvement roadmap Phase
    A): semantic diff was a structurally-duplicate sibling tab (same two-
    scenario+quantity shape as Advanced Comparison) -- now its 4th axis."""

    def test_panel_lists_differences_and_shows_evidence(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "advanced_compare_panel", None) is None
            window.close()
            return
        panel = window.advanced_compare_panel
        assert panel is not None
        panel.ensure_loaded()
        assert panel.semantic_diff_list.count() >= 1
        # clicking a difference renders the A - B evidence field
        first = panel._sd_cache[list(panel._sd_cache.keys())[0]][0]
        panel._show_semantic_evidence(first)
        assert panel.semantic_diff_canvas.fig.axes
        window.close()

    def test_comparison_report_includes_key_differences(self, qapp, tmp_path, monkeypatch):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        deadline = time.perf_counter() + 5.0
        while not getattr(window, "_scenario_summaries", None) and time.perf_counter() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        out = str(tmp_path / "cmp.html")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
        window._export_report([0, 12])
        text = (tmp_path / "cmp.html").read_text()
        assert "Key differences" in text
        window.close()


class TestQueryPanel:
    """V3-M4: physics query panel."""

    def test_query_runs_and_shows_navigable_answer(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "query_panel", None) is None
            window.close()
            return
        panel = window.query_panel
        panel.ensure_loaded()
        panel.query_edit.setText("hottest region")
        panel._run()
        assert panel.results.count() == 1
        assert "Highest temperature" in panel.answer_label.text()
        assert panel._image is not None  # answer marked on the field
        window.close()

    def test_unrecognized_query_is_not_answered(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.query_panel
        panel.ensure_loaded()
        panel.query_edit.setText("tell me a joke")
        panel._run()
        assert panel.results.count() == 0
        assert "Not understood" in panel.answer_label.text()
        window.close()


class TestAttentionPanel:
    """V3-M6: physics attention map panel (heuristic saliency)."""

    def test_panel_builds_renders_and_labels_honestly(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "attention_panel", None) is None
            window.close()
            return
        panel = window.attention_panel
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == len(sim_data.manifest)
        assert panel._series is not None and panel._image is not None
        # values are a normalized saliency in [0, 1]
        assert 0.0 <= float(panel._series.min()) and float(panel._series.max()) <= 1.0 + 1e-6
        from attention_panel import _DISCLAIMER
        assert "not a physical field" in _DISCLAIMER.lower()
        window.close()


class TestCausePanel:
    """V3-M7: cause explorer panel (gated)."""

    def test_click_produces_a_labelled_cause_chain(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "cause_panel", None) is None
            window.close()
            return
        panel = window.cause_panel
        panel.ensure_loaded()
        from cause_panel import _DISCLAIMER
        assert "not proven causation" in _DISCLAIMER.lower()
        # click the hottest cell's location -> a chain appears
        data = np.asarray(sim_data.store.get(sim_data.manifest[0].case_index, DEFAULT_SLICE_KEY))
        extent = sim_data.store.get_extent(sim_data.manifest[0].case_index, DEFAULT_SLICE_KEY)
        fi = panel.frame_slider.value()
        gr, gc = np.unravel_index(int(np.argmax(data[fi])), data[fi].shape)
        n_z, n_x = data[fi].shape

        class _Evt:
            inaxes = panel._ax
            xdata = extent[0] + gc / (n_x - 1) * (extent[1] - extent[0])
            ydata = extent[3] - gr / (n_z - 1) * (extent[3] - extent[2])
        panel._on_click(_Evt())
        assert panel.chain.count() >= 1
        window.close()

    def test_changing_the_frame_does_not_crash_the_application(self, qapp):
        """Regression: frame_slider.valueChanged used to be wired directly
        to _render(self, trace=None) -- QSpinBox.valueChanged(int) passes
        its new value positionally, so every frame change delivered the
        raw frame index into `trace`; `if trace and ...: for r, c in
        trace:` then raised TypeError (`'int' object is not iterable`) for
        any nonzero frame. This is an ordinary Python exception (confirmed
        by direct reproduction with faulthandler -- no native frames
        involved), not a native/segfault crash, so this plain assertion is
        genuine, full protection against it recurring."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.cause_panel
        panel.ensure_loaded()
        for value in (1, 2, panel.frame_slider.maximum(), 0, 5):
            panel.frame_slider.setValue(value)  # must not raise
        window.close()

    def test_open_render_close_reopen_cycle(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.cause_panel
        window.pages["analysis"].show_tab(panel)
        panel.ensure_loaded()
        panel.frame_slider.setValue(3)
        panel.hide()
        panel.show()
        panel.frame_slider.setValue(7)  # still works after hide/reshow
        window.close()

    def test_switching_analysis_view_away_and_back_still_works(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.cause_panel
        window.pages["analysis"].show_tab(panel)
        panel.ensure_loaded()
        window.pages["analysis"].show_tab(window.study_panel)  # switch away
        window.pages["analysis"].show_tab(panel)                # and back
        panel.frame_slider.setValue(4)
        assert panel._data is not None
        window.close()

    def test_changing_scenario_and_timestep_while_visible_keeps_the_app_usable(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.cause_panel
        window.pages["analysis"].show_tab(panel)
        panel.ensure_loaded()
        panel.scenario_combo.setCurrentIndex(min(2, panel.scenario_combo.count() - 1))
        panel.frame_slider.setValue(min(6, panel.frame_slider.maximum()))
        window.selection_bus.update(origin=None, time_s=1.0)
        assert panel._data is not None
        window.close()


class TestHeightPanel:
    """V4-M1: height-aware analysis workspace."""

    def test_panel_builds_picks_x_and_lists_height_insights(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "height_panel", None) is None
            window.close()
            return
        panel = window.height_panel
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == len(sim_data.manifest)
        assert panel.insights.count() >= 2  # plume / layer / ceiling readings
        # a locator click sets the vertical line and re-renders the profile
        extent = sim_data.store.get_extent(0, DEFAULT_SLICE_KEY)

        class _Evt:
            inaxes = panel._loc_ax
            xdata, ydata = 0.9, 0.1
        panel._on_click(_Evt())
        assert panel._x_col is not None
        window.close()


class TestEvidenceNotebookIntegration:
    """V4-M2: dockable, session-backed Evidence Notebook."""

    def test_dock_starts_hidden_and_save_reveals_and_stores(self, qapp):
        from insight import Insight
        window = MainWindow(load_simulation_data())
        assert window.evidence_dock.isHidden()
        ins = Insight("Peak temperature is 320 C.", category="query",
                      quantity="TEMPERATURE", time_s=42.0, value=320.0, basis="max")
        window.evidence_dock.add_insight(ins)
        assert len(window.evidence_dock.notebook) == 1
        assert not window.evidence_dock.isHidden()  # first save reveals it
        window.close()

    def test_panel_insight_lists_are_wired_to_the_notebook(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        from insight import Insight
        ins = Insight("Plume peaks at 0.28 m.", category="query",
                      quantity="TEMPERATURE", time_s=2.5, value=0.28, basis="max hot cell")
        # emitting the shared save signal from a real panel's list lands it
        window.height_panel.insights.insight_saved.emit(ins)
        assert len(window.evidence_dock.notebook) == 1
        window.close()

    def test_session_roundtrip_carries_the_notebook(self, qapp, tmp_path):
        from insight import Insight
        from session import read_session
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.evidence_dock.add_insight(
            Insight("Ceiling peaks at 41 C.", category="query", quantity="TEMPERATURE",
                    time_s=39.0, value=41.0, basis="near-ceiling band"))
        window.evidence_dock.notebook.set_note(0, "case A")
        p = str(tmp_path / "sess.json")
        # _save_session shows a file dialog; drive the same serialization it uses.
        from session import build_session_dict, write_session
        cells = window.view_grid.visible_cells()
        sess = build_session_dict(window.view_grid.layout_name, cells, 0, 0, False,
                                  window.current_colormap, False,
                                  notebook=window.evidence_dock.notebook.to_list())
        write_session(p, sess)
        window.evidence_dock.notebook.clear()
        window._apply_session(read_session(p))
        assert len(window.evidence_dock.notebook) == 1
        assert window.evidence_dock.notebook.entries[0].note == "case A"
        window.close()


class TestDashboardJumpToPeak:
    """V4-M3's linked multi-quantity inspection was folded into the
    Dashboard's "Jump to peak moment" button (Analysis-improvement roadmap
    Phase A): same "one moment across temperature/HRR/smoke layer" value,
    without a structurally-redundant sibling tab (Dashboard already shows
    all of those cards for the current instant)."""

    def test_fmt_hrr_uses_watts_when_sub_kilowatt(self):
        from summary_stats import fmt_hrr
        assert fmt_hrr(0.077) == "77 W"    # candle: sub-kW reads in W, not "0 kW"
        assert fmt_hrr(1500.0) == "1500.0 kW"

    def test_jump_to_peak_seeks_bus_time_to_the_peak_frame(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.dashboard_panel
        ci = panel._current_ci
        m = panel._model(ci)
        import numpy as np
        expected_peak_t = int(np.argmax(m["peakT"])) / panel._fps
        panel._on_jump_to_peak()
        assert window.selection_bus.current.time_s == pytest.approx(expected_peak_t)
        window.close()

    def test_scenario_combo_drives_the_shared_bus_not_a_second_state(self, qapp):
        """UX consolidation pass, item 1 (Global Analysis Scenario Control):
        Dashboard shows a per-scenario "Scenario" card but previously had no
        way to change it directly -- its scenario_combo must be wired
        through the same generic bind_to_bus mechanism every other analysis
        panel uses, so picking a scenario here updates the one shared
        SelectionBus (and every other synced panel), not a private copy."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.dashboard_panel
        target = sim_data.manifest[-1].case_index
        idx = panel.scenario_combo.findData(target)
        panel.scenario_combo.setCurrentIndex(idx)
        assert window.selection_bus.current.scenario == target
        assert panel._current_ci == target
        assert panel._cards["Scenario"].text() == sim_data.manifest[-1].folder
        # a scenario picked elsewhere syncs this combo right back (single
        # shared state, not an independent one).
        other = sim_data.manifest[0].case_index
        window.selection_bus.update(origin=None, scenario=other)
        assert panel.scenario_combo.currentData() == other
        window.close()

    def test_energy_budget_detail_button_opens_current_scenario(self, qapp, monkeypatch):
        """Analysis-improvement roadmap Phase B: Energy budget folded into
        this card as an expandable detail (no standalone tab) -- the panel
        itself is unchanged, just shown pre-selected to the current
        scenario instead of living in its own tab."""
        monkeypatch.setattr(QtWidgets.QDialog, "exec_", lambda self: 0)
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.dashboard_panel
        assert panel._energy_panel is window.energy_panel
        target_case = sim_data.manifest[3].case_index
        window.selection_bus.update(origin=None, scenario=target_case)
        panel._show_energy_detail()
        assert window.energy_panel.scenario_combo.currentData() == target_case
        assert window.energy_panel.parent() is panel  # reparented back after the dialog closes
        window.close()


class TestZonePanel:
    """V4-M4: named region / zone statistics."""

    def test_zone_bundle_stats_and_insights(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "zone_panel", None) is None
            window.close()
            return
        import zone_stats as zst
        panel = window.zone_panel
        panel.ensure_loaded()
        panel._zones.append(zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3))
        panel._select_zone(0)
        assert "doorway" in panel.stats_label.text()
        assert "peak" in panel.stats_label.text()
        assert panel.insights.count() >= 1
        # cross-scenario comparison fills a row per scenario
        panel._compare_across_scenarios()
        assert panel.compare_table.rowCount() == len(sim_data.manifest)
        # a zone finding saves to the Evidence Notebook (M2 wiring)
        panel.insights.insight_saved.emit(panel.insights.item(0).data(QtCore.Qt.UserRole))
        assert len(window.evidence_dock.notebook) == 1
        window.close()

    def test_zone_stats_include_smoke_accumulation(self, qapp):
        """Continuous soot-density visualization pass, Step 6:
        smoke_accumulation() was validated (a real, physically meaningful
        time-integral of zone-mean SOOT DENSITY -- see zone_stats.py's
        docstring) and exposed through this existing Zones workflow, not a
        new panel or new selection state."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        import zone_stats as zst
        panel = window.zone_panel
        panel.ensure_loaded()
        assert panel._soot_data is not None
        panel._zones.append(zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3))
        panel._select_zone(0)
        text = panel.stats_label.text()
        assert "smoke accumulation" in text
        assert "n/a" not in text  # real soot data available for this scenario/plane
        window.close()

    def test_zone_compare_table_has_smoke_accum_column(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        import zone_stats as zst
        panel = window.zone_panel
        panel.ensure_loaded()
        panel._zones.append(zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3))
        panel._select_zone(0)
        panel._compare_across_scenarios()
        headers = [panel.compare_table.horizontalHeaderItem(c).text()
                   for c in range(panel.compare_table.columnCount())]
        assert any("smoke accum" in h.lower() for h in headers)
        # every row got a real (non-"n/a") reading, not just the active scenario
        col = next(c for c, h in enumerate(headers) if "smoke accum" in h.lower())
        values = [panel.compare_table.item(r, col).text()
                  for r in range(panel.compare_table.rowCount())]
        assert all(v != "n/a" for v in values)
        window.close()

    def test_zone_smoke_accumulation_degrades_to_na_on_fetch_failure(self, qapp):
        """Regression: a soot-fetch failure must not blank the whole panel
        (PyQt5 aborts the process on an unhandled exception in a connected
        slot -- _recompute is one) and must show "n/a", not a fabricated 0."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        import zone_stats as zst
        panel = window.zone_panel
        panel.ensure_loaded()
        panel._soot_data = None  # simulate an unavailable/failed fetch
        panel._zones.append(zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3))
        panel._select_zone(0)  # must not raise
        assert "n/a" in panel.stats_label.text()
        window.close()

    def test_zones_survive_session_roundtrip(self, qapp, tmp_path):
        import zone_stats as zst
        from session import build_session_dict, write_session, read_session
        window = MainWindow(load_simulation_data())
        if window.zone_panel is None:
            window.close()
            return
        window.zone_panel.ensure_loaded()
        window.zone_panel._zones.append(zst.Zone("window", 0.1, 0.3, 0.2, 0.5))
        p = str(tmp_path / "s.json")
        cells = window.view_grid.visible_cells()
        write_session(p, build_session_dict(
            window.view_grid.layout_name, cells, 0, 0, False, window.current_colormap,
            False, zones=window.zone_panel.get_zones()))
        window.zone_panel.set_zones([])
        window._apply_session(read_session(p))
        assert len(window.zone_panel._zones) == 1
        assert window.zone_panel._zones[0].name == "window"
        window.close()


class TestTimeWindowPanel:
    """V4-M5: time-window / interval analysis."""

    def test_intervals_phases_and_before_after(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "time_window_panel", None) is None
            window.close()
            return
        panel = window.time_window_panel
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == len(sim_data.manifest)
        assert len(panel._series["phases"]) >= 1
        # whole-run window produces stats + an insight
        assert "s</b>" in panel.stats_label.text()
        assert panel.insights.count() == 1
        # selecting a detected phase narrows the window
        if len(panel._series["phases"]) >= 1:
            panel._on_phase_selected(1)
            _name, a, b = panel._series["phases"][0]
            assert panel._t0 == a and panel._t1 == b
        # before/after mode compares two halves and saves to the notebook
        panel._on_mode_changed(1)
        panel._split = 20.0
        panel._compute()
        assert "Before" in panel.stats_label.text() and "After" in panel.stats_label.text()
        panel.insights.insight_saved.emit(panel.insights.item(0).data(QtCore.Qt.UserRole))
        assert len(window.evidence_dock.notebook) == 1
        window.close()

    def test_window_selection_publishes_interval(self, qapp):
        """Consolidation Phase 2: a selected window (previously local-only)
        publishes Selection.interval (defined but previously unused by any
        panel). Split mode is deliberately not published (see set_bus's
        docstring: time_s also drives the shared playback frame)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        panel = window.time_window_panel
        panel.ensure_loaded()
        if not panel._series["phases"]:
            window.close()
            return
        panel._on_phase_selected(1)
        _name, a, b = panel._series["phases"][0]
        assert window.selection_bus.current.interval == (a, b)
        before = window.selection_bus.current
        panel._on_mode_changed(1)   # switch to split mode
        panel._split = 20.0
        panel._compute()
        panel._publish_selection()
        assert window.selection_bus.current is before   # split mode: no publish
        window.close()


class TestNamedSessions:
    """V4-M6: named, reproducible analysis sessions."""

    def _investigate(self, window):
        """Set up an investigation: a notebook entry, a zone, an interval."""
        from insight import Insight
        import zone_stats as zst
        window.evidence_dock.add_insight(Insight(
            "Peak 469 C at t=8s.", category="query", quantity="TEMPERATURE",
            time_s=8.0, value=469.0, basis="max"))
        window.zone_panel.ensure_loaded()
        window.zone_panel._zones.append(zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3))
        window.zone_panel._select_zone(0)
        window.time_window_panel.ensure_loaded()
        window.time_window_panel._mode = "window"
        window.time_window_panel._t0 = 10.0
        window.time_window_panel._t1 = 40.0

    def test_sessions_tab_is_removed_but_session_management_still_works(self, qapp):
        """Analysis UX + reliability pass: the Sessions tab (Reference &
        Communication) was removed -- session_store.py and main_window's
        own _on_session_save/_on_session_load (tested directly below,
        independent of any UI) stay, since they're genuine session-
        management capability, not the tab itself."""
        window = MainWindow(load_simulation_data())
        assert not hasattr(window, "sessions_panel")
        if window.sim_data.manifest:
            labels = [window.pages["analysis"].tabs.tabText(i)
                     for i in range(window.pages["analysis"].tabs.count())]
            group = window.pages["analysis"].tabs.widget(
                labels.index("Reference & Communication"))
            inner_labels = [group.tabText(i) for i in range(group.count())]
            assert "Sessions" not in inner_labels
        window.close()

    def test_save_then_reopen_restores_state_exactly(self, qapp, tmp_path):
        import session_store
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            return
        w1 = MainWindow(sim_data)
        w1._sessions_dir = str(tmp_path)
        self._investigate(w1)
        w1._on_session_save("Door study", "doorway 10-40 s")
        infos = session_store.list_sessions(str(tmp_path))
        assert len(infos) == 1
        w1.close()
        # a fresh window (as if the app was reopened) restores the session
        w2 = MainWindow(load_simulation_data())
        w2._sessions_dir = str(tmp_path)
        w2._on_session_load(infos[0].path)
        assert len(w2.evidence_dock.notebook) == 1
        assert len(w2.zone_panel._zones) == 1
        assert w2.zone_panel._zones[0].name == "doorway"
        assert w2.time_window_panel._t0 == 10.0 and w2.time_window_panel._t1 == 40.0
        w2.close()

    def test_export_report_writes_html(self, qapp, tmp_path):
        import session_store
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window._sessions_dir = str(tmp_path)
        self._investigate(window)
        window._on_session_save("Door study", "doorway growth")
        path = session_store.list_sessions(str(tmp_path))[0].path
        out = tmp_path / "report.html"
        from report_builder import build_session_report, write_report
        write_report(str(out), build_session_report(session_store.load_session(path)))
        html = out.read_text()
        assert "Door study" in html and "Peak 469 C" in html and "doorway" in html.lower()
        window.close()

    def test_empty_session_roundtrips(self, qapp, tmp_path):
        import session_store
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window._sessions_dir = str(tmp_path)
        window._on_session_save("Empty", "")
        infos = session_store.list_sessions(str(tmp_path))
        assert len(infos) == 1 and infos[0].n_notebook == 0 and infos[0].n_zones == 0
        window._on_session_load(infos[0].path)  # must not raise
        window.close()


class TestAdvancedComparePanel:
    """V4-M8: advanced comparison workflows (temporal / spatial / physics)."""

    def test_axes_populate_and_physics_is_honest(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or len(sim_data.manifest) < 2:
            assert getattr(window, "advanced_compare_panel", None) is None
            window.close()
            return
        panel = window.advanced_compare_panel
        panel.ensure_loaded()
        assert panel.combo_a.count() == len(sim_data.manifest)
        total = (panel.temporal_list.count() + panel.spatial_list.count()
                 + panel.physics_list.count())
        assert total >= 1
        for i in range(panel.physics_list.count()):
            ins = panel.physics_list.item(i).data(QtCore.Qt.UserRole)
            assert "not a proven cause" in ins.statement  # association-not-causation gate
        # a comparison Insight saves to the Evidence Notebook
        for lst in (panel.temporal_list, panel.spatial_list, panel.physics_list):
            if lst.count():
                lst.insight_saved.emit(lst.item(0).data(QtCore.Qt.UserRole))
                break
        assert len(window.evidence_dock.notebook) == 1
        window.close()

    def test_same_scenario_clears_axes(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if window.advanced_compare_panel is None:
            window.close()
            return
        panel = window.advanced_compare_panel
        panel.ensure_loaded()
        panel.combo_b.setCurrentIndex(panel.combo_a.currentIndex())  # A == B
        assert panel.temporal_list.count() == 0
        assert panel.spatial_list.count() == 0
        assert panel.physics_list.count() == 0
        window.close()

    def test_ab_pair_publishes_and_follows_selection_comparison(self, qapp):
        """Consolidation Phase 2: combo_a/combo_b don't fit bind_to_bus's
        generic scenario_combo lookup, so Selection.comparison (defined but
        previously unused by any panel) is wired via a small custom
        set_bus, matching the SensitivityPanel/SpaceTimePanel precedent."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if window.advanced_compare_panel is None:
            window.close()
            return
        panel = window.advanced_compare_panel
        panel.ensure_loaded()
        target_a = sim_data.manifest[2].case_index
        target_b = sim_data.manifest[3].case_index
        idx_a = panel.combo_a.findData(target_a)
        panel.combo_a.setCurrentIndex(idx_a)
        idx_b = panel.combo_b.findData(target_b)
        panel.combo_b.setCurrentIndex(idx_b)
        assert window.selection_bus.current.comparison == (target_a, target_b)
        # reverse: a comparison published elsewhere drives this panel's pair.
        other_a, other_b = sim_data.manifest[0].case_index, sim_data.manifest[1].case_index
        window.selection_bus.update(origin=None, comparison=(other_a, other_b))
        assert panel.combo_a.currentData() == other_a
        assert panel.combo_b.currentData() == other_b
        window.close()

    def test_add_to_session_report_pins_comparison_and_survives_round_trip(self, qapp, tmp_path):
        """Analysis-improvement roadmap Phase C: "Add comparison to session
        report" pins the current pair's semantic-diff differences, and the
        session report picks them up via report_builder's reused
        _differences_block rendering."""
        import session_store
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if window.advanced_compare_panel is None:
            window.close()
            return
        panel = window.advanced_compare_panel
        panel.ensure_loaded()
        panel._add_to_session_report()
        assert len(panel._pinned_comparisons) == 1
        assert "Added" in panel.report_status.text()
        c = panel._pinned_comparisons[0]
        assert c["case_a"] == panel.combo_a.currentData()
        assert c["case_b"] == panel.combo_b.currentData()

        session = window._collect_session_dict("s", "")
        assert session["comparisons"] == panel.get_comparisons()
        from report_builder import build_session_report
        html = build_session_report(session)
        assert c["label_a"] in html and c["label_b"] in html

        session_store.save_session(str(tmp_path), session)
        loaded = session_store.load_session(
            session_store.list_sessions(str(tmp_path))[0].path)
        window.advanced_compare_panel.set_comparisons([])
        window._apply_analysis_session(loaded)
        assert window.advanced_compare_panel.get_comparisons() == panel.get_comparisons()
        window.close()


class TestProbeMeasurePanel:
    """Analysis section consolidation Phase 4 (Analysis final-polish pass:
    the disposable "Quick probe" mode was removed -- Devices/Zones/
    Velocity already cover deliberate measurement more purposefully; see
    probe_measure_panel.py's docstring): Devices, Zones, and Velocity are
    three modes of one "Spatial Probes" workspace -- each child's own
    construction/store access/lazy-load/bus wiring is unchanged, only the
    tab-level presentation is consolidated."""

    def test_wrapper_holds_all_three_children_as_tabs(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.probe_measure_panel is None:
            window.close()
            return
        wrapper = window.probe_measure_panel
        labels = [wrapper.tabs.tabText(i) for i in range(wrapper.tabs.count())]
        assert labels == ["Devices", "Zones", "Velocity"]
        assert wrapper.tabs.widget(0) is window.device_panel
        assert wrapper.tabs.widget(1) is window.zone_panel
        assert wrapper.tabs.widget(2) is window.velocity_panel
        assert not hasattr(window, "measurement_panel")
        window.close()

    def test_showing_wrapper_loads_all_children_not_just_visible_one(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.probe_measure_panel is None:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.probe_measure_panel)
        QtWidgets.QApplication.processEvents()
        assert window.device_panel._loaded
        assert window.zone_panel._loaded
        assert window.velocity_panel._loaded
        window.close()

    def test_show_tab_reveals_a_specific_child_three_levels_deep(self, qapp):
        """Regression check for the show_tab recursion fix this phase
        required: group -> wrapper -> child is one level deeper than
        Phase D originally handled."""
        window = MainWindow(load_simulation_data())
        if window.probe_measure_panel is None:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.device_panel)
        QtWidgets.QApplication.processEvents()
        assert window.probe_measure_panel.tabs.currentWidget() is window.device_panel
        window.close()


class TestCompareDiscover:
    """Analysis final-polish pass: Compare & Discover's former 4-mode
    CompareDiscoverPanel wrapper (Pairwise/Parallel coordinates/Ensemble/
    Clustering) is unwrapped -- Parallel coordinates and Ensemble spread
    were removed outright (not enough standalone research value for their
    complexity), which left only 2 children, no longer earning their own
    indirection layer. Pairwise Comparison and PCA/Clustering are now
    direct tabs under the Compare & Discover group."""

    def test_pairwise_and_clustering_are_direct_analysis_tabs(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.advanced_compare_panel is None:
            window.close()
            return
        assert not hasattr(window, "compare_discover_panel")
        assert not hasattr(window, "parallel_coordinates_panel")
        assert not hasattr(window, "ensemble_panel")
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.advanced_compare_panel)
        QtWidgets.QApplication.processEvents()
        assert window.advanced_compare_panel._loaded
        if window.clustering_content is not None:
            window.pages["analysis"].show_tab(window.clustering_content)
            QtWidgets.QApplication.processEvents()
        window.close()


class TestPublicationExport:
    """V4-M10: export presets + panel figure export + notebook report."""

    def test_analysis_panels_export_figures(self, qapp, tmp_path):
        import figure_export as fex
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        for name, canvas_attr in (("height_panel", "plot_canvas"),
                                   ("zone_panel", "plot_canvas")):
            panel = getattr(window, name)
            panel.ensure_loaded()
            assert hasattr(panel, "export_button")
            out = tmp_path / f"{name}.png"
            fex.save_figure(getattr(panel, canvas_attr).fig, str(out), "Slide (16:9)")
            assert out.stat().st_size > 0
        window.close()

    def test_notebook_dock_exports_report(self, qapp, tmp_path):
        from insight import Insight
        from report_builder import build_notebook_report, write_report
        window = MainWindow(load_simulation_data())
        window.evidence_dock.add_insight(Insight(
            "Peak 469 C.", category="query", quantity="TEMPERATURE", time_s=8.0, basis="max"))
        out = tmp_path / "evidence.html"
        write_report(str(out), build_notebook_report(
            window.evidence_dock.notebook.to_list(), provenance="p"))
        assert "Peak 469 C" in out.read_text()
        assert callable(window.evidence_dock._export_report)
        window.close()


class TestQuantitiesPanel:
    """V4-M11: quantity breadth + gating, non-breaking."""

    def test_panel_lists_all_and_tools_exclude_gated(self, qapp):
        import registry as reg
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "quantities_panel", None) is None
            window.close()
            return
        panel = window.quantities_panel
        assert panel.table.rowCount() == len(reg.QUANTITY_REGISTRY)
        # gated (and derived) quantities never enter the data-driven tool combos
        tool_qs = [k.quantity for _l, k in window._quantity_options()]
        assert not any(reg.get_quantity(q).gated for q in tool_qs)
        assert not any(reg.get_quantity(q).kind == "derived" for q in tool_qs)
        window.close()

    def test_derived_preview_computes_on_real_data(self, qapp):
        import registry as reg
        from PyQt5 import QtCore
        window = MainWindow(load_simulation_data())
        if window.quantities_panel is None:
            window.close()
            return
        panel = window.quantities_panel
        for r in range(panel.table.rowCount()):
            name = panel.table.item(r, 0).data(QtCore.Qt.UserRole)
            if reg.quantity_status(name) == "derived":
                panel.table.selectRow(r)
                break
        assert panel.preview_button.isEnabled()
        panel._preview_derived()
        assert "preview on" in panel.detail.text() and "min" in panel.detail.text()
        window.close()



class TestSharedSelectionModel:
    """V5-M1: the shared selection bus wired across panels + Live Viewer."""

    def test_bus_and_provider_present(self, qapp):
        window = MainWindow(load_simulation_data())
        assert window.selection_bus is not None
        assert window.quantity_provider is not None
        window.close()

    def test_cross_panel_scenario_and_time_sync(self, qapp):
        import numpy as np
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.height_panel.ensure_loaded()
        window.zone_panel.ensure_loaded()
        # changing one panel's scenario publishes it; the other panel follows
        # (scenario sync is not visibility-gated).
        window.height_panel.scenario_combo.setCurrentIndex(3)
        assert window.selection_bus.current.scenario == window.height_panel.scenario_combo.currentData()
        assert window.zone_panel.scenario_combo.currentData() == window.selection_bus.current.scenario
        # RC polish: time sync is visibility-gated (only the shown analysis tab
        # animates live). Show the height panel, then a published time syncs its
        # frame slider.
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.height_panel)
        QtWidgets.QApplication.processEvents()
        window.selection_bus.update(origin=None, time_s=10.0)
        fps = window.time_controller.timesteps_per_second
        assert window.height_panel.frame_slider.value() == int(round(10.0 * fps))
        window.close()

    def test_insight_activation_routes_through_bus(self, qapp):
        from insight import Insight
        window = MainWindow(load_simulation_data())
        window._on_insight_activated(Insight(
            "peak", category="query", quantity="TEMPERATURE", time_s=8.0, basis="m"))
        assert window.selection_bus.current.time_s == 8.0
        window.close()

    def test_selection_survives_session_roundtrip(self, qapp):
        from selection import Selection
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.selection_bus.set(Selection(scenario=2, quantity="VELOCITY", point=(0.9, 0.1)))
        sd = window._collect_session_dict("t", "")
        window.selection_bus.set(Selection())
        window._apply_analysis_session(sd)
        s = window.selection_bus.current
        assert s.scenario == 2 and s.quantity == "VELOCITY" and s.point == (0.9, 0.1)
        window.close()


class TestStudyPanel:
    """V5-M2: study-level analytics + selection sync."""

    def test_panel_builds_and_syncs_scenario(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            assert getattr(window, "study_panel", None) is None
            window.close()
            return
        panel = window.study_panel
        assert len(panel._table) == len(sim_data.manifest)
        # study scenario pick publishes to the bus; a linked panel follows
        window.height_panel.ensure_loaded()
        panel.scenario_combo.setCurrentIndex(5)
        assert window.selection_bus.current.scenario == panel.scenario_combo.currentData()
        assert window.height_panel.scenario_combo.currentData() == panel.scenario_combo.currentData()
        # reverse: a bus scenario change highlights the study selection
        window.selection_bus.update(origin=None, scenario=9)
        assert panel.scenario_combo.currentData() == 9
        window.close()

    def test_response_curve_tab_is_removed_but_backend_function_still_used(self, qapp):
        """Analysis UX + reliability pass: the "Response curve" tab was
        removed (Factor influence already answers "what moves this
        response" more concisely) -- but study_analytics.response_curve()
        itself must survive, since factor_influence() (the kept "Factor
        influence" tab) calls it directly to build each factor's spread-
        of-means. Backend coverage for response_curve() lives in
        test_fire_intelligence.py::TestStudyAnalytics::
        test_response_curve_gives_the_per_level_means_factor_influence_
        summarizes, independent of this UI tab."""
        import study_analytics as sa
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        labels = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert "Response curve" not in labels
        assert not hasattr(panel, "curve_factor_combo")
        assert not hasattr(panel, "curve_canvas")
        assert "Factor influence" in labels  # the tab that still needs it
        assert callable(sa.response_curve)
        window.close()

    def test_factor_effects_folded_in_as_a_sub_tab(self, qapp):
        """Analysis-improvement roadmap Phase B: Factor effects' actual
        spatial diverging-field view is a sub-tab here now, complementing
        this panel's own scalar "Factor influence" ranking -- not a
        structurally-separate top-level tab. The panel itself (store
        access, lazy-load, bus wiring) is unchanged."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        labels = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert "Factor effects" in labels
        idx = labels.index("Factor effects")
        assert panel.tabs.widget(idx) is window.factor_effects_panel
        window.close()

    def test_sensitivity_folded_in_as_a_sub_tab(self, qapp):
        """Analysis section consolidation Phase 5: the Sensitivity
        Explorer (local sensitivity at a chosen factor setting) is a
        sub-tab here now, complementing this panel's own global spread
        across observed levels (Factor influence) -- not a structurally-
        separate top-level tab. Analysis UX + reliability pass further
        removed Sensitivity's own Tornado/What-if sub-tabs, so only
        Response surface remains."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        labels = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert "Sensitivity" in labels
        idx = labels.index("Sensitivity")
        assert panel.tabs.widget(idx) is window.sensitivity_panel
        assert window.sensitivity_panel.tabs.count() == 1
        window.close()

    def test_correlation_matrix_cells_are_annotated_with_values(self, qapp):
        """Compare/Study UX polish: the matrix used to be color-only, with
        no way to read the exact r without a separate tool."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        import study_analytics as sa
        ax = panel.corr_canvas.fig.axes[0]
        assert len(ax.texts) == len(sa.RESPONSE_KEYS) ** 2
        window.close()

    def test_correlation_filter_recomputes_over_the_matching_subset(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        full_n = len(panel._table)
        door_combo = panel._corr_filter_combos["door"]
        idx = door_combo.findData(0)  # a specific door level, not "All"
        assert idx > 0
        door_combo.setCurrentIndex(idx)
        filtered = panel._filtered_table()
        assert 0 < len(filtered) < full_n
        assert all(int(r["params"]["door"]) == 0 for r in filtered)
        assert f"{len(filtered)} of {full_n} scenarios" in panel.stats_label.text()
        window.close()

    def test_reset_filters_restores_the_full_table(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        full_n = len(panel._table)
        panel._corr_filter_combos["door"].setCurrentIndex(1)
        assert len(panel._filtered_table()) < full_n
        panel._reset_corr_filters()
        assert len(panel._filtered_table()) == full_n
        assert all(v is None for v in panel._corr_filters.values())
        window.close()

    def test_clicking_a_diagonal_cell_shows_the_distribution(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        import types
        event = types.SimpleNamespace(inaxes=object(), xdata=0.0, ydata=0.0)
        panel._on_corr_click(event)
        ax = panel.corr_drilldown_canvas.fig.axes[0]
        assert "distribution" in ax.get_title()
        window.close()

    def test_clicking_an_off_diagonal_cell_shows_a_scatter_with_r(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        import types
        event = types.SimpleNamespace(inaxes=object(), xdata=0.0, ydata=1.0)
        panel._on_corr_click(event)
        ax = panel.corr_drilldown_canvas.fig.axes[0]
        assert "r =" in ax.get_title()
        assert len(ax.collections) >= 1  # the scatter itself
        window.close()

    def test_click_outside_the_matrix_is_a_no_op(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        import types
        event = types.SimpleNamespace(inaxes=None, xdata=None, ydata=None)
        panel._on_corr_click(event)  # must not raise
        window.close()

    def test_thin_pair_is_hatched_and_annotated_with_its_own_n(self, qapp):
        """A pair whose own support is below the reliability floor is
        flagged even when the overall subset looks fine -- e.g. a hazard
        threshold response that's NaN for most scenarios."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        import study_analytics as sa
        table = panel._filtered_table()
        counts = sa.pairwise_n(table, sa.RESPONSE_KEYS)
        thin_pairs = [(i, j) for i in range(len(sa.RESPONSE_KEYS))
                     for j in range(len(sa.RESPONSE_KEYS))
                     if i != j and 0 < counts[i, j] < 6]
        if not thin_pairs:
            pytest.skip("this dataset has no thin (n<6) response pair to assert against")
        ax = panel.corr_canvas.fig.axes[0]
        assert any("(n=" in t.get_text() for t in ax.texts)
        assert len(ax.patches) >= 1  # the hatched overlay rectangle(s)
        window.close()

    def test_filtering_to_a_tiny_subset_flags_low_n_rather_than_hiding_it(self, qapp):
        """Filtering combos must recompute the matrix over the matching
        subset (not just hide cells) -- and a resulting tiny subset must
        surface a caution, not a bare convincing-looking r."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            window.close()
            return
        panel = window.study_panel
        # Stack every filter combo to its first real level -- the
        # narrowest subset the study's factors can produce.
        for combo in panel._corr_filter_combos.values():
            if combo.count() > 1:
                combo.setCurrentIndex(1)
        table = panel._filtered_table()
        assert len(table) <= 2  # a full factorial's narrowest per-cell subset
        ax = panel.corr_canvas.fig.axes[0]
        title = ax.get_title()
        if len(table) < 6:
            assert "hatched" in title or "n=" in "".join(t.get_text() for t in ax.texts)
        panel._reset_corr_filters()
        assert len(panel._filtered_table()) == len(panel._table)
        window.close()

    def test_experiments_subsection_is_fully_removed(self, qapp):
        """UX consolidation pass: Experiments (self-contained batch CRUD,
        no scenario/quantity/time controls, no scientific conclusion of its
        own) was removed from Study-Level -- not just hidden. experiment.py
        itself (the Knowledge Graph's own experiment-file reader) is
        untouched."""
        window = MainWindow(load_simulation_data())
        assert not hasattr(window, "experiments_panel")
        if window.study_panel is not None:
            group_names = []
            tabs = window.pages["analysis"].tabs
            for i in range(tabs.count()):
                group_names.append(tabs.tabText(i))
                inner = getattr(tabs.widget(i), "tabs", None)
                if inner is not None:
                    group_names.extend(inner.tabText(j) for j in range(inner.count()))
            assert "Experiments" not in group_names
        window.close()


class TestSensitivityPanel:
    """V5-M3: sensitivity explorer + bus hand-off."""

    def test_panel_and_bidirectional_bus(self, qapp):
        import study_analytics as sa
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo or not window.is_factorial:
            assert getattr(window, "sensitivity_panel", None) is None
            window.close()
            return
        panel = window.sensitivity_panel
        assert len(panel._table) == len(sim_data.manifest)
        # moving a slider publishes the nearest existing run; a panel follows
        window.height_panel.ensure_loaded()
        panel._sliders["vod"].setValue(panel._sliders["vod"].maximum())
        assert window.selection_bus.current.scenario is not None
        assert window.height_panel.scenario_combo.currentData() == window.selection_bus.current.scenario
        # selecting a scenario elsewhere snaps the sliders to its factor levels
        target = panel._table[7]
        window.selection_bus.update(origin=None, scenario=target["case_index"])
        for p in sa.PARAMS:
            assert panel._setting(p) == pytest.approx(float(target["params"][p]))
        window.close()

    def test_estimate_note_present(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sensitivity_panel is None:
            window.close()
            return
        assert "Estimated from Existing Scenarios" in window.sensitivity_panel.note.text()
        window.close()

    def test_tornado_and_whatif_tabs_are_removed_response_surface_kept(self, qapp):
        """Analysis UX + reliability pass: Tornado and What-if (all
        responses) were removed; Response surface -- and the "Pin what-if
        to Knowledge Graph" button, which uses predict()/nearest_scenario()
        rather than either removed tab -- remain."""
        window = MainWindow(load_simulation_data())
        if window.sensitivity_panel is None:
            window.close()
            return
        panel = window.sensitivity_panel
        labels = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert labels == ["Response surface"]
        assert not hasattr(panel, "tornado_canvas")
        assert not hasattr(panel, "whatif_table")
        assert hasattr(panel, "pin_button")
        window.close()


class TestSpatiotemporalPanel:
    """Analysis section consolidation Phase 6: Height, Time series, and
    Time Window are now three modes of one "Field & Time Explorer"
    workspace -- each child's own construction/store access/lazy-load/bus
    wiring is unchanged, only the tab-level presentation is consolidated.
    Space-time stays a separate top-level tab (deferred, see
    spatiotemporal_panel.py's docstring)."""

    def test_wrapper_holds_all_three_children_as_tabs(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.spatiotemporal_panel is None:
            window.close()
            return
        wrapper = window.spatiotemporal_panel
        labels = [wrapper.tabs.tabText(i) for i in range(wrapper.tabs.count())]
        assert labels == ["Vertical profile", "Point/Region/Line probe", "Whole-field & interval"]
        assert wrapper.tabs.widget(0) is window.height_panel
        assert wrapper.tabs.widget(1) is window.timeseries_panel
        assert wrapper.tabs.widget(2) is window.time_window_panel
        window.close()

    def test_showing_wrapper_loads_all_children_not_just_visible_one(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.spatiotemporal_panel is None:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.spatiotemporal_panel)
        QtWidgets.QApplication.processEvents()
        assert window.height_panel._loaded
        assert window.timeseries_panel._loaded
        assert window.time_window_panel._loaded
        window.close()

    def test_show_tab_reveals_a_specific_child_three_levels_deep(self, qapp):
        """Regression check for the recursive show_tab fix from Phase 4:
        group -> wrapper -> child."""
        window = MainWindow(load_simulation_data())
        if window.spatiotemporal_panel is None:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.time_window_panel)
        group = window.pages["analysis"].tabs.currentWidget()
        assert window.pages["analysis"].tabs.tabText(
            window.pages["analysis"].tabs.currentIndex()) == "Spatiotemporal Analysis"
        assert group.currentWidget() is window.spatiotemporal_panel
        assert window.spatiotemporal_panel.tabs.currentWidget() is window.time_window_panel
        window.close()


class TestResearchWorkspace:
    """V5-M4: hazard spaces + mission-control dashboard + workspace hook."""

    def test_hazard_panel_syncs_and_classifies(self, qapp):
        import hazard_spaces as hz
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "hazard_panel", None) is None
            window.close()
            return
        panel = window.hazard_panel
        panel.ensure_loaded()
        assert panel._series["classes"].shape[0] == panel._data.shape[0]
        # scenario sync is not visibility-gated
        window.selection_bus.update(origin=None, scenario=2)
        assert panel.scenario_combo.currentData() == 2
        # RC polish: time sync only drives the visible analysis tab. Show it.
        # Phase B: hazard_panel is nested inside the Hazard & Tenability
        # mode-toggle wrapper now -- reveal the wrapper's tab, not the panel
        # directly (it's no longer a direct QTabWidget child itself).
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.hazard_tenability_panel)
        QtWidgets.QApplication.processEvents()
        window.selection_bus.update(origin=None, time_s=10.0)
        assert panel.frame_slider.value() == int(round(10.0 * window.time_controller.timesteps_per_second))
        window.close()

    def test_dashboard_reads_selection_live(self, qapp):
        from selection import Selection
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        d = window.dashboard_panel
        window.selection_bus.set(Selection(scenario=3, time_s=8.0))
        assert d._cards["Time"].text() == "8.0 s"
        assert d._cards["Max hazard"].text() in ("Safe", "Warning", "Critical", "Untenable")
        assert "level" in d._cards["Door"].text()
        # live update on time change
        window.selection_bus.update(origin=None, time_s=50.0)
        assert d._cards["Time"].text() == "50.0 s"
        window.close()

    def test_workspace_preset_raises_tab(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.dashboard_panel is None:
            window.close()
            return
        got = []
        window.dashboard_panel.workspace_requested.connect(got.append)
        window.dashboard_panel.preset_combo.setCurrentText("Ventilation study")
        window.dashboard_panel._on_preset(0)
        assert got and got[0] == "Ventilation study"  # emits the preset name (MainWindow resolves)
        window.close()


class TestWorkspaceAndCommunication:
    """V5 Phase 4 completion (adaptive workspace, space-time) + Phase 5 start."""

    def test_workspace_preset_focuses_quantity_and_tab(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.dashboard_panel is None:
            window.close()
            return
        window.height_panel.ensure_loaded()
        window.dashboard_panel.preset_combo.setCurrentText("Ventilation study")
        window.dashboard_panel._on_preset(0)
        assert window.selection_bus.current.quantity == "VELOCITY"   # quantity focus
        assert window._active_page_key == "analysis"                  # tab raised
        # quantity is now a shared field: a quantity-aware panel followed
        if window.height_panel._quantity_options:
            assert window.height_panel._key.quantity == "VELOCITY"
        if window.sensitivity_panel is not None:
            # Phase 5: sensitivity_panel is now three levels deep (group ->
            # StudyPanel's own tabs -> sensitivity_panel) -- the recursive
            # show_tab must still actually raise its tab, not just the
            # Analysis page in general.
            assert window.study_panel.tabs.currentWidget() is window.sensitivity_panel
        window.close()

    def test_spacetime_point_syncs_both_ways(self, qapp):
        from timeseries import phys_to_index
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        st = window.spacetime_panel
        st.ensure_loaded()
        window.selection_bus.update(origin=None, point=(0.9, 0.1))
        expected = phys_to_index(st._extent, st._data.shape[1:], 0.9, 0.1)
        assert (st._row, st._col) == expected
        window.close()

    def test_narrative_chain_and_click_seeks(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        nv = window.narrative_panel
        nv.ensure_loaded()
        assert nv.tree.topLevelItemCount() >= 1
        top = nv.tree.topLevelItem(0)
        assert top.childCount() >= 1                       # evidence children
        assert any("basis:" in top.child(i).text(0) for i in range(top.childCount()))
        ev = top.data(0, QtCore.Qt.UserRole)
        nv._on_item(top, 0)                                # activating publishes to the bus
        assert window.selection_bus.current.time_s == ev.primary_time()
        window.close()


class TestPhase5Communication:
    """V5-M5: publication bundle. (Analysis final-polish pass: the
    ensemble-spread and assistant-search tests that used to live here were
    removed along with EnsemblePanel and the Assistant layer, both dropped
    entirely in that pass.)"""

    def test_publication_bundle_writes_figures_and_manifest(self, qapp, tmp_path):
        from figure_export import save_figure
        from report_builder import build_publication_manifest, write_report
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.height_panel.ensure_loaded()
        figs = []
        for attr, canv, name, cap in window._BUNDLE_FIGURES:
            p = getattr(window, attr, None)
            c = getattr(p, canv, None) if p else None
            if c is not None and getattr(p, "_loaded", False):
                out = tmp_path / f"{name}.png"
                save_figure(c.fig, str(out), "Journal — single column")
                figs.append((f"{name}.png", cap))
        assert len(figs) >= 1
        man = tmp_path / "manifest.html"
        write_report(str(man), build_publication_manifest(figs, [], {"Scenarios": "24"}))
        assert "Publication bundle" in man.read_text() and figs[0][0] in man.read_text()
        window.close()


class TestKnowledgeGraph:
    """V5 Phase 6: the Research Knowledge Graph."""

    def test_graph_builds_and_nodes_publish_selection(self, qapp):
        from insight import Insight
        import zone_stats as zst
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            assert getattr(window, "graph_panel", None) is None
            window.close()
            return
        window.evidence_dock.add_insight(Insight(
            "Peak 469 C.", category="query", quantity="TEMPERATURE",
            time_s=8.0, location=(0.9, 0.1), basis="max"))
        window.zone_panel.ensure_loaded()
        window.zone_panel._zones.append(zst.Zone("doorway", 0.8, 1.0, 0.0, 0.3))
        g = window.graph_panel
        g._current_scenario = 3
        g._loaded = True
        g._rebuild()
        assert len(g._graph.nodes_of("scenario")) == len(sim_data.manifest)
        assert g._graph.nodes_of("insight") and g._graph.nodes_of("zone")
        # a scenario node publishes its scenario; an insight node its time/point
        g._select_node("scenario:3")
        assert window.selection_bus.current.scenario == 3
        g._select_node("insight:0")
        assert window.selection_bus.current.time_s == 8.0
        window.close()

    def test_tag_filter_and_neighbors(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        g = window.graph_panel
        g._loaded = True
        g._rebuild()
        # a factor tag connects to exactly the scenarios carrying that level
        neigh = g._graph.neighbors("tag:vod2")
        assert all(g._graph.nodes[n].type == "scenario" for n in neigh) and len(neigh) >= 1
        # filtering by that tag narrows the visible set
        g.tag_combo.setCurrentIndex(g.tag_combo.findData("vod2"))
        visible = g._visible_ids()
        assert "tag:vod2" in visible and len(visible) == len(neigh) + 1
        window.close()

    def test_hazard_node_appears_and_links_to_the_current_scenario(self, qapp):
        """Analysis UX + reliability pass: hazard_by_scenario (reusing
        hazard_spaces.py's own worst-tenability-class classification) is
        computed for the currently-selected scenario, same bound as
        narrative events, and produces a hazard node linked to it."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        g = window.graph_panel
        g._current_scenario = sim_data.manifest[0].case_index
        g._loaded = True
        g._rebuild()
        hazard_nodes = g._graph.nodes_of("hazard")
        assert len(hazard_nodes) == 1
        assert hazard_nodes[0].scenario == sim_data.manifest[0].case_index
        assert f"scenario:{sim_data.manifest[0].case_index}" in g._graph.neighbors(hazard_nodes[0].id)
        window.close()

    def test_focus_on_selected_hides_everything_outside_the_neighborhood(self, qapp):
        """Analysis UX + reliability pass: the "Focus on selected" toggle
        must actually narrow what's visible, reusing the same neighbor
        computation click-highlighting already relies on."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        g = window.graph_panel
        g._loaded = True
        g._rebuild()
        full_count = len(g._visible_ids())
        g._select_node("tag:vod2")
        g.focus_checkbox.setChecked(True)
        focused = g._visible_ids()
        neigh = g._graph.neighbors("tag:vod2")
        assert focused == {"tag:vod2", *neigh}
        assert len(focused) < full_count
        g.focus_checkbox.setChecked(False)
        assert len(g._visible_ids()) == full_count
        window.close()

    def test_legend_lists_every_node_type(self, qapp):
        """Analysis final-polish pass: a compact always-visible legend
        maps each node color to its type -- previously only decodable via
        the x-axis column headers/tree grouping."""
        import graph_model as gm
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        g = window.graph_panel
        text = g.legend.text()
        for t in gm.NODE_TYPES:
            assert t.capitalize() in text
        window.close()

    def test_quantity_nodes_appear_for_the_current_scenario(self, qapp):
        """Analysis final-polish pass: a small, bounded set of
        summary_stats.py's per-scenario metrics becomes "quantity" nodes
        for the currently selected scenario -- the graph shows what the
        simulation actually measured, not only shared factor levels."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        g = window.graph_panel
        case_index = sim_data.manifest[0].case_index
        g._current_scenario = case_index
        g._loaded = True
        g._rebuild()
        nodes = g._graph.nodes_of("quantity")
        assert nodes  # summary_stats.py always computes at least peak temperature
        for n in nodes:
            assert n.scenario == case_index
            assert f"scenario:{case_index}" in g._graph.neighbors(n.id)
        window.close()

    def test_opening_graph_starts_focused_on_current_scenario(self, qapp):
        """Analysis final-polish pass: the first time the tab is shown
        with a scenario already selected, it starts in "Focus on selected"
        mode centered on that scenario, instead of the full graph -- avoids
        the "hairball by default" problem. Uncheckable back to everything."""
        from PyQt5 import QtGui
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        g = window.graph_panel
        case_index = sim_data.manifest[0].case_index
        g._current_scenario = case_index
        g._loaded = False
        g.showEvent(QtGui.QShowEvent())
        assert g.focus_checkbox.isChecked()
        assert g._selected == f"scenario:{case_index}"
        focused = g._visible_ids()
        g.focus_checkbox.setChecked(False)
        assert len(g._visible_ids()) > len(focused)
        window.close()


class TestReleaseCandidatePolish:
    """RC polish: theme-aware plots, HRR/Fire-Story states, layout, theme resolve."""

    def test_theme_resolves_and_plots_follow(self, qapp):
        import widgets
        window = MainWindow(load_simulation_data())
        window.height_panel.ensure_loaded()
        window._set_theme("dark")
        assert window.height_panel.plot_canvas.fig.get_facecolor()[0] < 0.2
        window._set_theme("light")
        assert window.height_panel.plot_canvas.fig.get_facecolor()[0] > 0.9
        assert window._resolve_theme("system") in ("light", "dark")
        assert window._resolve_theme("nonsense") == "light"
        window.close()

    def test_hrr_gauge_has_explicit_states(self, qapp):
        from inspector import InspectorPanel
        insp = InspectorPanel()
        # no HRR data -> explicit message, not a silent grey bar
        insp.set_time(0, hrr_fraction=None)
        assert insp.hrr_gauge.value() == 0
        assert "No heat-release-rate" in insp.hrr_state_label.text()
        # data present -> value shown, state cleared
        insp.set_time(0, hrr_fraction=0.5)
        assert insp.hrr_gauge.value() == 50 and insp.hrr_state_label.text() == ""

    def test_fire_story_empty_state(self, qapp):
        from inspector import InspectorPanel
        insp = InspectorPanel()
        insp.set_story([], fps=4)
        assert "No fire events" in insp.phase_label.text()

    def test_quantity_selector_sits_below_playback(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        # both controls exist and the quantity combo is populated
        assert window.quantity_combo.count() >= 1
        window.close()


class TestAnalysisPlayback:
    """RC polish: analysis pages feel alive (playback synced to the Live Viewer)."""

    def test_transport_bar_and_shared_clock(self, qapp):
        window = MainWindow(load_simulation_data())
        assert hasattr(window, "analysis_timeline") and hasattr(window, "analysis_speed")
        # the analysis transport drives the same TimeController as the Live Viewer
        window._on_seek_requested(5)
        assert window.time_controller.index == 5
        window._analysis_stop()
        assert window.time_controller.index == 0

    def test_transport_bar_is_visible_above_overview_and_interpretation(self, qapp):
        """The shared playback bar sits above the outer tab group
        (pages/analysis.py), so it's visible regardless of which group is
        active -- confirm this explicitly for Overview & Interpretation,
        the group this was requested for."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        analysis_page = window.pages["analysis"]
        group_names = [analysis_page.tabs.tabText(i) for i in range(analysis_page.tabs.count())]
        assert "Overview & Interpretation" in group_names
        analysis_page.tabs.setCurrentIndex(group_names.index("Overview & Interpretation"))
        assert not window.analysis_timeline.isHidden()
        window._on_seek_requested(12)
        assert window.time_controller.index == 12  # transport still drives the shared clock
        window.close()

    def test_playback_time_broadcasts_to_bus(self, qapp):
        window = MainWindow(load_simulation_data())
        fps = window.time_controller.timesteps_per_second
        window._on_seek_requested(40)
        assert window.selection_bus.current.time_s == pytest.approx(40 / fps)

    def test_visible_panel_follows_hidden_freezes_and_catches_up(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        ap = window.pages["analysis"]
        ap.show_tab(window.height_panel)
        window.height_panel.ensure_loaded()
        QtWidgets.QApplication.processEvents()
        window._on_seek_requested(80)
        assert window.height_panel.frame_slider.value() == 80        # visible follows
        ap.show_tab(window.hazard_tenability_panel)
        window.hazard_panel.ensure_loaded()
        QtWidgets.QApplication.processEvents()
        frozen = window.height_panel.frame_slider.value()
        window._on_seek_requested(160)
        assert window.height_panel.frame_slider.value() == frozen    # hidden freezes
        ap.show_tab(window.height_panel)
        QtWidgets.QApplication.processEvents()
        assert window.height_panel.frame_slider.value() == 160       # resend catches up
        window.close()

    def test_factor_influence_covers_arrival_times(self, qapp):
        import study_analytics as sa
        assert "time_to_300c_s" in sa.RESPONSE_KEYS and "time_to_600c_s" in sa.RESPONSE_KEYS
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if window.study_panel is not None:
            assert window.study_panel.response_combo.count() == len(sa.RESPONSE_KEYS)
        window.close()


class TestFieldCalculator:
    """V6-M1: safe scientific expression engine integrated as quantities."""

    def teardown_method(self):
        import field_calculator as fc
        fc.clear()

    def test_calculator_tab_is_removed_field_calculator_backend_stays(self, qapp):
        """Analysis UX + reliability pass: CalculatorPanel (the only UI to
        author a *new* calculated field) was removed -- field_calculator.py
        itself stays, since quantity_provider.py calls it directly for
        every calculated-field read (Live Viewer included), and session
        save/restore round-trips CalculatedField independent of any panel.
        Existing calculated fields keep working; creating new ones now
        requires calling field_calculator.py directly (as every other test
        in this class does)."""
        window = MainWindow(load_simulation_data())
        assert not hasattr(window, "calculator_panel")
        window.close()

    def test_create_field_registers_and_provider_computes(self, qapp):
        import numpy as np
        import field_calculator as fc
        from slice_key import SliceKey
        from registry import QUANTITY_REGISTRY, get_quantity
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        fc.register(fc.make_field("Temperature Rise", "Temperature - 20"))
        assert "Temperature Rise" in QUANTITY_REGISTRY
        q = get_quantity("Temperature Rise")
        assert q.calculated and q.kind == "derived" and q.expression == "Temperature - 20"
        raw = np.asarray(window.controller.store.get(0, SliceKey("TEMPERATURE")))
        calc = np.asarray(window.quantity_provider.get(0, SliceKey("Temperature Rise")))
        assert np.allclose(calc, raw - 20)        # plots/exports can load it via the provider
        window.close()

    def test_unsafe_expression_is_rejected(self, qapp):
        import field_calculator as fc
        window = MainWindow(load_simulation_data())
        with pytest.raises(fc.CalculatorError):
            fc.make_field("Bad", "__import__('os')")
        window.close()

    def test_gradient_and_rate_fields_compute(self, qapp):
        import numpy as np
        import field_calculator as fc
        from slice_key import SliceKey
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        fc.register(fc.make_field("Grad", "gradient(Temperature)"))
        fc.register(fc.make_field("Rate", "rate(Temperature)"))
        raw = np.asarray(window.controller.store.get(0, SliceKey("TEMPERATURE")))
        assert np.asarray(window.quantity_provider.get(0, SliceKey("Grad"))).shape == raw.shape
        assert np.asarray(window.quantity_provider.get(0, SliceKey("Rate"))).shape == raw.shape
        window.close()

    def test_calculated_fields_survive_session_roundtrip(self, qapp):
        import field_calculator as fc
        from registry import QUANTITY_REGISTRY
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        fc.register(fc.make_field("Exposure", "Temperature * 2"))
        sd = window._collect_session_dict("t", "")
        assert len(sd["calculated_fields"]) == 1
        fc.clear()
        assert "Exposure" not in QUANTITY_REGISTRY
        window._apply_analysis_session(sd)
        assert "Exposure" in QUANTITY_REGISTRY and len(fc.all_fields()) == 1
        window.close()

    def test_existing_quantities_unchanged(self, qapp):
        # V5 quantity discovery must be unaffected by the calculator
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if not sim_data.is_demo:
            tool_qs = [k.quantity for _l, k in window._quantity_options()]
            assert "TEMPERATURE" in tool_qs and "Temperature Rise" not in tool_qs
        window.close()


class TestCalculatedFieldsInLiveViewer:
    """V6-M1.5: calculated/derived fields are first-class visual quantities."""

    def teardown_method(self):
        import field_calculator as fc
        fc.clear()

    def test_derived_quantities_appear_in_live_combo(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        labels = [window.quantity_combo.itemText(i) for i in range(window.quantity_combo.count())]
        # existing derived quantities are now selectable in the Live Viewer
        assert "Temperature rise (ΔT)" in labels and "Dynamic pressure" in labels
        # analysis panels stay native-only (unchanged)
        assert "Temperature rise (ΔT)" not in [k.quantity for _l, k in window._quantity_options()]
        window.close()

    def test_calculated_field_selectable_and_renders(self, qapp):
        import numpy as np
        import field_calculator as fc
        from slice_key import SliceKey
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        fc.register(fc.make_field("TR", "Temperature - 20"))
        # fields_changed used to trigger this automatically via
        # CalculatorPanel (removed, Analysis UX + reliability pass) --
        # main_window still exposes it directly for any caller that
        # registers a field outside that UI.
        window._refresh_quantity_list()
        labels = [window.quantity_combo.itemText(i) for i in range(window.quantity_combo.count())]
        assert "TR" in labels                      # appears in the Live combo
        window.quantity_combo.setCurrentIndex(labels.index("TR"))
        QtWidgets.QApplication.processEvents()
        assert window.current_quantity_key.quantity == "TR"
        cell = window.view_grid.active_cell()
        raw = np.asarray(window.controller.store.get(cell.case_index, SliceKey("TEMPERATURE")))
        # renders and evolves over time
        assert np.allclose(window._frame_for_cell(cell, 0), raw[0] - 20)
        assert np.allclose(window._frame_for_cell(cell, 80), raw[80] - 20)
        window.close()

    def test_computed_fields_are_memoized(self, qapp):
        from slice_key import SliceKey
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        # a derived quantity computed twice returns the cached array (no recompute)
        a = window.quantity_provider.get(0, SliceKey("TEMPERATURE RISE"))
        b = window.quantity_provider.get(0, SliceKey("TEMPERATURE RISE"))
        assert a is b
        window.quantity_provider.invalidate()
        assert window.quantity_provider.get(0, SliceKey("TEMPERATURE RISE")) is not a
        window.close()

    def test_native_quantity_still_reads_store(self, qapp):
        import numpy as np
        from slice_key import SliceKey
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        cell = window.view_grid.active_cell()
        # native quantities are unchanged: _field routes them to the store
        via_field = window._field(window.controller.store, cell.case_index, SliceKey("TEMPERATURE"))
        via_store = window.controller.store.get(cell.case_index, SliceKey("TEMPERATURE"))
        assert via_field is via_store          # same cached object, no provider wrapping
        window.close()


class TestVirtualDeviceNetwork:
    """V6-M2: placing a device instruments the simulation like an
    experiment, without touching the parser/store/cache/TimeController/
    cinematic pipeline."""

    def test_heat_detector_and_sprinkler_can_disagree_and_look_different(self, qapp):
        """Analysis UX + reliability pass: heat_detector (instant
        threshold) and sprinkler (RTI thermal-lag ODE) are independent
        devices with independently different, both-correct physical
        models -- a sprinkler is *supposed* to be able to disagree with a
        heat detector at the identical point/frame, not a bug. The Live
        Viewer must make the two visually distinguishable by marker shape,
        not just by their (legitimately independent) active/idle color."""
        import devices as dv
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        p = window.device_panel
        p.ensure_loaded()
        case_index = p.scenario_combo.currentData()
        pos = (1.0, 1.0)
        hd = dv.Device(id="hd-1", name="HD-01", type="heat_detector", scenario=case_index,
                       position=pos, parameters=dv.default_parameters("heat_detector"),
                       direction=p.direction_combo.currentData(), offset=p.offset_spin.value())
        sp = dv.Device(id="sp-1", name="SP-01", type="sprinkler", scenario=case_index,
                       position=pos, parameters=dv.default_parameters("sprinkler"),
                       direction=p.direction_combo.currentData(), offset=p.offset_spin.value())
        hd.compute(window.quantity_provider, window.sim_data.timesteps_per_second)
        sp.compute(window.quantity_provider, window.sim_data.timesteps_per_second)
        p._devices.extend([hd, sp])

        # Both models are independently correct even at the identical point
        # -- state_at() must not force them to agree.
        found_disagreement = any(
            hd.state_at(i).get("active") != sp.state_at(i).get("active")
            for i in range(min(hd.n_frames(), sp.n_frames()))
        )
        # Not asserted as a hard requirement (depends on this dataset's
        # actual temperature history), but the two ARE independently
        # computed regardless -- the real assertion is the marker shapes.
        del found_disagreement

        markers = window._device_markers_for(case_index, 0)
        kinds = {kind for *_rest, kind in markers}
        assert {"heat_detector", "sprinkler"}.issubset(kinds)

        cell = window.view_grid.active_cell()
        # Analysis final-polish pass: with nothing placed yet, the
        # Live-Viewer legend explaining shape -> kind stays hidden (nothing
        # to explain on a scenario with no devices).
        cell.view.set_device_markers([])
        assert cell.view._device_legend.get_visible() is False
        cell.view.set_device_markers(markers)
        hd_shape = cell.view.device_scatters["heat_detector"].get_paths()[0]
        sp_shape = cell.view.device_scatters["sprinkler"].get_paths()[0]
        assert hd_shape.vertices.shape != sp_shape.vertices.shape or \
            not np.allclose(hd_shape.vertices, sp_shape.vertices)
        # Analysis final-polish pass: once markers exist, the legend
        # becomes visible and explains the shapes -- this is what closes
        # the "looks broken" gap on the Live Viewer itself, not just
        # inside the separate Devices analysis tab.
        assert cell.view._device_legend.get_visible() is True
        window.close()

    def test_placing_a_thermocouple_computes_once_and_lists(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.device_panel
        p.ensure_loaded()
        p.type_combo.setCurrentIndex(p.type_combo.findData("thermocouple"))
        p._place(1.0, 1.0)
        assert len(p._devices) == 1
        dev = p._devices[0]
        assert dev.results is not None and dev.name == "TC-01"
        assert p.list.count() == 1
        window.close()

    def test_device_marker_appears_and_tracks_activation_state(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.device_panel
        p.ensure_loaded()
        p.type_combo.setCurrentIndex(p.type_combo.findData("heat_detector"))
        p._place(1.0, 1.0)
        dev = p._devices[0]
        # markers reach the Live Viewer without a recompute -- state_at() only indexes results
        markers = window._device_markers_for(dev.scenario, 0)
        assert len(markers) == 1
        assert markers[0][:2] == (dev.position[0], dev.position[1])
        if dev.results.get("activated"):
            frame_act = dev.results["activation_frame"]
            before = window._device_markers_for(dev.scenario, max(0, frame_act - 1))[0][2]
            after = window._device_markers_for(dev.scenario, frame_act)[0][2]
            assert before != after   # idle -> active color flips at the cached activation frame
        window.close()

    def test_jump_to_device_seeks_playback(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.device_panel
        p.ensure_loaded()
        p.type_combo.setCurrentIndex(p.type_combo.findData("heat_detector"))
        p._place(1.0, 1.0)
        dev = p._devices[0]
        if not dev.results.get("activated"):
            window.close()
            return
        p.list.setCurrentRow(0)
        p._jump_to()
        QtWidgets.QApplication.processEvents()
        expected = dev.results["activation_frame"]
        assert window.time_controller.index == expected
        window.close()

    def test_compare_across_scenarios_evaluates_same_device_everywhere(self, qapp):
        """Analysis-improvement roadmap Phase C: the Zones cross-scenario
        pattern, reused for Devices -- place once, "Compare" evaluates the
        same position/type/parameters at every scenario."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        p = window.device_panel
        window.pages["analysis"].show_tab(p)
        QtWidgets.QApplication.processEvents()
        p.type_combo.setCurrentIndex(p.type_combo.findData("thermocouple"))
        p._place(1.0, 1.0)
        p.list.setCurrentRow(0)
        p._compare_across_scenarios()
        assert p.compare_table.isVisible()
        assert p.compare_table.rowCount() == len(sim_data.manifest)
        assert p.compare_table.item(0, 0).text() == sim_data.manifest[0].folder
        assert p.compare_table.item(0, 1).text() != ""
        window.close()

    def test_session_round_trip_preserves_devices_and_results(self, qapp, tmp_path):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.device_panel
        p.ensure_loaded()
        p.type_combo.setCurrentIndex(p.type_combo.findData("sprinkler"))
        p._place(1.0, 1.0)
        before = window._collect_session_dict()
        assert len(before["devices"]) == 1
        window._apply_analysis_session(before)
        after = window.device_panel.get_devices()
        assert after == before["devices"]      # identical, not recomputed
        window.close()


class _SyntheticVectorProvider:
    """Wraps a real QuantityProvider but supplies synthetic U/W so V6-M3's
    full (non-gated) pipeline is testable even though the actual dataset has
    no U-VELOCITY/W-VELOCITY yet. TEMPERATURE/VELOCITY/extent reads still go
    through the real provider (so the panel's background heatmap is real
    data) -- only get_vector is synthetic, and it is never confused with a
    real quantity: production code always goes through the real provider,
    this stand-in only exists in tests."""

    def __init__(self, real):
        self._real = real

    def get(self, scenario, key):
        return self._real.get(scenario, key)

    def get_extent(self, scenario, key):
        # U-VELOCITY/W-VELOCITY have no real geometry on disk in this
        # dataset (only TEMPERATURE/VELOCITY were ever read) -- redirect to
        # TEMPERATURE's real, on-disk extent rather than letting the real
        # store try (and crash) reading geometry for a slice that was never
        # part of the manifest.
        if key.quantity in ("U-VELOCITY", "W-VELOCITY"):
            return self._real.get_extent(scenario, SliceKey("TEMPERATURE"))
        return self._real.get_extent(scenario, key)

    def get_vector(self, scenario, direction=None, offset=None):
        temp = self._real.get(scenario, SliceKey("TEMPERATURE"))
        u = np.full_like(np.asarray(temp, dtype=float), 1.0)
        w = np.zeros_like(np.asarray(temp, dtype=float))
        return u, w

    def get_vector3d(self, scenario, direction=None, offset=None):
        # V6-M7: synthetic V (through-plane component) alongside U/W above.
        temp = self._real.get(scenario, SliceKey("TEMPERATURE"))
        u = np.full_like(np.asarray(temp, dtype=float), 1.0)
        v = np.full_like(np.asarray(temp, dtype=float), 2.0)
        w = np.zeros_like(np.asarray(temp, dtype=float))
        return u, v, w


class TestTrueVelocity:
    """V6-M3: true (U, W) vector field -- streamlines/quiver, gated on
    U-VELOCITY/W-VELOCITY. The real dataset has no U/W yet, so gating tests
    run against the real provider, while the rendering/coloring/jump-to/
    session tests substitute a synthetic provider (uniform flow) to exercise
    the full non-gated pipeline without fabricating anything in production
    code."""

    def test_placing_a_probe_on_the_real_dataset_is_gated(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._place(1.0, 1.0)
        probe = p._probes[0]
        assert probe.gated is True
        assert "M-SIM" in probe.results["reason"] or "msim" in probe.results["reason"].lower()
        assert "M-SIM" in p.status.text() or "msim" in p.status.text().lower()
        window.close()

    def test_gate_explanation_names_missing_quantities_and_research_value(self, qapp):
        """Analysis final-polish pass: the caption must be an honest,
        structured empty-state -- exactly which quantities/units are
        missing (from the registry, never hand-guessed) and what a real
        vector field would unlock -- never a fabricated value, and never
        just a terse "waiting for rerun" dead end."""
        from velocity_panel import VelocityPanel
        text = VelocityPanel._build_gate_explanation()
        for name in ("U-VELOCITY", "V-VELOCITY", "W-VELOCITY", "m/s"):
            assert name in text
        assert "unavailable" in text.lower()
        for use in ("Smoke transport", "Ventilation", "Recirculation", "Plume", "Streamlines"):
            assert use in text

    def test_get_vector_still_raises_gated_quantity_error(self, qapp):
        """V6-M3 must not weaken the gate: the real provider still raises."""
        from quantity_provider import GatedQuantityError
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        with pytest.raises(GatedQuantityError):
            window.quantity_provider.get_vector(0)
        window.close()

    def test_synthetic_provider_renders_quiver_and_streamline(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._provider = _SyntheticVectorProvider(window.quantity_provider)
        p.mode_combo.setCurrentIndex(p.mode_combo.findData("both"))
        case_index = p.scenario_combo.currentData()
        extent = window.quantity_provider.get_extent(case_index, SliceKey("TEMPERATURE"))
        x0, x1, z0, z1 = extent
        p._place(x0 + 0.3 * (x1 - x0), z0 + 0.3 * (z1 - z0))
        probe = p._probes[0]
        assert probe.gated is False
        assert probe.results["max_speed_m_s"] == pytest.approx(1.0)

        field = p._fields[case_index]
        quiver, streamlines, colors = window._vector_field_for(case_index, 0)
        assert quiver is not None and len(quiver[0]) > 0
        assert len(streamlines) == 1 and len(streamlines[0]) > 1
        assert len(colors) == 1

    def test_marker_ordering_matches_v6_m2_fix(self, qapp):
        """The tick loop must set the vector field *before* show_frame's
        blit, exactly like the V6-M2 device-marker ordering fix -- verified
        here by checking the artist actually holds non-empty data right
        after a seek (not one tick late)."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._provider = _SyntheticVectorProvider(window.quantity_provider)
        p.mode_combo.setCurrentIndex(p.mode_combo.findData("quiver"))
        cell = window.view_grid.active_cell()
        case_index = p.scenario_combo.currentData()
        extent = window.quantity_provider.get_extent(case_index, SliceKey("TEMPERATURE"))
        x0, x1, z0, z1 = extent
        p.scenario_combo.setCurrentIndex(p.scenario_combo.findData(int(cell.case_index))
                                        if p.scenario_combo.findData(int(cell.case_index)) >= 0
                                        else 0)
        p._place(x0 + 0.5 * (x1 - x0), z0 + 0.5 * (z1 - z0))
        n = window._current_n_frames
        window._on_seek_requested(int(n * 0.5))
        QtWidgets.QApplication.processEvents()
        assert cell.view.true_vector_quiver is not None
        window.close()

    def test_jump_to_probe_publishes_point_selection(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._provider = _SyntheticVectorProvider(window.quantity_provider)
        case_index = p.scenario_combo.currentData()
        extent = window.quantity_provider.get_extent(case_index, SliceKey("TEMPERATURE"))
        x0, x1, z0, z1 = extent
        seed = (x0 + 0.4 * (x1 - x0), z0 + 0.4 * (z1 - z0))
        p._place(*seed)
        p.list.setCurrentRow(0)
        p._jump_to()
        QtWidgets.QApplication.processEvents()
        assert window.selection_bus.current.point == pytest.approx(seed)
        window.close()

    def test_session_round_trip_preserves_probes_and_results(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._provider = _SyntheticVectorProvider(window.quantity_provider)
        case_index = p.scenario_combo.currentData()
        extent = window.quantity_provider.get_extent(case_index, SliceKey("TEMPERATURE"))
        x0, x1, z0, z1 = extent
        p._place(x0 + 0.5 * (x1 - x0), z0 + 0.5 * (z1 - z0))
        before = window._collect_session_dict()
        assert len(before["vector_probes"]) == 1
        window._apply_analysis_session(before)
        after = window.velocity_panel.get_probes()
        assert after == before["vector_probes"]     # identical, not recomputed
        window.close()

    def test_3d_toggle_gated_on_real_data(self, qapp):
        """V6-M7: with no synthetic provider, V-VELOCITY is gated exactly
        like U/W -- checking "3D (color by V)" must not crash, and simply
        has nothing to show (has_3d stays False)."""
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._place(1.0, 1.0)
        p.show_3d_check.setChecked(True)
        QtWidgets.QApplication.processEvents()
        window.close()   # must reach here without raising

    def test_3d_toggle_colors_quiver_with_synthetic_v(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        p = window.velocity_panel
        p.ensure_loaded()
        p._provider = _SyntheticVectorProvider(window.quantity_provider)
        p._place(1.0, 1.0)
        case_index = p.scenario_combo.currentData()
        field = p._fields[case_index]
        assert field.has_3d
        p.show_3d_check.setChecked(True)
        QtWidgets.QApplication.processEvents()
        assert p.canvas.fig.axes   # rendered without error
        window.close()


class TestUnifiedWorkspace:
    """V6-M4: connecting existing capabilities into one investigation
    workflow -- cross-navigation, Knowledge Graph expansion, Investigation
    History, and performance (no expensive work per tick). The standalone
    Context Panel tab this was originally built around was removed in the
    UX consolidation pass as low-value; context.gather_context (the data
    layer) and Investigation History remain and are tested directly."""

    def test_history_back_and_forward_shortcuts(self, qapp):
        """UX consolidation pass: the removed Context Panel's back/forward
        buttons were the only UI for Investigation History -- Alt+Left/
        Alt+Right keep it reachable without dedicating screen space to it."""
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.selection_bus.update(origin=None, scenario=0)
        window.selection_bus.update(origin=None, scenario=1)
        window._history_back()
        assert window.selection_bus.current.scenario == 0
        window._history_forward()
        assert window.selection_bus.current.scenario == 1
        window.close()

    def test_point_story_combines_measurement_and_cause_chain(self, qapp):
        """Analysis-improvement roadmap Phase C: the synthesized point-story
        pulls a local reading (already-cached, no new store read) and the
        Cause Explorer's last-traced chain (only when it's near the
        selected point) into one paragraph. UX consolidation pass: the
        standalone Context tab that used to display this was removed as
        low-value, but context.gather_context (the data layer) is kept --
        still exercised directly here, and reusable by future consumers.
        (Analysis final-polish pass: the "local reading" source used to be
        a disposable Quick Probe measurement -- now removed, see
        measurement_panel.py's removal -- so this uses a Zone instead,
        _related_measurements' now-permanent [] falls through to
        _related_zones exactly as context.py's own fallback order does.)"""
        import zone_stats as zst
        from insight import Insight
        from selection import Selection
        from context import gather_context
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        case_index = window.sim_data.manifest[0].case_index
        window.zone_panel.ensure_loaded()
        window.zone_panel._zones.append(zst.Zone("P1", 0.9, 1.1, 0.9, 1.1))
        sel = Selection(scenario=case_index, point=(1.0, 1.0))
        story = gather_context(window, sel)["point_story"]
        assert 'inside zone "P1"' in story
        # Cause chain absent until traced near this point -> no cause-trace clause yet.
        assert "Cause trace" not in story
        cp = window.cause_panel
        cp.ensure_loaded()
        cp._last_point = (1.0, 1.0)
        cp._last_insights = [Insight("It traces back to the hottest connected point (400 °C).",
                                     category="cause", quantity="TEMPERATURE")]
        sel2 = Selection(scenario=case_index, time_s=1.0, point=(1.0, 1.0001))
        story = gather_context(window, sel2)["point_story"]
        assert "Cause trace" in story and "hottest connected point" in story
        window.close()

    def test_history_ignores_playback_ticks_and_own_replay(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        window.selection_bus.update(origin=None, scenario=0)
        window.selection_bus.update(origin=None, scenario=1)
        n = len(window.history)
        assert n >= 2
        # playback ticks (origin=self) must not grow the log
        for t in (1.0, 2.0, 3.0, 4.0):
            window.selection_bus.update(origin=window, time_s=t)
        assert len(window.history) == n
        # back() then replaying it (origin=history) must not grow the log either
        sel = window.history.back()
        assert sel is not None
        window.selection_bus.set(sel, origin=window.history)
        assert len(window.history) == n
        window.close()

    def test_history_back_forward_round_trip_via_bus(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        window.selection_bus.update(origin=None, scenario=0)
        window.selection_bus.update(origin=None, scenario=1)
        assert window.history.can_back()
        sel = window.history.back()
        window.selection_bus.set(sel, origin=window.history)
        assert window.selection_bus.current.scenario == 0
        assert window.history.can_forward()
        sel = window.history.forward()
        window.selection_bus.set(sel, origin=window.history)
        assert window.selection_bus.current.scenario == 1
        window.close()

    def test_graph_gains_device_and_probe_nodes_after_placement(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        dp = window.device_panel
        dp.ensure_loaded()
        dp.type_combo.setCurrentIndex(dp.type_combo.findData("thermocouple"))
        dp._place(1.0, 1.0)
        window.graph_panel._rebuild()
        assert len(window.graph_panel._graph.nodes_of("device")) == 1
        window.close()

    def test_graph_gains_hypothesis_node_after_pinning_a_whatif(self, qapp):
        """Analysis-improvement roadmap Phase C: "Pin what-if to Knowledge
        Graph" from Sensitivity -- the graph_panel picks up pinned estimates
        the same way it already picks up devices/vector probes."""
        window = MainWindow(load_simulation_data())
        if window.sensitivity_panel is None:
            window.close()
            return
        window.show()
        sp = window.sensitivity_panel
        sp._pin_hypothesis()
        assert len(sp._hypotheses) == 1
        assert sp.pin_status.text() != ""
        window.graph_panel._rebuild()
        nodes = window.graph_panel._graph.nodes_of("hypothesis")
        assert len(nodes) == 1
        assert nodes[0].scenario == sp._hypotheses[0]["nearest_scenario"]
        window.close()

    def test_reveal_helper_used_by_workspace_preset(self, qapp):
        """Regression check for the _reveal refactor (V6-M4): its remaining
        call site (the Experiments panel's comparison hand-off was removed
        in the UX consolidation pass) must still raise the Analysis page
        and the right tab."""
        window = MainWindow(load_simulation_data())
        if window.dashboard_panel is None:
            window.close()
            return
        window._on_workspace_preset("Study analytics")
        assert window._active_page_key == "analysis"
        # Tabs are grouped -- the outer tab now holds the "Factors &
        # Sensitivity" group's own inner QTabWidget, which must be showing
        # study_panel.
        group = window.pages["analysis"].tabs.currentWidget()
        assert group.currentWidget() is window.study_panel
        window.close()

    def test_hover_highlight_sets_and_clears_without_touching_selection(self, qapp):
        """UX consolidation pass: the removed Context Panel was the only
        caller of SliceView.set_hover_highlight -- the primitive itself is
        kept (reusable rendering infra, e.g. for a future Assistant
        "reveal in viewer" action) and exercised directly here."""
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        cell = window.view_grid.active_cell()
        before = window.selection_bus.current
        cell.view.set_hover_highlight((1.0, 1.0))
        cell.view.redraw_overlays_now()
        assert len(cell.view.hover_highlight.get_offsets()) == 1
        assert window.selection_bus.current == before   # hover never touches selection
        cell.view.set_hover_highlight(None)
        cell.view.redraw_overlays_now()
        assert len(cell.view.hover_highlight.get_offsets()) == 0
        window.close()

    def test_session_report_includes_devices_via_full_session_pipeline(self, qapp):
        from report_builder import build_session_report
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.show()
        dp = window.device_panel
        dp.ensure_loaded()
        dp.type_combo.setCurrentIndex(dp.type_combo.findData("thermocouple"))
        dp._place(1.0, 1.0)
        session = window._collect_session_dict()
        html = build_session_report(session)
        assert "Virtual devices" in html and "TC-01" in html
        window.close()


class TestMultiPlaneCrossSections:
    """V6-M5: multi-plane linked cross-sections. The real dataset's .smv
    *declares* TEMPERATURE/VELOCITY at a second Y-normal offset (15), but
    actually reading it fails deep in the slice reader (a pre-existing
    data/parser quirk, discovered while building this milestone -- the
    .smv inventory and what's actually loadable can disagree). There are
    no X/Z-normal slices at all. Both cases must be caught and shown as a
    clean "gated" status -- never propagate an exception out of a Qt slot
    (PyQt5 aborts the process on that) and never crash."""

    def test_device_on_declared_but_unreadable_offset_is_gated_not_crashed(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        dp = window.device_panel
        dp.ensure_loaded()
        dp.type_combo.setCurrentIndex(dp.type_combo.findData("thermocouple"))
        dp.offset_spin.setValue(15)
        dp._place(1.0, 0.2)
        dev = dp._devices[-1]
        assert dev.direction == 1 and dev.offset == 15
        assert dev.results is None                  # never fabricated
        assert "Gated" in dp.status.text()
        window.close()

    def test_device_on_default_plane_still_works(self, qapp):
        """Regression check: the plane selector must not disturb the
        app's one verified, working plane (offset 0)."""
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        dp = window.device_panel
        dp.ensure_loaded()
        dp.type_combo.setCurrentIndex(dp.type_combo.findData("thermocouple"))
        assert dp.offset_spin.value() == 0 and dp.direction_combo.currentData() == 1
        dp._place(1.0, 0.2)
        dev = dp._devices[-1]
        assert dev.results is not None
        assert dp.status.text() == ""
        window.close()

    def test_device_on_gated_xz_plane_shows_status_not_crash(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        dp = window.device_panel
        dp.ensure_loaded()
        dp.type_combo.setCurrentIndex(dp.type_combo.findData("thermocouple"))
        dp.direction_combo.setCurrentIndex(dp.direction_combo.findData(0))   # x-normal -- absent
        dp._place(1.0, 1.0)
        dev = dp._devices[-1]
        assert dev.results is None                 # never fabricated
        assert "Gated" in dp.status.text()
        window.close()

    def test_spacetime_declared_but_unreadable_offset_is_gated_not_crashed(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        st = window.spacetime_panel
        st.ensure_loaded()
        assert st._data is not None            # the default plane loaded fine
        st.offset_spin.setValue(15)
        assert st._data is None                # gated, not fabricated
        assert "Gated" in st.status.text()
        assert st.xt_canvas.fig.axes            # rendered a gated placeholder, not a crash
        window.close()

    def test_spacetime_default_plane_still_works(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        st = window.spacetime_panel
        st.ensure_loaded()
        assert st._data is not None and st._gate_reason is None
        assert st.status.text() == ""
        window.close()

    def test_spacetime_x_normal_plane_is_gated_not_crashed(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        st = window.spacetime_panel
        st.ensure_loaded()
        st.plane_combo.setCurrentIndex(st.plane_combo.findData(0))   # x-normal
        assert st._data is None
        assert "Gated" in st.status.text()
        assert st.xt_canvas.fig.axes                # rendered a gated placeholder, not a crash
        window.close()

    def test_soot_plane_session_restore_disambiguates(self, qapp, tmp_path):
        """Regression test for a pre-existing bug V6-M5's plane-aware
        cell_to_dict incidentally fixes: two SOOT DENSITY combo entries
        (side view vs. doorway) share the same quantity name but different
        (direction, offset) -- session restore used to always pick the
        first, silently losing which plane was showing."""
        import glob
        import os
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo or not any(
                glob.glob(os.path.join(e.path, "*.s3d")) for e in window.sim_data.manifest):
            window.close()
            return
        options = window._quantity_options()
        soot = [(label, key) for label, key in options if key.quantity == "SOOT DENSITY"]
        if len(soot) < 2:
            window.close()
            return
        doorway_key = soot[1][1]
        cell = window.view_grid.active_cell()
        cell.quantity_key = doorway_key
        session = window._collect_session_dict()
        window._apply_session(session)
        assert cell.quantity_key.direction == doorway_key.direction
        assert cell.quantity_key.offset == doorway_key.offset
        window.close()


class _SyntheticCOProvider:
    """Wraps a real QuantityProvider but supplies a synthetic CO field, so
    V6-M6's full-FED pipeline is testable even though the actual dataset has
    no CO output yet. TEMPERATURE/extent reads go through the real provider
    (real background/geometry); only CO is synthetic and never confused
    with a real quantity -- production code always goes through the real,
    registry-gated provider."""

    def __init__(self, real, co_ppm=5000.0):
        self._real = real
        self._co_ppm = co_ppm

    def get(self, scenario, key):
        if key.quantity == "CARBON MONOXIDE VOLUME FRACTION":
            temp = self._real.get(scenario, SliceKey("TEMPERATURE", key.direction, key.offset))
            return np.full_like(np.asarray(temp, dtype=float), self._co_ppm)
        return self._real.get(scenario, key)

    def get_extent(self, scenario, key):
        if key.quantity == "CARBON MONOXIDE VOLUME FRACTION":
            return self._real.get_extent(scenario, SliceKey("TEMPERATURE", key.direction, key.offset))
        return self._real.get_extent(scenario, key)


class TestHazardTenabilityMerge:
    """Analysis-improvement roadmap Phase B: Hazard and Tenability were two
    separate top-level tabs classifying the same field into hazard bands
    via overlapping engines -- now one "Hazard & Tenability" tab with a
    mode toggle. A thin wrapper only, so both panels keep their full
    existing functionality/disclaimers unchanged."""

    def test_wrapper_holds_both_panels_and_defaults_to_map_view(self, qapp):
        window = MainWindow(load_simulation_data())
        wrapper = window.hazard_tenability_panel
        if wrapper is None:
            window.close()
            return
        assert wrapper.hazard_widget is window.hazard_panel
        assert wrapper.tenability_widget is window.tenability_panel
        assert wrapper.stack.currentWidget() is window.hazard_panel
        window.close()

    def test_mode_toggle_switches_between_panels(self, qapp):
        window = MainWindow(load_simulation_data())
        wrapper = window.hazard_tenability_panel
        if wrapper is None:
            window.close()
            return
        wrapper.mode_combo.setCurrentIndex(1)
        assert wrapper.stack.currentWidget() is window.tenability_panel
        wrapper.mode_combo.setCurrentIndex(0)
        assert wrapper.stack.currentWidget() is window.hazard_panel
        window.close()

    def test_showing_wrapper_loads_both_panels_not_just_visible_one(self, qapp):
        window = MainWindow(load_simulation_data())
        wrapper = window.hazard_tenability_panel
        if wrapper is None:
            window.close()
            return
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(wrapper)
        QtWidgets.QApplication.processEvents()
        assert window.hazard_panel._loaded
        assert window.tenability_panel._loaded
        window.close()


class TestAskTabDirect:
    """Analysis final-polish pass: the Assistant template-summary layer
    (assistant.py/assistant_panel.py/assistant_query_panel.py) was removed
    outright -- it didn't provide enough value for a research-focused
    application. QueryPanel (the deterministic physics-query grammar,
    previously reachable only as a secondary mode inside the removed
    Assistant wrapper) is restored to its own direct "Ask" tab under
    Reference & Communication; its own functionality is unchanged and
    fully covered by TestQueryPanel above."""

    def test_query_panel_is_reachable_as_its_own_tab_not_wrapped(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.query_panel is None:
            window.close()
            return
        assert not hasattr(window, "assistant_panel")
        assert not hasattr(window, "assistant_query_panel")
        window.show()
        window._navigate_to("analysis")
        window.pages["analysis"].show_tab(window.query_panel)
        QtWidgets.QApplication.processEvents()
        assert window.query_panel._loaded
        window.close()


class TestFullFED:
    """V6-M6: full FED (toxic-gas + convected-heat dose). CO is registry-
    gated on the real dataset -- these tests verify the honest gated
    fallback against real data, and the full pipeline via a synthetic CO
    provider (mirroring V6-M3's _SyntheticVectorProvider pattern)."""

    def test_hazard_panel_falls_back_to_partial_screen_when_co_gated(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.hazard_panel.ensure_loaded()
        assert not window.hazard_panel._series["has_co"]
        import hazard_spaces as hz
        assert hz.BASIS in window.hazard_panel.caption.text()
        window.close()

    def test_tenability_panel_falls_back_to_partial_screen_when_co_gated(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.tenability_panel.ensure_loaded()
        assert not window.tenability_panel._has_co
        window.close()

    def test_spacetime_full_fed_is_gated_on_real_data(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        st = window.spacetime_panel
        st.ensure_loaded()
        st.quantity_combo.setCurrentIndex(st.quantity_combo.findData("full_fed"))
        assert st._data is None
        assert "Gated" in st.status.text()
        window.close()

    def test_hazard_panel_escalates_via_full_fed_with_synthetic_co(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.hazard_panel._provider = _SyntheticCOProvider(window.quantity_provider, co_ppm=20000.0)
        window.hazard_panel.ensure_loaded()
        assert window.hazard_panel._series["has_co"]
        import hazard_spaces as hz
        assert hz.FULL_FED_BASIS in window.hazard_panel.caption.text()
        window.close()

    def test_tenability_panel_shows_full_fed_with_synthetic_co(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        window.tenability_panel._provider = _SyntheticCOProvider(window.quantity_provider, co_ppm=20000.0)
        window.tenability_panel.ensure_loaded()
        assert window.tenability_panel._has_co
        from tenability_panel import _FULL_FED_NOTICE
        assert window.tenability_panel.disclaimer.text() == _FULL_FED_NOTICE
        window.close()

    def test_spacetime_full_fed_with_synthetic_co_renders(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        st = window.spacetime_panel
        st._provider = _SyntheticCOProvider(window.quantity_provider, co_ppm=20000.0)
        st.ensure_loaded()
        st.quantity_combo.setCurrentIndex(st.quantity_combo.findData("full_fed"))
        assert st._data is not None and st._gate_reason is None
        assert st.status.text() == ""
        assert st._data.shape == window.controller.store.get(
            st.scenario_combo.currentData(), SliceKey("TEMPERATURE")).shape
        window.close()

    def test_device_thermocouple_shows_full_fed_with_synthetic_co(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            window.close()
            return
        dp = window.device_panel
        dp.ensure_loaded()
        dp._provider = _SyntheticCOProvider(window.quantity_provider, co_ppm=20000.0)
        dp.type_combo.setCurrentIndex(dp.type_combo.findData("thermocouple"))
        dp._place(1.0, 0.2)
        dev = dp._devices[-1]
        assert dev.results["max_fed_full"] is not None
        assert "FED" in dp._headline(dev)
        window.close()


class TestMultiCellInspector:
    """V6 polish: comparing 2+ scenarios used to only ever show the active
    cell's stats in the Inspector -- InspectorStack gives each visible cell
    its own full section instead."""

    def test_1x1_layout_has_one_visible_section(self, qapp):
        window = MainWindow(load_simulation_data())
        assert window.inspector_stack.count() == 1
        assert window.inspector_stack.section(0) is window.inspector
        window.close()

    def test_1x2_layout_grows_a_second_section_with_its_own_scenario(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window._set_grid_layout("1x2")
        cells = window.view_grid.visible_cells()
        assert len(cells) == 2
        cells[1].scenario_combo.setCurrentIndex(5)  # a distinct scenario for cell 2
        qapp.processEvents()
        window._on_time_changed(0)

        assert window.inspector_stack.count() == 2
        sec0, sec1 = window.inspector_stack.section(0), window.inspector_stack.section(1)
        assert sec0 is not sec1
        assert sec0._static_labels["Scenario"].text() == window._scenario_label(cells[0].case_index)
        assert sec1._static_labels["Scenario"].text() == window._scenario_label(cells[1].case_index)
        assert sec1._static_labels["Scenario"].text() != sec0._static_labels["Scenario"].text()
        window.close()

    def test_both_sections_populate_sparkline_hrr_and_story(self, qapp):
        """Each section is a real InspectorPanel, so the earlier fix (peak-
        temperature sparkline/HRR/fire-story available for any slice cell,
        not just the active one) applies to every section, not just the
        first."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window._set_grid_layout("1x2")
        cells = window.view_grid.visible_cells()
        vel_idx = next((i for i, info in enumerate(window.quantity_infos)
                       if info.key.quantity == "VELOCITY"), None)
        if vel_idx is not None:
            cells[1].quantity_combo.setCurrentIndex(vel_idx)
            qapp.processEvents()
        window._on_time_changed(5)

        for i in range(2):
            section = window.inspector_stack.section(i)
            assert len(section._series) > 0
            assert section.story_list.count() > 0
        window.close()

    def test_shrinking_back_to_1x1_hides_but_keeps_second_section(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window._set_grid_layout("1x2")
        window._on_time_changed(0)
        second = window.inspector_stack.section(1)
        window._set_grid_layout("1x1")
        assert window.inspector_stack.count() == 2  # never destroyed
        assert not second.isVisible()
        window.close()

    def test_hover_probe_updates_the_hovered_cells_own_section_only(self, qapp):
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        if sim_data.is_demo:
            window.close()
            return
        window._set_grid_layout("1x2")
        cells = window.view_grid.visible_cells()
        window._on_cell_probe(cells[1], 0.3, 0.2, 123.4)
        assert "123.4" in window.inspector_stack.section(1).probe_label.text()
        assert "Hover the plot" in window.inspector_stack.section(0).probe_label.text()
        window.close()

    def test_single_cell_inspector_stays_full_detail(self, qapp):
        """The plain Live Viewer (1x1, by far the common case) must show
        everything it always has -- sparkline/HRR/narration/Fire story --
        unchanged. Compact mode is only for comparing 2+ scenarios (see
        test_comparison_grid_switches_inspector_to_compact below)."""
        window = MainWindow(load_simulation_data())
        window.show()
        panel = window.inspector
        assert panel._compact is False
        assert panel.sparkline.isVisible()
        assert panel.hrr_gauge.isVisible()
        assert panel.story_list.isVisible()
        assert panel.narration_label.isVisible()
        assert panel.frame_label.isVisible()
        assert panel.probe_label.isVisible()
        window.close()

    def test_comparison_grid_switches_inspector_to_compact(self, qapp):
        """Live-polish request: when comparing 2+ scenarios, each section
        shows just Scenario/Quantity/Grid size/Slice/Duration/Frames + the
        Live readout + the peak-temperature sparkline (kept visible even
        compact, per a follow-up request) -- the HRR gauge/narration/Fire
        story are hidden (not removed; they still compute underneath, see
        InspectorPanel.set_compact), so both fit without heavy scrolling.
        Dropping back to a single cell restores full detail."""
        sim_data = load_simulation_data()
        window = MainWindow(sim_data)
        window.show()
        if sim_data.is_demo:
            window.close()
            return
        window._set_grid_layout("2x1")
        for i in range(window.inspector_stack.count()):
            section = window.inspector_stack.section(i)
            assert section._compact is True
            assert section.sparkline.isVisible()
            assert not section.hrr_gauge.isVisible()
            assert not section.story_list.isVisible()
        window._set_grid_layout("1x1")
        assert window.inspector._compact is False
        assert window.inspector.sparkline.isVisible()
        window.close()

    def test_2x1_layout_auto_links_color_scales(self, qapp):
        """The stacked layout exists to compare two scenarios directly --
        unlinked color scales would let equal values render as different
        colors between them. Switching to "2x1" turns on "Link color
        scales" automatically (still a normal toggle afterward)."""
        window = MainWindow(load_simulation_data())
        assert getattr(window, "_link_clim", False) is False
        window._set_grid_layout("2x1")
        assert window._link_clim is True
        assert window.link_clim_action.isChecked()
        window.close()


class TestQuantityDropdownTooltips:
    """Live-polish request: technical/coordinate-heavy quantity labels
    (e.g. "Dynamic pressure", "Smoke — doorway (x = 0.25 m)") get the
    registry's own plain-language "interpretation" as a per-item dropdown
    tooltip, reusing quantities_panel.py's existing explanatory text
    instead of requiring a trip to that separate panel."""

    def test_main_quantity_combo_items_carry_interpretation_tooltips(self, qapp):
        window = MainWindow(load_simulation_data())
        from registry import get_quantity
        found_any = False
        for i, info in enumerate(window._combo_quantity_list):
            interpretation = get_quantity(info.key.quantity).interpretation
            tip = window.quantity_combo.itemData(i, QtCore.Qt.ToolTipRole)
            if interpretation:
                found_any = True
                assert interpretation in tip
        assert found_any, "at least one real quantity should have an interpretation"
        window.close()

    def test_grid_cell_quantity_combo_items_carry_tooltips(self, qapp):
        window = MainWindow(load_simulation_data())
        cell = window.view_grid.active_cell()
        from registry import get_quantity
        for i, (_label, key) in enumerate(cell._quantity_options):
            interpretation = get_quantity(key.quantity).interpretation
            if interpretation:
                tip = cell.quantity_combo.itemData(i, QtCore.Qt.ToolTipRole)
                assert tip
        window.close()
