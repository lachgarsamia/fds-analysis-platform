"""Unit + integration tests for the FireLab nav-rail shell (roadmap
Phase 1): NavRail, Page lifecycle, and MainWindow's page-switching
behavior. Kept small -- the ~30 pre-existing integration tests already
exercise Live Viewer content; these focus on what's new: page identity,
lazy placeholder builds, and playback pausing on navigation away."""

import pytest
from PyQt5 import QtWidgets

from data_provider import load_simulation_data
from main_window import MainWindow
from nav import NavRail
from pages.base import Page
from pages.placeholder import PlaceholderPage


class TestNavRail:
    def test_first_entry_active_by_default(self, qapp):
        rail = NavRail([("home", "Home"), ("live", "Live Viewer")])
        assert rail._buttons["home"].isChecked()

    def test_click_emits_page_selected(self, qapp):
        rail = NavRail([("home", "Home"), ("live", "Live Viewer")])
        received = []
        rail.page_selected.connect(received.append)
        rail._buttons["live"].click()
        assert received == ["live"]

    def test_set_active_checks_the_right_button(self, qapp):
        rail = NavRail([("home", "Home"), ("live", "Live Viewer")])
        rail.set_active("live")
        assert rail._buttons["live"].isChecked()
        assert not rail._buttons["home"].isChecked()

    def test_collapse_shrinks_width_and_relabels(self, qapp):
        rail = NavRail([("home", "Home")])
        expanded_width = rail.width()
        rail.set_collapsed(True)
        assert rail.width() < expanded_width
        assert rail._buttons["home"].text() == "1"


class TestPageLifecycle:
    def test_base_page_hooks_are_no_ops(self, qapp):
        page = Page()
        page.on_enter()  # must not raise
        page.on_leave()

    def test_placeholder_builds_once(self, qapp):
        page = PlaceholderPage()
        assert page.layout() is None
        page.on_enter()
        assert page.layout() is not None
        child_count = page.layout().count()
        page.on_enter()  # a second on_enter must not rebuild/duplicate content
        assert page.layout().count() == child_count


class TestMainWindowPageSwitching:
    def test_boots_into_live(self, qapp):
        """UI/UX modernization pass: Simulation Viewer (LivePage) is now
        the default/main page, not Home."""
        window = MainWindow(load_simulation_data())
        assert window._active_page_key == "live"
        assert window.page_stack.currentWidget() is window.pages["live"]
        window.close()

    def test_live_content_built_eagerly_regardless_of_active_page(self, qapp):
        """The Live page's real content (view_grid, timeline, ...) must
        exist as soon as MainWindow is constructed, not only once the user
        navigates there -- every pre-existing test/call site assumes this."""
        window = MainWindow(load_simulation_data())
        assert window._active_page_key == "live"
        assert window.view_grid is not None
        assert window.timeline is not None
        window.close()

    def test_navigate_to_live_and_back_preserves_time_index(self, qapp):
        window = MainWindow(load_simulation_data())
        window.time_controller.seek(3)
        window._navigate_to("live")
        window._navigate_to("dataset")
        window._navigate_to("live")
        assert window.time_controller.index == 3
        window.close()

    def test_leaving_live_while_playing_pauses(self, qapp):
        window = MainWindow(load_simulation_data())
        window._navigate_to("live")
        window.time_controller.play()
        assert window.time_controller.is_playing()
        window._navigate_to("home")
        assert not window.time_controller.is_playing()
        window.close()

    def test_demo_mode_pages_navigate_without_crash(self, qapp, monkeypatch):
        """FireLab roadmap Phase 4: Dataset/Analysis/Compare must degrade
        gracefully (no manifest, nothing to browse/analyze/compare)
        instead of crashing -- regression for a real bug found where the
        Analysis page's on_enter() callback assumed attributes that only
        exist when a manifest is present."""
        monkeypatch.setattr("data_provider.list_scenario_folders", lambda *a, **kw: [])
        sim_data = load_simulation_data()
        assert sim_data.is_demo
        window = MainWindow(sim_data)
        for key in ("home", "compare", "dataset", "analysis", "export", "live"):
            window._navigate_to(key)
        window.close()

    def test_compare_preset_configures_a_1x2_difference_grid(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._apply_compare_preset("door")
        assert window._active_page_key == "live"
        cells = window.view_grid.visible_cells()
        assert len(cells) == 2
        assert cells[0].cell_type == "slice"
        assert cells[0].quantity_key.quantity == "VELOCITY"
        assert cells[1].cell_type == "difference"
        assert cells[1].quantity_key.quantity == "VELOCITY"
        manifest = {e.case_index: e for e in window.sim_data.manifest}
        a = manifest[cells[1].case_index_a]
        b = manifest[cells[1].case_index_b]
        assert a.door != b.door
        assert (a.candles, a.vod, a.voc) == (b.candles, b.vod, b.voc)
        window.close()
