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
from pages.analysis import AnalysisPage
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


class TestAnalysisPageGrouping:
    """Analysis-improvement roadmap Phase D: tabs re-grouped by
    investigation stage, with Experimental collapsed by default."""

    def test_panels_are_grouped_and_experimental_starts_collapsed(self, qapp):
        page = AnalysisPage(
            context_content=QtWidgets.QLabel("Context"),
            study_content=QtWidgets.QLabel("Study"),
            advanced_compare_content=QtWidgets.QLabel("Compare axes"),
            graph_content=QtWidgets.QLabel("Graph"),
            fire_mri_content=QtWidgets.QLabel("Fire MRI"),
            attention_content=QtWidgets.QLabel("Attention"))
        assert page.tabs.count() == 5
        group_names = [page.tabs.tabText(i) for i in range(page.tabs.count())]
        assert group_names == ["Core Investigation", "Study-Level", "Comparison",
                               "Interpretation & Communication", "Experimental"]
        experimental = page.tabs.widget(group_names.index("Experimental"))
        assert experimental.tabs.count() == 2   # Fire MRI, Attention
        assert experimental.tabs.isHidden()     # collapsed by default
        experimental.toggle.setChecked(True)
        assert not experimental.tabs.isHidden()

    def test_only_supplied_panels_form_a_group(self, qapp):
        """A group with nothing supplied gets no tab at all (same "only
        supplied surfaces get a tab" rule the flat layout already had)."""
        page = AnalysisPage(study_content=QtWidgets.QLabel("Study"),
                            sensitivity_content=QtWidgets.QLabel("Sensitivity"))
        assert page.tabs.count() == 1
        assert page.tabs.tabText(0) == "Study-Level"

    def test_show_tab_reveals_nested_panel_and_expands_experimental(self, qapp):
        fire_mri = QtWidgets.QLabel("Fire MRI")
        page = AnalysisPage(fire_mri_content=fire_mri,
                            attention_content=QtWidgets.QLabel("Attention"),
                            study_content=QtWidgets.QLabel("Study"))
        page.show_tab(fire_mri)
        experimental = page.tabs.currentWidget()
        assert page.tabs.tabText(page.tabs.currentIndex()) == "Experimental"
        assert experimental.tabs.currentWidget() is fire_mri
        assert not experimental.tabs.isHidden()   # auto-expanded on reveal

    def test_tab_shown_fires_on_outer_and_inner_switch(self, qapp):
        calls = []
        page = AnalysisPage(
            study_content=QtWidgets.QLabel("Study"),
            sensitivity_content=QtWidgets.QLabel("Sensitivity"),
            graph_content=QtWidgets.QLabel("Graph"))
        page.tab_shown.connect(lambda: calls.append(1))
        study_group = page.tabs.widget(0)   # Study-Level: Study, Sensitivity
        study_group.setCurrentIndex(1)      # inner switch, no outer change
        assert len(calls) == 1
        page.tabs.setCurrentIndex(1)        # outer switch to Interpretation & Communication
        assert len(calls) == 2


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

    def test_compare_preset_configures_a_stacked_linked_grid(self, qapp):
        """Compare presets show scenario A above scenario B as two plain
        slices (not a computed A-B difference) of the same quantity,
        stacked (2x1) with color scales linked -- a direct visual
        comparison, not a delta."""
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._apply_compare_preset("door")
        assert window._active_page_key == "live"
        assert window.view_grid.layout_name == "2x1"
        assert window._link_clim is True
        cells = window.view_grid.visible_cells()
        assert len(cells) == 2
        assert cells[0].cell_type == "slice"
        assert cells[1].cell_type == "slice"
        assert cells[0].quantity_key.quantity == "VELOCITY"
        assert cells[1].quantity_key.quantity == "VELOCITY"
        manifest = {e.case_index: e for e in window.sim_data.manifest}
        a = manifest[cells[0].case_index]
        b = manifest[cells[1].case_index]
        assert a.door != b.door
        assert (a.candles, a.vod, a.voc) == (b.candles, b.vod, b.voc)
        window.close()

    def test_reclicking_live_viewer_resets_a_leftover_compare_grid(self, qapp):
        """Compare and the plain Live Viewer must be independent: a
        preset's comparison grid must not stick around as "the" Live
        Viewer -- re-clicking "Live Viewer" (even though it's already the
        active page) is the user's way of asking for their own plain view
        back."""
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._apply_compare_preset("door")
        assert len(window.view_grid.visible_cells()) == 2
        assert window._compare_active is True

        window._navigate_to("live")  # re-clicking the already-active nav entry

        assert window._compare_active is False
        assert window.view_grid.layout_name == "1x1"
        assert len(window.view_grid.visible_cells()) == 1
        window.close()

    def test_navigating_away_and_back_to_live_resets_a_leftover_compare_grid(self, qapp):
        window = MainWindow(load_simulation_data())
        if window.sim_data.is_demo:
            pytest.skip("real dataset not present")
        window._apply_compare_preset("door")
        window._navigate_to("analysis")
        window._navigate_to("live")
        assert window._compare_active is False
        assert window.view_grid.layout_name == "1x1"
        window.close()
