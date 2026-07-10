"""Unit tests for views.py: SliceView (M2.2.1) and GridCell/ViewGrid
(M2.2.2/2.2.3). Pure widget-layer tests -- no ScenarioStore/controller
involved, cells are handed synthetic frames directly."""

import numpy as np
import pytest

from views import SliceView, GridCell, ViewGrid
from slice_key import SliceKey


FRAME = np.ones((49, 101), dtype=np.float32) * 42.0


class TestSliceView:
    def test_init_plot_then_show_frame(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=20.0, vmax=300.0, colorbar_label="Temperature (°C)")
        assert view.heatmap.get_array().shape == (49, 101)
        assert view.heatmap.get_clim() == (20.0, 300.0)
        assert view.colorbar.ax.get_ylabel() == "Temperature (°C)"

        new_frame = np.zeros((49, 101), dtype=np.float32)
        view.show_frame(new_frame)
        assert (view.heatmap.get_array() == 0).all()

    def test_set_cmap_and_clim(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=10.0, colorbar_label="x")
        view.set_cmap("viridis")
        assert view.heatmap.get_cmap().name == "viridis"
        view.set_clim(1.0, 99.0)
        assert view.heatmap.get_clim() == (1.0, 99.0)

    def test_set_title(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=10.0, colorbar_label="x")
        view.set_title("c1_d1 · TEMP")
        assert view.ax.get_title() == "c1_d1 · TEMP"


class TestGridCell:
    SCENARIOS = [("c1_d0_vod0_voc0", 0), ("c1_d0_vod0_voc1", 1), ("c2_d1_vod2_voc1", 23)]
    QUANTITIES = [("Temperature", SliceKey("TEMPERATURE", 1, 0)), ("Air speed", SliceKey("VELOCITY", 1, 0))]

    def test_defaults_to_first_option(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        assert cell.case_index == 0
        assert cell.quantity_key == SliceKey("TEMPERATURE", 1, 0)

    def test_empty_options_disable_combos(self, qapp):
        cell = GridCell([], [])
        assert not cell.scenario_combo.isEnabled()
        assert not cell.quantity_combo.isEnabled()
        assert cell.case_index == 0
        assert cell.quantity_key is None

    def test_scenario_combo_change_emits_signal(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        received = []
        cell.scenario_selected.connect(lambda c, case_index: received.append((c, case_index)))
        cell.scenario_combo.setCurrentIndex(2)
        assert received == [(cell, 23)]
        assert cell.case_index == 23

    def test_quantity_combo_change_emits_signal(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        received = []
        cell.quantity_selected.connect(lambda c, key: received.append((c, key)))
        cell.quantity_combo.setCurrentIndex(1)
        assert received == [(cell, SliceKey("VELOCITY", 1, 0))]
        assert cell.quantity_key == SliceKey("VELOCITY", 1, 0)

    def test_set_scenario_silently_does_not_emit(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        received = []
        cell.scenario_selected.connect(lambda c, case_index: received.append(case_index))
        cell.set_scenario_silently(23)
        assert received == []
        assert cell.case_index == 23
        assert cell.scenario_combo.currentIndex() == 2

    def test_set_quantity_silently_does_not_emit(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        received = []
        cell.quantity_selected.connect(lambda c, key: received.append(key))
        cell.set_quantity_silently(SliceKey("VELOCITY", 1, 0))
        assert received == []
        assert cell.quantity_key == SliceKey("VELOCITY", 1, 0)

    def test_click_emits_activated(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        received = []
        cell.activated.connect(received.append)
        cell.activated.emit(cell)  # mousePressEvent needs a real QMouseEvent; test the contract directly
        assert received == [cell]

    def test_set_active_changes_border_style(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell.set_active(False)
        inactive_style = cell.styleSheet()
        cell.set_active(True)
        active_style = cell.styleSheet()
        assert active_style != inactive_style


class TestViewGrid:
    SCENARIOS = [(f"c{i}", i) for i in range(24)]
    QUANTITIES = [("Temperature", SliceKey("TEMPERATURE", 1, 0)), ("Air speed", SliceKey("VELOCITY", 1, 0))]

    def test_starts_1x1_with_one_active_cell(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        assert grid.layout_name == "1x1"
        assert len(grid.visible_cells()) == 1
        assert grid.active_cell() is grid.visible_cells()[0]

    def test_growing_grid_creates_new_cells_and_fires_cell_created(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        created = []
        grid.cell_created.connect(created.append)
        grid.set_layout("2x2")
        assert grid.layout_name == "2x2"
        assert len(grid.visible_cells()) == 4
        assert len(created) == 3, "3 new cells beyond the original 1x1 cell"

    def test_shrinking_grid_preserves_cell_state(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        grid.set_layout("2x2")
        cells = grid.visible_cells()
        cells[3].set_scenario_silently(15)
        grid.set_layout("1x1")
        assert len(grid.visible_cells()) == 1
        grid.set_layout("2x2")
        assert grid.visible_cells()[3].case_index == 15, "cell state must survive a shrink/regrow cycle"

    def test_growing_does_not_recreate_existing_cells(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        first_cell = grid.visible_cells()[0]
        grid.set_layout("2x2")
        assert grid.visible_cells()[0] is first_cell

    def test_activating_a_cell_updates_active_cell_and_emits(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        grid.set_layout("2x2")
        cells = grid.visible_cells()
        received = []
        grid.active_cell_changed.connect(received.append)
        cells[2].activated.emit(cells[2])
        assert grid.active_cell() is cells[2]
        assert received == [cells[2]]

    def test_activating_already_active_cell_is_a_no_op(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        received = []
        grid.active_cell_changed.connect(received.append)
        active = grid.active_cell()
        active.activated.emit(active)
        assert received == []

    def test_shrinking_below_active_index_resets_active_to_zero(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        grid.set_layout("2x2")
        cells = grid.visible_cells()
        cells[3].activated.emit(cells[3])
        assert grid.active_cell() is cells[3]
        grid.set_layout("1x1")
        assert grid.active_cell() is cells[0]

    def test_cell_scenario_selected_relays_with_originating_cell(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        grid.set_layout("1x2")
        cells = grid.visible_cells()
        received = []
        grid.cell_scenario_selected.connect(lambda cell, case_index: received.append((cell, case_index)))
        cells[1].scenario_combo.setCurrentIndex(5)
        assert received == [(cells[1], 5)]

    def test_apply_accent_reaches_every_cell(self, qapp):
        grid = ViewGrid(self.SCENARIOS, self.QUANTITIES)
        grid.set_layout("2x2")
        grid.apply_accent("#FF0000")
        for cell in grid.visible_cells():
            assert cell._accent == "#FF0000"

    def test_empty_scenario_options_still_produces_a_usable_grid(self, qapp):
        """Demo mode: no manifest, so scenario_options is []."""
        grid = ViewGrid([], self.QUANTITIES)
        assert grid.active_cell().case_index == 0
        assert not grid.active_cell().scenario_combo.isEnabled()
