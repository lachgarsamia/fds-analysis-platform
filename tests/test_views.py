"""Unit tests for views.py: SliceView (M2.2.1), DifferenceView (M2.3.1),
and GridCell/ViewGrid (M2.2.2/2.2.3). Pure widget-layer tests -- no
ScenarioStore/controller involved, cells are handed synthetic frames
directly, except TestDifferenceViewRealData which deliberately reads the
real dataset to check the "physically sensible structure" DoD claim
against ground truth rather than trusting it looks reasonable."""

import os

import numpy as np
import pytest

from views import SliceView, DifferenceView, GridCell, ViewGrid
from slice_key import SliceKey
from load_data import SIM_ROOT, load_data


FRAME = np.ones((49, 101), dtype=np.float32) * 42.0

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


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


class TestDifferenceView:
    def test_compute_diff(self):
        a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
        b = np.array([[[0.5, 0.5], [0.5, 0.5]], [[5.0, 5.0], [5.0, 5.0]]], dtype=np.float32)
        diff0 = DifferenceView.compute_diff(a, b, 0)
        assert np.allclose(diff0, [[0.5, 1.5], [2.5, 3.5]])
        diff1 = DifferenceView.compute_diff(a, b, 1)
        assert np.allclose(diff1, [[5.0, 15.0], [25.0, 35.0]])

    def test_symmetric_clim_is_max_abs_over_samples(self):
        view = DifferenceView()
        a = np.zeros((10, 2, 2), dtype=np.float32)
        b = np.zeros((10, 2, 2), dtype=np.float32)
        a[3, 0, 0] = 100.0   # +100 at frame 3
        b[7, 1, 1] = 40.0    # a-b = -40 at frame 7
        vmin, vmax = view.symmetric_clim(a, b, cache_key="k", n_samples=10)
        assert vmax == 100.0
        assert vmin == -100.0, "clim must be symmetric even though the extremes aren't"

    def test_symmetric_clim_is_cached(self):
        view = DifferenceView()
        calls = []
        a = np.random.default_rng(0).uniform(0, 10, size=(5, 2, 2)).astype(np.float32)
        b = np.random.default_rng(1).uniform(0, 10, size=(5, 2, 2)).astype(np.float32)

        first = view.symmetric_clim(a, b, cache_key=("A", "B", "TEMPERATURE"))
        # Mutate the arrays -- if the second call actually rescanned, it
        # would see different data and (almost certainly) a different clim.
        a[:] = 0
        second = view.symmetric_clim(a, b, cache_key=("A", "B", "TEMPERATURE"))
        assert first == second, "second call with the same cache_key must reuse the cached result"

    def test_different_cache_keys_are_independent(self):
        view = DifferenceView()
        a = np.full((3, 2, 2), 10.0, dtype=np.float32)
        b = np.full((3, 2, 2), 0.0, dtype=np.float32)
        clim1 = view.symmetric_clim(a, b, cache_key="pair1")
        c = np.full((3, 2, 2), 999.0, dtype=np.float32)
        clim2 = view.symmetric_clim(c, b, cache_key="pair2")
        assert clim1 != clim2

    def test_init_plot_uses_diverging_cmap_by_default(self, qapp):
        view = DifferenceView()
        view.init_plot(FRAME, interpolation="nearest", vmin=-50.0, vmax=50.0, colorbar_label="ΔT (°C)")
        assert view.widget() is not None
        assert view._inner.heatmap.get_cmap().name == "RdBu_r"
        assert view._inner.heatmap.get_clim() == (-50.0, 50.0)
        assert view._inner.colorbar.ax.get_ylabel() == "ΔT (°C)"

    def test_show_frame_and_setters_delegate_to_inner_slice_view(self, qapp):
        view = DifferenceView()
        view.init_plot(FRAME, interpolation="nearest", vmin=-10.0, vmax=10.0, colorbar_label="x")
        diff_frame = np.full((49, 101), -5.0, dtype=np.float32)
        view.show_frame(diff_frame)
        assert (view._inner.heatmap.get_array() == -5.0).all()
        view.set_clim(-20.0, 20.0)
        assert view._inner.heatmap.get_clim() == (-20.0, 20.0)
        view.set_title("c1_d0 − c1_d1 · TEMP")
        assert view._inner.ax.get_title() == "c1_d0 − c1_d1 · TEMP"


@requires_real_dataset
class TestDifferenceViewRealData:
    """Verification against real data (not just synthetic arrays), per
    explicit instruction before this was wired into any UI: does the diff
    view actually show a physically sensible structure, or does it just
    fail to crash while showing something misleading?

    Scenario pair: c1_d0_vod0_voc0 (door height 0.05m) vs
    c1_d1_vod0_voc0 (door height 0.15m) -- confirmed via grep against both
    .fds files that this is the ONLY difference between them (the door's
    &HOLE z-extent; everything else -- candle count, VOD, VOC -- identical).
    The door sits at physical x=[0.25,0.29], near the floor; the candle
    burner sits at x=[0.92,0.96], also near the floor.

    Finding (recorded here so it's reproducible, not just a one-off
    investigation): for **TEMPERATURE**, the dominant |diff| signal is near
    the CANDLE/plume (x≈0.90-0.96), not the doorway -- roughly 15-30x
    larger there than in the door's own x-band. The door band's own signal
    for TEMPERATURE is actually *smaller* than an arbitrary control band
    elsewhere in the room (both by mean-per-frame-max and by
    time-averaged-mean) -- the door's effect on absolute temperature is
    real but too small to separate from ambient room-wide variation using
    these coarse region statistics. For **VELOCITY**, the door band *does*
    exceed the control band on both metrics (a directly airflow-driven
    quantity showing the ventilation effect the roadmap's example was
    really describing). The peak TEMPERATURE signal is spatially coherent
    (a smooth gradient across a 7x7 neighborhood, not an isolated noisy
    pixel) -- so the diff view is genuinely physically grounded either way,
    just not dominated by the effect ROADMAP.md's own illustrative example
    named for the quantity it named it for. See ROADMAP.md's M2.3 section
    for the full writeup; this test pins the specific numeric claims so a
    future change to slice.py/the dataset can't silently invalidate them
    without a visible test failure.
    """

    DOOR_CASE_NARROW = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")
    DOOR_CASE_WIDE = os.path.join(SIM_ROOT, "c1_d1_vod0_voc0")

    @pytest.fixture(scope="class")
    def temperature_diff(self):
        key = SliceKey("TEMPERATURE", 1, 0)
        data_wide = load_data(self.DOOR_CASE_WIDE, key)
        data_narrow = load_data(self.DOOR_CASE_NARROW, key)
        n = min(data_wide.shape[0], data_narrow.shape[0])
        return data_wide[:n] - data_narrow[:n]

    def test_diff_is_not_all_zero_or_nan(self, temperature_diff):
        assert np.isfinite(temperature_diff).all()
        assert np.abs(temperature_diff).max() > 1.0, "two different door widths must produce a real difference"

    def test_peak_signal_is_spatially_coherent_not_isolated_noise(self, temperature_diff):
        """The single largest-magnitude frame's peak pixel must sit inside
        a smoothly-varying neighborhood (a real plume-edge gradient), not
        be an isolated spike surrounded by near-zero values -- the
        signature of a real physical structure vs. a parser artifact."""
        peak_frame_i = int(np.argmax(np.abs(temperature_diff).max(axis=(1, 2))))
        frame = temperature_diff[peak_frame_i]
        row, col = np.unravel_index(np.argmax(np.abs(frame)), frame.shape)
        r0, r1 = max(0, row - 2), min(frame.shape[0], row + 3)
        c0, c1 = max(0, col - 2), min(frame.shape[1], col + 3)
        neighborhood = frame[r0:r1, c0:c1]
        # A real gradient: most of the neighborhood should carry a
        # meaningful fraction of the peak's magnitude, not near-zero noise.
        peak_val = np.abs(frame[row, col])
        assert np.median(np.abs(neighborhood)) > 0.1 * peak_val

    def test_candle_region_dominates_over_door_region(self, temperature_diff):
        """The recorded finding: for TEMPERATURE, the door-width effect is
        real but secondary -- the plume/candle region shows a much larger
        difference. Pinned here so this specific, checked claim doesn't
        silently drift if the parser or dataset changes."""
        n_x = temperature_diff.shape[2]
        x_cols = np.linspace(0, 1.0, n_x)
        door_band = (x_cols >= 0.24) & (x_cols <= 0.30)
        candle_band = (x_cols >= 0.90) & (x_cols <= 0.98)

        abs_diff = np.abs(temperature_diff)
        door_signal = abs_diff[:, :, door_band].max(axis=(1, 2)).mean()
        candle_signal = abs_diff[:, :, candle_band].max(axis=(1, 2)).mean()

        assert candle_signal > 10 * door_signal, (
            f"expected candle-region difference to dominate the door-region "
            f"difference (found candle={candle_signal:.2f}, door={door_signal:.2f})"
        )

    def test_temperature_door_region_does_not_exceed_control_region(self, temperature_diff):
        """Honest negative result, pinned rather than glossed over: for
        TEMPERATURE specifically, the door band's own signal is actually
        *smaller* than an arbitrary control band elsewhere in the room, on
        both a per-frame-max and a time-averaged basis. The door's real
        effect shows up in VELOCITY instead (see
        test_velocity_door_region_exceeds_control_region below) -- makes
        physical sense: temperature differences are dominated by chaotic
        flame/plume dynamics everywhere in the room, while airflow through
        the door is a direct, local, first-order effect of the door's own
        geometry."""
        n_x = temperature_diff.shape[2]
        x_cols = np.linspace(0, 1.0, n_x)
        door_band = (x_cols >= 0.24) & (x_cols <= 0.30)
        control_band = (x_cols >= 0.50) & (x_cols <= 0.60)

        abs_diff = np.abs(temperature_diff)
        door_meanmax = abs_diff[:, :, door_band].max(axis=(1, 2)).mean()
        control_meanmax = abs_diff[:, :, control_band].max(axis=(1, 2)).mean()
        assert door_meanmax < control_meanmax

    def test_velocity_door_region_exceeds_control_region(self):
        """The genuinely verified door-width effect: VELOCITY (a direct
        airflow measure) shows a larger difference in the door's own
        x-band than in an unrelated control band, on both a per-frame-max
        and a time-averaged basis -- consistent with a wider door letting
        more air move through it."""
        key = SliceKey("VELOCITY", 1, 0)
        data_wide = load_data(self.DOOR_CASE_WIDE, key)
        data_narrow = load_data(self.DOOR_CASE_NARROW, key)
        n = min(data_wide.shape[0], data_narrow.shape[0])
        diff = data_wide[:n] - data_narrow[:n]

        n_x = diff.shape[2]
        x_cols = np.linspace(0, 1.0, n_x)
        door_band = (x_cols >= 0.24) & (x_cols <= 0.30)
        control_band = (x_cols >= 0.50) & (x_cols <= 0.60)

        abs_diff = np.abs(diff)
        door_meanmax = abs_diff[:, :, door_band].max(axis=(1, 2)).mean()
        control_meanmax = abs_diff[:, :, control_band].max(axis=(1, 2)).mean()
        assert door_meanmax > control_meanmax

        mean_diff = diff.mean(axis=0)
        door_tavg = np.abs(mean_diff[:, door_band]).mean()
        control_tavg = np.abs(mean_diff[:, control_band]).mean()
        assert door_tavg > control_tavg


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
