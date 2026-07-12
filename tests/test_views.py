"""Unit tests for views.py: SliceView (M2.2.1), DifferenceView/EnsembleView
(M2.3.1/2.3.2), and GridCell/ViewGrid (M2.2.2/2.2.3). Pure widget-layer
tests -- no ScenarioStore/controller involved, cells are handed synthetic
frames directly, except TestDifferenceViewRealData which deliberately
reads the real dataset to check the "physically sensible structure" DoD
claim against ground truth rather than trusting it looks reasonable."""

import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from views import SliceView, DifferenceView, EnsembleView, GridCell, ViewGrid
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

    def test_init_plot_without_extent_still_works(self, qapp):
        """extent is optional -- callers that don't have geometry (or
        don't need physical coordinates) get the pre-M2.6 pixel-index
        behavior, not a crash."""
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=10.0, colorbar_label="x")
        assert view.value_at(5, 5) is None, "no extent means no meaningful physical-coordinate lookup"


class TestSliceViewProbe:
    """Corner/known-pixel accuracy for value_at() (M2.6.1's DoD: "probe
    accurate at corners"), and the row<->physical-z flip-awareness the
    class docstring documents (see SliceView's own docstring for the
    reasoning verified against real data before this was written)."""

    def _view_with_known_frame(self, extent=(0.0, 1.0, 0.0, 0.48)):
        # 4x4 frame, every cell a distinct value so row/col mixups fail loudly.
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        view = SliceView()
        view.init_plot(frame, cmap="viridis", interpolation="nearest",
                        vmin=0.0, vmax=15.0, colorbar_label="x", extent=extent)
        return view, frame

    def test_value_at_top_left_corner_is_row0_col0(self, qapp):
        view, frame = self._view_with_known_frame()
        # extent=(x0,x1,z0,z1): top-left of the *displayed image* is
        # (x0, z1) -- row 0 is physically at the top (z1), per origin='upper'.
        assert view.value_at(0.0, 0.48) == frame[0, 0]

    def test_value_at_bottom_right_corner_is_last_row_last_col(self, qapp):
        view, frame = self._view_with_known_frame()
        assert view.value_at(1.0, 0.0) == frame[-1, -1]

    def test_value_at_top_right_and_bottom_left(self, qapp):
        view, frame = self._view_with_known_frame()
        assert view.value_at(1.0, 0.48) == frame[0, -1]
        assert view.value_at(0.0, 0.0) == frame[-1, 0]

    def test_value_at_out_of_bounds_returns_none(self, qapp):
        view, _frame = self._view_with_known_frame()
        assert view.value_at(-0.5, 0.2) is None
        assert view.value_at(1.5, 0.2) is None

    def test_value_at_without_extent_returns_none(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=10.0, colorbar_label="x")
        assert view.value_at(0.5, 0.5) is None

    def test_enable_probe_calls_back_on_synthetic_motion_event(self, qapp):
        view, frame = self._view_with_known_frame()
        received = []
        view.enable_probe(lambda x, z, v: received.append((x, z, v)))

        class FakeEvent:
            inaxes = view.ax
            xdata = 0.0
            ydata = 0.48

        view._on_mouse_move(FakeEvent())
        assert received == [(0.0, 0.48, float(frame[0, 0]))]

    def test_probe_callback_gets_none_when_mouse_leaves_axes(self, qapp):
        view, _frame = self._view_with_known_frame()
        received = []
        view.enable_probe(lambda x, z, v: received.append((x, z, v)))

        class FakeEventOutside:
            inaxes = None
            xdata = None
            ydata = None

        view._on_mouse_move(FakeEventOutside())
        assert received == [(None, None, None)]

    def test_disable_probe_disconnects(self, qapp):
        view, _frame = self._view_with_known_frame()
        received = []
        view.enable_probe(lambda x, z, v: received.append((x, z, v)))
        view.disable_probe()

        class FakeEvent:
            inaxes = view.ax
            xdata = 0.0
            ydata = 0.48

        view._on_mouse_move(FakeEvent())
        assert received == [], "no callback should fire after disable_probe()"


@requires_real_dataset
class TestSliceViewProbeRealData:
    """The exact cross-check performed during development: a known
    peak-difference pixel from M2.3's investigation (row=46, col=96 in the
    flipped/displayed array, physically x=0.96, z=0.02) must round-trip
    through value_at() to the same real TEMPERATURE value stored in the
    array itself -- confirms the flip-aware extent convention is right on
    real data, not just a hand-built 4x4 synthetic frame."""

    def test_known_pixel_matches_real_data(self):
        key = SliceKey("TEMPERATURE", 1, 0)
        case_dir = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")
        data = load_data(case_dir, key)
        view = SliceView()
        view.init_plot(data[300], cmap="gist_heat", interpolation="nearest",
                        vmin=20.0, vmax=300.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        assert view.value_at(0.96, 0.02) == pytest.approx(float(data[300][46, 96]))

    def test_all_four_corners_match_real_data(self):
        key = SliceKey("TEMPERATURE", 1, 0)
        case_dir = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")
        data = load_data(case_dir, key)
        frame = data[300]
        view = SliceView()
        view.init_plot(frame, cmap="gist_heat", interpolation="nearest",
                        vmin=20.0, vmax=300.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        assert view.value_at(0.0, 0.48) == pytest.approx(float(frame[0, 0]))
        assert view.value_at(1.0, 0.0) == pytest.approx(float(frame[-1, -1]))
        assert view.value_at(1.0, 0.48) == pytest.approx(float(frame[0, -1]))
        assert view.value_at(0.0, 0.0) == pytest.approx(float(frame[-1, 0]))


class TestSliceViewIsotherms:
    def test_disabled_by_default(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x")
        assert not view.isotherms_enabled

    def test_enabling_with_levels_draws_a_contour(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        view.set_isotherm_levels([60, 100])
        view.set_isotherms_enabled(True)
        assert view.isotherms_enabled
        assert view._contour_artist is not None

    def test_disabling_clears_the_contour(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        view.set_isotherm_levels([60])
        view.set_isotherms_enabled(True)
        view.set_isotherms_enabled(False)
        assert not view.isotherms_enabled
        assert view._contour_artist is None

    def test_show_frame_redraws_contour_each_call_while_enabled(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        view.set_isotherm_levels([50])
        view.set_isotherms_enabled(True)
        first_artist = view._contour_artist
        new_frame = np.full((49, 101), 75.0, dtype=np.float32)
        view.show_frame(new_frame)
        assert view._contour_artist is not None
        assert view._contour_artist is not first_artist, "contour must be a fresh artist each frame, not reused"

    def test_show_frame_does_not_redraw_contour_when_disabled(self, qapp):
        """Off-state must stay on the cheap blit path -- no contour work
        at all, confirming M2.6's "off-state performance unchanged" DoD."""
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        view.set_isotherm_levels([50])  # levels set, but never enabled
        new_frame = np.full((49, 101), 75.0, dtype=np.float32)
        view.show_frame(new_frame)
        assert view._contour_artist is None

    def test_no_levels_set_draws_no_contour_even_if_enabled(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        view.set_isotherms_enabled(True)
        assert view._contour_artist is None

    def test_setting_levels_while_enabled_redraws_immediately(self, qapp):
        view = SliceView()
        view.init_plot(FRAME, cmap="gist_heat", interpolation="nearest",
                        vmin=0.0, vmax=100.0, colorbar_label="x", extent=(0.0, 1.0, 0.0, 0.48))
        view.set_isotherms_enabled(True)
        assert view._contour_artist is None  # no levels yet
        view.set_isotherm_levels([60, 100, 300])
        assert view._contour_artist is not None


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


class TestEnsembleView:
    def _arrays(self):
        # 3 "scenarios", 4 timesteps, 2x2 frames -- distinct, hand-computable values.
        a = np.full((4, 2, 2), 10.0, dtype=np.float32)
        b = np.full((4, 2, 2), 20.0, dtype=np.float32)
        c = np.full((4, 2, 2), 30.0, dtype=np.float32)
        return [a, b, c]

    def test_compute_composite_mean(self):
        result = EnsembleView.compute_composite(self._arrays(), index=0, stat="mean")
        assert np.allclose(result, 20.0)  # mean(10,20,30)

    def test_compute_composite_min_max(self):
        arrays = self._arrays()
        assert np.allclose(EnsembleView.compute_composite(arrays, 0, "min"), 10.0)
        assert np.allclose(EnsembleView.compute_composite(arrays, 0, "max"), 30.0)

    def test_compute_composite_std(self):
        arrays = self._arrays()
        result = EnsembleView.compute_composite(arrays, 0, "std")
        expected = np.std([10.0, 20.0, 30.0])
        assert np.allclose(result, expected)

    def test_compute_composite_varies_per_frame_index(self):
        a = np.array([[[1.0]], [[100.0]]], dtype=np.float32)
        b = np.array([[[3.0]], [[300.0]]], dtype=np.float32)
        assert np.allclose(EnsembleView.compute_composite([a, b], 0, "mean"), 2.0)
        assert np.allclose(EnsembleView.compute_composite([a, b], 1, "mean"), 200.0)

    def test_compute_composite_rejects_unknown_stat(self):
        with pytest.raises(ValueError):
            EnsembleView.compute_composite(self._arrays(), 0, "median")

    def test_cmap_for_std_is_always_viridis(self):
        assert EnsembleView.cmap_for("std", quantity_cmap="gist_heat") == "viridis"
        assert EnsembleView.cmap_for("std", quantity_cmap="viridis") == "viridis"

    def test_cmap_for_mean_min_max_uses_quantity_cmap(self):
        for stat in ("mean", "min", "max"):
            assert EnsembleView.cmap_for(stat, quantity_cmap="gist_heat") == "gist_heat"

    def test_label_for_std_uses_sigma_notation(self):
        label = EnsembleView.label_for("std", quantity_label="Temperature", unit="°C")
        assert label == "σ(Temperature) (°C)"

    def test_label_for_mean_min_max(self):
        assert EnsembleView.label_for("mean", "Temperature", "°C") == "Mean Temperature (°C)"
        assert EnsembleView.label_for("min", "Temperature", "°C") == "Min Temperature (°C)"
        assert EnsembleView.label_for("max", "Temperature", "°C") == "Max Temperature (°C)"

    def test_std_vmax_matches_manual_computation(self):
        view = EnsembleView()
        arrays = self._arrays()
        vmax = view.std_vmax(arrays, cache_key="grp1", n_samples=4)
        expected = np.std([10.0, 20.0, 30.0])  # constant across all frames here
        assert vmax == pytest.approx(expected)

    def test_std_vmax_is_cached(self):
        view = EnsembleView()
        arrays = self._arrays()
        first = view.std_vmax(arrays, cache_key="grp1")
        arrays[0][:] = 9999.0  # mutate after caching
        second = view.std_vmax(arrays, cache_key="grp1")
        assert first == second

    def test_init_plot_and_show_frame_delegate_to_inner_slice_view(self, qapp):
        view = EnsembleView()
        view.init_plot(FRAME, cmap="viridis", interpolation="nearest",
                        vmin=0.0, vmax=5.0, colorbar_label="σ(Temperature) (°C)")
        assert view._inner.heatmap.get_cmap().name == "viridis"
        assert view._inner.heatmap.get_clim() == (0.0, 5.0)
        composite = np.full((49, 101), 3.5, dtype=np.float32)
        view.show_frame(composite)
        assert (view._inner.heatmap.get_array() == 3.5).all()


@requires_real_dataset
class TestEnsembleViewRealData:
    """Lighter-weight real-data smoke check than DifferenceView's (not
    independently requested, but cheap and worth doing given the data is
    already used elsewhere in this file): confirms the composite math
    produces sane, finite, correctly-ordered results (min <= mean <= max,
    std >= 0) across a real selection of scenarios, not just synthetic
    constants."""

    CASES = ["c1_d0_vod0_voc0", "c1_d1_vod0_voc0", "c2_d0_vod0_voc0", "c2_d1_vod0_voc0"]

    @pytest.fixture(scope="class")
    def arrays(self):
        key = SliceKey("TEMPERATURE", 1, 0)
        loaded = [load_data(os.path.join(SIM_ROOT, case), key) for case in self.CASES]
        n = min(a.shape[0] for a in loaded)
        return [a[:n] for a in loaded]

    def test_min_mean_max_ordering_holds_at_a_real_frame(self, arrays):
        index = 200
        mean = EnsembleView.compute_composite(arrays, index, "mean")
        mn = EnsembleView.compute_composite(arrays, index, "min")
        mx = EnsembleView.compute_composite(arrays, index, "max")
        assert (mn <= mean + 1e-4).all()
        assert (mean <= mx + 1e-4).all()

    def test_std_is_non_negative_and_finite(self, arrays):
        std = EnsembleView.compute_composite(arrays, 200, "std")
        assert np.isfinite(std).all()
        assert (std >= 0).all()
        assert std.max() > 0, "4 different real scenarios must not be bit-identical everywhere"


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


class FakeEntry:
    """Duck-types manifest.ScenarioEntry's shape (case_index/folder/
    candles/door/vod/voc) without importing manifest.py, matching
    views.py's own deliberate lack of a data-layer dependency."""

    def __init__(self, case_index, folder, candles, door, vod, voc):
        self.case_index = case_index
        self.folder = folder
        self.candles = candles
        self.door = door
        self.vod = vod
        self.voc = voc


class TestGridCellTypeSwitching:
    SCENARIOS = [("c1_d0_vod0_voc0", 0), ("c1_d0_vod0_voc1", 1), ("c2_d1_vod2_voc1", 23)]
    QUANTITIES = [("Temperature", SliceKey("TEMPERATURE", 1, 0)), ("Air speed", SliceKey("VELOCITY", 1, 0))]

    def test_starts_as_slice_type(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        assert cell.cell_type == "slice"
        assert isinstance(cell.view, SliceView)
        assert hasattr(cell, "scenario_combo")

    def test_switch_to_difference_replaces_view_and_header(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("difference")
        assert cell.cell_type == "difference"
        assert isinstance(cell.view, DifferenceView)
        assert hasattr(cell, "scenario_combo_a")
        assert hasattr(cell, "scenario_combo_b")
        assert not hasattr(cell, "scenario_combo")

    def test_switch_to_ensemble_replaces_view_and_header(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("ensemble")
        assert cell.cell_type == "ensemble"
        assert isinstance(cell.view, EnsembleView)
        assert hasattr(cell, "ensemble_select_button")
        assert hasattr(cell, "stat_combo")

    def test_switch_back_to_slice_rebuilds_scenario_combo(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("difference")
        cell._set_cell_type("slice")
        assert cell.cell_type == "slice"
        assert isinstance(cell.view, SliceView)
        assert hasattr(cell, "scenario_combo")

    def test_switching_to_same_type_is_a_no_op(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        view_before = cell.view
        cell._set_cell_type("slice")
        assert cell.view is view_before

    def test_type_changed_signal_emits_new_type(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        received = []
        cell.type_changed.connect(lambda c, t: received.append(t))
        cell._set_cell_type("ensemble")
        assert received == ["ensemble"]

    def test_difference_combo_defaults_to_first_two_scenarios(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("difference")
        assert cell.case_index_a == 0
        assert cell.case_index_b == 1

    def test_difference_combo_change_emits_signal(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("difference")
        received = []
        cell.difference_scenarios_changed.connect(lambda c, a, b: received.append((a, b)))
        cell.scenario_combo_a.setCurrentIndex(2)
        assert received == [(23, 1)]
        assert cell.case_index_a == 23

    def test_ensemble_starts_with_no_scenarios_selected(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("ensemble")
        assert cell.ensemble_case_indices == []
        assert cell.ensemble_select_button.text() == "0 scenarios selected…"

    def test_ensemble_stat_combo_change_emits_signal(self, qapp):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES)
        cell._set_cell_type("ensemble")
        cell.ensemble_case_indices = [0, 1]
        received = []
        cell.ensemble_changed.connect(lambda c, indices, stat: received.append((indices, stat)))
        cell.stat_combo.setCurrentIndex(EnsembleView.STATS.index("std"))
        assert received == [([0, 1], "std")]
        assert cell.ensemble_stat == "std"

    def test_ensemble_picker_updates_selection_and_button(self, qapp, monkeypatch):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES,
                         manifest_entries=[FakeEntry(0, "c1_d0_vod0_voc0", 0, 0, 0, 0),
                                           FakeEntry(1, "c1_d0_vod0_voc1", 0, 0, 0, 1)])
        cell._set_cell_type("ensemble")
        received = []
        cell.ensemble_changed.connect(lambda c, indices, stat: received.append(indices))

        monkeypatch.setattr(
            "views.EnsemblePickerDialog.exec_",
            lambda self: (self.list_widget.item(0).setCheckState(QtCore.Qt.Checked),
                          QtWidgets.QDialog.Accepted)[1],
        )
        cell._open_ensemble_picker()
        assert received == [[0]]
        assert cell.ensemble_select_button.text() == "1 scenario selected…"

    def test_ensemble_picker_cancel_does_not_change_selection(self, qapp, monkeypatch):
        cell = GridCell(self.SCENARIOS, self.QUANTITIES,
                         manifest_entries=[FakeEntry(0, "c1_d0_vod0_voc0", 0, 0, 0, 0)])
        cell._set_cell_type("ensemble")
        received = []
        cell.ensemble_changed.connect(lambda c, indices, stat: received.append(indices))
        monkeypatch.setattr("views.EnsemblePickerDialog.exec_", lambda self: QtWidgets.QDialog.Rejected)
        cell._open_ensemble_picker()
        assert received == []
        assert cell.ensemble_case_indices == []


class TestEnsemblePickerDialog:
    ENTRIES = [
        FakeEntry(0, "c1_d0_vod0_voc0", candles=0, door=0, vod=0, voc=0),
        FakeEntry(1, "c1_d0_vod0_voc1", candles=0, door=0, vod=0, voc=1),
        FakeEntry(2, "c1_d1_vod0_voc0", candles=0, door=1, vod=0, voc=0),
        FakeEntry(3, "c2_d0_vod1_voc0", candles=1, door=0, vod=1, voc=0),
    ]

    def test_initial_selection_is_pre_checked(self, qapp):
        from views import EnsemblePickerDialog
        dialog = EnsemblePickerDialog(self.ENTRIES, initial_selection=[1, 3])
        assert sorted(dialog.selected_case_indices()) == [1, 3]

    def test_select_all_and_none(self, qapp):
        from views import EnsemblePickerDialog
        dialog = EnsemblePickerDialog(self.ENTRIES, initial_selection=[])
        dialog._set_all(QtCore.Qt.Checked)
        assert sorted(dialog.selected_case_indices()) == [0, 1, 2, 3]
        dialog._set_all(QtCore.Qt.Unchecked)
        assert dialog.selected_case_indices() == []

    def test_factor_filter_checks_only_matching_entries(self, qapp):
        from views import EnsemblePickerDialog
        dialog = EnsemblePickerDialog(self.ENTRIES, initial_selection=[])
        dialog._apply_filter("door", 1)
        assert dialog.selected_case_indices() == [2]

    def test_factor_filter_is_additive_not_exclusive(self, qapp):
        """Applying a second filter should add to, not replace, the first
        filter's checked entries -- a user building "all vod=open OR
        candles=2" incrementally, not just the last filter clicked."""
        from views import EnsemblePickerDialog
        dialog = EnsemblePickerDialog(self.ENTRIES, initial_selection=[])
        dialog._apply_filter("door", 1)   # -> {2}
        dialog._apply_filter("candles", 1)  # -> {2, 3}
        assert sorted(dialog.selected_case_indices()) == [2, 3]

    def test_empty_manifest_entries_produces_no_crash(self, qapp):
        from views import EnsemblePickerDialog
        dialog = EnsemblePickerDialog([], initial_selection=[])
        assert dialog.selected_case_indices() == []


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
