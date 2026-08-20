"""Tests for smoke-layer height (V2 roadmap M2.3; gradient-default
methodology correction, see layer_height.py's module docstring for the
Steckler et al. 1982 citation)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layer_height import smoke_layer_height_series  # noqa: E402

EXTENT = (0.0, 1.0, 0.0, 1.0)  # z0=0 (floor), z1=1 (ceiling)


# ------------------------------------------------------- half_integral
# (explicit method= now that "gradient" is the default -- same tests,
# same expectations as before the methodology correction, just an
# explicit opt-in to the specific method they exercise.)

def test_no_fire_returns_ceiling_height():
    data = np.full((3, 10, 5), 20.0)  # all at ambient
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="half_integral")
    np.testing.assert_allclose(heights, 1.0)


def test_uniform_hot_upper_half_gives_midpoint():
    # rows 0..4 = z in (1.0 .. 0.55], hot; rows 5..9 = z in [0.5..0.0), ambient.
    # row 0 = ceiling (z=1), row 9 = floor (z=0), per the app's convention.
    data = np.full((1, 10, 4), 20.0)
    data[0, :5, :] = 120.0  # top half hot, uniform excess
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="half_integral")
    # Uniform excess in the top half -> half the integral is reached
    # exactly at the midpoint of that hot band, i.e. z ~= 0.75.
    assert 0.7 < heights[0] < 0.8


def test_hotter_near_ceiling_gives_higher_layer_than_uniform():
    # Same total excess, but concentrated closer to the ceiling -> the
    # half-integral point should sit higher than the uniform case above.
    data = np.full((1, 10, 4), 20.0)
    data[0, 0, :] = 500.0  # single hot row right at the ceiling
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="half_integral")
    assert heights[0] > 0.9

def test_returns_one_height_per_frame():
    data = np.full((7, 10, 4), 20.0)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="half_integral")
    assert heights.shape == (7,)

def test_degenerate_single_row_returns_ceiling():
    data = np.full((2, 1, 4), 200.0)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="half_integral")
    np.testing.assert_allclose(heights, EXTENT[3])


# ------------------------------------------------------- gradient (default)

def test_gradient_is_the_default_method():
    """Calling with no method= at all must match method="gradient"
    exactly -- the methodology-correction requirement, not just an
    available option."""
    rng = np.random.default_rng(0)
    data = 20.0 + rng.uniform(0, 50, size=(4, 12, 5))
    default = smoke_layer_height_series(data, EXTENT, ambient_c=20.0)
    explicit = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    np.testing.assert_array_equal(default, explicit)


def test_no_fire_returns_ceiling_height_gradient():
    data = np.full((3, 10, 5), 20.0)  # all at ambient -- no signal at all
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    np.testing.assert_allclose(heights, 1.0)


def test_degenerate_single_row_returns_ceiling_gradient():
    data = np.full((2, 1, 4), 200.0)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    np.testing.assert_allclose(heights, EXTENT[3])


def test_gradient_locates_a_sharp_step_interface():
    """A clean step profile (hot upper layer, cool lower layer) -- the
    steepest gradient must sit at the step, the textbook case the
    Steckler et al. method targets directly."""
    n_z = 20
    data = np.full((1, n_z, 4), 20.0)
    data[0, :8, :] = 150.0  # rows 0..7 (near ceiling) hot; step at row 8
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    # row 8 of 20 descending from z=1 to z=0 -> z ~= 1 - 8/19 ~= 0.58
    assert 0.45 < heights[0] < 0.75


def test_median_filter_suppresses_a_single_frame_spike_but_keeps_the_plateau():
    """Requirement 2/3: a single injected outlier row must not derail the
    detected height (median filter, not clamped/dropped -- it's simply
    outvoted within its filter window), while a genuine stable plateau on
    neighboring frames still tracks sensibly."""
    n_t, n_z, n_x = 5, 20, 4
    data = np.full((n_t, n_z, n_x), 20.0)
    # Every frame: the same clean step interface at row 8 (matches the
    # sharp-step test above) -- a stable plateau this method should track
    # consistently across frames.
    data[:, :8, :] = 150.0
    # Inject a single-frame, single-row spike at frame 2, deep in the
    # otherwise-cool lower region (row 15) -- exactly the shape of the
    # real spurious spike observed on real data (one frame, one outlier
    # row, everything else stable).
    data[2, 15, :] = 900.0

    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    plateau_heights = heights[[0, 1, 3, 4]]
    # The plateau frames (no spike) must all agree closely with each other.
    assert np.ptp(plateau_heights) < 0.05
    # The spiked frame must NOT have been derailed toward the spike's row
    # (row 15, deep in the lower region, z ~= 1 - 15/19 ~= 0.21) -- it
    # should stay close to the same step interface as its neighbors.
    assert abs(heights[2] - plateau_heights.mean()) < 0.1


def test_gradient_output_is_never_clamped_to_the_smoothed_plateau_value():
    """Requirement 3: this module must not silently scrub a genuine,
    sustained discontinuity -- only single-frame noise is
    (correctly) absorbed by the median filter. A real, multi-frame step
    change in the interface height must still show up as a real change
    in the output series, not get flattened toward whatever the
    surrounding frames were."""
    n_t, n_z, n_x = 6, 20, 4
    data = np.full((n_t, n_z, n_x), 20.0)
    data[:3, :8, :] = 150.0    # first half: step at row 8 (higher layer)
    data[3:, :14, :] = 150.0   # second half: a real, sustained step at row 14 (lower layer)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    # A genuine, sustained descent must be visible, not smoothed away.
    assert heights[:3].mean() - heights[3:].mean() > 0.15


def test_returns_one_height_per_frame_gradient():
    data = np.full((7, 10, 4), 20.0)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient")
    assert heights.shape == (7,)


def test_unknown_method_raises():
    data = np.full((1, 10, 4), 20.0)
    try:
        smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ------------------------------------------------------- gradient_simple

def test_gradient_simple_locates_a_sharp_step_interface():
    n_z = 20
    data = np.full((1, n_z, 4), 20.0)
    data[0, :8, :] = 150.0  # step at row 8
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0, method="gradient_simple")
    assert 0.45 < heights[0] < 0.75


def test_gradient_simple_flat_profile_returns_a_room_extreme_not_garbage():
    # No fire: flat at ambient -> ceiling.
    no_fire = np.full((2, 10, 4), 20.0)
    heights = smoke_layer_height_series(no_fire, EXTENT, ambient_c=20.0, method="gradient_simple")
    np.testing.assert_allclose(heights, 1.0)

    # Fully engulfed: flat and hot -> floor.
    engulfed = np.full((2, 10, 4), 300.0)
    heights = smoke_layer_height_series(engulfed, EXTENT, ambient_c=20.0, method="gradient_simple")
    np.testing.assert_allclose(heights, 0.0)
