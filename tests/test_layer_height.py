"""Tests for smoke-layer height (V2 roadmap M2.3)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layer_height import smoke_layer_height_series  # noqa: E402

EXTENT = (0.0, 1.0, 0.0, 1.0)  # z0=0 (floor), z1=1 (ceiling)


def test_no_fire_returns_ceiling_height():
    data = np.full((3, 10, 5), 20.0)  # all at ambient
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0)
    np.testing.assert_allclose(heights, 1.0)


def test_uniform_hot_upper_half_gives_midpoint():
    # rows 0..4 = z in (1.0 .. 0.55], hot; rows 5..9 = z in [0.5..0.0), ambient.
    # row 0 = ceiling (z=1), row 9 = floor (z=0), per the app's convention.
    data = np.full((1, 10, 4), 20.0)
    data[0, :5, :] = 120.0  # top half hot, uniform excess
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0)
    # Uniform excess in the top half -> half the integral is reached
    # exactly at the midpoint of that hot band, i.e. z ~= 0.75.
    assert 0.7 < heights[0] < 0.8


def test_hotter_near_ceiling_gives_higher_layer_than_uniform():
    # Same total excess, but concentrated closer to the ceiling -> the
    # half-integral point should sit higher than the uniform case above.
    data = np.full((1, 10, 4), 20.0)
    data[0, 0, :] = 500.0  # single hot row right at the ceiling
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0)
    assert heights[0] > 0.9

def test_returns_one_height_per_frame():
    data = np.full((7, 10, 4), 20.0)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0)
    assert heights.shape == (7,)

def test_degenerate_single_row_returns_ceiling():
    data = np.full((2, 1, 4), 200.0)
    heights = smoke_layer_height_series(data, EXTENT, ambient_c=20.0)
    np.testing.assert_allclose(heights, EXTENT[3])
