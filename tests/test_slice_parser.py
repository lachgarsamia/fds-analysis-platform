"""Unit tests for the FDS binary parser (slice.py)."""

import pytest
import numpy as np
from fds.slice.slice import readDataOnly, readSlice


class TestSliceParser:
    """Tests for FDS .smv/.sf binary parsing."""

    def test_readDataOnly_shape(self, fixtures_dir):
        """Verify readDataOnly returns correct shape (481, 49, 101)."""
        data = readDataOnly(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert data.shape == (481, 49, 101), f"expected (481, 49, 101), got {data.shape}"

    def test_readDataOnly_dtype(self, fixtures_dir):
        """Verify data is float32 (memory optimized)."""
        data = readDataOnly(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert data.dtype == np.float32, f"expected float32, got {data.dtype}"

    def test_readDataOnly_not_none(self, fixtures_dir):
        """Verify readDataOnly does not silently return None."""
        data = readDataOnly(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert data is not None

    def test_readSlice_returns_tuple(self, fixtures_dir):
        """Verify readSlice returns a 5-tuple, not None."""
        result = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_readSlice_tuple_contents(self, fixtures_dir):
        """Verify readSlice tuple is (data, times, meshes, extent, norm_direction)."""
        data, times, meshes, extent, norm_direction = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert data.shape == (481, 49, 101)
        assert len(times) == 481
        assert meshes is not None
        assert extent is not None
        assert norm_direction is not None

    def test_times_strictly_increasing(self, fixtures_dir):
        """Verify timestep array is strictly monotonic increasing."""
        _, times, _, _, _ = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        diffs = np.diff(times)
        assert np.all(diffs > 0), "times must be strictly increasing"

    def test_times_length_matches_frames(self, fixtures_dir):
        """Verify times array length matches the number of frames."""
        data, times, _, _, _ = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert len(times) == data.shape[0]

    def test_frame_zero_ambient_temperature(self, fixtures_dir):
        """Spot check: frame 0 should be approximately ambient (~20°C)."""
        data, _, _, _, _ = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        frame_0_mean = np.mean(data[0, :, :])
        # Ambient is typically 20–25°C; allow some tolerance
        assert 15.0 <= frame_0_mean <= 30.0, (
            f"frame 0 mean {frame_0_mean}°C is outside "
            "expected ambient range [15, 30]"
        )

    def test_data_not_all_nan(self, fixtures_dir):
        """Verify data does not contain all NaN values."""
        data, _, _, _, _ = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert not np.all(np.isnan(data)), "data should not be entirely NaN"

    def test_data_not_all_zeros(self, fixtures_dir):
        """Verify data is not frozen at zero."""
        data, _, _, _, _ = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert not np.allclose(data, 0.0), "data should not be entirely zero"

    def test_temperature_increases_over_time(self, fixtures_dir):
        """Sanity check: spatial mean temperature should not decrease monotonically."""
        data, _, _, _, _ = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        # Compute spatial mean per frame
        frame_means = np.mean(data, axis=(1, 2))
        # Fire grows, so peak should occur after frame 0
        assert np.max(frame_means) > np.mean(frame_means[0:5]), (
            "temperature should not be uniformly decreasing "
            "(fire grows over time)"
        )
