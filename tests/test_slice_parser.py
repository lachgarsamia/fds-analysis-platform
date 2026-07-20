"""Unit tests for the FDS binary parser (slice.py)."""

import time

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
        """Verify readSlice tuple is (mesh, extent, data, mask, times)."""
        mesh, extent, data, mask, times = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert data.shape == (481, 49, 101)
        assert len(times) == 481
        assert mesh is not None
        assert extent is not None
        assert mask is not None

    def test_times_strictly_increasing(self, fixtures_dir):
        """Verify timestep array is strictly monotonic increasing."""
        mesh, extent, data, mask, times = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        diffs = np.diff(times)
        assert np.all(diffs > 0), "times must be strictly increasing"

    def test_times_length_matches_frames(self, fixtures_dir):
        """Verify times array length matches the number of frames."""
        mesh, extent, data, mask, times = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert len(times) == data.shape[0]

    def test_frame_zero_ambient_temperature(self, fixtures_dir):
        """Spot check: frame 0 should be approximately ambient (~20°C)."""
        mesh, extent, data, mask, times = readSlice(
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
        mesh, extent, data, mask, times = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert not np.all(np.isnan(data)), "data should not be entirely NaN"

    def test_data_not_all_zeros(self, fixtures_dir):
        """Verify data is not frozen at zero."""
        mesh, extent, data, mask, times = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        assert not np.allclose(data, 0.0), "data should not be entirely zero"

    def test_temperature_increases_over_time(self, fixtures_dir):
        """Sanity check: spatial mean temperature should not decrease monotonically."""
        mesh, extent, data, mask, times = readSlice(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        # Compute spatial mean per frame
        frame_means = np.mean(data, axis=(1, 2))
        # Fire grows, so peak should occur after frame 0
        assert np.max(frame_means) > np.mean(frame_means[0:5]), (
            "temperature should not be uniformly decreasing "
            "(fire grows over time)"
        )

    def test_cold_parse_under_500ms(self, fixtures_dir):
        """Vectorized read must parse one scenario in well under 0.5s (M1.2 DoD)."""
        t0 = time.perf_counter()
        data = readDataOnly(
            fixtures_dir, direction=1, offset=0, quantity="TEMPERATURE"
        )
        elapsed = time.perf_counter() - t0
        assert data is not None
        assert elapsed < 0.5, f"cold parse took {elapsed:.3f}s, expected <0.5s"


import os  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present")


@requires_real_dataset
class TestOuterEdgeColumn:
    """V2 roadmap M0.1: adjudication of the edge-column discrepancy filed
    in docs/spike-parser-validation.md §3. fdsreader's own per-mesh
    subslice (its independent raw .sf decode, before its to_global
    stitching) reports 42.58 C at the x=1.0 boundary node -- exactly what
    combineSlices reports. fdsreader's to_global() is what duplicates the
    x=0.99 value into the edge; our parser reads the true FDS value.
    Pinned so a future combineSlices change can't silently start padding
    the outer edge the way fdsreader's global stitcher does."""

    def test_outer_edge_is_the_true_distinct_fds_value(self):
        _mesh, _extent, data, _mask, _times = readSlice(
            os.path.join(SIM_ROOT, "c1_d0_vod0_voc0"),
            direction=1, offset=0, quantity="TEMPERATURE")
        assert data.shape[1:] == (49, 101)
        edge = data[329, 6, 100]      # x = 1.0 (outer boundary node)
        neighbor = data[329, 6, 99]   # x = 0.99
        # The genuine FDS value at the edge, distinct from its neighbor --
        # NOT a padded duplicate of it (fdsreader's to_global artifact).
        assert abs(edge - 42.58) < 0.1
        assert abs(neighbor - 82.41) < 0.1
        assert abs(edge - neighbor) > 30.0
