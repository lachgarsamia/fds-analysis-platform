"""Unit tests for the FDS binary parser (slice.py)."""

import time

import pytest
import numpy as np
from fds.slice.slice import readDataOnly, readSlice
import fds.slice.slice as fds_slice_module


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

# TestOuterEdgeColumn/TestVelocityCrossValidation/TestVectorVelocityCrossValidation
# below pin exact values from fds/sim/ specifically (parser-regression
# guards -- see each class's docstring), not "whatever the app's current
# live default happens to be". They used to import load_data.SIM_ROOT as a
# shortcut to "a real dataset on disk", back when that was always fds/sim/;
# now that SIM_ROOT points at fds/sim_stage1_prep/ (a different re-run,
# different pinned values), importing it here would silently point these
# regression guards at the wrong dataset -- so this hardcodes the specific
# reference dataset directly instead, independent of the app's own default.
SIM_ROOT = os.path.join(os.path.dirname(__file__), '..', 'fds', 'sim')  # noqa: E402

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


@requires_real_dataset
class TestVelocityCrossValidation:
    """VELOCITY cross-validation (follow-up to TestOuterEdgeColumn): the
    M1.3s spike (docs/spike-parser-validation.md) only ever numerically
    checked TEMPERATURE -- VELOCITY was confirmed present/readable but never
    verified cell-by-cell. Closed here, against the existing
    fds/sim/c1_d0_vod0_voc0/ dataset, before Pleiades spends compute on
    U/W-VELOCITY through this same code path (combineSlices, slice.py:412).

    Method: fdsreader==1.11.7 (scratch venv, not a project dependency) was
    used to decode each of the 24 raw per-mesh VELOCITY subslices directly
    (Slice.subslices / SubSlice.data -- fdsreader's independent .sf decode,
    *before* its own to_global() stitching), then reassembled into a global
    (t, z, x) grid using combineSlices' own offset/overwrite arithmetic
    (min/max extent, dx, off1/off2, last-mesh-wins on the y=0 duplicate
    seam). Compared against our parser's combineSlices output across the
    FULL grid (49x101 = 4,949 points x 481 frames = 2,380,469 cells) --
    not just the edge, since VELOCITY's failure mode (if any) was unknown.

    Result: exact bit-for-bit match, max abs diff 0.0 m/s, every cell,
    every frame -- including the domain edges (x=0, x=1.0, z=0, z=0.48)
    and the y=0 mesh-boundary duplication that produces the doubled 24
    subslices in the first place. No discrepancy of any kind was found, so
    (unlike TEMPERATURE) there was nothing to adjudicate -- this pins the
    already-clean result as a regression guard, using fdsreader's
    documented to_global() padding artifact (reproduced here too, e.g.
    frame 329/row 6/col 100: our 0.1026 vs to_global's padded 0.1807,
    identical signature to TEMPERATURE's) only as corroborating evidence
    that a real discrepancy would in fact show up if introduced.

    Values pinned to 4 decimal places (well inside float32 precision) since
    the underlying diff is exactly 0.0, not merely small."""

    def test_full_grid_values_at_domain_edges_and_interior(self):
        _mesh, _extent, data, _mask, _times = readSlice(
            os.path.join(SIM_ROOT, "c1_d0_vod0_voc0"),
            direction=1, offset=0, quantity="VELOCITY")
        assert data.shape == (481, 49, 101)

        # Same frame/row the TEMPERATURE edge-column test uses, so both
        # tests are directly comparable -- plus the far x, z edges.
        assert abs(data[329, 6, 100] - 0.1026) < 1e-3    # x = 1.0 edge
        assert abs(data[329, 6, 0] - 0.0792) < 1e-3      # x = 0.0 edge
        assert abs(data[329, 0, 50] - 0.1277) < 1e-3     # z = 0.0 edge
        assert abs(data[329, 48, 50] - 0.1168) < 1e-3    # z = 0.48 edge
        # An interior point, away from every domain boundary.
        assert abs(data[200, 24, 50] - 0.0298) < 1e-3

    def test_no_nan_or_placeholder_at_mesh_seams(self):
        """The y=0 plane is duplicated across 24 mesh subslices (12 unique
        x,z footprints x 2 abutting meshes) -- combineSlices' last-mesh-wins
        overwrite must actually fill every cell, not leave an uninitialized
        seam. (Cross-validated exactly against fdsreader's own raw per-mesh
        decode for this same reason -- see class docstring.)"""
        _mesh, _extent, data, _mask, _times = readSlice(
            os.path.join(SIM_ROOT, "c1_d0_vod0_voc0"),
            direction=1, offset=0, quantity="VELOCITY")
        assert not np.any(np.isnan(data))


_PLEIADES_DIR = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0_pleiades")
requires_pleiades_dataset = pytest.mark.skipif(
    not os.path.isdir(_PLEIADES_DIR), reason="c1_d0_vod0_voc0_pleiades/ dataset not present")


@requires_pleiades_dataset
class TestVectorVelocityCrossValidation:
    """U-VELOCITY / W-VELOCITY cross-validation against the real Pleiades
    run (c1_d0_vod0_voc0_pleiades/, FDS-6.11.1) that first produced these
    signed components -- closing the gap TestVelocityCrossValidation left
    (that test only had the scalar VELOCITY magnitude to check; there was
    no real U/W output anywhere until this run).

    Method: identical to TestVelocityCrossValidation -- fdsreader==1.11.7
    (scratch venv) decoded each raw per-mesh subslice directly
    (Slice.subslices / SubSlice.data, never to_global()), reassembled with
    combineSlices' own offset arithmetic, and diffed against our parser's
    output across the full grid, every frame, both quantities, AND both
    the default (PBY=0.000, 49x101 nodes) and CELL_CENTERED (PBY=-0.005,
    48x100 cells) variants of each. Result: exact bit-for-bit match, max
    abs diff 0.0 m/s, every combination -- and, since U/W are signed
    (unlike VELOCITY's magnitude), an explicit sign check was run too:
    zero sign disagreements on any cell where both readings exceed
    0.01 m/s (~1.8-2.3M qualifying cells per combination), including at
    every mesh-boundary/domain-edge column and row. combineSlices does not
    flip a component's sign at a stitching seam.

    One real finding, not a defect: fds.slice.slice.findSlices() selects
    slices by physical-offset proximity (`abs(slice_offset - offset) <
    1.5*slice_delta`, slice.py:406), and 1.5 mesh-cells (~0.015 m) is
    wider than the 0.005 m gap between the default and CELL_CENTERED
    planes -- so readSlice(direction=1, offset=0) actually matches BOTH
    variants' subslices (36, not 24) and hands them all to combineSlices().
    It still returns exactly the default-plane values today, but only
    because the default subslices are processed after the smaller
    CELL_CENTERED ones in file order and their larger extent fully
    overwrites them -- correct today by write-order coincidence, not by
    explicit selection. There is currently no offset value that isolates
    the CELL_CENTERED variant alone through the public readSlice()
    interface. This test therefore reads the CELL_CENTERED arrays by
    filtering `readSliceInfos()`'s slice list to `.centered` directly and
    calling `combineSlices()` on just those -- both already-public
    functions in slice.py, called differently, not modified -- to validate
    the data those files actually contain. Not fixed here (test file only,
    per scope); worth a real fix (e.g. an explicit `centered` argument to
    findSlices) before anything depends on selecting CELL_CENTERED data
    through the normal read path.

    Values pinned to 4 decimal places since the underlying diff is exactly
    0.0, not merely small -- includes negative (reversed-direction) values
    specifically, since a sign-only bug would not show up in a magnitude
    tolerance check."""

    def _cell_centered(self, quantity):
        smv_fn = fds_slice_module.scanDirectory(_PLEIADES_DIR)
        sc = fds_slice_module.readSliceInfos(os.path.join(_PLEIADES_DIR, smv_fn))
        meshes = fds_slice_module.readMeshes(os.path.join(_PLEIADES_DIR, smv_fn))
        centered_slices = [s for s in sc.slices
                           if s.quantity == quantity and s.norm_direction == 1 and s.centered]
        assert len(centered_slices) == 12, (quantity, len(centered_slices))
        for sid in centered_slices:
            sid.readAllTimes(_PLEIADES_DIR)
            sid.readData(_PLEIADES_DIR)
            sid.mapData(meshes)
        _mesh, _extent, data, _mask, _times = fds_slice_module.combineSlices(centered_slices)
        return data

    def test_u_velocity_default_variant(self):
        _mesh, _extent, data, _mask, _times = readSlice(
            _PLEIADES_DIR, direction=1, offset=0, quantity="U-VELOCITY")
        assert data.shape == (481, 49, 101)
        assert not np.any(np.isnan(data))
        assert abs(data[329, 22, 94] - (-0.5653)) < 1e-3   # negative -- reversed-direction flow
        assert abs(data[329, 15, 96] - 0.2143) < 1e-3
        assert abs(data[329, 6, 0]) < 1e-3                  # x=0.0 edge
        assert abs(data[329, 6, 100]) < 1e-3                # x=1.0 edge

    def test_w_velocity_default_variant(self):
        _mesh, _extent, data, _mask, _times = readSlice(
            _PLEIADES_DIR, direction=1, offset=0, quantity="W-VELOCITY")
        assert data.shape == (481, 49, 101)
        assert not np.any(np.isnan(data))
        assert abs(data[329, 21, 99] - (-0.1398)) < 1e-3   # negative -- reversed-direction flow
        assert abs(data[329, 16, 97] - 0.8060) < 1e-3
        assert abs(data[329, 6, 0] - (-0.0155)) < 1e-3      # x=0.0 edge, also negative
        assert abs(data[329, 6, 100] - 0.0051) < 1e-3       # x=1.0 edge

    def test_u_velocity_cell_centered_variant(self):
        data = self._cell_centered("U-VELOCITY")
        assert data.shape == (481, 48, 100)
        assert not np.any(np.isnan(data))
        assert abs(data[329, 21, 93] - (-0.5653)) < 1e-3
        assert abs(data[329, 21, 97] - 0.3696) < 1e-3

    def test_w_velocity_cell_centered_variant(self):
        data = self._cell_centered("W-VELOCITY")
        assert data.shape == (481, 48, 100)
        assert not np.any(np.isnan(data))
        assert abs(data[329, 3, 93] - (-0.2352)) < 1e-3
        assert abs(data[329, 12, 96] - 0.9667) < 1e-3
