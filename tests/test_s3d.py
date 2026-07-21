"""Tests for the `.s3d` volumetric reader (V2 roadmap M2.1).

Pure decoder logic (Fortran records, RLE, `.sz` sidecar) is tested
against small synthetic byte streams built by hand -- same philosophy
as tests/fixtures/ for the .sf parser: known bytes in, known values
out. `read_smoke3d_infos` runs against the real trimmed fixture .smv
(it has SMOKF3D/SMOKG3D metadata even though the binary .s3d payloads
were trimmed out). `extract_soot_y0_plane` needs real binary data and
is skipped when fds/sim/ isn't present, matching test_views.py's
requires_real_dataset convention.
"""

import io
import os
import struct

import numpy as np
import pytest

from fds.s3d import s3d
from load_data import SIM_ROOT


def _fortran_record(payload: bytes) -> bytes:
    length = struct.pack('<i', len(payload))
    return length + payload + length


class TestFortranRecord:
    def test_round_trip(self):
        f = io.BytesIO(_fortran_record(b'abcd'))
        assert s3d._read_fortran_record(f) == b'abcd'

    def test_returns_none_at_eof(self):
        f = io.BytesIO(b'')
        assert s3d._read_fortran_record(f) is None

    def test_marker_mismatch_raises(self):
        bad = struct.pack('<i', 4) + b'abcd' + struct.pack('<i', 5)
        f = io.BytesIO(bad)
        with pytest.raises(ValueError):
            s3d._read_fortran_record(f)


class TestReadS3DHeader:
    def test_parses_bounds_after_two_leading_ints(self):
        payload = struct.pack('<8i', 1, 0, 0, 25, 0, 15, 0, 16)
        f = io.BytesIO(_fortran_record(payload))
        assert s3d.read_s3d_header(f) == (0, 25, 0, 15, 0, 16)


class TestDecodeRLE:
    def test_all_literal_bytes(self):
        payload = bytes([1, 2, 3, 4])
        out = s3d.decode_rle(payload, npts=4)
        np.testing.assert_array_equal(out, [1, 2, 3, 4])

    def test_escape_run(self):
        payload = bytes([255, 7, 5])  # value=7, repeated 5 times
        out = s3d.decode_rle(payload, npts=5)
        np.testing.assert_array_equal(out, [7, 7, 7, 7, 7])

    def test_mixed_literal_and_run(self):
        payload = bytes([9, 255, 3, 2, 8])  # literal 9, then (3,3), then literal 8
        out = s3d.decode_rle(payload, npts=4)
        np.testing.assert_array_equal(out, [9, 3, 3, 8])

    def test_short_decode_raises(self):
        payload = bytes([1, 2])
        with pytest.raises(ValueError):
            s3d.decode_rle(payload, npts=5)


class TestReadS3DSeries:
    def test_single_frame_round_trip(self, tmp_path):
        # 2x2x1 node grid -> npts=4, one frame, all-literal RLE payload.
        header = _fortran_record(struct.pack('<8i', 1, 0, 0, 1, 0, 1, 0, 0))
        time_rec = _fortran_record(struct.pack('<f', 0.25))
        npts_rec = _fortran_record(struct.pack('<2i', 4, 4))
        data_rec = _fortran_record(bytes([10, 20, 30, 40]))
        path = tmp_path / "synthetic.s3d"
        path.write_bytes(header + time_rec + npts_rec + data_rec)

        times, bounds, levels = s3d.read_s3d_series(str(path))
        assert bounds == (0, 1, 0, 1, 0, 0)
        np.testing.assert_allclose(times, [0.25])
        assert levels.shape == (1, 2, 2, 1)
        # Fortran (column-major) order: byte 0 -> (x=0,y=0), byte 1 ->
        # (x=1,y=0) [x varies fastest], byte 2 -> (x=0,y=1), byte 3 -> (x=1,y=1).
        np.testing.assert_array_equal(levels[0, :, :, 0], [[10, 30], [20, 40]])

    def test_frame_npts_mismatch_raises(self, tmp_path):
        header = _fortran_record(struct.pack('<8i', 1, 0, 0, 1, 0, 1, 0, 0))
        time_rec = _fortran_record(struct.pack('<f', 0.0))
        npts_rec = _fortran_record(struct.pack('<2i', 3, 3))  # wrong: header implies 4
        data_rec = _fortran_record(bytes([1, 2, 3]))
        path = tmp_path / "bad.s3d"
        path.write_bytes(header + time_rec + npts_rec + data_rec)
        with pytest.raises(ValueError):
            s3d.read_s3d_series(str(path))


class TestReadSzUpperBounds:
    def test_parses_fourth_column(self, tmp_path):
        path = tmp_path / "x.s3d.sz"
        path.write_text("0.0 7072 84 0.5\n0.25 7072 90 0.64\n")
        bounds = s3d.read_sz_upper_bounds(str(path))
        np.testing.assert_allclose(bounds, [0.5, 0.64])

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "x.s3d.sz"
        path.write_text("0.0 7072 84 0.5\nnot enough cols\n0.25 7072 90 0.64\n")
        bounds = s3d.read_sz_upper_bounds(str(path))
        np.testing.assert_allclose(bounds, [0.5, 0.64])


class TestY0MeshIds:
    def test_selects_only_ymin_zero_meshes(self):
        class FakeMesh:
            def __init__(self, y0, y1):
                self.ranges = [[0, 1], [y0, y1], [0, 1]]

        class FakeCollection:
            meshes = [FakeMesh(-0.15, 0.0), FakeMesh(0.0, 0.15), FakeMesh(0.15, 0.30)]

        assert s3d.y0_mesh_ids(FakeCollection()) == [1]


@pytest.fixture
def fixtures_smv():
    return os.path.join(os.path.dirname(__file__), "fixtures", "c1_d0_vod0_voc0",
                         "c1_d0_vod0_voc0.smv")


class TestReadSmoke3DInfos:
    def test_finds_soot_density_and_temperature_across_all_meshes(self, fixtures_smv):
        infos = s3d.read_smoke3d_infos(fixtures_smv)
        soot = [i for i in infos if i.quantity == "SOOT DENSITY"]
        temp = [i for i in infos if i.quantity == "TEMPERATURE"]
        assert len(soot) == 24
        assert len(temp) == 24
        assert soot[0].units == "kg/m3"
        assert soot[0].filename.endswith(".s3d")
        assert {i.mesh_id for i in soot} == set(range(24))


requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


@requires_real_dataset
class TestExtractSootY0PlaneRealData:
    """Real-data verification, per the app's M1.3s-standard convention:
    checks structural correctness against ground truth (the .sf grid
    shape) and physical plausibility, not just "it runs"."""

    CASE_DIR = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")

    def test_matches_sf_grid_shape_after_stitching(self):
        from load_data import load_data
        from slice_key import SliceKey
        times, extent, frames = s3d.extract_soot_y0_plane(self.CASE_DIR)
        sf_data = load_data(self.CASE_DIR, key=SliceKey('TEMPERATURE'))
        assert frames.shape[1:] == sf_data.shape[1:]  # (n_z, n_x) matches exactly

    def test_values_are_nonnegative_and_grow_over_time(self):
        times, extent, frames = s3d.extract_soot_y0_plane(self.CASE_DIR)
        assert np.all(frames >= 0.0)
        early_nonzero = np.mean(frames[: len(frames) // 4] > 0)
        late_nonzero = np.mean(frames[-len(frames) // 4:] > 0)
        assert late_nonzero >= early_nonzero

    def test_extent_is_physically_sane(self):
        times, extent, frames = s3d.extract_soot_y0_plane(self.CASE_DIR)
        x0, x1, z0, z1 = extent
        assert x1 > x0
        assert z1 > z0


@requires_real_dataset
class TestLoadDataSootDispatch:
    """M2.2: load_data() routes SOOT DENSITY keys to the .s3d reader and
    scales to mg/m3, while extent comes from soot_plane_geometry."""

    CASE_DIR = os.path.join(SIM_ROOT, "c1_d0_vod0_voc0")

    def test_load_data_soot_matches_scaled_extract(self):
        from load_data import load_data, SOOT_DISPLAY_SCALE
        from slice_key import SliceKey, AXIS_TO_DIRECTION
        key = SliceKey("SOOT DENSITY", AXIS_TO_DIRECTION['y'], 0, 0.0)
        via_loader = load_data(self.CASE_DIR, key)
        _times, _extent, frames = s3d.extract_soot_plane(self.CASE_DIR, axis='y', offset=0.0)
        np.testing.assert_allclose(via_loader, frames * SOOT_DISPLAY_SCALE)

    def test_load_slice_geometry_soot_extent_matches_extract(self):
        from load_data import load_slice_geometry
        from slice_key import SliceKey, AXIS_TO_DIRECTION
        key = SliceKey("SOOT DENSITY", AXIS_TO_DIRECTION['x'], 0, 0.25)
        _mesh, extent, _mask = load_slice_geometry(self.CASE_DIR, key)
        _times, extract_extent, _frames = s3d.extract_soot_plane(self.CASE_DIR, axis='x', offset=0.25)
        assert tuple(extent) == tuple(extract_extent)
