"""`.s3d` volumetric smoke-data reader (V2 roadmap M2.1 -- "science backbone"),
sibling to `fds/slice/slice.py`, same validation standard (docs/spike-s3d.md,
docs/spike-s3d-v2.md).

Scope, per the M2.1 spike's conditional GO: SOOT DENSITY only, extracted at
the y=0 plane (the same plane the app's existing `.sf` TEMPERATURE/VELOCITY
slices use) via the 12-of-24 submeshes whose face sits at y=0 -- not a
full-domain 3D reconstruction. TEMPERATURE (`SMOKG3D`) is excluded: the
spike found its upper-bound sidecar data is identically zero in this
dataset (a genuine data limitation, not a decode bug).

Binary format (confirmed by the Tier-3 spike via manual byte inspection
cross-checked against `fdsreader`'s own decoder): each `.s3d` file is a
sequence of Fortran unformatted sequential records (4-byte leading/
trailing length markers, matching slice.py's `fds_fortran_backward`
convention) -- one header record (7 int32: a `ONE=1` sentinel + 6 mesh
node-index bounds), then per frame: a TIME record (1 float32), an
(NPTS, NCHARS) record (2 int32), and an NCHARS-byte RLE payload record.
RLE scheme: a `0xFF` escape byte followed by (value, repeat_count),
else a literal byte is one output value. Raw decoded values are 0-255
quantization levels, not physical units -- the matching `.sz` text
sidecar (`time, npts, nchars, upper_bound` per line) supplies the
per-frame rescale factor: physical = level/255 * upper_bound.

Not yet wired into the UI (ScenarioStore/DataKey registry) -- this is
the reader module only, per M2.1's scope.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

import fds.slice.slice as fds_slice

ESCAPE_BYTE = 255


def _read_fortran_record(f) -> bytes | None:
    """One Fortran unformatted sequential record's payload bytes, or
    None at EOF. Raises ValueError if the leading/trailing length
    markers disagree (a corrupted or misaligned read)."""
    lead = f.read(4)
    if len(lead) < 4:
        return None
    (length,) = struct.unpack('<i', lead)
    payload = f.read(length)
    trail = f.read(4)
    if len(trail) < 4:
        raise ValueError("truncated file: missing trailing record marker")
    (trail_length,) = struct.unpack('<i', trail)
    if trail_length != length:
        raise ValueError(f"Fortran record marker mismatch: lead={length} trail={trail_length}")
    return payload


def read_s3d_header(f) -> tuple:
    """(i1, i2, j1, j2, k1, k2) node-index bounds from the file's one
    header record. Real files measured 8 int32 (not the 7 the Tier-3
    spike doc's rounded description implied): a ONE=1 sentinel, one
    further unidentified int (always 0 on the scenarios checked here --
    not chased further), then the 6 bounds. Verified directly against
    real `.s3d` bytes for this M2.1 pass, not assumed from the prior
    spike's prose."""
    payload = _read_fortran_record(f)
    values = struct.unpack('<8i', payload)
    return values[2:]


def decode_rle(payload: bytes, npts: int) -> np.ndarray:
    """Decodes one frame's RLE payload into `npts` raw uint8 levels.
    Sequential by construction (token boundaries aren't knowable without
    scanning) -- see module docstring for the escape scheme."""
    out = np.empty(npts, dtype=np.uint8)
    buf = payload
    oi = 0
    i = 0
    n = len(buf)
    while i < n and oi < npts:
        b = buf[i]
        if b == ESCAPE_BYTE:
            value = buf[i + 1]
            count = buf[i + 2]
            out[oi:oi + count] = value
            oi += count
            i += 3
        else:
            out[oi] = b
            oi += 1
            i += 1
    if oi != npts:
        raise ValueError(f"RLE payload decoded to {oi} points, expected {npts}")
    return out


def read_s3d_series(path: str) -> tuple:
    """Reads an entire `.s3d` file: (times, bounds, levels).
    bounds = (i1,i2,j1,j2,k1,k2) node-index bounds from the header.
    levels: uint8 array, shape (n_frames, nx, ny, nz), nx=i2-i1+1 etc.
    -- matches the node-centered (IJK+1) grid shape `fdsreader` also
    reports for this format (cross-checked in the M2.1 spike)."""
    times = []
    frames = []
    with open(path, 'rb') as f:
        bounds = read_s3d_header(f)
        i1, i2, j1, j2, k1, k2 = bounds
        nx, ny, nz = i2 - i1 + 1, j2 - j1 + 1, k2 - k1 + 1
        npts = nx * ny * nz
        while True:
            time_payload = _read_fortran_record(f)
            if time_payload is None:
                break
            (time_val,) = struct.unpack('<f', time_payload)
            npts_payload = _read_fortran_record(f)
            frame_npts, _nchars = struct.unpack('<2i', npts_payload)
            if frame_npts != npts:
                raise ValueError(f"frame NPTS {frame_npts} != header-derived {npts}")
            rle_payload = _read_fortran_record(f)
            # order='F': FDS is a Fortran program: its arrays are
            # column-major (first index varies fastest in the flat byte
            # stream), unlike numpy's default C order. Verified directly
            # against fdsreader's own decode for this M2.2 pass -- a
            # C-order reshape gave matching nonzero *counts* per frame
            # (same total decoded bytes) but scrambled *positions*,
            # which a same-axis y=0 boundary slice mostly masked (small
            # apparent diff) while an x=const slice exposed clearly.
            levels = decode_rle(rle_payload, npts).reshape((nx, ny, nz), order='F')
            times.append(time_val)
            frames.append(levels)
    return np.asarray(times, dtype=np.float64), bounds, np.stack(frames, axis=0)


def read_sz_upper_bounds(path: str) -> np.ndarray:
    """Per-frame rescale factor from the `.sz` text sidecar (`time npts
    nchars upper_bound`, whitespace-separated, no header row)."""
    bounds = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            bounds.append(float(parts[3]))
    return np.asarray(bounds, dtype=np.float32)


@dataclass(frozen=True)
class Smoke3DInfo:
    quantity: str
    label: str
    units: str
    filename: str
    mesh_id: int  # 0-indexed, matching fds.slice.slice's SLCF convention


def read_smoke3d_infos(smv_path: str) -> list:
    """Every SMOKF3D/SMOKG3D declaration in a `.smv` file (both keywords
    share the same binary `.s3d`/`.sz` format -- confirmed in the M2.1
    spike; SMOKF3D vs SMOKG3D reflects the FDS output category, not a
    format difference)."""
    import mmap

    infos = []
    with open(smv_path, 'r') as infile:
        with mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as s:
            for keyword in (b'SMOKF3D', b'SMOKG3D'):
                # mmap.find()'s start defaults to the object's *current*
                # position, not 0 -- verified directly (not assumed from
                # docs) after this cost a real bug: the SMOKF3D loop's own
                # seek()/readline() calls left the position past most of
                # the file, so an unqualified find() for SMOKG3D next
                # silently missed all but the last mesh's declarations.
                cpos = s.find(keyword, 0)
                while cpos > 0:
                    s.seek(cpos)
                    header_line = s.readline().split()
                    mesh_id = int(header_line[1]) - 1
                    filename = s.readline().decode().strip()
                    quantity = s.readline().decode().strip()
                    label = s.readline().decode().strip()
                    units = s.readline().decode().strip()
                    infos.append(Smoke3DInfo(quantity, label, units, filename, mesh_id))
                    cpos = s.find(keyword, cpos + 1)
    return infos


_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

# For each slicing axis, which of the two remaining physical axes plots
# as the plane's row (vertical) vs column (horizontal) -- z is preferred
# as "vertical" whenever it survives, matching the app's existing
# top-down XZ slice convention (row 0 = ceiling).
_PLANE_AXES = {'x': (2, 1), 'y': (2, 0), 'z': (1, 0)}  # axis -> (row_axis, col_axis)


def y0_mesh_ids(mesh_collection, tol: float = 1e-9) -> list:
    """Back-compat alias for boundary_mesh_ids(mesh_collection, 'y', 0.0)
    -- kept so M2.1 code/tests written against the fixed y=0 plane don't
    need to change."""
    ids, _local_index = boundary_mesh_ids(mesh_collection, 'y', 0.0, tol)
    return ids


def boundary_mesh_ids(mesh_collection, axis: str, offset: float, tol: float = 1e-9) -> tuple:
    """M2.2 (any-plane slicing): 0-indexed ids of every mesh whose face
    along `axis` sits at `offset`, plus which local node-index (0 or -1)
    that face is at. Prefers each mesh's *min* face; only falls back to
    the *max* face if no mesh has a min face there (true at the domain's
    own outer max boundary, where no further mesh exists past it) --
    picking one consistent side avoids decoding + stitching the same
    physical plane twice from both neighbors of an interior boundary."""
    idx = _AXIS_INDEX[axis]
    min_matches = [i for i, m in enumerate(mesh_collection.meshes)
                   if abs(m.ranges[idx][0] - offset) < tol]
    if min_matches:
        return min_matches, 0
    max_matches = [i for i, m in enumerate(mesh_collection.meshes)
                   if abs(m.ranges[idx][1] - offset) < tol]
    return max_matches, -1


def _plane_layout(root_dir: str, axis: str, offset: float):
    """Shared geometry setup for extract_soot_plane / soot_plane_geometry
    (M2.2): resolves the target submeshes, plane axes, physical extent,
    grid spacings, and cell counts -- everything except the (expensive)
    per-submesh RLE data decode. Returns a dict of the pieces both
    callers need."""
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be one of {sorted(_AXIS_INDEX)}, got {axis!r}")

    smv_fn = fds_slice.scanDirectory(root_dir)
    if smv_fn is None:
        raise FileNotFoundError(f"no .smv file found in {root_dir}")
    smv_path = os.path.join(root_dir, smv_fn)

    mesh_collection = fds_slice.readMeshes(smv_path)
    mesh_ids, local_index = boundary_mesh_ids(mesh_collection, axis, offset)
    if not mesh_ids:
        raise ValueError(f"no mesh face found at {axis}={offset} in {root_dir}")
    target_mesh_ids = set(mesh_ids)

    infos = [info for info in read_smoke3d_infos(smv_path)
              if info.quantity == 'SOOT DENSITY' and info.mesh_id in target_mesh_ids]
    if not infos:
        raise ValueError(f"no SOOT DENSITY SMOKF3D data found at {axis}={offset} in {root_dir}")

    row_axis, col_axis = _PLANE_AXES[axis]
    row0 = min(mesh_collection.meshes[i.mesh_id].ranges[row_axis][0] for i in infos)
    row1 = max(mesh_collection.meshes[i.mesh_id].ranges[row_axis][1] for i in infos)
    col0 = min(mesh_collection.meshes[i.mesh_id].ranges[col_axis][0] for i in infos)
    col1 = max(mesh_collection.meshes[i.mesh_id].ranges[col_axis][1] for i in infos)

    sample_mesh = mesh_collection.meshes[infos[0].mesh_id]
    d_row = sample_mesh.axes[row_axis][1] - sample_mesh.axes[row_axis][0]
    d_col = sample_mesh.axes[col_axis][1] - sample_mesh.axes[col_axis][0]
    n_row = int(round((row1 - row0) / d_row)) + 1
    n_col = int(round((col1 - col0) / d_col)) + 1

    return {
        'mesh_collection': mesh_collection, 'infos': infos, 'local_index': local_index,
        'row_axis': row_axis, 'col_axis': col_axis,
        'extent': (col0, col1, row0, row1),
        'd_row': d_row, 'd_col': d_col, 'n_row': n_row, 'n_col': n_col,
    }


def soot_plane_geometry(root_dir: str, axis: str = 'y', offset: float = 0.0) -> tuple:
    """(mesh, extent, mask)-shaped geometry for a SOOT plane, matching
    fds.slice.slice.readSliceGeometry's return shape so ScenarioStore's
    get_extent() can consume it unchanged (M2.2). extent = [col0, col1,
    row0, row1] physical meters. Computed from `.smv` mesh metadata only
    -- no `.s3d` data decode -- so probe/isotherm coordinate mapping
    never forces a cold volumetric read just to learn the axes, same
    philosophy as readSliceGeometry vs readSlice."""
    layout = _plane_layout(root_dir, axis, offset)
    c0, c1, r0, r1 = layout['extent']
    return None, [c0, c1, r0, r1], None


def extract_soot_plane(root_dir: str, axis: str = 'y', offset: float = 0.0) -> tuple:
    """SOOT DENSITY at an arbitrary axis-aligned plane for one scenario
    (M2.2 -- any-plane slicing, generalizing M2.1's fixed y=0 extraction).
    `offset` must land exactly on a mesh boundary (no interpolation
    between mesh faces) -- same scope limit M2.1 already had for y=0,
    now stated for any axis rather than being implicit.

    Returns (times, extent, frames): extent = (col_min, col_max,
    row_min, row_max) in physical meters; frames: float32, shape
    (n_times, n_row, n_col), kg/m3, row 0 = the row axis's *max*
    coordinate (matches load_data.py's ceiling-first convention when
    the row axis is z; for axis='z' slices the row axis is y and "row 0
    = y max" is the equivalent flip, kept for consistency rather than
    physical meaning).

    Stitches by physical-coordinate placement (mirrors
    fds.slice.slice.combineSliceGeometry), same as M2.1.
    """
    layout = _plane_layout(root_dir, axis, offset)
    mesh_collection = layout['mesh_collection']
    infos = layout['infos']
    local_index = layout['local_index']
    row_axis, col_axis = layout['row_axis'], layout['col_axis']
    col0, col1, row0, row1 = layout['extent']
    d_row, d_col = layout['d_row'], layout['d_col']
    n_row, n_col = layout['n_row'], layout['n_col']

    times = None
    stitched = None
    for info in infos:
        mesh = mesh_collection.meshes[info.mesh_id]
        s3d_path = os.path.join(root_dir, info.filename)
        sz_path = s3d_path + '.sz'
        frame_times, _bounds, levels = read_s3d_series(s3d_path)
        upper_bounds = read_sz_upper_bounds(sz_path)
        n_frames = min(len(frame_times), len(upper_bounds))
        physical = (levels[:n_frames].astype(np.float32) / 255.0
                    * upper_bounds[:n_frames, None, None, None])
        tile = np.take(physical, local_index, axis=1 + _AXIS_INDEX[axis])  # drop the sliced axis

        if times is None:
            times = frame_times[:n_frames]
            stitched = np.zeros((n_frames, n_row, n_col), dtype=np.float32)
        # tile's remaining two axes are in ascending (mesh-axis-index)
        # order; reorder/transpose to (row, col) per _PLANE_AXES.
        remaining = [a for a in (0, 1, 2) if a != _AXIS_INDEX[axis]]
        tile = np.transpose(tile, (0,) + tuple(1 + remaining.index(a) for a in (row_axis, col_axis)))
        row_off = int(round((mesh.ranges[row_axis][0] - row0) / d_row))
        col_off = int(round((mesh.ranges[col_axis][0] - col0) / d_col))
        stitched[:, row_off:row_off + tile.shape[1], col_off:col_off + tile.shape[2]] = tile

    stitched = np.flip(stitched, axis=1)  # row 0 = row-axis max, matching load_data.py's ceiling-first convention
    return times, (col0, col1, row0, row1), stitched


def extract_soot_y0_plane(root_dir: str) -> tuple:
    """Back-compat alias for extract_soot_plane(root_dir, 'y', 0.0) --
    M2.1's original entry point, kept so existing callers/tests are
    unaffected by M2.2's generalization."""
    return extract_soot_plane(root_dir, axis='y', offset=0.0)
