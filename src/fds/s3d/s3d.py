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
            levels = decode_rle(rle_payload, npts).reshape(nx, ny, nz)
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


def y0_mesh_ids(mesh_collection, tol: float = 1e-9) -> list:
    """0-indexed ids of meshes whose Y-min face sits at y=0 -- the M2.1
    spike's finding that only half the domain's submeshes (the ones on
    one side of the y=0 boundary) are needed to extract that plane."""
    return [i for i, m in enumerate(mesh_collection.meshes) if abs(m.ranges[1][0]) < tol]


def extract_soot_y0_plane(root_dir: str) -> tuple:
    """SOOT DENSITY at the y=0 plane for one scenario: (times, extent,
    frames). extent = (x0, x1, z0, z1) in physical meters, matching
    load_data.py's convention. frames: float32 array, shape
    (n_times, n_z, n_x), physical units (kg/m3), row 0 = ceiling (z1) --
    same vertical-flip convention load_data.load_data() applies, so this
    is drop-in compatible with the existing SliceView rendering path
    whenever M2.1's follow-on wires it in.

    Stitches submesh tiles by physical-coordinate placement (mirrors
    fds.slice.slice.combineSliceGeometry's approach) rather than
    assuming a fixed tile order, so node-centered submeshes' one shared
    boundary node per seam is handled the same way the existing .sf
    stitcher already does -- not a new algorithm.
    """
    smv_fn = fds_slice.scanDirectory(root_dir)
    if smv_fn is None:
        raise FileNotFoundError(f"no .smv file found in {root_dir}")
    smv_path = os.path.join(root_dir, smv_fn)

    mesh_collection = fds_slice.readMeshes(smv_path)
    target_mesh_ids = set(y0_mesh_ids(mesh_collection))

    infos = [info for info in read_smoke3d_infos(smv_path)
              if info.quantity == 'SOOT DENSITY' and info.mesh_id in target_mesh_ids]
    if not infos:
        raise ValueError(f"no SOOT DENSITY SMOKF3D data found at y=0 in {root_dir}")

    x0 = min(mesh_collection.meshes[i.mesh_id].ranges[0][0] for i in infos)
    x1 = max(mesh_collection.meshes[i.mesh_id].ranges[0][1] for i in infos)
    z0 = min(mesh_collection.meshes[i.mesh_id].ranges[2][0] for i in infos)
    z1 = max(mesh_collection.meshes[i.mesh_id].ranges[2][1] for i in infos)

    sample_mesh = mesh_collection.meshes[infos[0].mesh_id]
    dx = sample_mesh.axes[0][1] - sample_mesh.axes[0][0]
    dz = sample_mesh.axes[2][1] - sample_mesh.axes[2][0]
    n_x = int(round((x1 - x0) / dx)) + 1
    n_z = int(round((z1 - z0) / dz)) + 1

    times = None
    stitched = None
    for info in infos:
        mesh = mesh_collection.meshes[info.mesh_id]
        s3d_path = os.path.join(root_dir, info.filename)
        sz_path = s3d_path + '.sz'
        frame_times, _bounds, levels = read_s3d_series(s3d_path)
        upper_bounds = read_sz_upper_bounds(sz_path)
        n_frames = min(len(frame_times), len(upper_bounds))
        # j-index 0 is this mesh's Y-min face -- the y=0 boundary, per
        # y0_mesh_ids's selection.
        tile = (levels[:n_frames, :, 0, :].astype(np.float32) / 255.0
                * upper_bounds[:n_frames, None, None])  # (n_frames, nx, nz)

        if times is None:
            times = frame_times[:n_frames]
            stitched = np.zeros((n_frames, n_z, n_x), dtype=np.float32)
        x_off = int(round((mesh.ranges[0][0] - x0) / dx))
        z_off = int(round((mesh.ranges[2][0] - z0) / dz))
        stitched[:, z_off:z_off + tile.shape[2], x_off:x_off + tile.shape[1]] = \
            np.transpose(tile, (0, 2, 1))

    stitched = np.flip(stitched, axis=1)  # row 0 = ceiling (z1), matching load_data.py
    return times, (x0, x1, z0, z1), stitched
