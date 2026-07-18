import os
import logging

import numpy as np

import fds.slice.slice as fds
from slice_key import SliceKey, DEFAULT_SLICE_KEY, SOOT_QUANTITY, DIRECTION_TO_AXIS

logger = logging.getLogger(__name__)

# SOOT DENSITY is stored in kg/m3 but is tiny (~1e-2 kg/m3 peak for these
# candle fires); the app displays it in mg/m3 (x1e6) so its values and the
# integer display-scale slider read in human-friendly numbers, same as
# TEMPERATURE's degrees. The low-level fds/s3d reader stays in kg/m3
# (its cross-validated unit); only this display-facing loader scales.
SOOT_DISPLAY_SCALE = 1.0e6

# fds/sim/ is resolved relative to this file, not the process cwd, so the
# loader works regardless of where the application is launched from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.join(_SRC_DIR, '..', 'fds', 'sim')

# Deprecated aliases for DEFAULT_SLICE_KEY's fields -- kept because
# ScenarioStore's disk-cache filenames were already built from these names
# before M2.1 (see git history); not worth a filename-format migration for
# a purely-derived cache. Prefer slice_key.DEFAULT_SLICE_KEY in new code.
QUANTITY = DEFAULT_SLICE_KEY.quantity
DIRECTION = DEFAULT_SLICE_KEY.direction
OFFSET = DEFAULT_SLICE_KEY.offset


def load_data(root_dir: str, key: SliceKey = DEFAULT_SLICE_KEY) -> np.ndarray:
    """Load one slice for one scenario folder, shape (n_times, n_row, n_col).

    For SOOT DENSITY (M2.2) the data is a plane extracted from the
    volumetric `.s3d` files (key.plane_pos gives the physical position
    along key.direction's axis) rather than a `.sf` slice -- already
    ceiling-first flipped by extract_soot_plane, so it's not re-flipped
    here, and scaled to mg/m3 for display.
    """
    if key.quantity == SOOT_QUANTITY:
        from fds.s3d.s3d import extract_soot_plane
        axis = DIRECTION_TO_AXIS[key.direction]
        _times, _extent, frames = extract_soot_plane(root_dir, axis=axis, offset=key.plane_pos)
        return frames * SOOT_DISPLAY_SCALE
    data = fds.readDataOnly(root_dir, direction=key.direction, offset=key.offset, quantity=key.quantity)
    data = np.flip(data, axis=1)
    return data


def load_slice_geometry(root_dir: str, key: SliceKey = DEFAULT_SLICE_KEY):
    """Return (mesh, extent, mask) for one slice without reading frame data."""
    if key.quantity == SOOT_QUANTITY:
        from fds.s3d.s3d import soot_plane_geometry
        axis = DIRECTION_TO_AXIS[key.direction]
        return soot_plane_geometry(root_dir, axis=axis, offset=key.plane_pos)
    return fds.readSliceGeometry(root_dir, direction=key.direction, offset=key.offset, quantity=key.quantity)


def check_scenario_count(n_scenarios: int, c: int, d: int, vod: int, voc: int):
    """Warn if the folder count on disk doesn't match the assumed factor-level counts."""
    expected = c * d * vod * voc
    if n_scenarios != expected:
        logger.warning(
            "found %d scenario folders in %s but factor levels (c=%d, d=%d, vod=%d, voc=%d) "
            "imply %d scenarios; data_matrix indexing may not match folder contents",
            n_scenarios, SIM_ROOT, c, d, vod, voc, expected)
