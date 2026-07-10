import os
import logging

import numpy as np

import fds.slice.slice as fds
from slice_key import SliceKey, DEFAULT_SLICE_KEY

logger = logging.getLogger(__name__)

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
    """Load one (quantity, direction, offset) slice for one scenario folder.

    Returns an array of shape (n_times, n_y, n_x).
    """
    data = fds.readDataOnly(root_dir, direction=key.direction, offset=key.offset, quantity=key.quantity)
    data = np.flip(data, axis=1)
    return data


def check_scenario_count(n_scenarios: int, c: int, d: int, vod: int, voc: int):
    """Warn if the folder count on disk doesn't match the assumed factor-level counts."""
    expected = c * d * vod * voc
    if n_scenarios != expected:
        logger.warning(
            "found %d scenario folders in %s but factor levels (c=%d, d=%d, vod=%d, voc=%d) "
            "imply %d scenarios; data_matrix indexing may not match folder contents",
            n_scenarios, SIM_ROOT, c, d, vod, voc, expected)
