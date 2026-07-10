"""Identifies one (quantity, normal-direction, offset) slice within a
scenario's .smv/.sf files -- the unit ScenarioStore now caches on, instead
of the previously-implicit "always TEMPERATURE" assumption.

M2.1: readSliceInfos already parses every slice the .smv file describes
(TEMPERATURE and VELOCITY both exist on disk today, see ROADMAP.md); this
module is the seam that exposes that inventory to the rest of the app
without every caller re-deriving it from fds.slice.slice directly.
"""

from dataclasses import dataclass
from typing import NamedTuple

import fds.slice.slice as fds

# The slice plane every scenario has been read from since M1.1: normal to
# y (direction=1), offset 0. Kept as the default SliceKey so existing
# callers that don't pass one explicitly see unchanged behavior.
DEFAULT_QUANTITY = 'TEMPERATURE'
DEFAULT_DIRECTION = 1
DEFAULT_OFFSET = 0


@dataclass(frozen=True)
class SliceKey:
    """Identifies a readable slice: which physical quantity, which axis
    it's normal to, and at what offset along that axis. Hashable (frozen)
    so it can be used directly as a cache dict key."""
    quantity: str
    direction: int = DEFAULT_DIRECTION
    offset: int = DEFAULT_OFFSET


DEFAULT_SLICE_KEY = SliceKey(DEFAULT_QUANTITY, DEFAULT_DIRECTION, DEFAULT_OFFSET)


class SliceInfo(NamedTuple):
    """A SliceKey plus the human-readable label/units the .smv file itself
    records for it (e.g. VELOCITY -> "vel", "m/s")."""
    key: SliceKey
    label: str
    units: str


def available_slices(root_dir: str) -> list:
    """Inventory every distinct (quantity, direction, offset) slice a
    scenario's .smv file describes, deduplicated (a slice can be split
    across multiple meshes, which readSliceInfos returns as separate
    entries with identical quantity/direction/offset).

    Returns a list of SliceInfo, in first-seen order. Raises the same way
    fds.slice.slice does if root_dir has no .smv file -- callers that want
    a graceful fallback (e.g. demo-data mode with no real scenario on
    disk) should catch that themselves.
    """
    smv_fn = fds.scanDirectory(root_dir)
    if smv_fn is None:
        raise FileNotFoundError(f"no .smv file found in {root_dir}")

    import os
    sc = fds.readSliceInfos(os.path.join(root_dir, smv_fn))

    seen = {}
    for s in sc.slices:
        key = SliceKey(s.quantity, s.norm_direction, s.norm_offset)
        if key not in seen:
            seen[key] = SliceInfo(key, s.label, s.units)
    return list(seen.values())
