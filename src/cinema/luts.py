"""Black-body-inspired fire LUT with an alpha ramp.

Reuses this app's existing 'fds_fire' color stops (colormaps.py) but adds
a per-stop alpha channel: ambient-temperature cells render fully
transparent (revealing a dark cinema backdrop instead of filling a
rectangle), fading in through the app's own hazard-band thresholds
(config.ISOTHERM_LEVELS['TEMPERATURE'] = [60, 100, 300], normalized here
against the same 20-300 C default window colormaps.py's FIRE_NAME stops
are calibrated against) so the fire appears to float in the room rather
than paint over it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import colormaps as _colormaps  # 'fds_fire' stops, same calibration window

# Alpha stops at the same normalized positions as colormaps._FIRE_STOPS
# (0.0 = 20C ambient, 1.0 = 300C): transparent until just past the
# "hazardous" band floor (60C), then ramps to fully opaque by the
# "clearly hot" mark (100C).
_ALPHA_STOPS = [
    (0.000, 0),
    (0.100, 0),
    (0.143, 40),
    (0.286, 190),
    (0.500, 255),
    (1.000, 255),
]


def _build_alpha_ramp(n: int) -> np.ndarray:
    xs = np.array([p for p, _ in _ALPHA_STOPS])
    ys = np.array([a for _, a in _ALPHA_STOPS], dtype=np.float64)
    return np.interp(np.linspace(0.0, 1.0, n), xs, ys)


def build_fire_rgba_lut(n: int = 256) -> np.ndarray:
    """(n, 4) uint8 RGBA lookup table, index 0 = ambient, index n-1 = hottest."""
    cmap = LinearSegmentedColormap.from_list(
        _colormaps.FIRE_NAME, _colormaps._FIRE_STOPS, N=n
    )
    rgba = (np.array(cmap(np.linspace(0.0, 1.0, n))) * 255.0).astype(np.uint8)
    rgba[:, 3] = _build_alpha_ramp(n).astype(np.uint8)
    return rgba


FIRE_RGBA_LUT = build_fire_rgba_lut()
