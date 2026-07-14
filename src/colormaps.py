"""Custom colormaps calibrated to this dataset's real temperature/speed
ranges (GUI modernization pass).

Registered under matplotlib's global colormap registry at import time, so
every existing call site that already passes a colormap as a plain string
(config.QUANTITY_DISPLAY, imshow(cmap=...), the View > Colormap menu)
keeps working unchanged -- these just add two more valid names.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------
# TEMPERATURE: "fds_fire" -- black -> deep red -> orange -> pale yellow,
# a fire/blackbody-inspired progression (not literal blackbody radiation
# physics, which wouldn't visibly glow until ~500C -- this is gas
# temperature, not a radiating surface -- but calibrated, not arbitrary).
#
# Calibration: QUANTITY_DISPLAY['TEMPERATURE']['vmin'] is always
# AMBIENT_C=20 and the default view's vmax is slider_default=300 (the
# out-of-the-box view every user sees first), so stops are placed at real
# fractions of THAT 20-300C window, landing on the app's own existing
# fire-safety hazard bands (config.ISOTHERM_LEVELS['TEMPERATURE'] =
# [60, 100, 300], from docs/spike-parser-validation.md's hazard-band
# proposal) rather than an arbitrary even split:
#   20C  (ambient, ISOTHERM band floor)      -> position 0.000 -> black
#   60C  ("hazardous" band starts)           -> position 0.143 -> just-visible red
#   100C (clearly hot)                       -> position 0.286 -> red
#   160C (hazard-band midpoint)              -> position 0.500 -> red-orange
#   220C                                     -> position 0.714 -> orange
#   260C                                     -> position 0.857 -> yellow-orange
#   300C ("severe" band starts, = default vmax) -> position 1.000 -> pale yellow
# The real 24-scenario ensemble's actual peak is 469.3C (verified directly
# against fds/sim/, not assumed) -- well past this window, which is
# expected: dragging the temperature slider above 300 stretches this same
# black->pale-yellow progression across the wider range, exactly as any
# colormap responds to a wider clim.
#
# The top stop is deliberately a pale warm yellow (#FFF3C4), not pure
# white: widgets.py's MplCanvas now fixes the plot background to pure
# white (#FFFFFF) regardless of app theme, so a colormap that also tops
# out at pure white would make the hottest -- most important -- pixels
# visually disappear into the surrounding canvas. Staying just short of
# white keeps peak temperatures visible as data, not background.
FIRE_NAME = "fds_fire"

_FIRE_STOPS = [
    (0.000, "#000000"),
    (0.143, "#380C02"),
    (0.286, "#8B1A00"),
    (0.500, "#E85D04"),
    (0.714, "#FFA200"),
    (0.857, "#FFD166"),
    (1.000, "#FFF3C4"),
]

# -----------------------------------------------------------------------
# VELOCITY: "fds_flow" -- deep blue -> teal -> green -> yellow -> red, a
# conventional flow/motion palette (wind-speed-map style) rather than a
# generic sequential colormap, per the GUI modernization request. No
# directional/vector data exists in this dataset (confirmed by direct
# investigation: FDS's QUANTITY='VELOCITY' record is the unsigned speed
# magnitude, not U/V/W-VELOCITY; the parser's binary layout carries one
# float per cell; real sampled data is 0.0 to ~3.7 m/s with zero negative
# values across every scenario checked) -- this only reads as "faster /
# slower", never as a direction, and doesn't pretend otherwise.
#
# Calibration: vmin is always 0.0 and the default view's vmax is
# slider_default=2 (m/s), so stops are placed at real fractions of that
# 0-2 m/s window -- the out-of-the-box view every user sees first, and
# comfortably covers the observed ~0-3.7 m/s range in the real dataset
# once the vmax slider is opened up a bit further.
FLOW_NAME = "fds_flow"

_FLOW_STOPS = [
    (0.00, "#1A3A8F"),
    (0.25, "#1B7FBF"),
    (0.50, "#1FB37A"),
    (0.75, "#D9C43C"),
    (1.00, "#E8491C"),
]


def build_fire_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(FIRE_NAME, _FIRE_STOPS, N=256)


def build_flow_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(FLOW_NAME, _FLOW_STOPS, N=256)


def register_custom_colormaps() -> None:
    """Idempotent: matplotlib raises if a name is already registered, so
    re-importing this module (e.g. across multiple tests in one process)
    must not crash."""
    for name, builder in ((FIRE_NAME, build_fire_colormap), (FLOW_NAME, build_flow_colormap)):
        if name not in mpl.colormaps:
            mpl.colormaps.register(builder(), name=name)


register_custom_colormaps()
