"""Shared configuration constants for the scenario design matrix and playback.

Kept in one place so main.py, data_provider.py, and simulation_controller.py
don't each hardcode (or, worse, disagree on) the same values -- this is the
single-source-of-truth fix for the "same magic numbers duplicated in three
places" issue identified during the initial codebase audit.
"""

import colormaps as _colormaps  # noqa: F401 -- registers 'fds_fire'/'fds_flow' with matplotlib before QUANTITY_DISPLAY references them by name

# Scenario design matrix: (candle count, door width, vertical-opening-door mode,
# vertical-opening-candle mode) -- see fds/generate_sim.py and protocol/2019_05_21.tex.
N_CANDLES, N_DOORS, N_VOD, N_VOC = 2, 2, 3, 2

# Default control-panel selection shown when the app starts.
DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC = 0, 1, 0, 0

# FDS writes one slice frame every DT_SLCF=0.25s in the source input deck, i.e. 4 frames/second.
FRAMES_PER_SECOND = 4

# Number of (scenario, quantity) entries kept resident in memory at once by
# ScenarioStore (M2.1 keys the cache on (scenario, SliceKey), not just
# scenario). Sized for the largest grid layout M2.2 offers -- a 2x2 grid can
# show up to 4 distinct (case, key) combos at once -- plus a small buffer so
# switching the active cell's scenario/quantity doesn't immediately evict a
# still-visible grid cell (was 4, single-view-only, pre-M2.2).
SCENARIO_CACHE_SIZE = 6

# Ambient (room) temperature, degrees Celsius -- the color scale's fixed lower
# bound (M1.3.2), so vmin no longer freezes at whatever frame 0 happens to show.
AMBIENT_C = 20.0

# Per-quantity display defaults (M2.1): the colormap, color-scale bounds, and
# plain-language label/unit shown when the user switches which quantity the
# heatmap displays. VELOCITY's slider range/default (1-10, default 2 m/s) is
# an engineering estimate from the on-disk dataset (observed magnitudes
# ~0-4 m/s across sampled scenarios), not a physically-derived bound --
# adjustable via the existing slider same as TEMPERATURE's.
#
# 'fds_fire'/'fds_flow' (GUI modernization pass, colormaps.py) are this
# app's own calibrated colormaps -- black/red/orange/yellow fire
# progression and a blue-to-red flow palette, each calibrated to this
# dataset's real observed range rather than a generic gradient (see
# colormaps.py's own module docstring for the exact calibration). The
# stock options (gist_heat/inferno/viridis/cividis) stay available in the
# View > Colormap menu; these are just the new defaults.
QUANTITY_DISPLAY = {
    'TEMPERATURE': {
        'label': 'Temperature',
        'unit': '°C',
        'cmap': 'fds_fire',
        'vmin': AMBIENT_C,
        'slider_min': 50, 'slider_max': 1000, 'slider_default': 300,
    },
    'VELOCITY': {
        'label': 'Air speed',
        'unit': 'm/s',
        'cmap': 'fds_flow',
        'vmin': 0.0,
        'slider_min': 1, 'slider_max': 10, 'slider_default': 2,
    },
    # SOOT DENSITY (M2.2, from volumetric `.s3d` data) is shown in mg/m3
    # (load_data.SOOT_DISPLAY_SCALE); grayscale reads as smoke. Slider
    # range spans the observed ~0-8000 mg/m3 across sampled scenarios --
    # an engineering estimate like VELOCITY's, adjustable via the slider.
    'SOOT DENSITY': {
        'label': 'Smoke (soot)',
        'unit': 'mg/m³',
        'cmap': 'gray_r',
        'vmin': 0.0,
        'slider_min': 100, 'slider_max': 10000, 'slider_default': 3000,
    },
}

# Default isotherm/contour levels per quantity (M2.6.2), keyed the same way
# as QUANTITY_DISPLAY. TEMPERATURE's are the M1.3s.4 hazard-band proposal
# (docs/spike-parser-validation.md §4: <60°C / 60-300°C / >300°C as general
# fire-safety reference points, not derived from this study's own data --
# pending domain-expert review per that spike's own caveat).
#
# VELOCITY's are "speed bands" (GUI modernization pass), reusing the exact
# same overlay mechanism as TEMPERATURE's isotherms unchanged (SliceView's
# _redraw_isotherms()/set_isotherm_levels() just draw contours at whatever
# levels this dict hands them, with no quantity-specific logic) -- 1/2/3 m/s
# span the real observed range (0-~3.7 m/s across sampled scenarios) and
# cross the default 2 m/s view meaningfully.
ISOTHERM_LEVELS = {
    'TEMPERATURE': [60, 100, 300],
    'VELOCITY': [1.0, 2.0, 3.0],
}
