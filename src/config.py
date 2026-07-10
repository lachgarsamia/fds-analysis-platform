"""Shared configuration constants for the scenario design matrix and playback.

Kept in one place so main.py, data_provider.py, and simulation_controller.py
don't each hardcode (or, worse, disagree on) the same values -- this is the
single-source-of-truth fix for the "same magic numbers duplicated in three
places" issue identified during the initial codebase audit.
"""

# Scenario design matrix: (candle count, door width, vertical-opening-door mode,
# vertical-opening-candle mode) -- see fds/generate_sim.py and protocol/2019_05_21.tex.
N_CANDLES, N_DOORS, N_VOD, N_VOC = 2, 2, 3, 2

# Default control-panel selection shown when the app starts.
DEFAULT_CANDLES, DEFAULT_DOOR, DEFAULT_VOD, DEFAULT_VOC = 0, 1, 0, 0

# FDS writes one slice frame every DT_SLCF=0.25s in the source input deck, i.e. 4 frames/second.
FRAMES_PER_SECOND = 4

# Number of scenarios kept resident in memory at once by ScenarioStore.
SCENARIO_CACHE_SIZE = 4

# Ambient (room) temperature, degrees Celsius -- the color scale's fixed lower
# bound (M1.3.2), so vmin no longer freezes at whatever frame 0 happens to show.
AMBIENT_C = 20.0

# Per-quantity display defaults (M2.1): the colormap, color-scale bounds, and
# plain-language label/unit shown when the user switches which quantity the
# heatmap displays. VELOCITY's slider range/default (1-10, default 2 m/s) is
# an engineering estimate from the on-disk dataset (observed magnitudes
# ~0-4 m/s across sampled scenarios), not a physically-derived bound --
# adjustable via the existing slider same as TEMPERATURE's.
QUANTITY_DISPLAY = {
    'TEMPERATURE': {
        'label': 'Temperature',
        'unit': '°C',
        'cmap': 'gist_heat',
        'vmin': AMBIENT_C,
        'slider_min': 50, 'slider_max': 1000, 'slider_default': 300,
    },
    'VELOCITY': {
        'label': 'Air speed',
        'unit': 'm/s',
        'cmap': 'viridis',
        'vmin': 0.0,
        'slider_min': 1, 'slider_max': 10, 'slider_default': 2,
    },
}
