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
