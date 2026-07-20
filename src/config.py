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

# Per-quantity display defaults and hazard bands (M2.1/M2.6) now live in
# the structured quantity registry (registry.py, M0.2) -- the single
# source of truth for label/unit/colormap/scale/hazard-levels/kind.
# QUANTITY_DISPLAY and ISOTHERM_LEVELS are kept here as derived views over
# it so every existing `config.QUANTITY_DISPLAY[q]['cmap']` /
# `config.ISOTHERM_LEVELS.get(q)` / `config.AMBIENT_C` call site is
# unchanged; new code should prefer registry.get_quantity(q).
from registry import AMBIENT_C, QUANTITY_REGISTRY, display_dict, isotherm_dict  # noqa: E402,F401

QUANTITY_DISPLAY = display_dict()
ISOTHERM_LEVELS = isotherm_dict()
