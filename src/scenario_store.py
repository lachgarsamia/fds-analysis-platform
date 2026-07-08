"""On-demand scenario data access with a bounded in-memory LRU cache.

Replaces the previous eager-load-everything approach: instead of parsing all
24 (or however many) scenarios' FDS slice files at startup, only the scenario
actually being displayed is loaded, and a small number of recently-used
scenarios are kept resident so switching between them doesn't always cost a
full re-parse.
"""

import os
import glob
import logging
import threading
from collections import OrderedDict

import numpy as np

from load_data import load_data, SIM_ROOT

logger = logging.getLogger(__name__)


def list_scenario_folders(sim_root: str = SIM_ROOT) -> list:
    """Return sorted scenario directory paths under sim_root."""
    return sorted(f for f in glob.glob(os.path.join(sim_root, '*')) if os.path.isdir(f))


def build_data_matrix(c: int, d: int, vod: int, voc: int) -> np.ndarray:
    """Map [candle_idx, door_idx, vod_idx, voc_idx] -> linear scenario index.

    This assumes list_scenario_folders() sorts scenario folders in exactly the
    same order as this nested loop counts: first grouped by candle setting,
    then door, then vertical-opening-door, then vertical-opening-candle. This
    holds for the current c<n>_d<n>_vod<n>_voc<n> folder naming (verified
    against fds/sim/ -- note the on-disk folders use 1-indexed candle labels
    c1/c2, not the 0-indexed c0/c1 that fds/generate_sim.py currently writes;
    only the lexicographic *order* matters here, not the literal digit values).
    """
    data_matrix = np.zeros((c, d, vod, voc), dtype=int)
    counter = 0
    for i in range(c):
        for j in range(d):
            for k in range(vod):
                for l in range(voc):
                    data_matrix[i, j, k, l] = counter
                    counter += 1
    return data_matrix


class ScenarioStore:
    """Loads scenario temperature arrays on demand, caching the last `cache_size` used.

    Thread-safe: UpdatePlot (background QThread) and Main (GUI thread) can both
    call get() concurrently without corrupting the cache.
    """

    def __init__(self, folders: list, cache_size: int = 4):
        if not folders:
            raise ValueError("no scenario folders provided")
        self.folders = folders
        self.cache_size = cache_size
        self._cache = OrderedDict()  # scenario_index -> ndarray, ordered least- to most-recently used
        self._lock = threading.Lock()

    @property
    def n_scenarios(self) -> int:
        return len(self.folders)

    def get(self, scenario_index: int) -> np.ndarray:
        """Return the (n_times, n_y, n_x) temperature array for a scenario, loading it if needed."""
        with self._lock:
            cached = self._cache.get(scenario_index)
            if cached is not None:
                self._cache.move_to_end(scenario_index)
                return cached

            data = load_data(self.folders[scenario_index])
            self._cache[scenario_index] = data
            self._cache.move_to_end(scenario_index)
            while len(self._cache) > self.cache_size:
                evicted_index, _ = self._cache.popitem(last=False)
                logger.debug("evicting scenario %d from cache", evicted_index)
            return data
