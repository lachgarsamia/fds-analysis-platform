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
from slice_key import SliceKey, DEFAULT_SLICE_KEY

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

    def __init__(self, folders: list, cache_size: int = 4, cache_dir: str = None):
        if not folders:
            raise ValueError("no scenario folders provided")
        self.folders = folders
        self.cache_size = cache_size
        # Disk cache is opt-in (None = disabled) so tests that use fake,
        # nonexistent folder paths don't touch the real filesystem.
        self.cache_dir = cache_dir
        self._cache = OrderedDict()  # scenario_index -> ndarray, ordered least- to most-recently used
        self._lock = threading.Lock()

    @property
    def n_scenarios(self) -> int:
        return len(self.folders)

    def is_cached(self, scenario_index: int, key: SliceKey = DEFAULT_SLICE_KEY) -> bool:
        """Whether (scenario_index, key) is already resident in the
        in-memory LRU cache -- a call to get() would return immediately
        with no disk I/O. Read-only inspection of existing state under the
        existing lock; doesn't change get()'s locking granularity or
        thread-safety (M1.4.4: lets the caller decide whether a scenario
        switch needs a background prefetch before it can redraw without
        blocking)."""
        with self._lock:
            return (scenario_index, key) in self._cache

    def get(self, scenario_index: int, key: SliceKey = DEFAULT_SLICE_KEY) -> np.ndarray:
        """Return the (n_times, n_y, n_x) array for one scenario's slice,
        loading it if needed. `key` defaults to the TEMPERATURE slice every
        scenario was read as before M2.1, so existing single-arg callers
        are unaffected."""
        cache_key = (scenario_index, key)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

            data = self._load_with_disk_cache(scenario_index, key)
            self._cache[cache_key] = data
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                (evicted_index, evicted_key_slice), _ = self._cache.popitem(last=False)
                logger.debug("evicting scenario %d (%s) from cache", evicted_index, evicted_key_slice)
            return data

    def _cache_path(self, folder: str, key: SliceKey) -> str:
        case = os.path.basename(os.path.normpath(folder))
        return os.path.join(self.cache_dir, f"{case}_{key.quantity}_dir{key.direction}_off{key.offset}.npy")

    @staticmethod
    def _is_cache_fresh(folder: str, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        source_files = glob.glob(os.path.join(folder, '*.sf')) + glob.glob(os.path.join(folder, '*.smv'))
        if not source_files:
            return False
        cache_mtime = os.path.getmtime(cache_path)
        newest_source_mtime = max(os.path.getmtime(f) for f in source_files)
        return cache_mtime >= newest_source_mtime

    def _load_with_disk_cache(self, scenario_index: int, key: SliceKey) -> np.ndarray:
        folder = self.folders[scenario_index]
        if self.cache_dir is None:
            return load_data(folder, key)

        cache_path = self._cache_path(folder, key)
        if self._is_cache_fresh(folder, cache_path):
            try:
                return np.load(cache_path, mmap_mode='r')
            except (OSError, ValueError) as e:
                logger.warning("disk cache at %s is unreadable (%s); re-parsing", cache_path, e)

        data = load_data(folder, key)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, data)
        except OSError as e:
            logger.warning("could not write disk cache at %s (%s); continuing without it", cache_path, e)
        return data
