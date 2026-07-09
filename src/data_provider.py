"""
data_provider.py
-----------------
Wraps the existing lazy-loading data layer (scenario_store.ScenarioStore) so
the GUI never has to know how FDS scenario data is parsed or cached, and
never has to bare-`except` a load failure into a hard `sys.exit`.

Note: an earlier reference version of this module wrapped `load_data.load_all_data`,
which eagerly loaded every scenario into one dense array. That function was
removed when the data layer was reworked to load scenarios on demand (see
scenario_store.py) -- eager loading took 36s and ~450MB just for the array;
lazy loading takes ~2s and ~115MB for the default scenario. This module wraps
ScenarioStore directly so that improvement isn't undone.

If the real dataset (fds/sim/) is not present, a synthetic demo dataset is
generated so the app is still runnable and demonstrable -- clearly labeled
as demo data in the window title and status bar, never silently.
"""

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import N_CANDLES, N_DOORS, N_VOD, N_VOC, FRAMES_PER_SECOND, SCENARIO_CACHE_SIZE
from load_data import check_scenario_count, SIM_ROOT
from scenario_store import ScenarioStore, list_scenario_folders, build_data_matrix


class DataLoadError(Exception):
    """Raised when simulation data cannot be loaded, with a user-facing message."""

    def __init__(self, message: str, technical_detail: str = ""):
        super().__init__(message)
        self.message = message
        self.technical_detail = technical_detail


class ScenarioSource(Protocol):
    """Interface both ScenarioStore and DemoScenarioStore satisfy.

    Lets the controller/view stay agnostic to whether they're driving real
    FDS data or the synthetic fallback.
    """

    def get(self, scenario_index: int) -> np.ndarray: ...


@dataclass
class SimulationData:
    """Container for the loaded dataset handle + metadata the UI needs."""
    store: ScenarioSource
    data_matrix: np.ndarray     # shape: (candles, door, vod, voc) -> case index
    timesteps_per_second: int
    is_demo: bool = False


class DemoScenarioStore:
    """Synthetic heat-map data so the UI can run without the real dataset.

    Produces a smoothly moving hot spot whose intensity/position depends on
    the scenario index, purely so the interface has something plausible to
    render and animate. Generated lazily per scenario (same .get() interface
    as ScenarioStore) rather than all at once.
    """

    def __init__(self, n_scenarios: int, n_timesteps: int = 40, h: int = 80, w: int = 120):
        self.n_scenarios = n_scenarios
        self.n_timesteps = n_timesteps
        self.h = h
        self.w = w
        self._cache = {}

    def get(self, scenario_index: int) -> np.ndarray:
        if scenario_index in self._cache:
            return self._cache[scenario_index]

        rng = np.random.default_rng(42 + scenario_index)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        data = np.zeros((self.n_timesteps, self.h, self.w), dtype=np.float32)
        base_temp = 20 + 15 * (scenario_index % 5)
        cx0 = rng.uniform(self.w * 0.3, self.w * 0.7)
        cy0 = rng.uniform(self.h * 0.3, self.h * 0.7)
        for t in range(self.n_timesteps):
            cx = cx0 + 10 * np.sin(t / 6.0 + scenario_index)
            cy = cy0 + 6 * np.cos(t / 8.0 + scenario_index)
            spread = 12 + 6 * np.sin(t / 10.0)
            data[t] = base_temp + 200 * np.exp(
                -(((xx - cx) ** 2) / (2 * spread ** 2) + ((yy - cy) ** 2) / (2 * spread ** 2)))

        self._cache[scenario_index] = data
        return data


def load_simulation_data(cache_size: int = SCENARIO_CACHE_SIZE) -> SimulationData:
    """Load real FDS scenario data if present under fds/sim/, else fall back to demo data.

    Raises DataLoadError with a user-facing message on unrecoverable failure
    (e.g. fds/sim/ exists but a scenario's files are missing/corrupt).
    """
    try:
        folders = list_scenario_folders()
    except Exception as e:
        raise DataLoadError(
            "Something went wrong while looking for simulation data.",
            f"Original error: {type(e).__name__}: {e}") from e

    if not folders:
        data_matrix = build_data_matrix(N_CANDLES, N_DOORS, N_VOD, N_VOC)
        demo_store = DemoScenarioStore(n_scenarios=N_CANDLES * N_DOORS * N_VOD * N_VOC)
        return SimulationData(store=demo_store, data_matrix=data_matrix,
                               timesteps_per_second=FRAMES_PER_SECOND, is_demo=True)

    try:
        check_scenario_count(len(folders), N_CANDLES, N_DOORS, N_VOD, N_VOC)
        data_matrix = build_data_matrix(N_CANDLES, N_DOORS, N_VOD, N_VOC)
        cache_dir = os.path.join(SIM_ROOT, '.cache')
        store = ScenarioStore(folders, cache_size=cache_size, cache_dir=cache_dir)
        return SimulationData(store=store, data_matrix=data_matrix,
                               timesteps_per_second=FRAMES_PER_SECOND, is_demo=False)
    except Exception as e:
        raise DataLoadError(
            "Something went wrong while loading the simulation data.",
            "Check that the dataset was downloaded and unpacked correctly (git-lfs). "
            "To unpack split archives on Linux/macOS:\n"
            "cat sim.tar.gz.a* > ./sim.tar.gz && tar -xf sim.tar.gz\n\n"
            f"Original error: {type(e).__name__}: {e}") from e
