"""In-app source for M3.2's trained-model predictions (3.2.5).

Duck-types data_provider.ScenarioSource (get/is_cached/get_extent) so it
plugs into the same GridCell/DifferenceView rendering MainWindow already
uses for real scenario data (see main_window.py's _store_for_cell) --
"1x3 grid reuse" per the milestone's own phrasing, not a parallel display
system.

Reads predictions/manifest.json + predictions/<case_index>.npy, written
by ml/rollout.py's export_full_scenario_predictions(). Absent entirely
(is_available False) if that directory/manifest doesn't exist -- e.g.
nobody has run the ml/ training pipeline yet -- mirroring the app's
existing demo-mode-absence convention (experiment browser, analytics
panel) rather than crashing or showing a broken button.
"""

from __future__ import annotations

import json
import os

import numpy as np

from slice_key import DEFAULT_SLICE_KEY

PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predictions")


class PredictionSource:
    """Case indices are whatever ml/rollout.py exported (the 4 held-out
    TEST scenarios) -- not contiguous/0-indexed the way a fresh
    ScenarioStore's would be, but the SAME real case_index the main store
    uses, so a prediction and its ground truth are always looked up with
    the identical index. Loaded eagerly at construction (a handful of
    ~10MB .npy files) so get()/is_cached() are always instant -- no
    background prefetch machinery is needed for it."""

    def __init__(self, real_store, predictions_dir: str = PREDICTIONS_DIR, enabled: bool = True):
        self._real_store = real_store
        self._arrays: dict = {}
        self.case_indices: list = []
        if not enabled:
            return
        manifest_path = os.path.join(predictions_dir, "manifest.json")
        if not os.path.isdir(predictions_dir) or not os.path.exists(manifest_path):
            return
        with open(manifest_path) as f:
            manifest = json.load(f)
        for case_index_str in manifest.get("cases", {}):
            case_index = int(case_index_str)
            npy_path = os.path.join(predictions_dir, f"{case_index}.npy")
            if os.path.exists(npy_path):
                self._arrays[case_index] = np.load(npy_path)
                self.case_indices.append(case_index)
        self.case_indices.sort()

    @property
    def is_available(self) -> bool:
        return bool(self._arrays)

    def get(self, scenario_index: int, key=None) -> np.ndarray:
        # key is accepted (ignored) for ScenarioSource interface parity --
        # predictions are TEMPERATURE-only (ml/rollout.py never exports
        # any other quantity).
        return self._arrays[scenario_index]

    def is_cached(self, scenario_index: int, key=None) -> bool:
        return scenario_index in self._arrays

    def get_extent(self, scenario_index: int, key=None) -> list:
        # Same physical geometry as the real scenario -- only the values
        # differ, never the room/mesh -- so this delegates to the real
        # store rather than re-parsing .smv geometry for an identical
        # answer.
        return self._real_store.get_extent(scenario_index, key or DEFAULT_SLICE_KEY)
