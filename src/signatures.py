"""Temporal Signature Engine (V3 Phase 0, Fire Intelligence Layer).

Compresses a scenario's full `(n_times, n_z, n_x)` field into a stack of
per-cell *temporal-aggregate maps* -- the substrate for the Fire MRI
(V3-M1). One pass over the cached array answers, for every location:
when did it first cross each hazard level, what was its peak and when,
how long did it stay dangerous, how fast did it cool, and how much total
thermal exposure did it accumulate.

Pure NumPy, Qt-free, deterministic. Cross-validated against the existing
per-scenario summary statistics (the peak channel's maximum equals
summary_stats.max_temp_c; first-crossing minima equal the time-to-threshold
scalars) -- see tests/test_signatures.py.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from registry import AMBIENT_C, get_quantity
from slice_key import DEFAULT_SLICE_KEY


@dataclass(frozen=True)
class SignatureSet:
    """Named per-cell maps for one (scenario, quantity), each shape
    (n_z, n_x). `channels` keys: 'peak', 'time_of_peak', 'cooling_rate',
    'thermal_dose', and per-level 'first_crossing_<L>' / 'duration_above_<L>'.
    `extent` is the physical (x0, x1, z0, z1)."""
    channels: dict
    extent: tuple
    fps: int
    levels: tuple
    unit: str

    def channel_names(self) -> list:
        return list(self.channels.keys())

    def map(self, name: str) -> np.ndarray:
        return self.channels[name]

    def at_cell(self, row: int, col: int) -> dict:
        """Every channel's value at one cell -- the Fire MRI probe readout."""
        return {name: float(m[row, col]) for name, m in self.channels.items()}


def compute_signatures(data: np.ndarray, extent: tuple, fps: int,
                       levels: tuple, ambient_c: float, unit: str = "") -> SignatureSet:
    """Pure temporal-signature computation. `data` is (n_t, n_z, n_x).
    `levels` are the hazard thresholds to compute first-crossing/duration
    for (typically the quantity's registry hazard_levels)."""
    fps = max(1, fps)
    arr = np.asarray(data, dtype=np.float64)
    n_t = arr.shape[0]
    dt = 1.0 / fps

    channels: dict = {}
    channels["peak"] = arr.max(axis=0)
    peak_idx = arr.argmax(axis=0)
    channels["time_of_peak"] = peak_idx.astype(np.float64) * dt

    # Cooling rate: mean °C/s decrease from each cell's peak to the end of
    # the run (0 where the peak is at the last frame -- no cooling observed).
    t_end = (n_t - 1) * dt
    t_peak = channels["time_of_peak"]
    final = arr[-1]
    span = np.maximum(t_end - t_peak, dt)
    cooling = (channels["peak"] - final) / span
    cooling[peak_idx >= n_t - 1] = 0.0
    channels["cooling_rate"] = np.clip(cooling, 0.0, None)

    # Thermal dose: time-integrated exposure above ambient (°C·s) -- the
    # cumulative a single frame cannot show.
    channels["thermal_dose"] = np.clip(arr - ambient_c, 0.0, None).sum(axis=0) * dt

    for level in levels:
        exceed = arr > level
        ever = exceed.any(axis=0)
        first = exceed.argmax(axis=0).astype(np.float64) * dt
        first[~ever] = np.inf
        channels[f"first_crossing_{level:g}"] = first
        channels[f"duration_above_{level:g}"] = exceed.sum(axis=0) * dt

    return SignatureSet(channels=channels, extent=tuple(extent) if extent is not None else None,
                        fps=fps, levels=tuple(levels), unit=unit)


# ---------------------------------------------------------------- disk cache
def signature_cache_path(cache_dir: str, case_index: int, quantity: str) -> str:
    safe_q = quantity.replace(" ", "_")
    return os.path.join(cache_dir, f"signatures_{case_index}_{safe_q}.npz")


def _cache_fresh(cache_path: str, source_folder: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    sources = (glob.glob(os.path.join(source_folder, "*.sf"))
               + glob.glob(os.path.join(source_folder, "*.smv")))
    if not sources:
        return True  # nothing to invalidate against; trust the cache
    return os.path.getmtime(cache_path) >= max(os.path.getmtime(s) for s in sources)


def load_signatures(store, case_index: int, key, fps: int, cache_dir: str = None,
                    source_folder: str = None, levels: tuple = None) -> SignatureSet:
    """Compute (or load from an mtime-invalidated NPZ cache) the signature
    set for one (scenario, quantity). Mirrors ScenarioStore's own disk-cache
    contract: cache is opt-in via `cache_dir`."""
    key = key or DEFAULT_SLICE_KEY
    qinfo = get_quantity(key.quantity)
    if levels is None:
        levels = qinfo.hazard_levels or (AMBIENT_C * 3,)
    extent = store.get_extent(case_index, key)

    cache_path = None
    if cache_dir is not None:
        cache_path = signature_cache_path(cache_dir, case_index, key.quantity)
        if source_folder is not None and _cache_fresh(cache_path, source_folder):
            try:
                npz = np.load(cache_path, allow_pickle=False)
                channels = {name: npz[name] for name in npz.files if name != "__meta__"}
                return SignatureSet(channels=channels,
                                    extent=tuple(extent) if extent is not None else None,
                                    fps=max(1, fps), levels=tuple(levels), unit=qinfo.unit)
            except (OSError, ValueError, KeyError):
                pass  # corrupted cache -> recompute

    data = store.get(case_index, key)
    sig = compute_signatures(data, extent, fps, levels, AMBIENT_C, unit=qinfo.unit)
    if cache_path is not None:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path, **sig.channels)
        except OSError:
            pass
    return sig
