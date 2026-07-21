"""Frame Descriptor Engine (V3 Phase 0, Fire Intelligence Layer).

Per-*frame* scalar physics (as opposed to signatures.py's per-*cell*
temporal maps): for every timestep, a handful of numbers describing the
state of the fire -- spatial max/mean, hot-area fraction, smoke-layer
height, mean temperature gradient, and their frame-to-frame rates of
change. This `(n_frames, n_descriptors)` table is the raw material for
event detection (events.py), the state-space trajectory (V3-M5), and the
semantic diff (V3-M3).

Pure NumPy, Qt-free, deterministic. Reuses the existing per-frame
reductions (summary_stats' max-by-frame, layer_height's series) rather
than reinventing them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from registry import AMBIENT_C
from layer_height import smoke_layer_height_series

# "Hot" for the hot-area-fraction descriptor -- the same low hazard band
# used across the app (config.ISOTHERM_LEVELS / features.HOT_THRESHOLD_C).
HOT_THRESHOLD_C = 60.0

# The scalar descriptors, in a fixed order (also the state-space vector's
# column order). Rates of change are derived from these, appended below.
BASE_DESCRIPTORS = ("spatial_max", "spatial_mean", "hot_area_fraction",
                    "layer_height", "gradient_mag")


@dataclass(frozen=True)
class DescriptorTable:
    """Per-frame descriptor columns for one scenario. `columns` maps each
    descriptor name to a (n_frames,) array; `times` is the frame time axis
    (seconds)."""
    columns: dict
    times: np.ndarray
    fps: int

    @property
    def n_frames(self) -> int:
        return len(self.times)

    def column(self, name: str) -> np.ndarray:
        return self.columns[name]

    def as_matrix(self, names: list = None) -> np.ndarray:
        """(n_frames, n_selected) matrix for state-space embedding."""
        names = names or list(self.columns.keys())
        return np.column_stack([self.columns[n] for n in names])


def _rate(series: np.ndarray, fps: int) -> np.ndarray:
    """d/dt of a per-frame series (same length; forward difference, last
    value repeated)."""
    diff = np.diff(series) * fps
    if diff.size == 0:
        return np.zeros_like(series)
    return np.append(diff, diff[-1])


def compute_descriptors(data: np.ndarray, extent: tuple, fps: int,
                        ambient_c: float = AMBIENT_C,
                        hot_threshold_c: float = HOT_THRESHOLD_C) -> DescriptorTable:
    """Pure per-frame descriptor computation. `data` is (n_t, n_z, n_x)."""
    fps = max(1, fps)
    arr = np.asarray(data, dtype=np.float64)
    n_t = arr.shape[0]
    times = np.arange(n_t) / fps

    cols: dict = {}
    cols["spatial_max"] = arr.max(axis=(1, 2))
    cols["spatial_mean"] = arr.mean(axis=(1, 2))
    cols["hot_area_fraction"] = (arr > hot_threshold_c).mean(axis=(1, 2))
    if extent is not None:
        cols["layer_height"] = smoke_layer_height_series(arr, extent, ambient_c)
    else:
        cols["layer_height"] = np.zeros(n_t)
    # Mean in-plane gradient magnitude per frame (a proxy for structure /
    # front sharpness -- feeds the attention map and regime detection).
    # Guard degenerate axes (np.gradient needs >= 2 samples along an axis).
    gz = np.gradient(arr, axis=1) if arr.shape[1] >= 2 else np.zeros_like(arr)
    gx = np.gradient(arr, axis=2) if arr.shape[2] >= 2 else np.zeros_like(arr)
    cols["gradient_mag"] = np.sqrt(gz ** 2 + gx ** 2).mean(axis=(1, 2))

    # Rates of change, appended as their own descriptors.
    for name in ("spatial_max", "spatial_mean", "layer_height"):
        cols[f"d_{name}"] = _rate(cols[name], fps)

    return DescriptorTable(columns=cols, times=times, fps=fps)
