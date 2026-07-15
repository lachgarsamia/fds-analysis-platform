"""Per-frame temperature -> RGBA rendering (FireLab roadmap, Phase 2 task 1).

EffectsPipeline.render() replaces matplotlib's Normalize+cmap step: it
normalizes against an adaptive (auto-exposure) upper bound, applies a
filmic tone curve so hot cores saturate gracefully instead of clipping,
and looks the result up in the FireLUT's black-body-with-alpha ramp.
"""

from __future__ import annotations

import time

import numpy as np

from cinema.luts import FIRE_RGBA_LUT


class AutoExposure:
    """Camera-iris-style adaptive vmax: an EMA of a high percentile of
    each incoming frame, so faint early plumes stay visible and later
    flashover frames don't clip. `locked=True` freezes vmax (science-mode
    parity / manual slider control)."""

    def __init__(self, vmax_init: float, tau_frames: float = 8.0, percentile: float = 99.5):
        self.vmax = float(vmax_init)
        self._alpha = 1.0 / max(tau_frames, 1.0)
        self._percentile = percentile
        self.locked = False

    def update(self, frame: np.ndarray) -> float:
        if self.locked:
            return self.vmax
        target = float(np.percentile(frame, self._percentile))
        self.vmax += self._alpha * (target - self.vmax)
        return self.vmax


def filmic_tonemap(t: np.ndarray, shoulder: float = 0.6) -> np.ndarray:
    """Reinhard-style shoulder curve on already-normalized [0, 1] input:
    compresses highlights toward 1.0 gracefully instead of clipping, while
    staying close to linear at low values."""
    return t * (1.0 + t * shoulder) / (1.0 + t)


class EffectsPipeline:
    """Owns the auto-exposure state for one view cell; render() turns a
    raw temperature array into an RGBA uint8 image of the same shape."""

    def __init__(self, vmin: float, vmax_init: float):
        self.vmin = float(vmin)
        self.exposure = AutoExposure(vmax_init)
        self.last_cost_ms = 0.0

    def render(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        vmax = self.exposure.update(frame)
        span = max(vmax - self.vmin, 1e-6)
        t = np.clip((frame - self.vmin) / span, 0.0, 1.0)
        t = filmic_tonemap(t)
        idx = (t * (len(FIRE_RGBA_LUT) - 1)).astype(np.uint8)
        rgba = FIRE_RGBA_LUT[idx]
        self.last_cost_ms = (time.perf_counter() - t0) * 1000.0
        return rgba
