"""Per-frame temperature -> RGBA rendering (FireLab roadmap, Phase 2 task 1).

EffectsPipeline.render() replaces matplotlib's Normalize+cmap step: it
normalizes against an adaptive (auto-exposure) upper bound, applies a
filmic tone curve so hot cores saturate gracefully instead of clipping,
and looks the result up in the FireLUT's black-body-with-alpha ramp.
"""

from __future__ import annotations

import time

import numpy as np

from cinema.bloom import apply_bloom
from cinema.luts import FIRE_RGBA_LUT
from cinema.noise import FLICKER_TRACK
from cinema.smoke import SmokeSimulator, composite_over, smoke_rgba

# 1/f flicker amplitude: fraction of tonemapped intensity the pink-noise
# track can add/subtract per frame -- subtle on purpose (candle-like
# breathing, not a strobing screen).
FLICKER_AMPLITUDE = 0.05

# Bloom strength at hrr_intensity=1.0 (see EffectsPipeline.render).
BLOOM_STRENGTH = 0.8


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
        self._flicker_i = 0
        self._smoke: SmokeSimulator = None

    def render(self, frame: np.ndarray, hrr_intensity: float = 1.0,
               velocity_frame: np.ndarray = None) -> np.ndarray:
        """hrr_intensity: a scenario's current HRR(t) normalized to its own
        peak (1.0 = at-or-near peak), or 1.0 (neutral) if no HRR data is
        available -- scales both the flicker amplitude and the bloom
        strength, so the glow physically tracks the real heat-release
        curve instead of being a constant cosmetic overlay.

        velocity_frame: this cell's VELOCITY data at the same timestep, or
        None -- drives the smoke layer's Tier 2 advection (see
        cinema/smoke.py); Tier 1 (fixed upward drift) is used when it's
        absent."""
        t0 = time.perf_counter()
        vmax = self.exposure.update(frame)
        span = max(vmax - self.vmin, 1e-6)
        t = np.clip((frame - self.vmin) / span, 0.0, 1.0)
        t = filmic_tonemap(t)

        flicker = FLICKER_TRACK[self._flicker_i % len(FLICKER_TRACK)]
        self._flicker_i += 1
        t = np.clip(t * (1.0 + FLICKER_AMPLITUDE * hrr_intensity * flicker), 0.0, 1.0)

        idx = (t * (len(FIRE_RGBA_LUT) - 1)).astype(np.uint8)
        fire_rgba = FIRE_RGBA_LUT[idx]
        fire_rgba = apply_bloom(fire_rgba, t, strength=BLOOM_STRENGTH * hrr_intensity)

        if self._smoke is None or self._smoke.buffer.shape != frame.shape:
            self._smoke = SmokeSimulator(frame.shape, ambient_c=self.vmin)
        density = self._smoke.step(frame, velocity_frame)
        composited = composite_over(fire_rgba, smoke_rgba(density))

        self.last_cost_ms = (time.perf_counter() - t0) * 1000.0
        return composited
