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
from cinema.shimmer import HeatShimmer
from cinema.smoke import SmokeSimulator, composite_over, smoke_rgba

# 1/f flicker amplitude: fraction of tonemapped intensity the pink-noise
# track can add/subtract per frame -- candle-like breathing, not a
# strobing screen. Bumped slightly from the original 0.05 for a more
# visibly "alive" flame per the fire-realism pass.
FLICKER_AMPLITUDE = 0.07

# Bloom strength at hrr_intensity=1.0 (see EffectsPipeline.render).
# Bumped from the original 0.8 -- a stronger glow reads as a hotter,
# more incandescent flame instead of a flat-edged hot spot.
BLOOM_STRENGTH = 1.3

# Ambient backdrop (fire-realism pass): a very dim, warm radial falloff
# behind the fire so it reads as "floating in a dim room" rather than a
# pure black void -- centered low/wide like ambient floor-bounce light,
# composited under smoke and fire.
AMBIENT_STRENGTH = 0.05
AMBIENT_TINT = np.array([46.0, 32.0, 24.0], dtype=np.float32)  # dim warm ember-brown


def _ambient_backdrop(shape: tuple) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    cx, cy = nx / 2.0, ny * 0.85
    dist = np.hypot((xx - cx) / (nx * 0.65), (yy - cy) / (ny * 0.65))
    falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2
    out = np.empty(shape + (4,), dtype=np.uint8)
    out[..., 0] = AMBIENT_TINT[0]
    out[..., 1] = AMBIENT_TINT[1]
    out[..., 2] = AMBIENT_TINT[2]
    out[..., 3] = (falloff * AMBIENT_STRENGTH * 255.0).astype(np.uint8)
    return out


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
        self._shimmer = HeatShimmer()
        self._ambient_backdrop: np.ndarray = None

    def render(self, frame: np.ndarray, hrr_intensity: float = 1.0,
               velocity_frame: np.ndarray = None,
               soot_frame: np.ndarray = None, soot_ceiling: float = None) -> np.ndarray:
        """hrr_intensity: a scenario's current HRR(t) normalized to its own
        peak (1.0 = at-or-near peak), or 1.0 (neutral) if no HRR data is
        available -- scales both the flicker amplitude and the bloom
        strength, so the glow physically tracks the real heat-release
        curve instead of being a constant cosmetic overlay.

        velocity_frame: this cell's VELOCITY data at the same timestep, or
        None -- drives the smoke layer's Tier 2 advection (see
        cinema/smoke.py) when real soot data isn't available; Tier 1
        (fixed upward drift) is used when it's absent too.

        soot_frame/soot_ceiling (continuous soot-density visualization
        pass -- cinema/smoke.py's Tier 3): this cell's real SOOT DENSITY
        data and its normalization ceiling. When given, the smoke layer is
        a smoothed rendering of this *real* field instead of a
        temperature-derived proxy -- velocity_frame's Tier-2 advection is
        then skipped (FDS's own physics already computed where the smoke
        moved; re-advecting it would distort the real spatial pattern)."""
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
        if self._ambient_backdrop is None or self._ambient_backdrop.shape[:2] != frame.shape:
            self._ambient_backdrop = _ambient_backdrop(frame.shape)
        density = self._smoke.step(frame, velocity_frame, soot_frame=soot_frame, soot_ceiling=soot_ceiling)
        composited = composite_over(smoke_rgba(density), self._ambient_backdrop)
        composited = composite_over(fire_rgba, composited)
        composited = self._shimmer.warp(composited, frame, self.vmin)

        self.last_cost_ms = (time.perf_counter() - t0) * 1000.0
        return composited
