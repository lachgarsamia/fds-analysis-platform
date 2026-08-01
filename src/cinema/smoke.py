"""Smoke layer: Tier 1 temperature-derived haze + Tier 2 velocity-advected
dye + Tier 3 real soot density (FireLab roadmap Phase 2.1f tiers 1-2;
Tier 3 added by the continuous soot-density visualization pass).

Tier 1 (velocity_frame=None, soot_frame=None): a persistent density buffer
accumulates max(T - T_source, 0) with decay, advected by a fixed upward
drift + a gentle horizontal sway.

Tier 2 (velocity_frame given, soot_frame=None): the same accumulation, but
advected by a per-pixel field combining the real VELOCITY slice's
magnitude with a direction prior (up, blended with "away from the hot
core" via -grad T). Honest caveat: the stored VELOCITY slice is speed
magnitude only (no u/w components) -- true directional advection would
need M-SIM to add U-VELOCITY/W-VELOCITY slices to fds/template.fds
(flagged as a wishlist item for that milestone, not a blocker here).

Tier 3 (soot_frame given): the real SOOT DENSITY field (smoke_density.py's
same continuous, data-driven normalization the scientific overlay uses),
smoothed by an exponential blend toward it each frame for playback
stability -- no fake production or advection, since FDS's own physics
already computed both where soot was produced and where it moved; Tiers
1-2's simulated production/advection would only distort that real spatial
pattern. Takes priority over Tiers 1-2 whenever real data is available.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

import smoke_density as smd

SOURCE_THRESHOLD_C = 60.0        # config.ISOTHERM_LEVELS['TEMPERATURE'][0] -- reuse the existing hazard-band floor
PRODUCTION_SCALE = 1.0 / 400.0
DECAY = 0.985
MAX_DENSITY = 1.0

BUOYANCY_SPEED = 0.6              # tier 1: fixed upward drift, rows/frame
SWAY_AMPLITUDE = 0.4              # subtle horizontal jitter, columns
VELOCITY_SCALE = 0.5              # tier 2: m/s -> columns-or-rows/frame
UP_BIAS = 0.55                    # tier 2: blend weight of "straight up" vs "away from hot core"

# Tier 3: exponential-moving-average weight toward the real (normalized)
# soot field each frame -- purely a temporal smoothing choice for
# cinematic playback stability, not a physical model; the real field
# itself is the ground truth, this only keeps frame-to-frame changes from
# looking like a hard cut.
REAL_SOOT_BLEND = 0.35

SMOKE_TINT = np.array([120.0, 110.0, 100.0], dtype=np.float32)  # gray-brown haze
SMOKE_OPACITY = 0.75


class SmokeSimulator:
    """Owns one cell's persistent smoke-density buffer across frames."""

    def __init__(self, shape: tuple, ambient_c: float):
        self.buffer = np.zeros(shape, dtype=np.float32)
        self.ambient_c = ambient_c
        self._sway_t = 0
        ny, nx = shape
        self._yy, self._xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

    def step(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray = None,
             soot_frame: np.ndarray = None, soot_ceiling: float = None) -> np.ndarray:
        """soot_frame/soot_ceiling (Tier 3): when given, the buffer is
        blended toward the real, normalized SOOT DENSITY field instead of
        Tiers 1-2's simulated production/advection -- see the module
        docstring for why re-advecting real data would be wrong, not just
        redundant."""
        if soot_frame is not None and soot_ceiling is not None:
            self._blend_toward_real_soot(soot_frame, soot_ceiling)
        else:
            self._advect(temperature_frame, velocity_frame)
            self._produce_and_decay(temperature_frame)
        return self.buffer

    def _blend_toward_real_soot(self, soot_frame: np.ndarray, soot_ceiling: float) -> None:
        # max_alpha=1.0: this buffer's own [0, MAX_DENSITY] scale is
        # applied later by smoke_rgba()'s SMOKE_OPACITY -- the ceiling
        # mapping itself should still span the buffer's full range, not
        # additionally cap it here.
        target = smd.soot_alpha(soot_frame, soot_ceiling, max_alpha=MAX_DENSITY)
        self.buffer = self.buffer * (1.0 - REAL_SOOT_BLEND) + target * REAL_SOOT_BLEND

    def _sway(self) -> float:
        self._sway_t += 1
        return SWAY_AMPLITUDE * float(np.sin(self._sway_t * 0.05))

    def _advect(self, temperature_frame: np.ndarray, velocity_frame: np.ndarray) -> None:
        sway = self._sway()
        if velocity_frame is None:
            vy = np.full_like(self.buffer, -BUOYANCY_SPEED)
            vx = np.full_like(self.buffer, sway)
        else:
            grad_y, grad_x = np.gradient(temperature_frame)
            away_y, away_x = -grad_y, -grad_x  # points away from the hot core
            norm = np.hypot(away_y, away_x) + 1e-6
            dir_y, dir_x = away_y / norm, away_x / norm
            dir_y = dir_y * (1.0 - UP_BIAS) - UP_BIAS  # blend in a constant "straight up" bias
            dnorm = np.hypot(dir_y, dir_x) + 1e-6
            dir_y, dir_x = dir_y / dnorm, dir_x / dnorm
            speed = np.clip(velocity_frame, 0.0, None) * VELOCITY_SCALE + BUOYANCY_SPEED * 0.3
            vy, vx = dir_y * speed, dir_x * speed
            vx = vx + sway

        # Semi-Lagrangian backward trace: the value now at (y, x) came
        # from (y, x) - v one step ago.
        src_y = self._yy - vy
        src_x = self._xx - vx
        self.buffer = map_coordinates(self.buffer, [src_y, src_x], order=1, mode="nearest")

    def _produce_and_decay(self, temperature_frame: np.ndarray) -> None:
        production = np.clip(
            temperature_frame - self.ambient_c - SOURCE_THRESHOLD_C, 0.0, None
        ) * PRODUCTION_SCALE
        self.buffer = np.clip(self.buffer * DECAY + production, 0.0, MAX_DENSITY)


def smoke_rgba(density: np.ndarray) -> np.ndarray:
    """density (H, W) float -> (H, W, 4) uint8, a flat gray-brown tint
    whose alpha follows the density buffer."""
    alpha = np.clip(density * SMOKE_OPACITY, 0.0, 1.0)
    out = np.empty(density.shape + (4,), dtype=np.uint8)
    out[..., 0] = SMOKE_TINT[0]
    out[..., 1] = SMOKE_TINT[1]
    out[..., 2] = SMOKE_TINT[2]
    out[..., 3] = (alpha * 255.0).astype(np.uint8)
    return out


def composite_over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Standard straight-alpha Porter-Duff "A over B", both (H, W, 4)
    uint8. Used to composite the fire layer over the smoke layer, which
    itself sits over the (already-drawn) dark backdrop -- fire glows
    through smoke, smoke occludes the room, matching real depth ordering.
    """
    top_rgb = top[..., :3].astype(np.float32) / 255.0
    top_a = top[..., 3].astype(np.float32) / 255.0
    bot_rgb = bottom[..., :3].astype(np.float32) / 255.0
    bot_a = bottom[..., 3].astype(np.float32) / 255.0

    out_a = top_a + bot_a * (1.0 - top_a)
    out_rgb = top_rgb * top_a[..., None] + bot_rgb * bot_a[..., None] * (1.0 - top_a[..., None])
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = out_rgb / safe_a[..., None]

    out = np.empty_like(top)
    out[..., :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out[..., 3] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
    return out
