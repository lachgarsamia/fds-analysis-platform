"""Heat shimmer / refraction pass (FireLab roadmap Phase 2.1e).

The air above a heat source visibly distorts light passing through it --
a temperature-scaled, time-varying noise displacement warps the
composited image, applied only where the frame is actually hot (so cold
air, and the dark backdrop around it, never wobbles). Kept subtle by
design: the roadmap's own guidance is "sells realism at 10% strength,
looks like a broken screen at 50%".
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

_NOISE_SIZE = 128


def _make_tileable_noise(size: int, seed: int) -> np.ndarray:
    """A smooth, wraparound-safe (tileable) noise field: a periodic random
    field with its high frequencies filtered out, so scrolling through it
    with wraparound indexing never shows a seam and looks organic rather
    than static-y."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((size, size))
    freq_y = np.fft.fftfreq(size)[:, None]
    freq_x = np.fft.fftfreq(size)[None, :]
    radius = np.hypot(freq_y, freq_x)
    lowpass = np.exp(-((radius * size / 6.0) ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * lowpass))
    field /= np.max(np.abs(field)) + 1e-9
    return field.astype(np.float32)


_NOISE_A = _make_tileable_noise(_NOISE_SIZE, seed=7)
_NOISE_B = _make_tileable_noise(_NOISE_SIZE, seed=13)

SHIMMER_KNEE_C = 40.0     # degrees above ambient before shimmer starts
SHIMMER_STRENGTH = 1.2    # max pixel displacement at full heat -- deliberately low


def _sample_noise(noise: np.ndarray, shape: tuple, offset: tuple) -> np.ndarray:
    ny, nx = shape
    ys = (np.arange(ny)[:, None] + offset[0]).astype(np.float32) % noise.shape[0]
    xs = (np.arange(nx)[None, :] + offset[1]).astype(np.float32) % noise.shape[1]
    ys = np.broadcast_to(ys, shape)
    xs = np.broadcast_to(xs, shape)
    return map_coordinates(noise, [ys, xs], order=1, mode="wrap")


class HeatShimmer:
    """Owns the scroll offset that advances the noise field each call."""

    def __init__(self):
        self._t = 0.0

    def warp(self, image: np.ndarray, temperature_frame: np.ndarray, ambient_c: float) -> np.ndarray:
        """image: (H, W, C) uint8, any channel count. Returns a warped
        copy of the same shape/dtype."""
        self._t += 1.0
        shape = temperature_frame.shape
        excess = np.clip(temperature_frame - ambient_c - SHIMMER_KNEE_C, 0.0, None)
        heat_factor = np.clip(excess / 200.0, 0.0, 1.0)

        dy = _sample_noise(_NOISE_A, shape, (self._t * 0.6, self._t * 0.15)) * 0.7
        dy += _sample_noise(_NOISE_A, shape, (self._t * 1.3, -self._t * 0.2)) * 0.3
        dx = _sample_noise(_NOISE_B, shape, (self._t * 0.4, self._t * 0.25)) * 0.7
        dx += _sample_noise(_NOISE_B, shape, (-self._t * 0.9, self._t * 0.5)) * 0.3

        dy = dy * heat_factor * SHIMMER_STRENGTH
        dx = dx * heat_factor * SHIMMER_STRENGTH

        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        src_y = np.clip(yy + dy, 0, ny - 1)
        src_x = np.clip(xx + dx, 0, nx - 1)

        out = np.empty_like(image)
        for c in range(image.shape[-1]):
            out[..., c] = map_coordinates(image[..., c], [src_y, src_x], order=1, mode="nearest")
        return out
