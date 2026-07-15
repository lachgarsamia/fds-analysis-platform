"""Bloom / emissive glow pass.

Hot zones bleed light like a lens: pixels above a knee threshold seed a
multi-radius Gaussian glow, additively composited back in. Driven by the
same tonemapped intensity used for the FireLUT lookup (not the LUT's own
alpha), so the glow can spill into currently-transparent/ambient pixels --
that spill into the surrounding darkness is what reads as "glow", not a
hard-edged hot silhouette.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

KNEE = 0.45
SIGMAS = (2.0, 6.0, 16.0)
WEIGHTS = (0.5, 0.3, 0.2)
GLOW_TINT = np.array([1.0, 0.55, 0.15], dtype=np.float32)  # warm orange emissive spill


def apply_bloom(rgba: np.ndarray, intensity: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """rgba: (H, W, 4) uint8 from the FireLUT. intensity: (H, W) float in
    [0, 1], the tonemapped value that was looked up in the LUT. Returns a
    same-shape uint8 RGBA with the glow composited in."""
    energy = np.clip((intensity - KNEE) / max(1.0 - KNEE, 1e-6), 0.0, 1.0).astype(np.float32)
    halo = np.zeros_like(energy)
    for sigma, weight in zip(SIGMAS, WEIGHTS):
        halo += gaussian_filter(energy, sigma=sigma) * weight
    halo = np.clip(halo * strength, 0.0, 1.0)

    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    rgb_out = np.clip(rgb + GLOW_TINT * halo[..., None], 0.0, 1.0)
    alpha_out = np.clip(alpha + halo * 0.6, 0.0, 1.0)

    out = np.empty_like(rgba)
    out[..., :3] = (rgb_out * 255.0).astype(np.uint8)
    out[..., 3] = (alpha_out * 255.0).astype(np.uint8)
    return out
