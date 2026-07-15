"""Precomputed pink (1/f) noise track for the flicker effect.

A candle-like flame brightens/dims with 1/f ("pink") statistics, not white
noise's harsher frame-to-frame jitter -- generated once at import time and
cycled through per-frame by EffectsPipeline, not recomputed per render().
"""

from __future__ import annotations

import numpy as np


def generate_pink_noise(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic ~1/f noise track of length n, normalized to [-1, 1]."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid divide-by-zero at DC
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = pink / peak
    return pink.astype(np.float32)


FLICKER_TRACK = generate_pink_noise(2048)
