"""Sub-frame interpolation: blends two adjacent stored frames so playback
can be driven at a higher visual refresh rate than the data's native
sample rate (config.FRAMES_PER_SECOND), for smoother-looking motion."""

from __future__ import annotations

import numpy as np


def lerp_frames(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = min(max(t, 0.0), 1.0)
    return a + (b - a) * t
