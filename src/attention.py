"""Physics Attention Map (V3-M6, Fire Intelligence Layer).

A per-frame *saliency* field that makes stable regions fade and active
physics glow -- a "look here" cue. It is a normalized weighted sum of
where things are changing: temporal change of temperature (|dT/dt|),
temporal change of air speed (|dV/dt|), the spatial temperature gradient
(|grad T|), and the change in heat-release rate (|dHRR/dt|).

IMPORTANT: this is a **heuristic saliency map, not a physical field**. It
does not measure any single simulated quantity; it highlights activity by
combining rates of change. Every UI surface that shows it says so. Pure
NumPy, Qt-free, deterministic.
"""

from __future__ import annotations

import numpy as np

# Relative weights of the four activity cues. Deliberately simple and
# roughly equal (the HRR term, a per-frame scalar, is down-weighted since
# it modulates the whole frame uniformly rather than pointing anywhere).
DEFAULT_WEIGHTS = {"dtemp": 1.0, "grad": 1.0, "dvel": 1.0, "dhrr": 0.5}


def _norm(x: np.ndarray) -> np.ndarray:
    """Scale by the global maximum so components are comparable and stable
    frames (little change anywhere) stay near zero."""
    m = float(np.max(x)) if x.size else 0.0
    return x / m if m > 0 else np.zeros_like(x)


def attention_series(temp: np.ndarray, velocity: np.ndarray = None,
                     hrr_frames: np.ndarray = None, fps: int = 4,
                     weights: dict = None) -> np.ndarray:
    """Per-frame saliency, shape (n_t, n_z, n_x), values in [0, 1].
    `velocity` (same shape as temp) and `hrr_frames` ((n_t,) HRR per frame)
    are optional -- their cues are simply omitted when absent."""
    weights = weights or DEFAULT_WEIGHTS
    t = np.asarray(temp, dtype=np.float64)
    n_t, n_z, n_x = t.shape
    fps = max(1, fps)

    dtemp = np.abs(np.gradient(t, axis=0)) * fps if n_t > 1 else np.zeros_like(t)
    gz = np.gradient(t, axis=1) if n_z >= 2 else np.zeros_like(t)
    gx = np.gradient(t, axis=2) if n_x >= 2 else np.zeros_like(t)
    grad = np.sqrt(gz ** 2 + gx ** 2)

    sal = weights["dtemp"] * _norm(dtemp) + weights["grad"] * _norm(grad)

    if velocity is not None:
        v = np.asarray(velocity, dtype=np.float64)
        n = min(sal.shape[0], v.shape[0])
        dvel = np.abs(np.gradient(v, axis=0)) * fps if v.shape[0] > 1 else np.zeros_like(v)
        sal = sal[:n] + weights["dvel"] * _norm(dvel)[:n]

    if hrr_frames is not None:
        hrr = np.asarray(hrr_frames, dtype=np.float64)
        dhrr = np.abs(np.gradient(hrr)) * fps if hrr.size > 1 else np.zeros_like(hrr)
        dhrr = _norm(dhrr)
        n = min(sal.shape[0], dhrr.shape[0])
        sal = sal[:n] + weights["dhrr"] * dhrr[:n, None, None]

    return _norm(sal)
