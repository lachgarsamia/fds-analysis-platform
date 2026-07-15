"""Ember/flame particles (FireLab roadmap Phase 2.1g): a sparse,
punctuation-only effect rendered as a separate matplotlib scatter artist
on top of the fire+smoke composite -- not the content itself. Pure NumPy
structure-of-arrays pool, no per-particle Python objects.
"""

from __future__ import annotations

import numpy as np

MAX_PARTICLES = 80
SPAWN_TEMP_KNEE_C = 150.0   # degrees above ambient before a cell can spawn an ember
SPAWN_RATE = 2.0            # expected new particles per step at full heat
LIFETIME_FRAMES = 40.0      # steps; short-lived by design
BUOYANCY = -0.25            # constant upward accel, rows/step^2 (negative = toward row 0/ceiling)
JITTER = 0.15


class EmberParticles:
    """step() advects/culls/spawns; render_arrays() returns matplotlib
    scatter-ready (offsets, sizes, colors), embers are punctuation
    (sparse, tiny, bright, short-lived), not a smoke/fire substitute."""

    def __init__(self, shape: tuple, max_particles: int = MAX_PARTICLES, seed: int = 0):
        self.shape = shape
        self.max_particles = max_particles
        self._rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 2), dtype=np.float32)   # (row, col)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.age = np.zeros((0,), dtype=np.float32)
        self.life = np.zeros((0,), dtype=np.float32)
        self.size = np.zeros((0,), dtype=np.float32)

    def step(self, temperature_frame: np.ndarray, ambient_c: float, velocity_frame: np.ndarray = None) -> None:
        self._advect()
        self._cull()
        self._spawn(temperature_frame, ambient_c, velocity_frame)

    def _advect(self) -> None:
        if len(self.pos) == 0:
            return
        self.vel[:, 0] += BUOYANCY * 0.1
        self.vel += self._rng.normal(0.0, JITTER, size=self.vel.shape).astype(np.float32)
        self.pos = self.pos + self.vel
        self.age += 1.0

    def _cull(self) -> None:
        if len(self.pos) == 0:
            return
        ny, nx = self.shape
        alive = (
            (self.age < self.life)
            & (self.pos[:, 0] >= -2) & (self.pos[:, 0] < ny + 2)
            & (self.pos[:, 1] >= -2) & (self.pos[:, 1] < nx + 2)
        )
        self.pos = self.pos[alive]
        self.vel = self.vel[alive]
        self.age = self.age[alive]
        self.life = self.life[alive]
        self.size = self.size[alive]

    def _spawn(self, temperature_frame: np.ndarray, ambient_c: float, velocity_frame: np.ndarray) -> None:
        room = self.max_particles - len(self.pos)
        if room <= 0:
            return
        excess = np.clip(temperature_frame - ambient_c - SPAWN_TEMP_KNEE_C, 0.0, None)
        total = excess.sum()
        if total <= 0:
            return
        n_spawn = min(room, int(self._rng.poisson(SPAWN_RATE)))
        if n_spawn <= 0:
            return

        probs = (excess / total).ravel()
        flat_idx = self._rng.choice(probs.size, size=n_spawn, p=probs)
        ny, nx = self.shape
        rows = (flat_idx // nx).astype(np.float32)
        cols = (flat_idx % nx).astype(np.float32)
        new_pos = np.stack([rows, cols], axis=1)

        lift = np.full(n_spawn, BUOYANCY, dtype=np.float32)
        if velocity_frame is not None:
            lift = lift - velocity_frame.ravel()[flat_idx].astype(np.float32) * 0.3
        new_vel = np.stack([lift, self._rng.normal(0.0, JITTER, size=n_spawn).astype(np.float32)], axis=1)

        self.pos = np.concatenate([self.pos, new_pos], axis=0)
        self.vel = np.concatenate([self.vel, new_vel], axis=0)
        self.age = np.concatenate([self.age, np.zeros(n_spawn, dtype=np.float32)], axis=0)
        self.life = np.concatenate(
            [self.life, self._rng.uniform(LIFETIME_FRAMES * 0.5, LIFETIME_FRAMES, size=n_spawn).astype(np.float32)],
            axis=0,
        )
        self.size = np.concatenate(
            [self.size, self._rng.uniform(3.0, 8.0, size=n_spawn).astype(np.float32)], axis=0
        )

    def render_arrays(self):
        """(offsets, sizes, colors) for scatter.set_offsets/set_sizes/
        set_facecolor -- offsets in (x, y) = (col, row) order, matching
        what a matplotlib scatter over an imshow-plotted array expects."""
        if len(self.pos) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        life_frac = np.clip(1.0 - self.age / np.maximum(self.life, 1e-6), 0.0, 1.0)
        sizes = self.size * (0.4 + 0.6 * life_frac)
        # White-hot when young, fading through orange to transparent as it cools/dies.
        colors = np.stack([
            np.ones_like(life_frac),
            np.clip(0.35 + 0.65 * life_frac, 0.0, 1.0),
            np.clip(0.15 * life_frac, 0.0, 1.0),
            life_frac,
        ], axis=1)
        offsets = self.pos[:, ::-1]
        return offsets, sizes, colors
