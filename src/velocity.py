"""True 3D velocity: streamlines / quiver / volume (V6-M3, the lead V6-1
capability). Gated on U-VELOCITY/W-VELOCITY -- the current FDS output only
has VELOCITY (speed magnitude, no direction); the in-plane vector field
needs the two signed components the M-SIM re-run adds (see
docs/msim-preparation.md section 3). This module never fabricates a
direction from the magnitude alone: every entry point reads through
QuantityProvider.get_vector(), which raises GatedQuantityError until the
real components exist -- that exception is never caught here, only in the
UI layer (velocity_panel.py), which surfaces it as a clear, honest status
instead of a broken plot.

VectorField is the compute-once/read-many engine, mirroring devices.py's
Device: `compute()` reads (U, W) once and derives speed/angle over the
whole (t, z, x) stack in one vectorized pass; `quiver_at()` is then a plain
array slice (O(1) per frame) and `streamline_at()` is a per-frame-memoized
integration (a first visit to a frame costs one bounded RK integration,
repeats are a dict lookup) -- so nothing here recomputes the underlying
field on every GUI tick.

Pure NumPy, Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from slice_key import SliceKey, DEFAULT_DIRECTION, DEFAULT_OFFSET
import measure as mz

METHODS = ("euler", "rk2", "rk4")
MODES = ("quiver", "streamlines", "both")
COLOR_BY = ("speed", "direction", "temperature")


def stride_for_target(shape: tuple, target_count: int) -> int:
    """Grid stride so a stride-subsampled (n_z, n_x) grid has approximately
    `target_count` points -- the "adaptive density" quiver control."""
    n_z, n_x = shape
    total = max(1, n_z * n_x)
    return max(1, int(round((total / max(1, target_count)) ** 0.5)))


def quiver_grid(u_frame: np.ndarray, w_frame: np.ndarray, extent: tuple, stride: int):
    """(xs, zs, us, ws) for every `stride`-th grid point, in physical
    coordinates -- a plain grid subsample. Positions are deterministic from
    `stride` alone, so a caller can keep them fixed across frames and only
    refresh U/V (see SliceView.set_vector_field)."""
    n_z, n_x = u_frame.shape
    rows = np.arange(0, n_z, max(1, stride))
    cols = np.arange(0, n_x, max(1, stride))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    x0, x1, z0, z1 = extent
    xs = x0 + (cc / max(n_x - 1, 1)) * (x1 - x0)
    zs = z1 - (rr / max(n_z - 1, 1)) * (z1 - z0)
    return xs.ravel(), zs.ravel(), u_frame[rr, cc].ravel(), w_frame[rr, cc].ravel()


def _uv_at(u_frame, w_frame, extent, x, z):
    return mz.probe_value(u_frame, extent, x, z), mz.probe_value(w_frame, extent, x, z)


def _euler_step(u_frame, w_frame, extent, x, z, h):
    u, w = _uv_at(u_frame, w_frame, extent, x, z)
    return x + h * u, z + h * w


def _rk2_step(u_frame, w_frame, extent, x, z, h):
    u1, w1 = _uv_at(u_frame, w_frame, extent, x, z)
    u2, w2 = _uv_at(u_frame, w_frame, extent, x + h * u1, z + h * w1)
    return x + h * 0.5 * (u1 + u2), z + h * 0.5 * (w1 + w2)


def _rk4_step(u_frame, w_frame, extent, x, z, h):
    k1u, k1w = _uv_at(u_frame, w_frame, extent, x, z)
    k2u, k2w = _uv_at(u_frame, w_frame, extent, x + h / 2 * k1u, z + h / 2 * k1w)
    k3u, k3w = _uv_at(u_frame, w_frame, extent, x + h / 2 * k2u, z + h / 2 * k2w)
    k4u, k4w = _uv_at(u_frame, w_frame, extent, x + h * k3u, z + h * k3w)
    dx = h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
    dz = h / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return x + dx, z + dz


_STEPPERS = {"euler": _euler_step, "rk2": _rk2_step, "rk4": _rk4_step}


def integrate_streamline(u_frame: np.ndarray, w_frame: np.ndarray, extent: tuple, seed: tuple,
                          step: float = 0.05, max_steps: int = 200, max_length: float = 5.0,
                          method: str = "rk4") -> list:
    """A single frozen-frame streamline from `seed` (x, z) -- the in-plane
    (u, w) field integrated with a fixed-step method (Euler/RK2/RK4),
    stopping at the domain edge, `max_length`, `max_steps`, or a near-zero
    local speed (a stalled point). 2D/in-plane only -- see module docstring
    and the V6-M3 PR limitations for the 3D extension this doesn't cover."""
    if method not in _STEPPERS:
        raise ValueError(f"unknown integration method: {method!r} (expected one of {METHODS})")
    stepper = _STEPPERS[method]
    x0, x1, z0, z1 = extent
    x, z = float(seed[0]), float(seed[1])
    points = [(x, z)]
    length = 0.0
    for _ in range(max(0, max_steps)):
        u, w = _uv_at(u_frame, w_frame, extent, x, z)
        if float(np.hypot(u, w)) < 1e-6:
            break
        nx, nz = stepper(u_frame, w_frame, extent, x, z, step)
        seg = float(np.hypot(nx - x, nz - z))
        if length + seg > max_length:
            break
        if not (min(x0, x1) <= nx <= max(x0, x1) and min(z0, z1) <= nz <= max(z0, z1)):
            break
        x, z = nx, nz
        length += seg
        points.append((x, z))
    return points


_STREAMLINE_CACHE_MAX = 64


class VectorField:
    """The whole-plane (U, W) vector field for one scenario/plane -- computed
    once (`compute()`) and read many times. `quiver_at()` is O(1) array
    indexing; `streamline_at()` is a bounded, per-frame-memoized integration.
    Duck-typed on `provider` (needs `.get_vector(scenario, direction, offset)`
    and `.get_extent(scenario, key)`) so tests can substitute a lightweight
    fake without any Qt/gating machinery."""

    def __init__(self, provider, scenario: int, direction: Optional[int] = None,
                 offset: Optional[int] = None):
        self._provider = provider
        self.scenario = scenario
        self.direction = DEFAULT_DIRECTION if direction is None else direction
        self.offset = DEFAULT_OFFSET if offset is None else offset
        self.u: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.speed: Optional[np.ndarray] = None
        self.angle: Optional[np.ndarray] = None
        self.extent = None
        self._streamline_cache: dict = {}

    def compute(self) -> None:
        """Reads U/W once via QuantityProvider.get_vector -- raises
        GatedQuantityError (propagated, never caught here) when the
        components aren't available. Never fabricates a fallback field."""
        u, w = self._provider.get_vector(self.scenario, self.direction, self.offset)
        self.u = np.asarray(u, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.speed = np.hypot(self.u, self.w)      # vectorized once -- O(1) per frame after this
        self.angle = np.arctan2(self.w, self.u)
        # Extent comes from VELOCITY (same plane, always real/non-gated),
        # never from U-VELOCITY/W-VELOCITY themselves: those have no slice
        # geometry on disk until the M-SIM re-run, and reading geometry for
        # a quantity that was never part of the manifest risks a hard crash
        # deep in the slice-file parser rather than a clean exception.
        self.extent = self._provider.get_extent(
            self.scenario, SliceKey("VELOCITY", self.direction, self.offset))
        self._streamline_cache.clear()

    @property
    def n_frames(self) -> int:
        return int(self.u.shape[0]) if self.u is not None else 0

    def quiver_at(self, frame_index: int, density: int = 400):
        """(xs, zs, us, ws) at `frame_index` -- the position grid depends
        only on `density`, so a caller can keep it fixed across frames and
        just refresh U/V (see SliceView.set_vector_field)."""
        i = min(max(frame_index, 0), self.n_frames - 1)
        stride = stride_for_target(self.u.shape[1:], density)
        return quiver_grid(self.u[i], self.w[i], self.extent, stride)

    def probe_speed(self, x: float, z: float) -> np.ndarray:
        """The (t,) local speed series at physical (x, z) -- same probe
        convention as devices.py, for a seed's own readout/export."""
        return np.array([mz.probe_value(self.speed[k], self.extent, x, z)
                         for k in range(self.n_frames)])

    def probe_angle(self, x: float, z: float) -> np.ndarray:
        return np.array([mz.probe_value(self.angle[k], self.extent, x, z)
                         for k in range(self.n_frames)])

    def streamline_at(self, seed: tuple, frame_index: int, step: float = 0.05,
                       max_steps: int = 200, max_length: float = 5.0,
                       method: str = "rk4") -> list:
        """A frozen-frame streamline from `seed`, memoized per (seed, frame,
        params): most playback/scrubbing revisits a small working set of
        frames, so repeats are a dict lookup; a first visit to a frame costs
        one bounded integration, never a whole-field recompute."""
        i = min(max(frame_index, 0), self.n_frames - 1)
        key = (round(seed[0], 4), round(seed[1], 4), i, step, max_steps, max_length, method)
        cached = self._streamline_cache.get(key)
        if cached is not None:
            return cached
        points = integrate_streamline(self.u[i], self.w[i], self.extent, seed,
                                      step=step, max_steps=max_steps,
                                      max_length=max_length, method=method)
        self._streamline_cache[key] = points
        while len(self._streamline_cache) > _STREAMLINE_CACHE_MAX:
            self._streamline_cache.pop(next(iter(self._streamline_cache)))
        return points


@dataclass
class VectorProbe:
    """A seed point placed on the vector field (V6-M3), mirroring devices.py's
    Device: a local reading, computed once from an already-computed
    VectorField and cached on `results`; playback only ever indexes it (via
    the field's own `streamline_at`, itself memoized). `results` is
    `{"gated": True, "reason": ...}` -- never a fabricated series -- when
    U/W aren't available for this probe's scenario."""
    id: str
    name: str
    scenario: int
    position: tuple
    results: Optional[dict] = None

    def compute(self, vector_field: "VectorField", fps: int) -> None:
        """Precomputes this probe's local speed/angle series from an
        already-computed VectorField -- probe_speed/probe_angle are cheap,
        O(n_frames) bilinear lookups, not a recompute of U/W. Called once at
        placement/scenario-change, never per GUI tick."""
        speed = vector_field.probe_speed(*self.position)
        angle = vector_field.probe_angle(*self.position)
        n = len(speed)
        fps = max(1, fps)
        self.results = {
            "time_s": (np.arange(n) / fps).tolist(),
            "speed_m_s": speed.tolist(),
            "angle_deg": np.degrees(angle).tolist(),
            "max_speed_m_s": float(np.max(speed)) if n else 0.0,
            "basis": "U/W-VELOCITY probed at (x, z), bilinearly interpolated each frame.",
        }

    def mark_gated(self, reason: str) -> None:
        """No U/W for this probe's scenario -- record why, fabricate nothing."""
        self.results = {"gated": True, "reason": reason}

    @property
    def gated(self) -> bool:
        return bool(self.results and self.results.get("gated"))

    def state_at(self, frame_index: int) -> dict:
        """This probe's readout at an already-computed frame -- current
        speed, for the panel/marker to display during playback without
        recomputing anything."""
        if not self.results or self.gated:
            return {"speed_m_s": None}
        speed = self.results["speed_m_s"]
        i = min(max(frame_index, 0), len(speed) - 1) if speed else 0
        return {"speed_m_s": float(speed[i]) if speed else None}

    def summary_insight(self):
        """A traceable Insight (V3 model) for this probe -- its peak speed,
        or the gate reason if U/W aren't available. Imported lazily so this
        module stays optional to import from pure-engine test contexts."""
        from insight import Insight
        if self.gated:
            return Insight(
                statement=f"Vector probe {self.name}: gated -- {self.results['reason']}",
                category="event", quantity="VELOCITY", location=tuple(self.position),
                basis=self.results["reason"])
        r = self.results or {}
        return Insight(
            statement=f"Vector probe {self.name}: peak speed {r.get('max_speed_m_s', 0.0):.1f} m/s",
            category="event", quantity="VELOCITY", location=tuple(self.position),
            value=r.get("max_speed_m_s"), unit="m/s", basis=r.get("basis", ""))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "scenario": int(self.scenario),
                "position": [float(self.position[0]), float(self.position[1])],
                "results": self.results}

    @classmethod
    def from_dict(cls, d: dict) -> "VectorProbe":
        pos = d.get("position", [0.0, 0.0])
        return cls(id=str(d.get("id", "")), name=str(d.get("name", "")),
                   scenario=int(d.get("scenario", 0)),
                   position=(float(pos[0]), float(pos[1])), results=d.get("results"))


def export_csv(probe: VectorProbe, path: str) -> None:
    """time_s, speed_m_s, angle_deg -- plus a metadata header (coordinates,
    scenario, basis) for traceability. Reuses timeseries.write_series_csv
    (same CSV convention as devices.py's export)."""
    from timeseries import write_series_csv
    r = probe.results or {}
    metadata = {
        "probe_name": probe.name,
        "position_x_m": f"{probe.position[0]:.4g}", "position_z_m": f"{probe.position[1]:.4g}",
        "scenario": probe.scenario,
    }
    if probe.gated:
        metadata["gated"] = True
        metadata["reason"] = r.get("reason", "")
        write_series_csv(path, "time_s", np.asarray([]), [], metadata=metadata)
        return
    metadata["basis"] = r.get("basis", "")
    time_s = np.asarray(r.get("time_s", []), dtype=float)
    speed = np.asarray(r.get("speed_m_s", []), dtype=float)
    angle = np.asarray(r.get("angle_deg", []), dtype=float)
    write_series_csv(path, "time_s", time_s,
                     [("speed_m_s", speed), ("angle_deg", angle)], metadata=metadata)


def volume_sample(provider, scenario: int, x: float, y: float, z: float):
    """V6-M3 seam, NOT implemented: a volumetric (U, V, W) sample at a true
    3D point, the way load_data.py's extract_soot_plane() reads volumetric
    SOOT DENSITY from `.s3d`. No `.s3d` U/V/W output exists yet -- FDS would
    need a volumetric velocity dump docs/msim-preparation.md doesn't
    currently request. Raises to make the seam explicit (prepared, not
    surfaced) rather than silently returning nothing or a fabricated value."""
    raise NotImplementedError(
        "volumetric velocity sampling is not available -- only in-plane U/W "
        "slices are prepared (see docs/msim-preparation.md section 3). A "
        "volumetric path would extend load_data.py's .s3d dispatch the way "
        "SOOT DENSITY does, once FDS volumetric velocity output exists.")
