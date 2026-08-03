"""Virtual Device Network (V6-M2): place a virtual instrument at a physical
point and get a physically meaningful, traceable measurement -- turning the
app from "visualizing fields" into "instrumenting a simulation like an
experiment".

Three device types, each a deterministic reduction of existing fields at a
point. No new physics model, no guessed inputs, no AI interpretation:

- thermocouple: samples TEMPERATURE at (x, z); reports the full history,
  its maximum, the peak heating rate, and time-to-threshold for the
  standard 60/100/300 C bands.
- heat_detector: a threshold comparator over the same history -- activates
  the instant temperature (optionally: rate of rise) crosses a set point.
- sprinkler: the standard RTI thermal-response ODE
  dT_link/dt = (sqrt(|u|)/RTI) * (T_gas - T_link), Euler-integrated using
  the local gas temperature and, where registered, the local air-speed
  field (VELOCITY). Falls back to a clearly-labelled reduced model (u held
  at 1.0 m/s) when velocity data isn't available -- it never invents a
  measured velocity.

Reuses QuantityProvider.get/get_extent (so a device reads TEMPERATURE RISE,
a Field-Calculator field, or any other registered quantity exactly like any
other consumer) and measure.probe_value for the bilinear point sample -- no
parallel measurement pipeline. A device's time series is computed once, at
creation/edit time, via `Device.compute()`, and cached on `results`;
playback only ever indexes the cached arrays (see device_panel.py) -- the
same memoize-once-read-many policy as the QuantityProvider itself. Pure
NumPy, Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from slice_key import SliceKey, DEFAULT_DIRECTION, DEFAULT_OFFSET
import measure as mz
import tenability as tn

KINDS = ("thermocouple", "heat_detector", "sprinkler")
KIND_LABELS = {"thermocouple": "Thermocouple", "heat_detector": "Heat detector",
               "sprinkler": "Sprinkler (RTI)"}

TC_THRESHOLDS = (60.0, 100.0, 300.0)   # standard thermocouple report bands, deg C


def probe_series(provider, scenario: int, quantity: str, x: float, z: float,
                 direction: int = DEFAULT_DIRECTION, offset: int = DEFAULT_OFFSET) -> np.ndarray:
    """The full (t,) time series of `quantity` at physical (x, z) on the
    given plane (V6-M5: direction/offset, defaulting to the app's usual
    plane), read through the QuantityProvider -- native, derived, or
    calculated, whatever is registered -- and bilinearly interpolated at
    the point each frame. An unavailable plane raises GatedQuantityError
    (from the provider), never silently falls back to a different one."""
    key = SliceKey(quantity, direction, offset)
    data = np.asarray(provider.get(scenario, key))
    extent = provider.get_extent(scenario, key)
    return np.array([mz.probe_value(data[k], extent, x, z) for k in range(data.shape[0])])


def _crossing_time(series: np.ndarray, fps: int, threshold: float) -> Optional[float]:
    """First time (s) `series` reaches/exceeds `threshold`, or None if it
    never does. A plain threshold crossing -- no interpolation, no guessing."""
    idx = np.where(np.asarray(series) >= threshold)[0]
    return float(idx[0] / fps) if idx.size else None


def _has_velocity(provider, scenario: int, direction: int = DEFAULT_DIRECTION,
                  offset: int = DEFAULT_OFFSET) -> bool:
    try:
        provider.get(scenario, SliceKey("VELOCITY", direction, offset))
        return True
    except Exception:
        return False


# ------------------------------------------------------------- device math
def compute_thermocouple(provider, scenario: int, x: float, z: float, fps: int,
                         direction: int = DEFAULT_DIRECTION, offset: int = DEFAULT_OFFSET) -> dict:
    fps = max(1, fps)
    temp = probe_series(provider, scenario, "TEMPERATURE", x, z, direction, offset)
    n = temp.shape[0]
    time_s = np.arange(n) / fps
    rate = np.gradient(temp) * fps if n >= 2 else np.zeros_like(temp)
    temp_field = temp.reshape(-1, 1, 1)
    fed_heat = tn.fed_heat_dose(temp_field, fps).reshape(-1)
    # V6-M6: full FED (toxic-gas + convected-heat dose) at this point, when
    # CO is available. Gated today (registry 'CARBON MONOXIDE VOLUME
    # FRACTION') -- a clean GatedQuantityError, never a fabricated gas dose.
    fed_full = None
    try:
        co = probe_series(provider, scenario, "CARBON MONOXIDE VOLUME FRACTION",
                          x, z, direction, offset)
        fed_full = tn.full_fed(temp_field, co.reshape(-1, 1, 1), fps).reshape(-1).tolist()
    except Exception:  # noqa: BLE001 - GatedQuantityError today; never fabricate a gas dose
        fed_full = None
    return {
        "time_s": time_s.tolist(),
        "temperature_C": temp.tolist(),
        "max_temperature_C": float(np.max(temp)) if n else 0.0,
        "heating_rate_C_per_s": float(np.max(rate)) if n else 0.0,
        "threshold_times_s": {f"{t:g}": _crossing_time(temp, fps, t) for t in TC_THRESHOLDS},
        "fed_heat": fed_heat.tolist(),
        "fed_full": fed_full,
        "max_fed_heat": float(np.max(fed_heat)) if n else 0.0,
        "max_fed_full": float(np.max(fed_full)) if fed_full else None,
        "basis": "TEMPERATURE probed at (x, z), bilinearly interpolated each frame.",
    }


def compute_heat_detector(provider, scenario: int, x: float, z: float, fps: int,
                          activation_temp: float = 74.0,
                          rise_threshold: Optional[float] = None,
                          direction: int = DEFAULT_DIRECTION, offset: int = DEFAULT_OFFSET) -> dict:
    """Activates the instant TEMPERATURE crosses `activation_temp`, or (if
    `rise_threshold` is set) the instant its rate of rise crosses it in
    deg C/s -- whichever comes first."""
    fps = max(1, fps)
    temp = probe_series(provider, scenario, "TEMPERATURE", x, z, direction, offset)
    n = temp.shape[0]
    time_s = np.arange(n) / fps
    t_temp = _crossing_time(temp, fps, activation_temp)
    t_rise = None
    if rise_threshold is not None and n >= 2:
        rate = np.gradient(temp) * fps
        t_rise = _crossing_time(rate, fps, rise_threshold)
    candidates = [t for t in (t_temp, t_rise) if t is not None]
    activation_time = min(candidates) if candidates else None
    activation_frame = int(round(activation_time * fps)) if activation_time is not None else None
    basis = f"threshold crossing: TEMPERATURE >= {activation_temp:g} C"
    if rise_threshold is not None:
        basis += f", or rate of rise >= {rise_threshold:g} C/s"
    return {
        "time_s": time_s.tolist(),
        "temperature_C": temp.tolist(),
        "activated": activation_time is not None,
        "activation_time_s": activation_time,
        "activation_frame": activation_frame,
        "activation_temperature_C": (float(temp[activation_frame])
                                     if activation_frame is not None else None),
        "basis": basis,
    }


def compute_sprinkler(provider, scenario: int, x: float, z: float, fps: int,
                      rti: float = 100.0, activation_temp: float = 68.0,
                      direction: int = DEFAULT_DIRECTION, offset: int = DEFAULT_OFFSET) -> dict:
    """Standard RTI thermal-response model:

        dT_link/dt = (sqrt(|u|) / RTI) * (T_gas - T_link)

    forward-Euler integrated at the simulation's own frame rate. Uses the
    local VELOCITY field for |u| when it is registered; otherwise falls
    back to a fixed u = 1.0 m/s (a minimum-convection reduced model) and
    labels the result as such -- never invents a measured velocity."""
    fps = max(1, fps)
    rti = max(1e-6, float(rti))
    temp = probe_series(provider, scenario, "TEMPERATURE", x, z, direction, offset)
    n = temp.shape[0]
    time_s = np.arange(n) / fps
    reduced_model = not _has_velocity(provider, scenario, direction, offset)
    u = (np.ones(n) if reduced_model
        else np.abs(probe_series(provider, scenario, "VELOCITY", x, z, direction, offset)))
    dt = 1.0 / fps
    link = np.empty(n)
    link[0] = temp[0] if n else 0.0
    for k in range(1, n):
        d = (np.sqrt(max(u[k - 1], 0.0)) / rti) * (temp[k - 1] - link[k - 1])
        link[k] = link[k - 1] + d * dt
    t_act = _crossing_time(link, fps, activation_temp)
    frame_act = int(round(t_act * fps)) if t_act is not None else None
    basis = ("RTI model: dT_link/dt = (sqrt(|u|)/RTI)*(T_gas - T_link), Euler-"
             f"integrated at {fps} fps, RTI={rti:g} (m·s)^0.5")
    basis += (" -- reduced model: local VELOCITY unavailable, u fixed at 1.0 m/s."
             if reduced_model else " using the local VELOCITY field.")
    return {
        "time_s": time_s.tolist(),
        "temperature_C": temp.tolist(),
        "link_temperature_C": link.tolist(),
        "activated": t_act is not None,
        "activation_time_s": t_act,
        "activation_frame": frame_act,
        "reduced_model": reduced_model,
        "basis": basis,
    }


_COMPUTE = {
    "thermocouple": lambda provider, dev, fps: compute_thermocouple(
        provider, dev.scenario, dev.position[0], dev.position[1], fps,
        direction=dev.direction, offset=dev.offset),
    "heat_detector": lambda provider, dev, fps: compute_heat_detector(
        provider, dev.scenario, dev.position[0], dev.position[1], fps,
        activation_temp=dev.parameters.get("activation_temp_C", 74.0),
        rise_threshold=dev.parameters.get("rise_threshold_C_per_s"),
        direction=dev.direction, offset=dev.offset),
    "sprinkler": lambda provider, dev, fps: compute_sprinkler(
        provider, dev.scenario, dev.position[0], dev.position[1], fps,
        rti=dev.parameters.get("rti", 100.0),
        activation_temp=dev.parameters.get("activation_temp_C", 68.0),
        direction=dev.direction, offset=dev.offset),
}

_DEFAULT_PARAMETERS = {
    "thermocouple": {},
    "heat_detector": {"activation_temp_C": 74.0},
    "sprinkler": {"rti": 100.0, "activation_temp_C": 68.0},
}


def default_parameters(device_type: str) -> dict:
    return dict(_DEFAULT_PARAMETERS.get(device_type, {}))


@dataclass
class Device:
    id: str
    name: str
    type: str                        # thermocouple | heat_detector | sprinkler
    scenario: int                    # which case this device is placed in
    position: tuple                  # physical (x, z), metres
    parameters: dict = field(default_factory=dict)
    results: Optional[dict] = None   # cached compute() output; None until computed
    direction: int = DEFAULT_DIRECTION   # V6-M5: which plane this device reads
    offset: int = DEFAULT_OFFSET

    def compute(self, provider, fps: int) -> None:
        """Evaluate this device's full time series once and cache it on
        `results`. Deterministic and idempotent -- call again after editing
        position/parameters to recompute. Playback never calls this; it only
        reads the cached `results`."""
        self.results = _COMPUTE[self.type](provider, self, fps)

    def n_frames(self) -> int:
        return len(self.results.get("time_s", [])) if self.results else 0

    def state_at(self, frame_index: int) -> dict:
        """The device's readout at one already-computed frame -- current
        temperature and activation state -- for the panel/marker to display
        during playback without recomputing anything."""
        r = self.results or {}
        temp = r.get("temperature_C") or r.get("link_temperature_C")
        i = min(max(frame_index, 0), len(temp) - 1) if temp else 0
        out = {"temperature_C": float(temp[i]) if temp else None}
        if self.type in ("heat_detector", "sprinkler"):
            frame_act = r.get("activation_frame")
            out["active"] = frame_act is not None and i >= frame_act
        return out

    def device_state_series(self) -> list:
        """A per-frame integer state column for CSV export: the count of
        TC_THRESHOLDS exceeded so far (thermocouple), or 0/1 before/after
        activation (heat_detector, sprinkler)."""
        r = self.results or {}
        n = len(r.get("time_s", []))
        if self.type == "thermocouple":
            temp = np.asarray(r.get("temperature_C", []), dtype=float)
            state = np.zeros(n, dtype=int)
            for t in TC_THRESHOLDS:
                state += (temp >= t).astype(int)
            return state.tolist()
        frame = r.get("activation_frame")
        if frame is None:
            return [0] * n
        return [0] * frame + [1] * (n - frame)

    def summary_insight(self):
        """A traceable Insight (V3 model) for this device's headline result
        -- the detector/sprinkler activation, or the thermocouple's peak
        reading. Imported lazily so this module stays optional to import
        from pure-engine test contexts."""
        from insight import Insight
        r = self.results or {}
        if not r:
            return None
        if self.type == "thermocouple":
            return Insight(
                statement=f"Thermocouple {self.name} peaked at {r['max_temperature_C']:.1f} °C",
                category="event", quantity="TEMPERATURE",
                location=tuple(self.position), value=r["max_temperature_C"], unit="°C",
                basis=r["basis"])
        kind = "Heat detector" if self.type == "heat_detector" else "Sprinkler"
        t = r.get("activation_time_s")
        if t is None:
            return Insight(statement=f"{kind} {self.name} did not activate.",
                           category="event", quantity="TEMPERATURE",
                           location=tuple(self.position), basis=r.get("basis", ""))
        if self.type == "heat_detector":
            temp_at = r.get("activation_temperature_C")
        else:
            frame_act = r.get("activation_frame")
            link = r.get("link_temperature_C") or []
            temp_at = float(link[frame_act]) if frame_act is not None and frame_act < len(link) else None
        statement = f"{kind} {self.name} activated at {t:.1f} s"
        if temp_at is not None:
            statement += f" (temperature {temp_at:.1f} °C)"
        return Insight(
            statement=statement, category="event", quantity="TEMPERATURE",
            time_s=float(t), location=tuple(self.position), value=temp_at, unit="°C",
            basis=r.get("basis", ""))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "type": self.type,
                "scenario": int(self.scenario),
                "position": [float(self.position[0]), float(self.position[1])],
                "parameters": dict(self.parameters), "results": self.results,
                "direction": int(self.direction), "offset": int(self.offset)}

    @classmethod
    def from_dict(cls, d: dict) -> "Device":
        pos = d.get("position", [0.0, 0.0])
        return cls(id=str(d.get("id", "")), name=str(d.get("name", "")),
                   type=str(d.get("type", "thermocouple")),
                   scenario=int(d.get("scenario", 0)),
                   position=(float(pos[0]), float(pos[1])),
                   parameters=dict(d.get("parameters", {})),
                   results=d.get("results"),
                   direction=int(d.get("direction", DEFAULT_DIRECTION)),
                   offset=int(d.get("offset", DEFAULT_OFFSET)))


def compare_across_scenarios(device: Device, provider, manifest: list, fps: int) -> list:
    """Evaluate `device` (same type/position/parameters/plane) at every
    scenario in `manifest`, holding everything but scenario fixed -- the
    devices analogue of Zone Statistics' cross-scenario zone_bundle sweep
    (place once, compare everywhere). Returns a list of (entry, computed
    Device | None) pairs; None marks a scenario where this device's plane
    is gated on that dataset (never fabricated)."""
    from dataclasses import replace
    out = []
    for entry in manifest:
        tmp = replace(device, scenario=entry.case_index, results=None)
        try:
            tmp.compute(provider, fps)
            out.append((entry, tmp))
        except Exception:  # noqa: BLE001 - a gated/missing plane on this scenario
            out.append((entry, None))
    return out


def export_csv(dev: Device, path: str) -> None:
    """time_s, temperature_C, device_state -- plus fed_heat/fed_full for a
    thermocouple (V6-M6; fed_full omitted when CO is gated) -- plus a
    metadata header (device type, coordinates, parameters, scenario, basis)
    for traceability. Reuses timeseries.write_series_csv (same CSV
    convention as every other export in the app)."""
    from timeseries import write_series_csv
    r = dev.results or {}
    time_s = np.asarray(r.get("time_s", []), dtype=float)
    temp = np.asarray(r.get("temperature_C", []), dtype=float)
    state = np.asarray(dev.device_state_series(), dtype=float)
    metadata = {
        "device_type": dev.type, "device_name": dev.name,
        "position_x_m": f"{dev.position[0]:.4g}", "position_z_m": f"{dev.position[1]:.4g}",
        "scenario": dev.scenario,
        "plane_direction": dev.direction, "plane_offset": dev.offset,   # V6-M5
    }
    for k, v in dev.parameters.items():
        metadata[f"param_{k}"] = v
    metadata["basis"] = r.get("basis", "")
    series = [("temperature_C", temp), ("device_state", state)]
    if dev.type == "thermocouple" and r.get("fed_heat"):
        series.append(("fed_heat", np.asarray(r["fed_heat"], dtype=float)))
        if r.get("fed_full"):
            series.append(("fed_full", np.asarray(r["fed_full"], dtype=float)))
    write_series_csv(path, "time_s", time_s, series, metadata=metadata)
