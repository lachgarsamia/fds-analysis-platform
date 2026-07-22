"""The Shared Selection Model (V5-M1, Phase 0/1) — Layer 2, the app's
canonical interaction object.

`Selection` is one immutable value describing *what the researcher is looking
at*: scenario, quantity, point, region, height, time, interval, phase, and
comparison state. It is a superset of the fields an `Insight` already carries,
the live viewer's state, and the session's persisted fragments — so every
subsystem can convert to and from it (see the adapters in later steps).

`SelectionContext` is a lightweight read façade so consumers say `ctx.scenario`
/ `ctx.frame(fps)` instead of unpacking the model everywhere; new fields
(`study`, `experiment`, `workspace`, …) can be added later without touching a
single consumer.

`SelectionBus` is the only Qt piece here: one signal, one current value, an
origin-guarded `set` that is a no-op when nothing changed — so a click that
lands on the same state cannot ricochet between panels.

Design: docs/architecture-selection-model.md. `Selection` and
`SelectionContext` are pure/Qt-free and unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

Point = Tuple[float, float]
Region = Tuple[float, float, float, float]
Interval = Tuple[float, float]
Comparison = Tuple[int, int]


@dataclass(frozen=True)
class Selection:
    scenario: Optional[int] = None            # case_index (primary)
    quantity: str = "TEMPERATURE"
    point: Optional[Point] = None             # physical (x, z)
    region: Optional[Region] = None           # physical (x0, x1, z0, z1)
    height: Optional[float] = None            # a chosen z
    time_s: Optional[float] = None            # instant
    interval: Optional[Interval] = None       # (t0, t1)
    phase: Optional[str] = None               # detected-phase name (events.py)
    comparison: Optional[Comparison] = None   # (scenario_a, scenario_b)

    def with_(self, **changes) -> "Selection":
        """A copy with some fields replaced (the model is immutable, so a
        change is always a new value)."""
        return replace(self, **changes)

    def frame(self, fps: int) -> Optional[int]:
        """Frame index for `time_s` at `fps`, or None."""
        if self.time_s is None:
            return None
        return int(round(self.time_s * max(1, fps)))

    def to_dict(self) -> dict:
        """JSON-safe plain values for the session (V5-M1). Only non-default
        fields are written, so an empty selection round-trips to {}."""
        out = {}
        for name in ("scenario", "quantity", "point", "region", "height",
                     "time_s", "interval", "phase", "comparison"):
            value = getattr(self, name)
            if value is not None and not (name == "quantity" and value == "TEMPERATURE"):
                out[name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        d = d or {}
        def _tuple(key):
            v = d.get(key)
            return tuple(v) if isinstance(v, (list, tuple)) else None
        return cls(scenario=d.get("scenario"), quantity=d.get("quantity", "TEMPERATURE"),
                   point=_tuple("point"), region=_tuple("region"), height=d.get("height"),
                   time_s=d.get("time_s"), interval=_tuple("interval"),
                   phase=d.get("phase"), comparison=_tuple("comparison"))

    @classmethod
    def from_insight(cls, insight, scenario=None) -> "Selection":
        """Build a Selection from any Insight-like object (duck-typed, so
        this module imports nothing). Mirror of Insight.to_selection."""
        t = getattr(insight, "time_s", None)
        time_s = float(t) if isinstance(t, (int, float)) else None
        interval = tuple(t) if isinstance(t, tuple) else None
        return cls(scenario=scenario,
                   quantity=getattr(insight, "quantity", None) or "TEMPERATURE",
                   point=getattr(insight, "location", None),
                   region=getattr(insight, "region", None),
                   time_s=time_s, interval=interval)


class SelectionContext:
    """A read-oriented façade over a `Selection` (Layer 2). Consumers depend
    on this, not on the model's fields, so the model can grow without churn.
    Optionally carries the QuantityProvider for the selected quantity."""

    def __init__(self, selection: Selection, provider=None):
        self._selection = selection
        self._provider = provider

    @property
    def selection(self) -> Selection:
        return self._selection

    @property
    def provider(self):
        return self._provider

    # --- convenience accessors ---
    @property
    def scenario(self):
        return self._selection.scenario

    @property
    def quantity(self) -> str:
        return self._selection.quantity

    @property
    def point(self):
        return self._selection.point

    @property
    def region(self):
        return self._selection.region

    @property
    def height(self):
        return self._selection.height

    @property
    def time_s(self):
        return self._selection.time_s

    @property
    def interval(self):
        return self._selection.interval

    @property
    def phase(self):
        return self._selection.phase

    @property
    def comparison(self):
        return self._selection.comparison

    def has_point(self) -> bool:
        return self._selection.point is not None

    def has_region(self) -> bool:
        return self._selection.region is not None

    def has_interval(self) -> bool:
        return self._selection.interval is not None

    def frame(self, fps: int):
        return self._selection.frame(fps)


# ------------------------------------------------------------------ the bus
from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402


class SelectionBus(QObject):
    """The single source of truth for the current `Selection` plus one signal.
    Panels connect to `changed` to react and call `set`/`update` (with their
    own `origin`) to drive. `set` is a no-op when the selection is unchanged,
    and consumers ignore changes whose `origin` is themselves -- together these
    close the feedback-loop failure mode by construction."""

    changed = pyqtSignal(object, object)  # (Selection, origin)

    def __init__(self, selection: Selection = None, parent=None):
        super().__init__(parent)
        self._current = selection if selection is not None else Selection()

    @property
    def current(self) -> Selection:
        return self._current

    def context(self, provider=None) -> SelectionContext:
        """A `SelectionContext` over the current selection."""
        return SelectionContext(self._current, provider)

    def set(self, selection: Selection, origin=None) -> bool:
        """Publish a new selection. Returns True if it changed (and emitted),
        False if it equalled the current one (no emit)."""
        if selection == self._current:
            return False
        self._current = selection
        self.changed.emit(selection, origin)
        return True

    def update(self, origin=None, **fields) -> bool:
        """Partial update: replace some fields of the current selection."""
        return self.set(self._current.with_(**fields), origin)

    def resend(self) -> None:
        """Re-emit the current selection unconditionally (RC polish): lets a
        panel that was hidden (and therefore skipped live updates) catch up to
        the current state the moment it becomes visible."""
        self.changed.emit(self._current, None)
