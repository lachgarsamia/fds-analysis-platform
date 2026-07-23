"""QuantityProvider (V5-M1 / Phase 0) — the Layer-1 computation layer.

Downstream code asks the provider for a field, never `ScenarioStore` directly:

    ScenarioStore
         │
    QuantityProvider.get(scenario, key)
         │
     ┌───┴──────────┬───────────────┐
    Raw          Derived          Future
    Temperature  dT/dt            FED
    Velocity     Gradient         Smoke toxicity
                 Thermal dose     M-SIM quantities

Raw quantities pass straight through to the store (unchanged). A *derived*
quantity (registry `kind == "derived"`) is computed from its source field via
`derived_quantities.derive`, so a new derived quantity is just a registry entry
+ a function and every tool supports it for free.

This module **wraps** the store and never modifies it (scenario_store is
off-limits). Qt-free.
"""

from __future__ import annotations

from collections import OrderedDict

from registry import get_quantity
from slice_key import SliceKey, DEFAULT_DIRECTION, DEFAULT_OFFSET, DIRECTION_TO_AXIS, available_slices
from derived_quantities import derive, source_quantity


class GatedQuantityError(RuntimeError):
    """Raised when a *registered but gated* quantity is requested before its
    data exists (the M-SIM re-run). Callers surface the gate_reason; they never
    fabricate the field. See docs/msim-preparation.md and ROADMAP-V6.md."""


class QuantityProvider:
    # V6-M1.5: computed (derived/calculated) results are memoized so playback
    # does not re-evaluate an expression every frame. Native quantities are NOT
    # memoized here -- the store already caches them. Bounded LRU, matching the
    # store's cache philosophy; invalidate() is called when definitions change.
    _MEMO_MAX = 8

    def __init__(self, store, fps: int = 1):
        self._store = store
        self._fps = max(1, fps)   # V6-M1: rate() needs the time step
        self._memo: OrderedDict = OrderedDict()
        self._plane_cache: dict = {}   # V6-M5: scenario -> {(quantity, direction, offset)}

    def invalidate(self) -> None:
        """Drop memoized computed fields (e.g. when a calculated field is
        created, edited, or removed)."""
        self._memo.clear()

    def _source_key(self, key: SliceKey) -> SliceKey:
        """For a derived quantity, the SliceKey of the raw field it reads."""
        src = source_quantity(key.quantity)
        return SliceKey(src, key.direction, key.offset) if src else key

    def _available_planes(self, scenario: int):
        """The (quantity, direction, offset) slices this scenario's .smv
        file actually describes -- parsed once (available_slices) and
        cached, never touched again per frame. Returns None (unknown,
        never gate on it) rather than an empty set when the inventory can't
        be determined -- e.g. `self._store` is a lightweight test double
        with no `.folders`, not a real ScenarioStore -- so this check only
        ever *adds* protection for the real store and never breaks a
        fake/mock one that worked fine before V6-M5."""
        if scenario not in self._plane_cache:
            try:
                folder = self._store.folders[scenario]
                infos = available_slices(folder)
                self._plane_cache[scenario] = {
                    (i.key.quantity, i.key.direction, i.key.offset) for i in infos}
            except Exception:
                self._plane_cache[scenario] = None
        return self._plane_cache[scenario]

    def _ensure_plane_available(self, scenario: int, key: SliceKey) -> None:
        """V6-M5: raise GatedQuantityError if (quantity, direction, offset)
        is positively confirmed absent from the .smv inventory -- checked
        against the already-parsed list, never by attempting a read.
        Querying geometry/data for a plane that was never extracted can
        crash the underlying slice-file parser instead of raising cleanly
        (the lesson from V6-M3's gated U/W-VELOCITY work); this guard is
        what keeps multi-plane support honest instead of merely hoping a
        bad plane never gets requested. Skipped for volumetric (`plane_pos`
        set) reads (a separate `.s3d` path `available_slices` doesn't
        inventory) and when the inventory itself can't be determined."""
        if key.plane_pos is not None:
            return
        planes = self._available_planes(scenario)
        if planes is None:
            return
        if (key.quantity, key.direction, key.offset) not in planes:
            axis = DIRECTION_TO_AXIS.get(key.direction, str(key.direction))
            raise GatedQuantityError(
                f"No {key.quantity} slice at direction={axis} ({key.direction}), "
                f"offset={key.offset} for this scenario -- only the planes present "
                f"in the .smv file are readable (see docs/msim-preparation.md).")

    def _compute(self, scenario: int, key: SliceKey, q):
        if getattr(q, "calculated", False):
            # V6-M1: a Field-Calculator field. Evaluate its expression, resolving
            # each dependency through this same provider (so a calculated field
            # can build on raw, derived, or other calculated fields), on the same
            # slice plane. Never executes raw code -- see field_calculator.
            import field_calculator as fc
            resolve = lambda dep_key: self.get(scenario, SliceKey(
                dep_key, key.direction, key.offset))
            return fc.evaluate(q.expression, resolve, self._fps)
        # a legacy single-source derived quantity (temperature rise, dyn.
        # pressure) -- routed through self.get() (not the store directly) so
        # its source read gets the same gating/plane checks as any other
        # quantity (V6-M5).
        source = self.get(scenario, self._source_key(key))
        return derive(key.quantity, source)   # elementwise over the whole (t,z,x) array

    def get(self, scenario: int, key: SliceKey = None):
        key = key or SliceKey("TEMPERATURE")
        q = get_quantity(key.quantity)
        if q.gated:
            # V6 hook: gated quantities (U/W-velocity, CO, pressure, visibility,
            # heat flux, soot mass) are registered but have no data until the
            # M-SIM re-run. When that lands, they become plain slice reads and
            # this guard falls through -- no other change needed.
            raise GatedQuantityError(q.gate_reason)
        if getattr(q, "calculated", False) or q.kind == "derived":
            ck = (scenario, key.quantity, key.direction, key.offset)
            cached = self._memo.get(ck)
            if cached is not None:
                self._memo.move_to_end(ck)
                return cached
            result = self._compute(scenario, key, q)
            self._memo[ck] = result
            self._memo.move_to_end(ck)
            while len(self._memo) > self._MEMO_MAX:
                self._memo.popitem(last=False)
            return result
        self._ensure_plane_available(scenario, key)
        return self._store.get(scenario, key)

    def get_vector(self, scenario: int, direction: int = None, offset: int = None):
        """V6-M3: the in-plane velocity vector field (U, W) for true
        streamlines / quiver. Reads U-VELOCITY/W-VELOCITY through the same
        `get()` path as any other quantity -- both are registered `gated=True`
        today (docs/msim-preparation.md §3), so `get()`'s own gate check raises
        GatedQuantityError here exactly as before. The moment the M-SIM re-run
        ungates them in the registry, this starts returning real arrays with
        no further code change -- the seam is real, not a hardcoded stub."""
        direction = DEFAULT_DIRECTION if direction is None else direction
        offset = DEFAULT_OFFSET if offset is None else offset
        u = self.get(scenario, SliceKey("U-VELOCITY", direction, offset))
        w = self.get(scenario, SliceKey("W-VELOCITY", direction, offset))
        return u, w

    def get_vector3d(self, scenario: int, direction: int = None, offset: int = None):
        """V6-M7: the full 3D velocity vector (U, V, W) for true 3D flow
        visualization. V-VELOCITY (the through-plane component) is
        registered `gated=True` today, same as U/W -- reads through the
        same get() path, so this raises GatedQuantityError immediately
        (before touching the store) exactly like get_vector() does, and
        starts returning real arrays with no code change once the M-SIM
        re-run ungates it. Never a 2D-to-3D fabrication: callers that only
        need the in-plane pair should keep using get_vector()."""
        direction = DEFAULT_DIRECTION if direction is None else direction
        offset = DEFAULT_OFFSET if offset is None else offset
        u = self.get(scenario, SliceKey("U-VELOCITY", direction, offset))
        v = self.get(scenario, SliceKey("V-VELOCITY", direction, offset))
        w = self.get(scenario, SliceKey("W-VELOCITY", direction, offset))
        return u, v, w

    def get_extent(self, scenario: int, key: SliceKey = None):
        key = key or SliceKey("TEMPERATURE")
        q = get_quantity(key.quantity)
        if q.gated:
            raise GatedQuantityError(q.gate_reason)
        if getattr(q, "calculated", False):
            # a calculated field shares its first dependency's grid/extent.
            import field_calculator as fc
            field = fc.get_field(key.quantity)
            dep = field.dependencies[0] if field and field.dependencies else "TEMPERATURE"
            return self.get_extent(scenario, SliceKey(dep, key.direction, key.offset))
        if q.kind == "derived":
            key = self._source_key(key)
        self._ensure_plane_available(scenario, key)
        return self._store.get_extent(scenario, key)

    @property
    def store(self):
        """Escape hatch for code that legitimately needs the raw store
        (e.g. availability checks). New reads should prefer get()."""
        return self._store
