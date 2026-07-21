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

from registry import get_quantity
from slice_key import SliceKey
from derived_quantities import derive, source_quantity


class QuantityProvider:
    def __init__(self, store):
        self._store = store

    def _source_key(self, key: SliceKey) -> SliceKey:
        """For a derived quantity, the SliceKey of the raw field it reads."""
        src = source_quantity(key.quantity)
        return SliceKey(src, key.direction, key.offset) if src else key

    def get(self, scenario: int, key: SliceKey = None):
        key = key or SliceKey("TEMPERATURE")
        if get_quantity(key.quantity).kind == "derived":
            source = self._store.get(scenario, self._source_key(key))
            return derive(key.quantity, source)   # elementwise; applies to the whole (t,z,x) array
        return self._store.get(scenario, key)

    def get_extent(self, scenario: int, key: SliceKey = None):
        key = key or SliceKey("TEMPERATURE")
        if get_quantity(key.quantity).kind == "derived":
            key = self._source_key(key)
        return self._store.get_extent(scenario, key)

    @property
    def store(self):
        """Escape hatch for code that legitimately needs the raw store
        (e.g. availability checks). New reads should prefer get()."""
        return self._store
