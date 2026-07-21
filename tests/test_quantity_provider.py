"""V5-M1 / Phase 0: the QuantityProvider computation layer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantity_provider import QuantityProvider  # noqa: E402
from slice_key import SliceKey  # noqa: E402


class _FakeStore:
    """Minimal store: returns a fixed array/extent for any scenario, keyed
    by quantity, and records which quantities were asked for."""
    def __init__(self):
        self.temperature = np.array([[[20.0, 120.0], [320.0, 20.0]]])  # (1,2,2)
        self.velocity = np.array([[[0.0, 2.0], [4.0, 0.0]]])
        self.extent = [0.0, 1.0, 0.0, 1.0]
        self.asked = []

    def get(self, scenario, key):
        self.asked.append(key.quantity)
        return {"TEMPERATURE": self.temperature, "VELOCITY": self.velocity}[key.quantity]

    def get_extent(self, scenario, key):
        return self.extent


class TestQuantityProvider:
    def test_raw_passes_through(self):
        store = _FakeStore()
        p = QuantityProvider(store)
        out = p.get(0, SliceKey("TEMPERATURE"))
        assert np.array_equal(out, store.temperature) and store.asked == ["TEMPERATURE"]

    def test_derived_temperature_rise(self):
        store = _FakeStore()
        p = QuantityProvider(store)
        out = p.get(0, SliceKey("TEMPERATURE RISE"))
        assert np.allclose(out, store.temperature - 20.0)   # computed from TEMPERATURE
        assert store.asked == ["TEMPERATURE"]               # read the source, not a "TEMPERATURE RISE" file

    def test_derived_dynamic_pressure_from_velocity(self):
        store = _FakeStore()
        out = QuantityProvider(store).get(0, SliceKey("DYNAMIC PRESSURE"))
        assert np.allclose(out, 0.5 * 1.2 * store.velocity ** 2)

    def test_extent_of_derived_delegates_to_source(self):
        store = _FakeStore()
        assert QuantityProvider(store).get_extent(0, SliceKey("TEMPERATURE RISE")) == store.extent
