"""V5-M1 / Phase 0: the QuantityProvider computation layer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import quantity_provider as qp_mod  # noqa: E402
from quantity_provider import QuantityProvider, GatedQuantityError  # noqa: E402
from slice_key import SliceKey, SliceInfo  # noqa: E402


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


class _FakeStoreWithFolders(_FakeStore):
    """A store shaped like the real ScenarioStore (has `.folders`), so the
    V6-M5 plane-availability check actually engages instead of failing open."""
    def __init__(self):
        super().__init__()
        self.folders = ["/fake/scenario0"]


class TestPlaneGating:
    """V6-M5: multi-plane reads must raise GatedQuantityError for a plane
    confirmed absent from the .smv inventory, but never break a test double
    that has no real inventory to check (fail-open, not fail-closed)."""

    def test_default_plane_available_passes_through(self, monkeypatch):
        store = _FakeStoreWithFolders()
        monkeypatch.setattr(qp_mod, "available_slices", lambda folder: [
            SliceInfo(SliceKey("TEMPERATURE", 1, 0), "temp", "C"),
            SliceInfo(SliceKey("VELOCITY", 1, 0), "vel", "m/s"),
        ])
        out = QuantityProvider(store).get(0, SliceKey("TEMPERATURE", 1, 0))
        assert np.array_equal(out, store.temperature)

    def test_second_real_offset_passes_through(self, monkeypatch):
        """Mirrors the real dataset: TEMPERATURE exists at both offset 0
        and offset 15 on the same (y-normal) direction."""
        store = _FakeStoreWithFolders()
        monkeypatch.setattr(qp_mod, "available_slices", lambda folder: [
            SliceInfo(SliceKey("TEMPERATURE", 1, 0), "temp", "C"),
            SliceInfo(SliceKey("TEMPERATURE", 1, 15), "temp", "C"),
        ])
        out = QuantityProvider(store).get(0, SliceKey("TEMPERATURE", 1, 15))
        assert np.array_equal(out, store.temperature)

    def test_absent_plane_raises_gated_quantity_error(self, monkeypatch):
        store = _FakeStoreWithFolders()
        monkeypatch.setattr(qp_mod, "available_slices", lambda folder: [
            SliceInfo(SliceKey("TEMPERATURE", 1, 0), "temp", "C"),
        ])
        with pytest.raises(GatedQuantityError):
            QuantityProvider(store).get(0, SliceKey("TEMPERATURE", 0, 0))   # x-normal -- not in inventory

    def test_get_extent_also_gates(self, monkeypatch):
        store = _FakeStoreWithFolders()
        monkeypatch.setattr(qp_mod, "available_slices", lambda folder: [
            SliceInfo(SliceKey("TEMPERATURE", 1, 0), "temp", "C"),
        ])
        with pytest.raises(GatedQuantityError):
            QuantityProvider(store).get_extent(0, SliceKey("TEMPERATURE", 2, 0))   # z-normal -- absent

    def test_inventory_parsed_once_and_cached(self, monkeypatch):
        store = _FakeStoreWithFolders()
        calls = []
        def fake_available_slices(folder):
            calls.append(folder)
            return [SliceInfo(SliceKey("TEMPERATURE", 1, 0), "temp", "C")]
        monkeypatch.setattr(qp_mod, "available_slices", fake_available_slices)
        p = QuantityProvider(store)
        for _ in range(5):
            p.get(0, SliceKey("TEMPERATURE", 1, 0))
        assert len(calls) == 1   # parsed once, cached -- never per read/tick

    def test_fails_open_when_store_has_no_folders(self):
        """A test double / lightweight store with no `.folders` (like
        _FakeStore itself) must behave exactly as before V6-M5 -- the
        inventory can't be determined, so the read proceeds rather than
        being incorrectly gated."""
        store = _FakeStore()   # no .folders
        out = QuantityProvider(store).get(0, SliceKey("TEMPERATURE", 0, 0))
        assert np.array_equal(out, store.temperature)

    def test_volumetric_plane_pos_skips_the_check(self, monkeypatch):
        """A `.s3d` volumetric read (plane_pos set) isn't in the .sf
        inventory at all -- the check must not gate it."""
        store = _FakeStoreWithFolders()
        monkeypatch.setattr(qp_mod, "available_slices", lambda folder: [])
        key = SliceKey("TEMPERATURE", 1, 0, plane_pos=0.5)
        out = QuantityProvider(store).get(0, key)
        assert np.array_equal(out, store.temperature)
