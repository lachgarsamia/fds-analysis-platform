"""Tests for Factor-Effect Field Maps (V2 roadmap M3.1, F2 flagship)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import factor_effects as fx  # noqa: E402
from load_data import SIM_ROOT  # noqa: E402
from slice_key import SliceKey  # noqa: E402

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present")


class FakeEntry:
    def __init__(self, case_index, candles, door, vod=0, voc=0):
        self.case_index = case_index
        self.candles, self.door, self.vod, self.voc = candles, door, vod, voc


class FakeStore:
    """Per-scenario constant fields so group means are hand-computable:
    scenario value = 10*candles + door."""
    def __init__(self, entries):
        self._val = {e.case_index: 10 * e.candles + e.door for e in entries}

    def get(self, case_index, key):
        return np.full((4, 3, 5), float(self._val[case_index]), dtype=np.float32)


class TestPureComputation:
    def _entries(self):
        # Full 2x2 factorial in (candles, door).
        return [FakeEntry(0, 0, 0), FakeEntry(1, 0, 1), FakeEntry(2, 1, 0), FakeEntry(3, 1, 1)]

    def test_factor_groups(self):
        groups = fx.factor_groups(self._entries(), 'door')
        assert groups == {0: [0, 2], 1: [1, 3]}

    def test_group_mean_streams_correct_mean(self):
        entries = self._entries()
        store = FakeStore(entries)
        # door=1 group: scenarios 1 (val 1) and 3 (val 11) -> mean 6.
        mean = fx.group_mean_series(store, [1, 3], SliceKey('X'))
        assert mean.shape == (4, 3, 5)
        np.testing.assert_allclose(mean, 6.0)

    def test_main_effect_is_high_minus_low_group_mean(self):
        entries = self._entries()
        store = FakeStore(entries)
        # door: high group mean = 6, low group (0:val0, 2:val10) mean = 5 -> effect 1.
        field = fx.main_effect_series(store, entries, 'door', SliceKey('X'))
        np.testing.assert_allclose(field, 1.0)
        # candles: high (2:10,3:11)=10.5, low (0:0,1:1)=0.5 -> effect 10.
        field_c = fx.main_effect_series(store, entries, 'candles', SliceKey('X'))
        np.testing.assert_allclose(field_c, 10.0)

    def test_main_effect_none_when_single_level(self):
        entries = [FakeEntry(0, 0, 0), FakeEntry(1, 0, 1)]  # candles constant
        store = FakeStore(entries)
        assert fx.main_effect_series(store, entries, 'candles', SliceKey('X')) is None

    def test_interaction_zero_for_additive_field(self):
        # value = 10*candles + door is additive -> interaction is exactly 0.
        entries = self._entries()
        store = FakeStore(entries)
        inter = fx.interaction_series(store, entries, 'candles', 'door', SliceKey('X'))
        np.testing.assert_allclose(inter, 0.0, atol=1e-6)

    def test_interaction_nonzero_for_multiplicative_field(self):
        entries = self._entries()

        class MultStore:
            def get(self, ci, key):
                e = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}[ci]
                return np.full((2, 2, 2), float((e[0] + 1) * (e[1] + 1)), dtype=np.float32)
        inter = fx.interaction_series(MultStore(), entries, 'candles', 'door', SliceKey('X'))
        # (2*2 - 2*1) - (1*2 - 1*1) = 2 - 1 = 1.
        np.testing.assert_allclose(inter, 1.0)

    def test_effect_magnitude_and_peak(self):
        field = np.array([[[-3.0, 1.0]], [[2.0, -4.0]]], dtype=np.float32)  # 2 frames
        # spatial mean|.| per frame: 2.0, 3.0 -> mean 2.5; peak 4.0
        assert fx.effect_magnitude(field) == pytest.approx(2.5)
        assert fx.effect_peak(field) == pytest.approx(4.0)


@requires_real_dataset
class TestFactorEffectRealData:
    """Validates against M2.3's pinned door-width finding, now at the
    factor-effect field level: for TEMPERATURE the door main effect is
    dominated by the candle/plume region, not the doorway; for VELOCITY
    the doorway carries real signal."""

    def _load(self):
        from data_provider import load_simulation_data
        sim = load_simulation_data()
        if sim.is_demo:
            pytest.skip("real dataset not present")
        return sim

    def _bands(self, sim, quantity):
        field = fx.main_effect_series(sim.store, sim.manifest, 'door', SliceKey(quantity))
        extent = sim.store.get_extent(0, SliceKey(quantity))
        x0, x1 = extent[0], extent[1]
        n_x = field.shape[2]
        x_cols = np.linspace(x0, x1, n_x)
        door = (x_cols >= 0.24) & (x_cols <= 0.30)
        candle = (x_cols >= 0.90) & (x_cols <= 0.98)
        abs_field = np.abs(field)
        door_sig = abs_field[:, :, door].max(axis=(1, 2)).mean()
        candle_sig = abs_field[:, :, candle].max(axis=(1, 2)).mean()
        return door_sig, candle_sig

    def test_temperature_door_effect_dominated_by_candle_region(self):
        sim = self._load()
        door_sig, candle_sig = self._bands(sim, 'TEMPERATURE')
        assert candle_sig > door_sig, (
            f"expected candle-region temperature effect to exceed the door "
            f"region (candle={candle_sig:.3f}, door={door_sig:.3f})")

    def test_velocity_door_effect_present_at_doorway(self):
        sim = self._load()
        door_sig, _candle = self._bands(sim, 'VELOCITY')
        assert door_sig > 0.0

    def test_effect_field_matches_slice_grid_shape(self):
        sim = self._load()
        field = fx.main_effect_series(sim.store, sim.manifest, 'candles', SliceKey('TEMPERATURE'))
        assert field.shape == (481, 49, 101)
