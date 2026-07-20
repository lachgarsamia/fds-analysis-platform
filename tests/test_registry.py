"""Tests for the quantity registry (V2 roadmap M0.2)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import registry  # noqa: E402
import config  # noqa: E402
from slice_key import DataKey, SliceKey  # noqa: E402


class TestRegistry:
    def test_get_quantity_known(self):
        q = registry.get_quantity("VELOCITY")
        assert q.name == "VELOCITY" and q.unit == "m/s" and q.kind == "slice2d"

    def test_get_quantity_unknown_falls_back_to_temperature(self):
        assert registry.get_quantity("NOPE").name == "TEMPERATURE"

    def test_kind_discriminates_slice_vs_volume(self):
        assert registry.get_quantity("TEMPERATURE").kind == "slice2d"
        assert registry.get_quantity("SOOT DENSITY").kind == "volume"


class TestDerivedViews:
    def test_display_dict_matches_registry(self):
        for name, q in registry.QUANTITY_REGISTRY.items():
            d = config.QUANTITY_DISPLAY[name]
            assert d["label"] == q.label and d["unit"] == q.unit
            assert d["cmap"] == q.cmap and d["vmin"] == q.vmin
            assert d["slider_default"] == q.slider_default

    def test_isotherm_dict_only_has_quantities_with_hazard_bands(self):
        assert config.ISOTHERM_LEVELS["TEMPERATURE"] == [60, 100, 300]
        assert config.ISOTHERM_LEVELS["VELOCITY"] == [1.0, 2.0, 3.0]
        assert "SOOT DENSITY" not in config.ISOTHERM_LEVELS  # no hazard bands declared

    def test_ambient_c_single_source(self):
        assert config.AMBIENT_C == registry.AMBIENT_C == 20.0


class TestDataKeyAlias:
    def test_datakey_is_slicekey(self):
        assert DataKey is SliceKey
        assert isinstance(SliceKey("TEMPERATURE"), DataKey)
