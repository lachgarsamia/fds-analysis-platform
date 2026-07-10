"""Unit tests for slice_key.py (M2.1): SliceKey identity and the .smv
quantity inventory."""

from slice_key import SliceKey, DEFAULT_SLICE_KEY, available_slices


class TestSliceKey:
    def test_slice_key_equality_and_hash(self):
        a = SliceKey("TEMPERATURE", 1, 0)
        b = SliceKey("TEMPERATURE", 1, 0)
        c = SliceKey("VELOCITY", 1, 0)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        # Usable as a dict key -- this is the whole reason it's frozen.
        d = {a: "temp-data"}
        assert d[b] == "temp-data"

    def test_default_slice_key_matches_pre_m2_1_constants(self):
        assert DEFAULT_SLICE_KEY.quantity == "TEMPERATURE"
        assert DEFAULT_SLICE_KEY.direction == 1
        assert DEFAULT_SLICE_KEY.offset == 0

    def test_available_slices_finds_temperature_and_velocity(self, fixtures_dir):
        """The fixture's .smv describes both quantities even though only
        TEMPERATURE's .sf data was kept on disk (see conftest.py) --
        available_slices() only reads .smv metadata, so it should still see
        VELOCITY."""
        infos = available_slices(fixtures_dir)
        quantities = {info.key.quantity for info in infos}
        assert quantities == {"TEMPERATURE", "VELOCITY"}

    def test_available_slices_deduplicates_across_meshes(self, fixtures_dir):
        """A (quantity, direction, offset) combo split across multiple
        meshes must appear once, not once per mesh."""
        infos = available_slices(fixtures_dir)
        keys = [info.key for info in infos]
        assert len(keys) == len(set(keys))

    def test_available_slices_includes_default_key(self, fixtures_dir):
        infos = available_slices(fixtures_dir)
        assert DEFAULT_SLICE_KEY in {info.key for info in infos}
