"""Tests for the Height-Aware Analysis Workspace (V4-M1), extended by the
Live-polish follow-up to route through field_fn/extent_fn (the same
computed-quantity dispatch main_window.py uses elsewhere) and to support
DYNAMIC PRESSURE, TEMPERATURE RISE, and SOOT DENSITY alongside the
original native slice2d quantities.
"""

import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import colormaps  # noqa: E402,F401 -- registers "fds_fire"/"fds_flow" etc. with
                   # matplotlib at import time; main_window.py normally
                   # guarantees this via its own import chain before any
                   # panel renders, which a standalone panel test bypasses.
from height_panel import HeightPanel, DYNAMIC_PRESSURE_QUANTITY  # noqa: E402
from slice_key import SliceKey  # noqa: E402


class FakeEntry:
    def __init__(self, case_index, folder):
        self.case_index = case_index
        self.folder = folder
        self.path = "/nonexistent"
        self.candles = self.door = self.vod = self.voc = 0


#: The regression this guards: SOOT DENSITY's doorway slice is read from a
#: much narrower plane (n_x=3) than the wide side-view slices every other
#: quantity in this suite shares (n_x=11) -- exactly the shape mismatch
#: that exposed the stale-_x_col IndexError this session.
WIDE_EXTENT = (0.0, 1.0, 0.0, 0.3)
NARROW_EXTENT = (0.2, 0.3, 0.0, 0.3)


def _make_field(n_times=5, n_z=6, n_x=11):
    t = np.arange(n_times, dtype=np.float32)[:, None, None]
    z = np.arange(n_z, dtype=np.float32)[None, :, None]
    x = np.arange(n_x, dtype=np.float32)[None, None, :]
    return t * 100.0 + z * 10.0 + x


def field_fn(case_index, key):
    if key.quantity == "SOOT DENSITY":
        return _make_field(n_x=3)
    return _make_field(n_x=11)


def extent_fn(case_index, key):
    if key.quantity == "SOOT DENSITY":
        return NARROW_EXTENT
    return WIDE_EXTENT


QUANTITY_OPTIONS = [
    ("Temperature", SliceKey("TEMPERATURE")),
    ("Dynamic pressure", SliceKey("DYNAMIC PRESSURE")),
    ("Temperature rise (ΔT)", SliceKey("TEMPERATURE RISE")),
    ("Smoke — doorway (x = 0.25 m)", SliceKey("SOOT DENSITY")),
]


@pytest.fixture
def panel(qapp):
    manifest = [FakeEntry(0, "case_a"), FakeEntry(1, "case_b")]
    p = HeightPanel(None, manifest, QUANTITY_OPTIONS, fps=4,
                    field_fn=field_fn, extent_fn=extent_fn)
    p.ensure_loaded()
    yield p
    p.deleteLater()


class TestHeightPanelRouting:
    def test_ensure_loaded_populates_scenarios(self, panel):
        assert panel.scenario_combo.count() == 2
        assert panel._loaded

    def test_quantity_options_include_derived_and_volume_kinds(self, panel):
        labels = [panel.quantity_combo.itemText(i)
                   for i in range(panel.quantity_combo.count())]
        assert labels == [label for label, _key in QUANTITY_OPTIONS]

    def test_field_fn_is_used_instead_of_bare_store_get(self, qapp):
        """Regression guard for the store.get() bypass bug class: a store
        that would raise if ever touched directly proves field_fn/extent_fn
        are the only data path, matching main_window.py's dispatch."""
        class ExplodingStore:
            def get(self, *a, **k):
                raise AssertionError("bypassed field_fn, called store.get() directly")

            def get_extent(self, *a, **k):
                raise AssertionError("bypassed extent_fn, called store.get_extent() directly")

        manifest = [FakeEntry(0, "case_a")]
        p = HeightPanel(ExplodingStore(), manifest, QUANTITY_OPTIONS, fps=4,
                         field_fn=field_fn, extent_fn=extent_fn)
        p.ensure_loaded()
        for i in range(p.quantity_combo.count()):
            p.quantity_combo.setCurrentIndex(i)
        p.deleteLater()


class TestHeightPanelQuantitySwitching:
    def test_all_quantities_selectable_without_crashing(self, panel):
        for i in range(panel.quantity_combo.count()):
            panel.quantity_combo.setCurrentIndex(i)
            assert panel._data is not None

    def test_stale_x_col_is_clamped_to_narrower_quantity(self, panel):
        """The actual bug found this session: clicking near the right edge
        of a wide (n_x=11) slice, then switching to the narrower (n_x=3)
        SOOT DENSITY doorway slice, used to raise IndexError inside
        ha.vertical_profile because _x_col was never re-clamped."""
        temp_idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                         if k.quantity == "TEMPERATURE")
        smoke_idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                          if k.quantity == "SOOT DENSITY")
        panel.quantity_combo.setCurrentIndex(temp_idx)
        panel._x_col = 10  # last valid column of the wide (n_x=11) slice
        panel.quantity_combo.setCurrentIndex(smoke_idx)  # n_x=3 here
        assert panel._x_col <= 2
        assert panel._data.shape[2] == 3

    def test_dynamic_pressure_gets_flow_forcing_caption_not_neutral_plane(self, panel):
        idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                    if k.quantity == "DYNAMIC PRESSURE")
        panel.quantity_combo.setCurrentIndex(idx)
        text = panel.caption.text()
        assert "flow-forcing profile" in text.lower()
        assert "neutral-plane finder" in text.lower() or "neutral plane" in text.lower()
        assert "not a neutral" in text.lower() or "not a neutral-plane" in text.lower()

    def test_other_quantities_get_default_caption(self, panel):
        temp_idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                         if k.quantity == "TEMPERATURE")
        panel.quantity_combo.setCurrentIndex(temp_idx)
        assert "click the map" in panel.caption.text().lower()


class TestHeightPanelSafetyNet:
    """The Live-polish crash-fix follow-up: PyQt5 aborts the whole process
    (qFatal -> abort(), no catchable Python exception) on an unhandled
    error inside a slot connected to a signal -- quantity_combo's
    currentIndexChanged is exactly such a slot, so _reload must never let
    an exception escape into Qt's event loop."""

    def test_exception_in_field_fn_is_caught_not_raised(self, qapp, caplog):
        def exploding_field_fn(case_index, key):
            if key.quantity == "DYNAMIC PRESSURE":
                raise RuntimeError("simulated resolver failure")
            return _make_field(n_x=11)

        manifest = [FakeEntry(0, "case_a")]
        p = HeightPanel(None, manifest, QUANTITY_OPTIONS, fps=4,
                         field_fn=exploding_field_fn, extent_fn=extent_fn)
        p.ensure_loaded()
        idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                    if k.quantity == "DYNAMIC PRESSURE")
        with caplog.at_level(logging.ERROR):
            p.quantity_combo.setCurrentIndex(idx)  # must not raise/abort
        assert "could not load this quantity" in p.caption.text().lower()
        assert any(r.levelno >= logging.ERROR for r in caplog.records)
        p.deleteLater()

    def test_recovers_after_a_failed_switch(self, qapp):
        """A caught failure on one quantity shouldn't wedge the panel --
        switching to a working quantity afterwards should render normally."""
        def exploding_field_fn(case_index, key):
            if key.quantity == "DYNAMIC PRESSURE":
                raise RuntimeError("simulated resolver failure")
            return _make_field(n_x=11)

        manifest = [FakeEntry(0, "case_a")]
        p = HeightPanel(None, manifest, QUANTITY_OPTIONS, fps=4,
                         field_fn=exploding_field_fn, extent_fn=extent_fn)
        p.ensure_loaded()
        bad_idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                        if k.quantity == "DYNAMIC PRESSURE")
        good_idx = next(i for i, (_l, k) in enumerate(QUANTITY_OPTIONS)
                         if k.quantity == "TEMPERATURE")
        p.quantity_combo.setCurrentIndex(bad_idx)
        p.quantity_combo.setCurrentIndex(good_idx)
        assert p._data is not None
        assert "click the map" in p.caption.text().lower()
        p.deleteLater()
