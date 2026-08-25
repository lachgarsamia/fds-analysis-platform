"""Rendering smoke test for CompositeFlowPanel: does a real composite
(filled W background + speed-scaled quiver + translucent temperature
zones) get produced from known-good U/W/TEMPERATURE data, without error,
and does the panel show an honest gate message (never a fabricated plot)
when either is unavailable -- same conventions as test_streamline_panel.py.

No fdsreader cross-validation here (rendering-only change; U/W-VELOCITY/
TEMPERATURE data itself is already validated elsewhere).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from composite_flow_panel import CompositeFlowPanel  # noqa: E402
from quantity_provider import GatedQuantityError  # noqa: E402
from slice_key import SliceKey  # noqa: E402


class FakeEntry:
    def __init__(self, case_index, folder):
        self.case_index = case_index
        self.folder = folder
        self.path = "/nonexistent"
        self.candles = self.door = self.vod = self.voc = 0


EXTENT = (0.0, 1.0, 0.0, 0.5)
N_TIMES, N_Z, N_X = 20, 12, 16


def _swirl_field(case_index):
    """Synthetic U/W with real, non-degenerate spatial structure (a simple
    rotational flow) so the quiver/pcolormesh have genuine data to draw --
    scaled by case_index so scenarios differ."""
    z = np.linspace(-1, 1, N_Z)
    x = np.linspace(-1, 1, N_X)
    xx, zz = np.meshgrid(x, z)
    scale = 1.0 + case_index
    u = -zz * scale
    w = xx * scale
    u = np.repeat(u[None, :, :], N_TIMES, axis=0)
    w = np.repeat(w[None, :, :], N_TIMES, axis=0)
    return u.astype(float), w.astype(float)


def _temp_field(case_index):
    """Synthetic TEMPERATURE with real spatial structure (a hot spot) so
    the isotherm contour has genuine levels to trace."""
    z = np.linspace(-1, 1, N_Z)
    x = np.linspace(-1, 1, N_X)
    xx, zz = np.meshgrid(x, z)
    t = 20.0 + (50.0 + 10.0 * case_index) * np.exp(-(xx ** 2 + zz ** 2))
    return np.repeat(t[None, :, :], N_TIMES, axis=0).astype(float)


class FakeProvider:
    """Mimics QuantityProvider's get_vector()/get_extent()/get() -- the
    methods VectorField and CompositeFlowPanel are duck-typed against."""

    def __init__(self, gated_vector_cases=(), gated_temp_cases=()):
        self._gated_vector = set(gated_vector_cases)
        self._gated_temp = set(gated_temp_cases)

    def get_vector(self, scenario, direction, offset):
        if scenario in self._gated_vector:
            raise GatedQuantityError("forced for test")
        return _swirl_field(scenario)

    def get_extent(self, scenario, key):
        return EXTENT

    def get(self, scenario, key: SliceKey):
        assert key.quantity == "TEMPERATURE"
        if scenario in self._gated_temp:
            raise GatedQuantityError("forced for test")
        return _temp_field(scenario)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def manifest():
    return [FakeEntry(0, "case_a"), FakeEntry(1, "case_b")]


@pytest.fixture
def panel(qapp, provider, manifest):
    p = CompositeFlowPanel(provider, manifest, fps=4)
    p.ensure_loaded()
    return p


def test_scenario_combo_populated_from_manifest(panel, manifest):
    assert panel.scenario_combo.count() == len(manifest)


def test_renders_all_three_layers_without_error(panel):
    """The whole point: a filled mesh (W), a quiver (direction+length),
    and filled contour bands (temperature zones) all get produced from
    known-good data."""
    ax = panel.canvas.fig.axes[0]
    kinds = {type(c).__name__ for c in ax.collections}
    assert any(k.endswith("QuadMesh") for k in kinds), "expected the W pcolormesh"
    assert "Quiver" in kinds, "expected the quiver"
    assert len(ax.collections) >= 3, "expected mesh + quiver + at least one temperature-zone collection"


def test_both_colorbars_are_added_and_labeled(panel):
    """Two independent colorbars: W velocity (Layer 1) and temperature
    (Layer 3's translucent zones) -- neither should be assumed to be
    axes[-1] since matplotlib appends them in the order added."""
    labels = [ax.get_ylabel() for ax in panel.canvas.fig.axes[1:]]
    assert len(labels) >= 2, "expected two colorbar axes alongside the plot axes"
    assert any("m/s" in l and "W" in l for l in labels), f"no W-velocity colorbar in {labels}"
    assert any("Temperature" in l for l in labels), f"no temperature colorbar in {labels}"


def test_quiver_length_is_speed_scaled_within_a_compressed_range(panel):
    """Layer 2 length now encodes speed sub-linearly (sqrt, floored) --
    not uniform (the old behavior) and not linear (which would crush the
    ~20x real speed range to invisibility) -- see QUIVER_SPEED_REF/
    QUIVER_LENGTH_FLOOR/QUIVER_LENGTH_MAX. Color still separately encodes
    magnitude too (Layer 1); this only asserts the length side."""
    ax = panel.canvas.fig.axes[0]
    quiver = [c for c in ax.collections if type(c).__name__ == "Quiver"][0]
    from composite_flow_panel import QUIVER_LENGTH_FLOOR, QUIVER_LENGTH_MAX
    mags = np.hypot(quiver.U, quiver.V)
    nonzero = mags[mags > 1e-9]
    assert nonzero.size > 0
    # not uniform: the synthetic swirl field has real speed variation
    assert not np.allclose(nonzero, nonzero[0])
    assert nonzero.min() >= QUIVER_LENGTH_FLOOR - 1e-9
    assert nonzero.max() <= QUIVER_LENGTH_MAX + 1e-9


def test_quiver_samples_the_unflipped_field_not_the_display_flip(panel):
    """Regression test for the quiver-orientation bug: an earlier version
    of this panel passed the already-vertically-flipped u/w arrays into
    velocity.quiver_grid, which computes each sample's z from the RAW
    (row 0 = ceiling) convention (see composite_flow_panel.py's module
    docstring) -- that double-flip silently mispaired each arrow's
    position with a *different* row's velocity, confirmed against
    measure.probe_value ground truth before the fix landed.

    _swirl_field's U is antisymmetric top-to-bottom: +scale at row 0
    (the ceiling) and -scale at the last row (the floor). A regression
    that re-introduces the double-flip would swap those, so the sample
    nearest the ceiling (z close to EXTENT's z1) should have positive U;
    the buggy version would render it negative."""
    ax = panel.canvas.fig.axes[0]
    quiver = [c for c in ax.collections if type(c).__name__ == "Quiver"][0]
    offsets = np.array(quiver.get_offsets())
    z1 = EXTENT[3]
    ceiling_idx = np.argmax(offsets[:, 1])
    assert offsets[ceiling_idx, 1] == pytest.approx(z1, abs=1e-6)
    assert quiver.U[ceiling_idx] > 0, "ceiling-row U should be positive (raw field), not negative (flipped)"


def test_quiver_grid_positions_are_identical_across_renders(panel):
    """Fixed-stride positions (velocity.quiver_grid) are deterministic
    from stride/extent alone -- re-rendering the same scenario must not
    move a single arrow, the same temporal-stability guarantee
    StreamlinePanel's fixed seeds give, with no cache needed here since
    nothing about the *positions* depends on frame data."""
    ax = panel.canvas.fig.axes[0]
    quiver = [c for c in ax.collections if type(c).__name__ == "Quiver"][0]
    before = np.array(quiver.get_offsets())
    panel._current_index = 5
    panel._render()
    ax2 = panel.canvas.fig.axes[0]
    quiver2 = [c for c in ax2.collections if type(c).__name__ == "Quiver"][0]
    after = np.array(quiver2.get_offsets())
    assert np.array_equal(before, after)


def test_axis_limits_match_the_vector_field_extent(panel):
    ax = panel.canvas.fig.axes[0]
    x0, x1, z0, z1 = EXTENT
    assert ax.get_xlim() == pytest.approx((x0, x1))
    assert ax.get_ylim() == pytest.approx((z0, z1))


def test_gated_velocity_shows_honest_message_not_a_fabricated_plot(qapp, manifest):
    gated_provider = FakeProvider(gated_vector_cases=(0,))
    p = CompositeFlowPanel(gated_provider, manifest, fps=4)
    p.ensure_loaded()
    ax = p.canvas.fig.axes[0]
    assert len(ax.collections) == 0
    assert p.status.text() != ""


def test_gated_temperature_shows_honest_message_not_a_fabricated_plot(qapp, manifest):
    """U/W are fine but TEMPERATURE isn't -- must still gate cleanly, not
    render a composite missing one of its three layers."""
    gated_provider = FakeProvider(gated_temp_cases=(0,))
    p = CompositeFlowPanel(gated_provider, manifest, fps=4)
    p.ensure_loaded()
    ax = p.canvas.fig.axes[0]
    assert len(ax.collections) == 0
    assert p.status.text() != ""


def test_switching_scenario_recomputes_the_fields(panel):
    """Different scenarios have different swirl scale/hot-spot magnitude
    -- confirms a real per-scenario recompute, not a cached/stuck first
    scenario."""
    field0 = panel._ensure_field(0)
    field1 = panel._ensure_field(1)
    assert not np.allclose(field0.speed, field1.speed)
    temp0 = panel._ensure_temperature(0)
    temp1 = panel._ensure_temperature(1)
    assert not np.allclose(temp0, temp1)
