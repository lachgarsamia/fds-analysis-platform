"""Rendering smoke test for StreamlinePanel: does a real matplotlib
streamplot get produced from known-good U/W data, without error, and does
the panel show an honest gate message (never a fabricated plot) when U/W
aren't available -- same conventions as test_smoke_layer_motion_panel.py.

No fdsreader cross-validation here (rendering-only change; U/W-VELOCITY
data itself is already validated elsewhere -- TestVectorVelocityCrossValidation).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamline_panel import StreamlinePanel  # noqa: E402
from quantity_provider import GatedQuantityError  # noqa: E402


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
    rotational flow) so streamplot has genuine streamlines to draw, not a
    flat/zero field -- scaled by case_index so scenarios differ."""
    z = np.linspace(-1, 1, N_Z)
    x = np.linspace(-1, 1, N_X)
    xx, zz = np.meshgrid(x, z)
    scale = 1.0 + case_index
    u = -zz * scale
    w = xx * scale
    u = np.repeat(u[None, :, :], N_TIMES, axis=0)
    w = np.repeat(w[None, :, :], N_TIMES, axis=0)
    return u.astype(float), w.astype(float)


class FakeProvider:
    """Mimics QuantityProvider's get_vector()/get_extent() -- the two
    methods velocity.VectorField is duck-typed against (see velocity.py's
    own docstring)."""

    def __init__(self, gated_cases=()):
        self._gated = set(gated_cases)

    def get_vector(self, scenario, direction, offset):
        if scenario in self._gated:
            raise GatedQuantityError("forced for test")
        return _swirl_field(scenario)

    def get_extent(self, scenario, key):
        return EXTENT


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def manifest():
    return [FakeEntry(0, "case_a"), FakeEntry(1, "case_b")]


@pytest.fixture
def panel(qapp, provider, manifest):
    p = StreamlinePanel(provider, manifest, fps=4)
    p.ensure_loaded()
    return p


def test_scenario_combo_populated_from_manifest(panel, manifest):
    assert panel.scenario_combo.count() == len(manifest)


def test_renders_a_real_streamplot_without_error(panel):
    """The whole point: a figure with actual streamline line-collection
    artists gets produced from known-good U/W data."""
    ax = panel.canvas.fig.axes[0]
    assert len(ax.collections) > 0, "expected at least one LineCollection from streamplot"


def test_colorbar_is_added_and_labeled_with_speed_units(panel):
    assert len(panel.canvas.fig.axes) >= 2, "expected a colorbar axes alongside the plot axes"
    cbar_ax = panel.canvas.fig.axes[-1]
    assert "m/s" in cbar_ax.get_ylabel()


def test_uses_perceptually_uniform_colormap_not_jet(panel):
    ax = panel.canvas.fig.axes[0]
    coll = ax.collections[0]
    assert coll.cmap.name in ("viridis", "plasma")
    assert coll.cmap.name != "jet"


def test_density_control_default_matches_app_default(panel):
    """0.9, not matplotlib's own 1.0 default -- visual clarity pass, phase
    2: lowered from 1.8 once seeding became feature-based (fire/door/vent
    clusters + a light background grid) instead of a blind 12x6 grid --
    the higher density was tuned for that grid's much denser coverage and
    read as clutter competing with the feature clusters and room outline
    for attention. See streamline_panel.py's DEFAULT_DENSITY."""
    assert panel.density_spin.value() == pytest.approx(0.9)


def test_density_is_a_tunable_constructor_parameter(qapp, provider, manifest):
    p = StreamlinePanel(provider, manifest, fps=4, density=2.5)
    assert p.density_spin.value() == pytest.approx(2.5)


def test_axis_limits_match_the_vector_field_extent(panel):
    ax = panel.canvas.fig.axes[0]
    x0, x1, z0, z1 = EXTENT
    assert ax.get_xlim() == pytest.approx((x0, x1))
    assert ax.get_ylim() == pytest.approx((z0, z1))


def test_gated_scenario_shows_honest_message_not_a_fabricated_plot(qapp, manifest):
    """No U/W data for this scenario -- must show the gate reason, never
    draw made-up streamlines."""
    gated_provider = FakeProvider(gated_cases=(0,))
    p = StreamlinePanel(gated_provider, manifest, fps=4)
    p.ensure_loaded()
    ax = p.canvas.fig.axes[0]
    assert len(ax.collections) == 0
    assert p.status.text() != ""


def test_switching_scenario_recomputes_the_field(panel, provider):
    """Different scenarios have different swirl scale -- the rendered
    speed range should differ, confirming a real per-scenario recompute
    (not a cached/stuck first-scenario field)."""
    field0 = panel._ensure_field(0)
    field1 = panel._ensure_field(1)
    assert not np.allclose(field0.speed, field1.speed)
