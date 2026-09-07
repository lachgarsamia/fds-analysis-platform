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


def _temp_field():
    """Synthetic TEMPERATURE (t, z, x): a hot blob near centre (~200 C)
    falling to ~20 C at the edges, so the colour map has real hot/cold to
    separate."""
    z = np.linspace(-1, 1, N_Z)
    x = np.linspace(-1, 1, N_X)
    xx, zz = np.meshgrid(x, z)
    frame = 20.0 + 180.0 * np.exp(-(xx ** 2 + zz ** 2) / 0.3)
    return np.repeat(frame[None, :, :], N_TIMES, axis=0).astype(float)


class FakeProvider:
    """Mimics QuantityProvider's get_vector()/get_extent()/get() -- the
    methods velocity.VectorField and StreamlinePanel are duck-typed
    against (see velocity.py's own docstring)."""

    def __init__(self, gated_cases=(), no_temperature=False):
        self._gated = set(gated_cases)
        self._no_temperature = no_temperature

    def get_vector(self, scenario, direction, offset):
        if scenario in self._gated:
            raise GatedQuantityError("forced for test")
        return _swirl_field(scenario)

    def get(self, scenario, key):
        # StreamlinePanel._ensure_temperature() calls this for the
        # TEMPERATURE slice (CHANGE 3).
        if self._no_temperature:
            raise GatedQuantityError("no TEMPERATURE for test")
        return _temp_field()

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


def test_colorbar_is_added_and_labeled_with_temperature(panel):
    assert len(panel.canvas.fig.axes) >= 2, "expected a colorbar axes alongside the plot axes"
    cbar_ax = panel.canvas.fig.axes[-1]
    assert "°C" in cbar_ax.get_ylabel() or "Temperature" in cbar_ax.get_ylabel()


def test_streamlines_coloured_by_temperature_with_turbo_and_a_data_driven_norm(panel):
    """Colour = local gas TEMPERATURE on turbo (perceptually-ordered,
    dark-ended -- never jet, never a washed-out mid-tone ramp), with a
    plain linear norm from ~ambient to a high percentile (measured
    against the real distribution, not a fixed diverging centre)."""
    from matplotlib.colors import Normalize
    ax = panel.canvas.fig.axes[0]
    coll = ax.collections[0]
    assert coll.cmap.name == "turbo"
    assert coll.cmap.name != "jet"
    assert isinstance(coll.norm, Normalize)
    # near ambient at the bottom, real spread above it
    assert coll.norm.vmin <= 21.0
    assert coll.norm.vmax > coll.norm.vmin + 15.0


def test_falls_back_to_speed_colour_when_temperature_unavailable(qapp, manifest):
    """No readable TEMPERATURE for a scenario -> colour by speed (viridis),
    never crash, never a fabricated temperature field."""
    p = StreamlinePanel(FakeProvider(no_temperature=True), manifest, fps=4)
    p.ensure_loaded()
    ax = p.canvas.fig.axes[0]
    coll = ax.collections[0]
    assert coll.cmap.name == "viridis"
    assert "m/s" in p.canvas.fig.axes[-1].get_ylabel()


def test_density_control_default_matches_app_default(panel):
    """1.0 -- matplotlib's own streamplot default. Later visual-tuning
    passes (bolder/denser styling, feature-based seeding, a room outline)
    were reverted back to this original look/behavior; only the
    frame-to-frame seed-stability fix (fixed start_points, see
    streamline_panel.py's _seed_points) was kept on top of it. See
    streamline_panel.py's DEFAULT_DENSITY."""
    assert panel.density_spin.value() == pytest.approx(1.0)


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
