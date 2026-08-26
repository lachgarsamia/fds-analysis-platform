"""Rendering smoke test for TracerFlowPanel: does a real tracer-particle
pool advect and render from known-good U/W data without error, and does
the panel show an honest gate message (never a fabricated plot) when U/W
aren't available -- same conventions as test_streamline_panel.py.

No fdsreader cross-validation here (rendering-only addition; U/W-VELOCITY
data itself is already validated elsewhere -- TestVectorVelocityCrossValidation).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracer_flow_panel import TracerFlowPanel  # noqa: E402
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
    rotational flow) so particles have genuine motion to advect through,
    not a flat/zero field -- scaled by case_index so scenarios differ."""
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
    p = TracerFlowPanel(provider, manifest, fps=4, n_particles=30)
    p.ensure_loaded()
    return p


def test_scenario_combo_populated_from_manifest(panel, manifest):
    assert panel.scenario_combo.count() == len(manifest)


def test_renders_a_real_particle_pool_without_error(panel):
    """The whole point: a figure with actual scatter/trail artists gets
    produced from known-good U/W data."""
    ax = panel.canvas.fig.axes[0]
    assert len(ax.collections) > 0, "expected scatter + trail LineCollection artists"


def test_colorbar_is_added_and_labeled_with_speed_units(panel):
    assert len(panel.canvas.fig.axes) >= 2, "expected a colorbar axes alongside the plot axes"
    cbar_ax = panel.canvas.fig.axes[-1]
    assert "m/s" in cbar_ax.get_ylabel()


def test_particle_count_control_matches_constructor_default(panel):
    assert panel.count_spin.value() == 30


def test_particle_count_is_a_tunable_constructor_parameter(qapp, provider, manifest):
    p = TracerFlowPanel(provider, manifest, fps=4, n_particles=75)
    assert p.count_spin.value() == 75


def test_axis_limits_match_the_vector_field_extent(panel):
    ax = panel.canvas.fig.axes[0]
    x0, x1, z0, z1 = EXTENT
    assert ax.get_xlim() == pytest.approx((x0, x1))
    assert ax.get_ylim() == pytest.approx((z0, z1))


def test_gated_scenario_shows_honest_message_not_a_fabricated_plot(qapp, manifest):
    """No U/W data for this scenario -- must show the gate reason, never
    draw made-up particles."""
    gated_provider = FakeProvider(gated_cases=(0,))
    p = TracerFlowPanel(gated_provider, manifest, fps=4, n_particles=30)
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


def test_particles_are_seeded_within_room_bounds(panel):
    """Room-bounded seeding, same convention as streamline_panel.py's
    _seed_points -- a freshly *spawned* particle should start inside the
    (extent-clamped) room, not scattered across the full domain including
    outside walls. Checked via _spawn_positions() directly rather than
    panel.pos after a render, since a live particle's advected position is
    only bounded by the domain extent, not the room, until it respawns."""
    from schematic import ROOM_X, ROOM_Z

    case_index = panel.scenario_combo.currentData()
    pool = panel._pools[case_index]
    fresh = pool._spawn_positions(200)
    x0, x1 = ROOM_X
    z0, z1 = ROOM_Z
    assert np.all(fresh[:, 0] >= x0 - 1e-9) and np.all(fresh[:, 0] <= x1 + 1e-9)
    assert np.all(fresh[:, 1] >= z0 - 1e-9) and np.all(fresh[:, 1] <= z1 + 1e-9)


def test_advection_moves_particles_between_frames(panel):
    """The whole point of a tracer panel over a static plot: particles
    actually move, driven by the real field, as playback advances."""
    case_index = panel.scenario_combo.currentData()
    pos_before = panel._pools[case_index].pos.copy()
    panel._current_index += 1
    panel._render()
    pos_after = panel._pools[case_index].pos.copy()
    assert not np.array_equal(pos_before, pos_after)


def test_rerender_at_same_frame_does_not_re_advect(panel):
    """Re-showing the tab or moving an unrelated control (particle count,
    reseed) must not silently double-advect a frame that hasn't actually
    changed on the shared playback clock."""
    case_index = panel.scenario_combo.currentData()
    panel._render()
    pos_a = panel._pools[case_index].pos.copy()
    panel._render()
    pos_b = panel._pools[case_index].pos.copy()
    assert np.array_equal(pos_a, pos_b)


def test_reseed_scatters_particles_to_new_positions(panel):
    case_index = panel.scenario_combo.currentData()
    pos_before = panel._pools[case_index].pos.copy()
    panel._on_reseed_clicked()
    pos_after = panel._pools[case_index].pos.copy()
    assert not np.array_equal(pos_before, pos_after)


def test_changing_particle_count_resizes_the_pool(panel):
    case_index = panel.scenario_combo.currentData()
    panel.count_spin.setValue(60)
    assert panel._pools[case_index].n == 60
