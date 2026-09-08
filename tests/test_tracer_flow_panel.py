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
    outside walls."""
    from schematic import ROOM_X, ROOM_Z

    case_index = panel.scenario_combo.currentData()
    pool = panel._pools[case_index]
    fresh = pool._spawn_positions(200)
    x0, x1 = ROOM_X
    z0, z1 = ROOM_Z
    assert np.all(fresh[:, 0] >= x0 - 1e-9) and np.all(fresh[:, 0] <= x1 + 1e-9)
    assert np.all(fresh[:, 1] >= z0 - 1e-9) and np.all(fresh[:, 1] <= z1 + 1e-9)


def test_advected_particles_stay_inside_the_room_except_through_openings(qapp):
    """Regression: a live (advected) particle is contained by the room's
    own walls/ceiling, not the far-off domain edge -- it may only leave
    through the doorway's z-span (left wall) or an open vent's x-span
    (ceiling), never a solid border. Before the fix, particles drifted
    out the corridor side and their dots/trails rendered outside the
    drawn room outline where there is no opening.

    Driven directly through _ParticlePool.step() with strong uniform
    fields aimed at each border, so every particle is pushed hard against
    a wall/ceiling within a few ticks."""
    import tracer_flow_panel as tfp

    room = (0.27, 1.0, 0.0, 0.22)
    open_spans, duct_top = [(0.32, 0.40), (0.86, 0.94)], 0.24
    door_span = (0.0, 0.06)
    nz, nx = 12, 16
    ext = (0.0, 1.0, 0.0, 0.5)

    for ux, wz in ((-3.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, -3.0)):
        pool = tfp._ParticlePool(200, room, open_vent_spans=open_spans,
                                 duct_top_z=duct_top, door_z_span=door_span, seed=1)
        u = np.full((nz, nx), ux)
        w = np.full((nz, nx), wz)
        for _ in range(40):
            pool.step(u, w, ext, dt=0.25)
        p = pool.pos
        assert np.all(p[:, 0] <= room[1] + 1e-9), "passed through the solid right wall"
        assert np.all(p[:, 1] >= room[2] - 1e-9), "passed through the solid floor"

        left_out = p[p[:, 0] < room[0] - 1e-9]
        assert np.all((left_out[:, 1] >= door_span[0] - 1e-9)
                      & (left_out[:, 1] <= door_span[1] + 1e-9)), \
            "left through a solid part of the left wall"
        assert np.all(left_out[:, 0] >= room[0] - tfp._DOOR_EXIT_DEPTH - 1e-9)

        top_out = p[p[:, 1] > room[3] + 1e-9]
        for px, pz in top_out:
            assert any(a - 1e-9 <= px <= b + 1e-9 for a, b in open_spans), \
                "left through solid ceiling"
            assert pz <= duct_top + 1e-9


def test_left_wall_is_solid_above_the_doorway(qapp):
    """Same leftward nudge just past the wall line: a particle at door
    height is allowed through the opening (sits in the exit allowance);
    one above the door lintel hits solid wall and respawns inside."""
    import tracer_flow_panel as tfp

    room = (0.27, 1.0, 0.0, 0.22)
    pool = tfp._ParticlePool(2, room, open_vent_spans=[], duct_top_z=None,
                             door_z_span=(0.0, 0.06), seed=0)
    pool.pos[:] = [[0.30, 0.03], [0.30, 0.15]]   # one at door height, one above the lintel
    pool.trail[:] = pool.pos[:, None, :]
    u = np.full((12, 16), -0.16)   # dx = -0.04 at dt=0.25 -> x = 0.26, just past the wall
    w = np.zeros((12, 16))
    pool.step(u, w, (0.0, 1.0, 0.0, 0.5), dt=0.25)

    at_door, above_lintel = pool.pos[0], pool.pos[1]
    assert room[0] - tfp._DOOR_EXIT_DEPTH - 1e-9 <= at_door[0] < room[0], \
        "a door-height particle should pass into the exit allowance, not respawn"
    assert above_lintel[0] >= room[0] - 1e-9, \
        "an above-lintel particle must be blocked by the solid wall and respawn inside"


def _shear_field(A=3.0, nz=41, nx=41):
    """Smooth, fast, non-uniform field (u ~ sin pi z, w ~ sin pi x) -- a
    curved flow where a single Euler step at dt=0.25 badly overshoots."""
    zc = np.linspace(0, 1, nz)[:, None]
    xc = np.linspace(0, 1, nx)[None, :]
    u = A * np.sin(np.pi * zc) * np.ones_like(xc)
    w = A * np.sin(np.pi * xc) * np.ones_like(zc)
    return u, w


def test_fast_field_triggers_adaptive_substepping(qapp):
    """A ~4 m/s field on a 25 mm grid at dt=0.25 s would leap ~40 cells in
    one Euler step -- step() must split it into many sub-steps. A slow
    field takes exactly one."""
    import tracer_flow_panel as tfp
    ext = (0.0, 1.0, 0.0, 1.0)
    pool = tfp._ParticlePool(50, (0.0, 1.0, 0.0, 1.0), seed=0)

    u, w = _shear_field(A=4.0)
    pool.step(u, w, ext, dt=0.25)
    assert pool._last_substeps > 10, f"fast field should sub-step a lot, got k={pool._last_substeps}"

    pool.step(u * 1e-3, w * 1e-3, ext, dt=0.25)
    assert pool._last_substeps == 1, "a slow field must not sub-step"


def test_substepping_tracks_the_curved_path_euler_overshoots(qapp):
    """Against a fine reference integration of the same field, the
    adaptive step lands close to the true trajectory; a forced single
    Euler step (k=1) overshoots it by a wide margin."""
    import tracer_flow_panel as tfp
    ext = (0.0, 1.0, 0.0, 1.0)
    u, w = _shear_field(A=2.0)          # ~2.8 m/s peak -- k hits the cap
    start = np.array([[0.5, 0.35]])

    def integrate(steps):
        p = tfp._ParticlePool(1, (0.0, 1.0, 0.0, 1.0), seed=0)
        p.pos[:] = start
        dt = 0.25 / steps
        for _ in range(steps):
            uu, ww = tfp._sample_uw(u, w, ext, p.pos[:, 0], p.pos[:, 1])
            p.pos[:, 0] += dt * uu
            p.pos[:, 1] += dt * ww
        return p.pos.copy()

    ref = integrate(8000)

    pool = tfp._ParticlePool(1, (0.0, 1.0, 0.0, 1.0), seed=0)
    pool.pos[:] = start
    pool.step(u, w, ext, dt=0.25)
    adaptive_err = float(np.hypot(*(pool.pos - ref).ravel()))

    saved = tfp._MAX_SUBSTEPS
    tfp._MAX_SUBSTEPS = 1
    try:
        pe = tfp._ParticlePool(1, (0.0, 1.0, 0.0, 1.0), seed=0)
        pe.pos[:] = start
        pe.step(u, w, ext, dt=0.25)
        euler_err = float(np.hypot(*(pe.pos - ref).ravel()))
    finally:
        tfp._MAX_SUBSTEPS = saved

    assert adaptive_err < 0.03, f"adaptive step should track the path, err={adaptive_err:.4f}"
    assert euler_err > 8 * adaptive_err, \
        f"single Euler step should overshoot far more (euler {euler_err:.4f} vs adaptive {adaptive_err:.4f})"


def test_fast_particle_trail_is_a_resolved_curve_not_a_dot(qapp):
    """Each sub-step feeds one trail slot, so after a fast step a moving
    particle's TRAIL_LEN slots trace an arc (uniform resolution, no
    coarse/fine seam) rather than collapsing to its endpoint."""
    import tracer_flow_panel as tfp
    ext = (0.0, 1.0, 0.0, 1.0)
    u, w = _shear_field(A=2.0)
    pool = tfp._ParticlePool(40, (0.0, 1.0, 0.0, 1.0), seed=1)
    pool.pos[:] = np.column_stack([np.full(40, 0.5), np.linspace(0.2, 0.8, 40)])
    pool.trail[:] = pool.pos[:, None, :]
    pool.step(u, w, ext, dt=0.25)

    moved = np.hypot(*(pool.trail[:, -1, :] - pool.trail[:, 0, :]).T) > 3e-3
    assert moved.any()
    bbox = np.hypot(*(pool.trail.max(axis=1) - pool.trail.min(axis=1)).T)
    assert np.all(bbox[moved] > 3e-3), "a fast particle's trail collapsed to ~a point"
    # slots advance monotonically along the path (no back-and-forth wobble)
    hop = np.hypot(*np.diff(pool.trail, axis=1).transpose(2, 0, 1))
    net = np.hypot(*(pool.trail[:, -1, :] - pool.trail[:, 0, :]).T)
    assert np.all(hop[moved].sum(axis=1) < 1.5 * net[moved] + 1e-6), "trail zig-zags instead of tracing the arc"


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
