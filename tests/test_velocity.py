"""V6-M3 True Velocity: validates the pure engine in velocity.py against
independently-derived closed-form flows (uniform flow, solid-body vortex),
never by calling the same code path twice."""

from __future__ import annotations

import numpy as np
import pytest

import velocity as vel
from quantity_provider import QuantityProvider, GatedQuantityError
from slice_key import SliceKey


def _grid(shape, extent, u_fn, w_fn, n_frames=1):
    """A (t, z, x) stack for both components from analytic u(x,z)/w(x,z)."""
    n_z, n_x = shape
    x0, x1, z0, z1 = extent
    xs = np.linspace(x0, x1, n_x)
    zs = np.linspace(z1, z0, n_z)   # row 0 = z1 (top), matching the app's convention
    xx, zz = np.meshgrid(xs, zs)
    u = u_fn(xx, zz)
    w = w_fn(xx, zz)
    u_stack = np.stack([u] * n_frames)
    w_stack = np.stack([w] * n_frames)
    return u_stack, w_stack


class FakeVectorStore:
    """A minimal ScenarioStore stand-in for QuantityProvider: TEMPERATURE is
    irrelevant here, only U-VELOCITY/W-VELOCITY matter."""

    def __init__(self, u_stack, w_stack, extent, gated=False):
        self._u = u_stack
        self._w = w_stack
        self._extent = extent
        self._gated = gated

    def get(self, scenario, key):
        if self._gated:
            raise AssertionError("store.get should never be reached when gated=True "
                                 "-- QuantityProvider.get() must raise before this")
        if key.quantity == "U-VELOCITY":
            return self._u
        if key.quantity == "W-VELOCITY":
            return self._w
        raise KeyError(key.quantity)

    def get_extent(self, scenario, key):
        return self._extent


class FakeProvider:
    """Duck-typed VectorField provider that bypasses the registry's gating
    entirely, for pure-engine math tests (gating itself is exercised
    separately, against the real QuantityProvider, in TestGating below)."""

    def __init__(self, u_stack, w_stack, extent, v_stack=None):
        self._u = u_stack
        self._w = w_stack
        self._v = v_stack
        self._extent = extent

    def get_vector(self, scenario, direction=None, offset=None):
        return self._u, self._w

    def get_vector3d(self, scenario, direction=None, offset=None):
        if self._v is None:
            from quantity_provider import GatedQuantityError
            raise GatedQuantityError("Requires the M-SIM cluster re-run")
        return self._u, self._v, self._w

    def get_extent(self, scenario, key):
        return self._extent


class TestStrideForTarget:
    def test_matches_target_approximately(self):
        stride = vel.stride_for_target((100, 100), target_count=100)
        n_points = len(range(0, 100, stride)) ** 2
        assert 50 <= n_points <= 400   # loose -- it's a heuristic, not exact

    def test_never_zero(self):
        assert vel.stride_for_target((10, 10), target_count=10_000) >= 1


class TestQuiverGrid:
    def test_grid_positions_and_stride(self):
        shape = (5, 5)
        extent = (0.0, 4.0, 0.0, 4.0)
        u = np.full(shape, 2.0)
        w = np.full(shape, 0.0)
        xs, zs, us, ws = vel.quiver_grid(u, w, extent, stride=2)
        # rows/cols 0, 2, 4 -> 3x3 = 9 points
        assert len(xs) == 9
        assert np.allclose(us, 2.0) and np.allclose(ws, 0.0)
        assert xs.min() == pytest.approx(0.0) and xs.max() == pytest.approx(4.0)


class TestUniformFlowStreamline:
    def test_straight_line_at_constant_speed(self):
        extent = (-5.0, 5.0, -5.0, 5.0)
        u_stack, w_stack = _grid((81, 81), extent, lambda x, z: np.full_like(x, 1.0),
                                 lambda x, z: np.full_like(x, 0.0))
        seed = (-2.0, 0.0)
        points = vel.integrate_streamline(u_stack[0], w_stack[0], extent, seed,
                                          step=0.1, max_steps=20, max_length=100.0,
                                          method="rk4")
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        assert np.allclose(zs, 0.0, atol=1e-8)                  # pure x-motion
        assert xs[-1] == pytest.approx(-2.0 + 0.1 * 20, abs=1e-6)  # dx/dt=1 -> exact

    def test_stops_at_max_length(self):
        extent = (-5.0, 5.0, -5.0, 5.0)
        u_stack, w_stack = _grid((41, 41), extent, lambda x, z: np.full_like(x, 1.0),
                                 lambda x, z: np.full_like(x, 0.0))
        points = vel.integrate_streamline(u_stack[0], w_stack[0], extent, (0.0, 0.0),
                                          step=0.1, max_steps=1000, max_length=1.0)
        total = sum(np.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
                   for i in range(len(points) - 1))
        assert total <= 1.0 + 1e-9

    def test_stops_at_domain_edge(self):
        extent = (0.0, 1.0, -1.0, 1.0)
        u_stack, w_stack = _grid((21, 21), extent, lambda x, z: np.full_like(x, 1.0),
                                 lambda x, z: np.full_like(x, 0.0))
        points = vel.integrate_streamline(u_stack[0], w_stack[0], extent, (0.9, 0.0),
                                          step=0.5, max_steps=100, max_length=100.0)
        assert points[-1][0] <= 1.0 + 1e-9


class TestVortexStreamline:
    """Solid-body rotation u=-k*z, w=k*x -- an affine field, so bilinear
    interpolation reproduces it exactly; RK4's only error is the fixed-step
    time discretization, tightly bounded for a short arc."""

    def test_matches_closed_form_rotation(self):
        k = 1.0
        extent = (-2.0, 2.0, -2.0, 2.0)
        u_stack, w_stack = _grid((161, 161), extent,
                                 lambda x, z: -k * z, lambda x, z: k * x)
        seed = (1.0, 0.0)
        step, n = 0.01, 100
        points = vel.integrate_streamline(u_stack[0], w_stack[0], extent, seed,
                                          step=step, max_steps=n, max_length=100.0,
                                          method="rk4")
        t = step * n
        expected_x, expected_z = np.cos(k * t), np.sin(k * t)
        assert points[-1][0] == pytest.approx(expected_x, abs=1e-3)
        assert points[-1][1] == pytest.approx(expected_z, abs=1e-3)
        # radius is conserved for true solid-body rotation
        radius = np.hypot(points[-1][0], points[-1][1])
        assert radius == pytest.approx(1.0, abs=1e-3)

    def test_rk4_more_accurate_than_euler(self):
        k = 1.0
        extent = (-2.0, 2.0, -2.0, 2.0)
        u_stack, w_stack = _grid((161, 161), extent,
                                 lambda x, z: -k * z, lambda x, z: k * x)
        seed = (1.0, 0.0)
        step, n = 0.05, 40   # a coarser step to make integration error visible
        t = step * n
        expected = np.array([np.cos(k * t), np.sin(k * t)])
        rk4 = vel.integrate_streamline(u_stack[0], w_stack[0], extent, seed,
                                       step=step, max_steps=n, max_length=100.0, method="rk4")
        euler = vel.integrate_streamline(u_stack[0], w_stack[0], extent, seed,
                                         step=step, max_steps=n, max_length=100.0, method="euler")
        err_rk4 = np.hypot(*(np.array(rk4[-1]) - expected))
        err_euler = np.hypot(*(np.array(euler[-1]) - expected))
        assert err_rk4 < err_euler


class TestUnknownMethod:
    def test_raises_value_error(self):
        extent = (0.0, 1.0, 0.0, 1.0)
        u = np.zeros((3, 3)); w = np.zeros((3, 3))
        with pytest.raises(ValueError):
            vel.integrate_streamline(u, w, extent, (0.5, 0.5), method="bogus")


class TestVectorField:
    def _field(self, n_frames=5):
        extent = (-2.0, 2.0, -2.0, 2.0)
        u_stack, w_stack = _grid((41, 41), extent, lambda x, z: np.full_like(x, 3.0),
                                 lambda x, z: np.full_like(x, 4.0), n_frames=n_frames)
        provider = FakeProvider(u_stack, w_stack, extent)
        field = vel.VectorField(provider, scenario=0)
        field.compute()
        return field

    def test_speed_and_angle_precomputed(self):
        field = self._field()
        assert field.speed.shape == field.u.shape
        assert np.allclose(field.speed, 5.0)          # 3-4-5 triangle
        assert np.allclose(field.angle, np.arctan2(4.0, 3.0))

    def test_quiver_at_matches_quiver_grid(self):
        field = self._field()
        xs, zs, us, ws = field.quiver_at(0, density=50)
        stride = vel.stride_for_target(field.u.shape[1:], 50)
        xs2, zs2, us2, ws2 = vel.quiver_grid(field.u[0], field.w[0], field.extent, stride)
        np.testing.assert_allclose(xs, xs2)
        np.testing.assert_allclose(us, us2)

    def test_probe_speed_is_constant(self):
        field = self._field()
        series = field.probe_speed(0.5, 0.5)
        assert series.shape == (field.n_frames,)
        assert np.allclose(series, 5.0)

    def test_streamline_at_is_memoized(self):
        field = self._field()
        a = field.streamline_at((-1.0, -1.0), frame_index=0)
        b = field.streamline_at((-1.0, -1.0), frame_index=0)
        assert a is b                       # cache hit, not recomputed
        field.compute()                     # recompute clears the cache
        c = field.streamline_at((-1.0, -1.0), frame_index=0)
        assert c is not a

    def test_n_frames(self):
        field = self._field(n_frames=7)
        assert field.n_frames == 7


class TestVectorField3D:
    """V6-M7: the optional true-3D enhancement (V-VELOCITY/compute_v) --
    the 2D (U, W) baseline must be completely unaffected either way."""

    def _field_2d_only(self, n_frames=5):
        extent = (-2.0, 2.0, -2.0, 2.0)
        u_stack, w_stack = _grid((41, 41), extent, lambda x, z: np.full_like(x, 3.0),
                                 lambda x, z: np.full_like(x, 4.0), n_frames=n_frames)
        provider = FakeProvider(u_stack, w_stack, extent)   # no v_stack -- V gated
        field = vel.VectorField(provider, scenario=0)
        field.compute()
        return field

    def _field_3d(self, n_frames=5):
        extent = (-2.0, 2.0, -2.0, 2.0)
        shape = (41, 41)
        u_stack, w_stack = _grid(shape, extent, lambda x, z: np.full_like(x, 3.0),
                                 lambda x, z: np.full_like(x, 4.0), n_frames=n_frames)
        v_stack, _ = _grid(shape, extent, lambda x, z: np.full_like(x, 12.0),
                           lambda x, z: np.full_like(x, 0.0), n_frames=n_frames)
        provider = FakeProvider(u_stack, w_stack, extent, v_stack=v_stack)
        field = vel.VectorField(provider, scenario=0)
        field.compute()
        return field, provider

    def test_has_3d_false_until_compute_v(self):
        field = self._field_2d_only()
        assert not field.has_3d
        assert field.v is None and field.speed3d is None

    def test_compute_v_requires_compute_first(self):
        extent = (-1.0, 1.0, -1.0, 1.0)
        u_stack, w_stack = _grid((5, 5), extent, lambda x, z: x, lambda x, z: z)
        provider = FakeProvider(u_stack, w_stack, extent)
        field = vel.VectorField(provider, scenario=0)
        with pytest.raises(RuntimeError):
            field.compute_v()

    def test_compute_v_raises_gated_when_v_absent(self):
        field = self._field_2d_only()
        with pytest.raises(GatedQuantityError):
            field.compute_v()
        assert not field.has_3d   # 2D field untouched by the failed attempt

    def test_compute_v_populates_speed3d(self):
        field, _provider = self._field_3d()
        field.compute_v()
        assert field.has_3d
        # 3-4-12-13 Pythagorean quadruple: sqrt(3^2+12^2+4^2) = 13
        assert np.allclose(field.speed3d, 13.0)

    def test_quiver_at_3d_matches_quiver_grid_with_v(self):
        field, _provider = self._field_3d()
        field.compute_v()
        xs, zs, us, ws, vs = field.quiver_at_3d(0, density=50)
        stride = vel.stride_for_target(field.u.shape[1:], 50)
        xs2, zs2, us2, ws2, vs2 = vel.quiver_grid(field.u[0], field.w[0], field.extent,
                                                   stride, v_frame=field.v[0])
        np.testing.assert_allclose(vs, vs2)
        assert np.allclose(vs, 12.0)

    def test_quiver_at_3d_requires_compute_v(self):
        field = self._field_2d_only()
        with pytest.raises(RuntimeError):
            field.quiver_at_3d(0)

    def test_quiver_at_unaffected_by_3d_availability(self):
        """quiver_at() (the pre-existing 4-tuple) must return identically
        whether or not compute_v() was ever called."""
        field, _provider = self._field_3d()
        before = field.quiver_at(0, density=50)
        field.compute_v()
        after = field.quiver_at(0, density=50)
        assert len(before) == 4 and len(after) == 4
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b)

    def test_probe_v_requires_compute_v(self):
        field = self._field_2d_only()
        with pytest.raises(RuntimeError):
            field.probe_v(0.0, 0.0)

    def test_probe_v_is_constant(self):
        field, _provider = self._field_3d()
        field.compute_v()
        series = field.probe_v(0.5, 0.5)
        assert np.allclose(series, 12.0)


class TestGating:
    """The real QuantityProvider.get_vector must still raise
    GatedQuantityError, unchanged, pointing at the M-SIM doc -- V6-M3 must
    not weaken this."""

    def test_get_vector_raises_when_ungated_data_absent(self):
        class DummyStore:
            def get(self, scenario, key):
                raise AssertionError("should never reach the store -- gate must fire first")

            def get_extent(self, scenario, key):
                raise AssertionError("should never reach the store")

        provider = QuantityProvider(DummyStore(), fps=1)
        with pytest.raises(GatedQuantityError) as exc:
            provider.get_vector(0)
        assert "msim-preparation" in str(exc.value) or "M-SIM" in str(exc.value)

    def test_vector_field_compute_propagates_gate(self):
        class DummyStore:
            def get(self, scenario, key):
                raise AssertionError("gate must fire before the store is touched")

            def get_extent(self, scenario, key):
                raise AssertionError("gate must fire before the store is touched")

        provider = QuantityProvider(DummyStore(), fps=1)
        field = vel.VectorField(provider, scenario=0)
        with pytest.raises(GatedQuantityError):
            field.compute()
        assert field.u is None                 # never fabricated a fallback

    def test_get_vector3d_raises_when_v_absent(self):
        """V6-M7: get_vector3d must gate exactly like get_vector -- V is
        registered gated=True, same as U/W, so this fires before the store
        is ever touched."""
        class DummyStore:
            def get(self, scenario, key):
                raise AssertionError("should never reach the store -- gate must fire first")

            def get_extent(self, scenario, key):
                raise AssertionError("should never reach the store")

        provider = QuantityProvider(DummyStore(), fps=1)
        with pytest.raises(GatedQuantityError) as exc:
            provider.get_vector3d(0)
        assert "msim-preparation" in str(exc.value) or "M-SIM" in str(exc.value)

    def test_vector_field_compute_v_propagates_gate(self):
        class DummyStore:
            def get(self, scenario, key):
                raise AssertionError("gate must fire before the store is touched")

            def get_extent(self, scenario, key):
                raise AssertionError("gate must fire before the store is touched")

        provider = QuantityProvider(DummyStore(), fps=1)
        field = vel.VectorField(provider, scenario=0)
        field.u = np.zeros((1, 1, 1))   # satisfy compute_v()'s precondition check
        field.w = np.zeros((1, 1, 1))
        with pytest.raises(GatedQuantityError):
            field.compute_v()
        assert field.v is None and not field.has_3d


def test_volume_sample_is_not_implemented():
    with pytest.raises(NotImplementedError):
        vel.volume_sample(provider=None, scenario=0, x=0.0, y=0.0, z=0.0)


class TestVectorProbe:
    def _field(self):
        extent = (-2.0, 2.0, -2.0, 2.0)
        u_stack, w_stack = _grid((41, 41), extent, lambda x, z: np.full_like(x, 3.0),
                                 lambda x, z: np.full_like(x, 4.0), n_frames=6)
        provider = FakeProvider(u_stack, w_stack, extent)
        field = vel.VectorField(provider, scenario=0)
        field.compute()
        return field

    def test_compute_populates_results(self):
        probe = vel.VectorProbe(id="p1", name="VP-01", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field(), fps=2)
        assert probe.results is not None and not probe.gated
        assert probe.results["max_speed_m_s"] == pytest.approx(5.0)
        assert len(probe.results["time_s"]) == 6

    def test_state_at_indexes_cached_speed(self):
        probe = vel.VectorProbe(id="p2", name="VP-02", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field(), fps=2)
        assert probe.state_at(3)["speed_m_s"] == pytest.approx(5.0)

    def test_mark_gated_fabricates_nothing(self):
        probe = vel.VectorProbe(id="p3", name="VP-03", scenario=0, position=(0.0, 0.0))
        probe.mark_gated("Requires the M-SIM cluster re-run")
        assert probe.gated
        assert probe.state_at(0)["speed_m_s"] is None

    def test_session_round_trip_is_identical(self):
        probe = vel.VectorProbe(id="p4", name="VP-04", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field(), fps=2)
        restored = vel.VectorProbe.from_dict(probe.to_dict())
        assert restored.to_dict() == probe.to_dict()

    def test_summary_insight_reports_peak_speed(self):
        probe = vel.VectorProbe(id="p5", name="VP-05", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field(), fps=2)
        ins = probe.summary_insight()
        assert "5.0" in ins.statement

    def test_summary_insight_reports_gate_reason(self):
        probe = vel.VectorProbe(id="p6", name="VP-06", scenario=0, position=(0.0, 0.0))
        probe.mark_gated("Requires the M-SIM cluster re-run")
        ins = probe.summary_insight()
        assert "gated" in ins.statement.lower()

    def test_export_csv(self, tmp_path):
        probe = vel.VectorProbe(id="p7", name="VP-07", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field(), fps=2)
        path = tmp_path / "vp.csv"
        vel.export_csv(probe, str(path))
        text = path.read_text()
        assert "time_s,speed_m_s,angle_deg" in text

    def _field_3d(self):
        extent = (-2.0, 2.0, -2.0, 2.0)
        shape = (41, 41)
        u_stack, w_stack = _grid(shape, extent, lambda x, z: np.full_like(x, 3.0),
                                 lambda x, z: np.full_like(x, 4.0), n_frames=6)
        v_stack, _ = _grid(shape, extent, lambda x, z: np.full_like(x, 12.0),
                           lambda x, z: np.full_like(x, 0.0), n_frames=6)
        provider = FakeProvider(u_stack, w_stack, extent, v_stack=v_stack)
        field = vel.VectorField(provider, scenario=0)
        field.compute()
        field.compute_v()
        return field

    def test_compute_with_3d_field_adds_speed3d(self):
        probe = vel.VectorProbe(id="p8", name="VP-08", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field_3d(), fps=2)
        assert probe.results["max_speed3d_m_s"] == pytest.approx(13.0)   # 3-4-12 -> 13
        assert probe.results["v_m_s"][0] == pytest.approx(12.0)
        assert "3D" in probe.results["basis"]

    def test_compute_without_3d_field_omits_speed3d(self):
        probe = vel.VectorProbe(id="p9", name="VP-09", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field(), fps=2)   # 2D-only field (no compute_v())
        assert "max_speed3d_m_s" not in probe.results
        assert "v_m_s" not in probe.results

    def test_summary_insight_prefers_3d_speed_when_available(self):
        probe = vel.VectorProbe(id="p10", name="VP-10", scenario=0, position=(0.5, 0.5))
        probe.compute(self._field_3d(), fps=2)
        ins = probe.summary_insight()
        assert "3D peak speed" in ins.statement and "13.0" in ins.statement

    def test_export_csv_gated_still_writes_metadata(self, tmp_path):
        probe = vel.VectorProbe(id="p8", name="VP-08", scenario=0, position=(0.0, 0.0))
        probe.mark_gated("Requires the M-SIM cluster re-run")
        path = tmp_path / "vp_gated.csv"
        vel.export_csv(probe, str(path))
        text = path.read_text()
        assert "gated" in text.lower()
