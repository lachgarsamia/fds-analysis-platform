"""V6-M2 Virtual Device Network: validates the pure engine in devices.py
against independently-derived expected values (never by calling the same
code path twice)."""

from __future__ import annotations

import numpy as np
import pytest

import devices as dv
import tenability as tn
from quantity_provider import QuantityProvider


class FakeStore:
    """A minimal ScenarioStore stand-in: uniform-per-frame fields, so a
    bilinear probe at any point returns exactly the frame's scalar value."""

    def __init__(self, temp, velocity=None, co=None, extent=(0.0, 2.0, 0.0, 2.0)):
        self._temp = temp
        self._velocity = velocity
        self._co = co
        self._extent = extent

    def get(self, scenario, key):
        if key.quantity == "TEMPERATURE":
            return self._temp
        if key.quantity == "VELOCITY":
            if self._velocity is None:
                raise KeyError("VELOCITY not available in this fixture")
            return self._velocity
        if key.quantity == "CARBON MONOXIDE VOLUME FRACTION":
            if self._co is None:
                from quantity_provider import GatedQuantityError
                raise GatedQuantityError("Requires the M-SIM cluster re-run")
            return self._co
        raise KeyError(key.quantity)

    def get_extent(self, scenario, key):
        return self._extent


def _uniform_field(values, shape=(3, 3)):
    return np.stack([np.full(shape, v, dtype=float) for v in values])


class TestThermocouple:
    def test_matches_probe_and_maximum(self):
        values = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_thermocouple(provider, 0, 1.0, 1.0, fps=1)
        assert r["temperature_C"] == pytest.approx(values)
        assert r["max_temperature_C"] == pytest.approx(120.0)

    def test_fed_heat_is_populated_and_monotonic(self):
        values = [20.0 + 10.0 * i for i in range(11)]
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_thermocouple(provider, 0, 1.0, 1.0, fps=1)
        assert len(r["fed_heat"]) == 11
        assert all(b > a for a, b in zip(r["fed_heat"], r["fed_heat"][1:]))
        assert r["max_fed_heat"] == pytest.approx(max(r["fed_heat"]))

    def test_fed_full_is_none_when_co_gated(self):
        """The real registry gates CO -- QuantityProvider raises
        GatedQuantityError before the store is ever touched, regardless of
        what the store itself would return."""
        values = [20.0 + 10.0 * i for i in range(5)]
        provider = QuantityProvider(FakeStore(_uniform_field(values), co=_uniform_field(values)), fps=1)
        r = dv.compute_thermocouple(provider, 0, 1.0, 1.0, fps=1)
        assert r["fed_full"] is None and r["max_fed_full"] is None

    def test_fed_full_combines_heat_and_gas_dose_when_co_available(self):
        """A duck-typed fake *provider* (not wrapped in the real
        QuantityProvider, which always gates CO via the registry) simulates
        the post-M-SIM-re-run case where CO is real."""
        n = 5
        temp = _uniform_field([200.0] * n)
        co = _uniform_field([5000.0] * n)
        fake_provider = FakeStore(temp, co=co)   # duck-typed: .get/.get_extent only
        r = dv.compute_thermocouple(fake_provider, 0, 1.0, 1.0, fps=1)
        assert r["fed_full"] is not None
        co_probed = np.full((n, 1, 1), 5000.0)   # the bilinearly-probed (uniform) CO value
        expected = np.asarray(r["fed_heat"]) + tn.fed_gas_dose(co_probed, 1).reshape(-1)
        np.testing.assert_allclose(r["fed_full"], expected)
        assert r["max_fed_full"] == pytest.approx(max(r["fed_full"]))

    def test_heating_rate(self):
        # 10 C/s ramp -> max heating rate is exactly 10.
        values = [20.0 + 10.0 * i for i in range(11)]
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_thermocouple(provider, 0, 1.0, 1.0, fps=1)
        assert r["heating_rate_C_per_s"] == pytest.approx(10.0)

    def test_threshold_crossing_times(self):
        # idx: 0..10 -> 20,30,...,120. >=60 first at idx 4 (t=4s); >=100 at idx 8 (t=8s).
        values = [20.0 + 10.0 * i for i in range(11)]
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_thermocouple(provider, 0, 1.0, 1.0, fps=1)
        tt = r["threshold_times_s"]
        assert tt["60"] == pytest.approx(4.0)
        assert tt["100"] == pytest.approx(8.0)
        assert tt["300"] is None   # never reached


class TestHeatDetector:
    def test_synthetic_ramp_activates(self):
        # 20 -> 100 C over 11 frames (10 C/frame @ fps=1); activation 74 C.
        values = [20.0 + 10.0 * i for i in range(11)]
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_heat_detector(provider, 0, 1.0, 1.0, fps=1, activation_temp=74.0)
        # first index with value >= 74 is idx 6 (value 80).
        assert r["activated"] is True
        assert r["activation_frame"] == 6
        assert r["activation_time_s"] == pytest.approx(6.0)
        assert r["activation_temperature_C"] == pytest.approx(80.0)

    def test_never_activates_below_threshold(self):
        values = [20.0] * 10
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_heat_detector(provider, 0, 1.0, 1.0, fps=1, activation_temp=74.0)
        assert r["activated"] is False
        assert r["activation_time_s"] is None
        assert r["activation_frame"] is None

    def test_basis_is_traceable(self):
        values = [20.0] * 5
        provider = QuantityProvider(FakeStore(_uniform_field(values)), fps=1)
        r = dv.compute_heat_detector(provider, 0, 1.0, 1.0, fps=1, activation_temp=74.0)
        assert "74" in r["basis"] and "threshold" in r["basis"]


class TestSprinklerRTI:
    def test_response_curve_matches_analytic_step_response(self):
        """Step gas temperature (ambient at t=0, constant Tg afterward) with
        no VELOCITY registered -> the reduced model (u == 1.0 m/s). The
        discrete RTI recurrence is linear, so its closed form is derived
        independently here (not by calling compute_sprinkler) and compared
        elementwise."""
        ambient, tg, rti, fps, n = 20.0, 300.0, 50.0, 2.0, 41
        temp = np.array([ambient] + [tg] * (n - 1))
        provider = QuantityProvider(FakeStore(_uniform_field(temp)), fps=int(fps))
        r = dv.compute_sprinkler(provider, 0, 1.0, 1.0, fps=int(fps),
                                 rti=rti, activation_temp=68.0)
        assert r["reduced_model"] is True

        dt = 1.0 / fps
        a = (np.sqrt(1.0) / rti) * dt   # u fixed at 1.0 in the reduced model
        expected = np.empty(n)
        expected[0] = ambient
        expected[1] = ambient  # temp[0]==link[0]==ambient -> no change on the first step
        for k in range(2, n):
            expected[k] = tg + (expected[k - 1] - tg) * (1 - a)
        np.testing.assert_allclose(r["link_temperature_C"], expected, rtol=1e-10)

        idx = np.where(expected >= 68.0)[0]
        expected_activation_time = float(idx[0] / fps) if idx.size else None
        assert r["activation_time_s"] == pytest.approx(expected_activation_time)

    def test_uses_velocity_when_available(self):
        ambient, tg, rti, fps, n = 20.0, 300.0, 50.0, 2.0, 21
        temp = np.array([ambient] + [tg] * (n - 1))
        velocity = np.full(n, 4.0)   # constant |u| = 4 m/s
        provider = QuantityProvider(
            FakeStore(_uniform_field(temp), velocity=_uniform_field(velocity)), fps=int(fps))
        r = dv.compute_sprinkler(provider, 0, 1.0, 1.0, fps=int(fps), rti=rti, activation_temp=68.0)
        assert r["reduced_model"] is False
        assert "VELOCITY" in r["basis"]
        # higher |u| -> faster response -> earlier (or equal) activation than the reduced model.
        reduced = dv.compute_sprinkler(
            QuantityProvider(FakeStore(_uniform_field(temp)), fps=int(fps)),
            0, 1.0, 1.0, fps=int(fps), rti=rti, activation_temp=68.0)
        assert r["activation_time_s"] <= reduced["activation_time_s"]


class TestDeviceLifecycle:
    def _provider(self):
        values = [20.0 + 10.0 * i for i in range(11)]
        return QuantityProvider(FakeStore(_uniform_field(values)), fps=1)

    def test_compute_populates_results_and_state(self):
        dev = dv.Device(id="d1", name="TC-01", type="thermocouple", scenario=0,
                        position=(1.0, 1.0))
        assert dev.results is None
        dev.compute(self._provider(), fps=1)
        assert dev.results is not None
        assert dev.n_frames() == 11
        s = dev.state_at(6)
        assert s["temperature_C"] == pytest.approx(80.0)

    def test_heat_detector_state_series_is_0_then_1(self):
        dev = dv.Device(id="d2", name="HD-01", type="heat_detector", scenario=0,
                        position=(1.0, 1.0), parameters={"activation_temp_C": 74.0})
        dev.compute(self._provider(), fps=1)
        state = dev.device_state_series()
        assert state[:6] == [0] * 6
        assert state[6:] == [1] * 5

    def test_session_round_trip_is_identical(self):
        dev = dv.Device(id="d3", name="TC-02", type="thermocouple", scenario=0,
                        position=(1.0, 1.0))
        dev.compute(self._provider(), fps=1)
        restored = dv.Device.from_dict(dev.to_dict())
        assert restored.to_dict() == dev.to_dict()

    def test_summary_insight_thermocouple(self):
        dev = dv.Device(id="d4", name="TC-03", type="thermocouple", scenario=0,
                        position=(1.0, 1.0))
        dev.compute(self._provider(), fps=1)
        ins = dev.summary_insight()
        assert "120.0" in ins.statement
        assert ins.basis

    def test_summary_insight_detector_activation(self):
        dev = dv.Device(id="d5", name="HD-02", type="heat_detector", scenario=0,
                        position=(1.0, 1.0), parameters={"activation_temp_C": 74.0})
        dev.compute(self._provider(), fps=1)
        ins = dev.summary_insight()
        assert "activated at 6.0 s" in ins.statement
        assert ins.time_s == pytest.approx(6.0)

    def test_export_csv(self, tmp_path):
        dev = dv.Device(id="d6", name="TC-04", type="thermocouple", scenario=0,
                        position=(1.0, 1.0))
        dev.compute(self._provider(), fps=1)
        path = tmp_path / "tc.csv"
        dv.export_csv(dev, str(path))
        text = path.read_text()
        assert "device_type" in text and "thermocouple" in text
        assert "time_s,temperature_C,device_state" in text


class FakeMultiPlaneStore:
    """A store with genuinely distinct data per (direction, offset) --
    lets a test tell whether a device actually read the plane it asked
    for, not just the app's default one."""

    def __init__(self, planes: dict, extent=(0.0, 2.0, 0.0, 2.0)):
        self._planes = planes   # {(direction, offset): temp_array}
        self._extent = extent

    def get(self, scenario, key):
        if key.quantity != "TEMPERATURE":
            raise KeyError(key.quantity)
        try:
            return self._planes[(key.direction, key.offset)]
        except KeyError:
            raise KeyError(f"no plane at direction={key.direction}, offset={key.offset}")

    def get_extent(self, scenario, key):
        return self._extent


class TestMultiPlane:
    """V6-M5: Device carries its own (direction, offset), defaulting to the
    app's usual plane -- devices on different planes read genuinely
    different data, and the default stays byte-identical to pre-V6-M5
    behavior."""

    def test_default_plane_unchanged(self):
        provider = QuantityProvider(
            FakeMultiPlaneStore({(1, 0): _uniform_field([20.0, 30.0, 40.0])}), fps=1)
        dev = dv.Device(id="p1", name="TC-01", type="thermocouple", scenario=0, position=(1.0, 1.0))
        assert dev.direction == 1 and dev.offset == 0   # DEFAULT_DIRECTION/DEFAULT_OFFSET
        dev.compute(provider, fps=1)
        assert dev.results["max_temperature_C"] == pytest.approx(40.0)

    def test_device_on_a_different_offset_reads_that_planes_data(self):
        provider = QuantityProvider(FakeMultiPlaneStore({
            (1, 0): _uniform_field([20.0, 30.0, 40.0]),
            (1, 15): _uniform_field([100.0, 200.0, 300.0]),
        }), fps=1)
        dev = dv.Device(id="p2", name="TC-02", type="thermocouple", scenario=0,
                        position=(1.0, 1.0), offset=15)
        dev.compute(provider, fps=1)
        assert dev.results["max_temperature_C"] == pytest.approx(300.0)

    def test_device_on_an_absent_plane_is_gated(self):
        from quantity_provider import GatedQuantityError
        provider = QuantityProvider(FakeMultiPlaneStore({(1, 0): _uniform_field([20.0])}), fps=1)
        dev = dv.Device(id="p3", name="TC-03", type="thermocouple", scenario=0,
                        position=(1.0, 1.0), direction=0)   # x-normal -- not in this fixture
        with pytest.raises(Exception):   # the fake raises KeyError; the real provider raises GatedQuantityError
            dev.compute(provider, fps=1)

    def test_session_round_trip_preserves_plane(self):
        dev = dv.Device(id="p4", name="TC-04", type="thermocouple", scenario=0,
                        position=(1.0, 1.0), direction=2, offset=7)
        restored = dv.Device.from_dict(dev.to_dict())
        assert restored.direction == 2 and restored.offset == 7

    def test_from_dict_defaults_plane_for_old_sessions(self):
        """A session saved before V6-M5 has no direction/offset keys --
        must restore to the app's default plane, not crash."""
        d = {"id": "p5", "name": "TC-05", "type": "thermocouple", "scenario": 0,
            "position": [1.0, 1.0]}
        restored = dv.Device.from_dict(d)
        assert restored.direction == 1 and restored.offset == 0
