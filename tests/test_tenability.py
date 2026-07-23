"""Tests for tenability screening (V2 roadmap M3.2)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tenability as tn  # noqa: E402


class TestTimeToUntenableField:
    def test_first_crossing_per_cell(self):
        # 3 frames, 1x2 grid. Cell 0 crosses 60 at frame 1, cell 1 never.
        data = np.array([
            [[20.0, 20.0]],
            [[70.0, 30.0]],
            [[90.0, 40.0]],
        ], dtype=np.float32)
        field = tn.time_to_untenable_field(data, 60.0, fps=2)
        assert field.shape == (1, 2)
        assert field[0, 0] == pytest.approx(0.5)  # frame 1 / fps 2
        assert np.isinf(field[0, 1])

    def test_cell_hot_from_start_is_zero(self):
        data = np.array([[[100.0]], [[100.0]]], dtype=np.float32)
        field = tn.time_to_untenable_field(data, 60.0, fps=4)
        assert field[0, 0] == 0.0

    def test_scalar_is_earliest_crossing_anywhere(self):
        data = np.array([
            [[20.0, 20.0]],
            [[20.0, 70.0]],   # cell 1 crosses at frame 1
            [[90.0, 90.0]],   # cell 0 crosses at frame 2
        ], dtype=np.float32)
        assert tn.time_to_untenable_scalar(data, 60.0, fps=1) == pytest.approx(1.0)

    def test_scalar_none_when_never_reached(self):
        data = np.full((5, 2, 2), 25.0, dtype=np.float32)
        assert tn.time_to_untenable_scalar(data, 60.0, fps=4) is None

    def test_untenable_fraction(self):
        data = np.array([[[70.0, 20.0], [80.0, 30.0]]], dtype=np.float32)  # 1 frame, 2x2
        assert tn.untenable_fraction(data, 60.0, 0) == pytest.approx(0.5)

    def test_threshold_is_strict_exceedance(self):
        data = np.full((2, 1, 1), 60.0, dtype=np.float32)  # exactly at threshold
        assert tn.time_to_untenable_scalar(data, 60.0, fps=4) is None


class TestFullFED:
    """V6-M6: full FED (toxic-gas + convected-heat dose), verified against
    independently-derived closed forms (a constant field's cumulative dose
    is just (frame_index+1) * per-frame increment), never by calling the
    same code path twice."""

    def test_fed_heat_dose_constant_temperature(self):
        T, fps, n = 300.0, 2, 5
        field = np.full((n, 1, 1), T)
        fed = tn.fed_heat_dose(field, fps)
        dt_min = 1.0 / (fps * 60.0)
        increment = np.exp(0.0273 * T - 5.1849) * dt_min
        expected = np.array([(k + 1) * increment for k in range(n)]).reshape(n, 1, 1)
        np.testing.assert_allclose(fed, expected)

    def test_fed_gas_dose_constant_co(self):
        co_ppm, fps, n = 5000.0, 2, 5
        field = np.full((n, 1, 1), co_ppm)
        fed = tn.fed_gas_dose(field, fps)
        dt_min = 1.0 / (fps * 60.0)
        increment = (co_ppm ** 1.036) / 35000.0 * dt_min
        expected = np.array([(k + 1) * increment for k in range(n)]).reshape(n, 1, 1)
        np.testing.assert_allclose(fed, expected)

    def test_full_fed_is_the_sum_of_both_doses(self):
        n = 6
        temp = np.full((n, 1, 1), 200.0)
        co = np.full((n, 1, 1), 3000.0)
        fps = 1
        combined = tn.full_fed(temp, co, fps)
        independently_summed = tn.fed_gas_dose(co, fps) + tn.fed_heat_dose(temp, fps)
        np.testing.assert_allclose(combined, independently_summed)

    def test_fed_increases_monotonically_over_time(self):
        n = 10
        temp = np.full((n, 1, 1), 250.0)
        co = np.full((n, 1, 1), 2000.0)
        fed = tn.full_fed(temp, co, fps=2)
        flat = fed[:, 0, 0]
        assert np.all(np.diff(flat) > 0)

    def test_time_to_fed_field_first_crossing_per_cell(self):
        fed_field = np.zeros((6, 1, 2))
        fed_field[:, 0, 0] = np.linspace(0.0, 1.5, 6)   # crosses 1.0 at frame 4
        fed_field[:, 0, 1] = 0.0                        # never crosses
        field = tn.time_to_fed_field(fed_field, fps=2)
        assert field.shape == (1, 2)
        assert field[0, 0] == pytest.approx(4 / 2)
        assert np.isinf(field[0, 1])

    def test_time_to_fed_scalar_detects_first_crossing(self):
        # A field engineered so cell (0,0) crosses FED=1.0 exactly at frame 4.
        fed_field = np.zeros((10, 1, 2))
        fed_field[:, 0, 0] = np.linspace(0.0, 2.0, 10)   # crosses 1.0 between frame 4 and 5
        fed_field[:, 0, 1] = 0.0                         # never crosses
        fps = 2
        t = tn.time_to_fed_scalar(fed_field, fps)
        expected_frame = int(np.argmax(fed_field[:, 0, 0] >= 1.0))
        assert t == pytest.approx(expected_frame / fps)

    def test_time_to_fed_scalar_none_when_never_reached(self):
        fed_field = np.full((5, 2, 2), 0.1)
        assert tn.time_to_fed_scalar(fed_field, fps=2) is None

    def test_gas_dose_never_negative_even_for_negative_input(self):
        """CO shouldn't ever be negative in real data, but the dose
        integral must not produce NaN/negative garbage if it somehow is
        (e.g. an interpolation artifact at a boundary)."""
        field = np.full((3, 1, 1), -5.0)
        fed = tn.fed_gas_dose(field, fps=1)
        assert np.all(fed >= 0.0) and not np.any(np.isnan(fed))


class FakeEntry:
    def __init__(self, case_index, folder, path):
        self.case_index = case_index
        self.folder = folder
        self.path = path
        self.candles = self.door = self.vod = self.voc = 0


class FakeProvider:
    """A QuantityProvider stand-in: TEMPERATURE is real; CO is gated (as it
    is in the real registry today), raising GatedQuantityError exactly as
    QuantityProvider.get() does -- so these tests exercise the same
    fallback-to-partial-screen path the real app takes."""

    def __init__(self, co=None):
        self.data = np.stack([np.full((3, 4), 20.0 + 30.0 * t, dtype=np.float32) for t in range(5)])
        self._co = co

    def get(self, case_index, key):
        if key.quantity == "CARBON MONOXIDE VOLUME FRACTION":
            if self._co is None:
                from quantity_provider import GatedQuantityError
                raise GatedQuantityError("Requires the M-SIM cluster re-run")
            return self._co
        return self.data

    def get_extent(self, case_index, key):
        return [0.0, 1.0, 0.0, 0.3]


# Backward-compatible alias: FakeProvider fully replaces the old FakeStore
# (same .get/.get_extent shape QuantityProvider itself has), just with the
# CO-gating semantics TenabilityPanel now depends on.
FakeStore = FakeProvider


class TestTenabilityPanel:
    def test_ensure_loaded_populates_and_reports(self, qapp):
        from tenability_panel import TenabilityPanel
        panel = TenabilityPanel(FakeProvider(), [FakeEntry(0, "case", "/x")], fps=4)
        panel.ensure_loaded()
        assert panel.scenario_combo.count() == 1
        assert "untenable" in panel.stats_label.text().lower()
        panel.deleteLater()

    def test_threshold_change_recomputes(self, qapp):
        from tenability_panel import TenabilityPanel
        panel = TenabilityPanel(FakeProvider(), [FakeEntry(0, "case", "/x")], fps=4)
        panel.ensure_loaded()
        before = panel.stats_label.text()
        panel.threshold_spin.setValue(500)  # never reached by the fake data
        assert panel.stats_label.text() != before
        assert "never" in panel.stats_label.text().lower()
        panel.deleteLater()

    def test_disclaimer_states_partial_and_no_co(self, qapp):
        from tenability_panel import TenabilityPanel, _DISCLAIMER
        assert "FED" in _DISCLAIMER
        assert "CO" in _DISCLAIMER
        assert "Partial" in _DISCLAIMER or "partial" in _DISCLAIMER

    def test_no_co_shows_partial_disclaimer(self, qapp):
        from tenability_panel import TenabilityPanel, _DISCLAIMER
        panel = TenabilityPanel(FakeProvider(), [FakeEntry(0, "case", "/x")], fps=4)
        panel.ensure_loaded()
        assert not panel._has_co
        assert panel.disclaimer.text() == _DISCLAIMER
        panel.deleteLater()

    def test_co_available_shows_full_fed(self, qapp):
        from tenability_panel import TenabilityPanel, _FULL_FED_NOTICE
        co = np.full((5, 3, 4), 5000.0, dtype=np.float32)
        panel = TenabilityPanel(FakeProvider(co=co), [FakeEntry(0, "case", "/x")], fps=4)
        panel.ensure_loaded()
        assert panel._has_co
        assert panel.disclaimer.text() == _FULL_FED_NOTICE
        assert "incapacitated" in panel.stats_label.text().lower()
        panel.deleteLater()
