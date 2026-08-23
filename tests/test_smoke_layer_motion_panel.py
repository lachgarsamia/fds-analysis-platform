"""Tests for the Smoke-Layer Motion panel (Analysis roadmap follow-up):
a position widget over layer_height.py's existing smoke-layer height
series (top plot) locked to one scrub cursor, with the shared "layer,
plume & ceiling over time" plot (height_panel.draw_layer_plume_ceiling_
over_time, also used by Field & Time Explorer) on the bottom -- replacing
what used to be a standalone descent-rate plot there (Spatiotemporal
Analysis consolidation pass). The descent-rate computation itself is
unchanged and still reported live in the status label; it's just no
longer its own plotted axis. The bottom plot is never labeled "velocity"
in the UI -- that name is reserved for the real FDS VELOCITY quantity.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smoke_layer_motion_panel import (  # noqa: E402
    SmokeLayerMotionPanel, _NO_DATA_CAPTION, _presmoke_frame_count)
from layer_height import smoke_layer_height_series  # noqa: E402
from slice_key import SliceKey  # noqa: E402
from quantity_provider import GatedQuantityError  # noqa: E402
from registry import AMBIENT_C  # noqa: E402


class FakeEntry:
    def __init__(self, case_index, folder):
        self.case_index = case_index
        self.folder = folder
        self.path = "/nonexistent"
        self.candles = self.door = self.vod = self.voc = 0


EXTENT = (0.0, 1.0, 0.0, 1.0)


def _descending_field(case_index, n_times=40, n_z=10, n_x=4):
    """A synthetic TEMPERATURE field whose hot band grows from the ceiling
    downward over time, so smoke_layer_height_series produces a real,
    monotonically-descending-then-settling curve -- not a flat/degenerate
    one -- per scenario (case_index shifts the growth rate so scenarios
    are distinguishable)."""
    data = np.full((n_times, n_z, n_x), 20.0)
    rate = 1 + case_index  # rows of hot band grow faster for a higher case_index
    for t in range(n_times):
        hot_rows = min(n_z, int(t / rate))
        data[t, :hot_rows, :] = 300.0
    return data


class FakeProvider:
    """Mimics QuantityProvider.get()/get_extent() -- gated_cases raises
    GatedQuantityError for TEMPERATURE on those case indices, same
    contract the real provider uses."""

    def __init__(self, gated_cases=()):
        self._gated = set(gated_cases)

    def get(self, case_index, key):
        if case_index in self._gated:
            raise GatedQuantityError("forced for test")
        return _descending_field(case_index)

    def get_extent(self, case_index, key):
        return EXTENT


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def manifest():
    return [FakeEntry(0, "case_a"), FakeEntry(1, "case_b")]


@pytest.fixture
def panel(qapp, provider, manifest):
    p = SmokeLayerMotionPanel(provider, manifest, fps=4)
    p.ensure_loaded()
    return p


def test_scenario_combo_populated_from_manifest(panel, manifest):
    assert panel.scenario_combo.count() == len(manifest)


def test_height_reuses_layer_height_exactly(panel, provider):
    """The panel must not reimplement the smoke-layer computation --
    its height series has to be byte-identical to calling
    layer_height.smoke_layer_height_series directly on the same data."""
    data = provider.get(0, SliceKey("TEMPERATURE"))
    expected = smoke_layer_height_series(data, EXTENT, AMBIENT_C)
    np.testing.assert_allclose(panel._series["height"], expected)


def test_rate_is_gradient_of_height_times_fps(panel):
    height = panel._series["height"]
    rate = panel._series["rate"]
    expected = np.gradient(height) * panel._fps
    np.testing.assert_allclose(rate, expected)


def test_rate_negative_while_height_descends(panel):
    """Where the height curve is actually decreasing, its derivative must
    be negative -- the core position/velocity relationship this widget
    exists to show."""
    height = panel._series["height"]
    rate = panel._series["rate"]
    descending = np.diff(height) < -1e-9
    idx = np.flatnonzero(descending)
    assert idx.size > 0
    # rate[i+1] is the derivative straddling the drop from height[i] to height[i+1]
    assert np.all(rate[idx + 1] < 0)


def test_switching_scenario_recomputes_from_that_scenarios_data(panel, provider):
    panel.scenario_combo.setCurrentIndex(0)
    height_a = panel._series["height"].copy()
    panel.scenario_combo.setCurrentIndex(1)
    height_b = panel._series["height"].copy()
    assert not np.allclose(height_a, height_b)
    np.testing.assert_allclose(height_b, smoke_layer_height_series(
        provider.get(1, SliceKey("TEMPERATURE")), EXTENT, AMBIENT_C))


def test_scrubbing_moves_cursor_on_both_panels_identically(panel):
    n = len(panel._series["height"])
    for frac in (0.2, 0.7):
        fi = int(n * frac)
        panel.frame_slider.setValue(fi)
        ax_h, ax_r, _jet_ax = panel.canvas.fig.axes
        cursor_h = [l.get_xdata()[0] for l in ax_h.lines if len(set(l.get_xdata())) == 1]
        cursor_r = [l.get_xdata()[0] for l in ax_r.lines if len(set(l.get_xdata())) == 1]
        t_expected = fi / panel._fps
        assert cursor_h and abs(cursor_h[0] - t_expected) < 1e-9
        assert cursor_r and abs(cursor_r[0] - t_expected) < 1e-9


def test_status_label_reports_height_and_rate_at_cursor(panel):
    panel.frame_slider.setValue(5)
    height = panel._series["height"][5]
    assert f"{height:.3f} m" in panel.status_label.text()


def test_gated_temperature_shows_honest_empty_state_not_a_guessed_curve(qapp, manifest):
    provider = FakeProvider(gated_cases=(0, 1))
    p = SmokeLayerMotionPanel(provider, manifest, fps=4)
    p.ensure_loaded()
    assert p._series is None
    assert p.caption.text() == _NO_DATA_CAPTION
    assert p.status_label.text() == ""
    # The canvas must not silently render a plausible-looking blank line
    # plot -- it should have exactly one placeholder axes with no data lines.
    axes = p.canvas.fig.axes
    assert len(axes) == 1
    assert axes[0].get_lines() == []


def test_recovers_real_data_once_provider_is_no_longer_gated(qapp, manifest):
    provider = FakeProvider(gated_cases=(0,))
    p = SmokeLayerMotionPanel(provider, manifest, fps=4)
    p.ensure_loaded()
    assert p._series is None
    provider._gated.clear()
    p._cache.clear()
    p._reload()
    assert p._series is not None


# ------------------------------------------------------------- cleanup pass
# Fix 1 (sign-convention docs) and Fix 2 (t=0 derivative-spike honesty).

def test_presmoke_frame_count_detects_leading_ceiling_run():
    height = np.array([1.0, 1.0, 1.0, 0.6, 0.4, 0.3])
    assert _presmoke_frame_count(height, ceiling_z=1.0) == 3


def test_presmoke_frame_count_ignores_a_later_return_to_ceiling():
    """Smoke genuinely clearing later (height returning to the ceiling
    value mid-run) must NOT be folded into the leading pre-smoke count --
    that's a real event, not a startup artifact."""
    height = np.array([1.0, 0.6, 0.4, 1.0, 1.0])
    assert _presmoke_frame_count(height, ceiling_z=1.0) == 1


def test_presmoke_frame_count_zero_when_smoke_present_from_frame_zero():
    height = np.array([0.6, 0.5, 0.4, 0.3])
    assert _presmoke_frame_count(height, ceiling_z=1.0) == 0


def test_underlying_rate_series_is_never_modified_by_the_display_mask(panel):
    """The cleanup pass masks what's *drawn*, never the real derivative --
    _series["rate"] must stay the complete, untouched np.gradient(height)*fps
    array, including its raw (contaminated) leading values."""
    height = panel._series["height"]
    expected = np.gradient(height) * panel._fps
    np.testing.assert_allclose(panel._series["rate"], expected)
    presmoke_n = panel._series["presmoke_n"]
    assert presmoke_n >= 1
    # The raw value at the contaminated index is still there, unmasked.
    assert np.isfinite(panel._series["rate"][presmoke_n - 1])


def test_presmoke_interval_is_shaded_on_both_panels(panel):
    ax_h, ax_r, _jet_ax = panel.canvas.fig.axes
    assert len(ax_h.patches) >= 1  # axvspan
    assert len(ax_r.patches) >= 1


def test_bottom_plot_is_the_shared_layer_plume_ceiling_plot_not_descent_rate(panel):
    """Spatiotemporal Analysis consolidation pass: the bottom plot is now
    height_panel.draw_layer_plume_ceiling_over_time's shared plot (same
    one Field & Time Explorer uses), not a standalone descent-rate axis --
    a title check alone wouldn't rule out a coincidentally-renamed rate
    plot, so this also checks for the twin ceiling-temperature axis and
    real plume-height data flowing through, both specific to the shared
    plot and absent from the old one."""
    fig = panel.canvas.fig
    assert len(fig.axes) == 3, "expected height + layer/plume/ceiling + its ceiling-temp twin axis"
    ax_h, ax_r, jet_ax = fig.axes
    assert ax_r.get_title() == "Layer, plume & ceiling over time"
    assert "descent rate" not in ax_r.get_title().lower()
    assert "ceiling" in jet_ax.get_ylabel().lower()
    assert panel._series["plume"] is not None
    assert panel._series["ceiling"] is not None
    assert len(panel._series["plume"]) == len(panel._series["height"])
    # The plume-height line (not the smoke-layer line, which the top plot
    # also draws in the same color) must actually be present on ax_r.
    plume_lines = [l for l in ax_r.get_lines() if l.get_label() == "plume height (m)"]
    assert len(plume_lines) == 1
    np.testing.assert_allclose(plume_lines[0].get_ydata(), panel._series["plume"])


def test_top_and_bottom_plots_show_the_smoothed_line_not_raw_height(panel):
    """Display-only smoothing (window=5): both the top plot's own line and
    the bottom shared plot's "smoke layer (m)" line must differ from
    panel._series["height"] (the raw signal) -- proving the smoothing is
    applied consistently to both, not just one, and confirming it's
    display-only (the raw series itself is untouched and still what the
    status label reads)."""
    import height_analysis as ha
    raw_height = panel._series["height"]
    expected = ha.smooth_layer_height(raw_height)

    ax_h, ax_r, _jet_ax = panel.canvas.fig.axes
    top_line = ax_h.get_lines()[0]
    np.testing.assert_allclose(top_line.get_ydata(), expected)
    assert not np.allclose(top_line.get_ydata(), raw_height)

    bottom_lines = [l for l in ax_r.get_lines() if l.get_label() == "smoke layer (m)"]
    assert len(bottom_lines) == 1
    np.testing.assert_allclose(bottom_lines[0].get_ydata(), expected)

    # panel._series["height"] itself must stay raw -- proof this never
    # touched layer_height.py's output, only what gets plotted.
    np.testing.assert_allclose(panel._series["height"], raw_height)


def test_status_label_reads_raw_height_not_smoothed(panel):
    """The status label pairs height with rate (np.gradient(raw height)*
    fps) in one reading -- showing a smoothed height there would desync
    the two numbers (a big rate next to a muted height), so the label
    keeps reading the true raw value even though the line above it is
    smoothed."""
    idx = panel._series["presmoke_n"] + 2
    panel.frame_slider.setValue(idx)
    raw_height_at_idx = panel._series["height"][idx]
    assert f"height {raw_height_at_idx:.3f} m" in panel.status_label.text()


def test_status_label_says_no_smoke_detected_during_presmoke_frames(panel):
    panel.frame_slider.setValue(0)
    text = panel.status_label.text()
    assert "no smoke detected yet" in text
    # Must not print a fabricated/misleading rate reading during this window.
    assert "m/s" not in text


def test_status_label_reports_real_rate_once_past_the_presmoke_window(panel):
    presmoke_n = panel._series["presmoke_n"]
    panel.frame_slider.setValue(presmoke_n + 2)
    text = panel.status_label.text()
    assert "m/s" in text
    assert "no smoke detected yet" not in text


def test_steepest_real_drop_still_aligns_with_rate_minimum_after_masking(panel):
    """The core position/velocity relationship must survive the honesty
    fix: outside the masked pre-smoke window, the steepest real height
    drop still lines up with the displayed rate curve's most negative
    point."""
    height = panel._series["height"]
    rate = panel._series["rate"]
    presmoke_n = panel._series["presmoke_n"]
    start = presmoke_n + 1
    real_diff = np.diff(height[start:])
    real_rate = rate[start:]
    steepest_idx = start + int(np.argmin(real_diff))
    most_negative_idx = start + int(np.argmin(real_rate))
    assert abs(steepest_idx - most_negative_idx) <= 2


def test_bottom_panel_never_labels_itself_velocity(panel):
    """The real FDS VELOCITY quantity is a different thing entirely --
    the bottom plot must not collide with that name anywhere a
    user actually reads."""
    ax_h, ax_r, jet_ax = panel.canvas.fig.axes
    for text in (ax_h.get_title(), ax_r.get_title(), ax_r.get_ylabel(),
                 ax_h.get_ylabel(), jet_ax.get_ylabel(), panel.caption.text()):
        assert "velocity" not in text.lower()
