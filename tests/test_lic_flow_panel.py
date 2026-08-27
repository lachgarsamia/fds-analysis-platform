"""Rendering + flicker-safety tests for LICFlowPanel: does a real composite
(speed color + LIC direction texture + temperature isotherms) get produced
from known-good U/W/TEMPERATURE data, without error; does the panel show an
honest gate message (never a fabricated plot) when either is unavailable;
and -- the property this panel lives or dies on for playback -- is the
per-scenario noise texture actually cached and reused across frames, not
regenerated (which would strobe), same conventions as
test_streamline_panel.py.

No fdsreader cross-validation here (rendering-only change; U/W-VELOCITY/
TEMPERATURE data itself is already validated elsewhere).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lic_flow_panel import (  # noqa: E402
    LICFlowPanel, compute_lic_texture, _contrast_stretch, _generate_noise,
)
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
    rotational flow) so the LIC texture has genuine direction to trace --
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
    methods VectorField and LICFlowPanel are duck-typed against."""

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
    p = LICFlowPanel(provider, manifest, fps=4)
    p.ensure_loaded()
    return p


def test_scenario_combo_populated_from_manifest(panel, manifest):
    assert panel.scenario_combo.count() == len(manifest)


def test_renders_all_layers_without_error(panel):
    """The whole point: a speed-color image, a LIC texture image on top of
    it, isotherm contour lines, and geometry lines all get produced from
    known-good data."""
    ax = panel.canvas.fig.axes[0]
    images = ax.get_images()
    assert len(images) == 2, "expected speed color + LIC texture images"
    # Isotherms (ax.contour) + geometry (ax.plot) both render as Line2D
    # artists on the axes.
    assert len(ax.lines) > 0, "expected isotherm and/or geometry lines"


def test_exactly_one_colorbar_for_speed(panel):
    """Only one filled/color field (speed) -- isotherms are lines, the LIC
    texture is a semi-transparent overlay, neither gets its own colorbar."""
    labels = [ax.get_ylabel() for ax in panel.canvas.fig.axes[1:]]
    assert len(labels) == 1, f"expected exactly one colorbar axes, got {labels}"
    assert "Speed" in labels[0] and "m/s" in labels[0]


def test_lic_texture_values_within_unit_range(panel):
    ax = panel.canvas.fig.axes[0]
    texture = np.array(ax.get_images()[1].get_array())
    assert texture.min() >= 0.0 - 1e-9
    assert texture.max() <= 1.0 + 1e-9
    assert not np.any(np.isnan(texture))


def test_noise_is_cached_per_scenario_not_regenerated_across_frames(panel):
    """The flicker-safety property this panel exists to guarantee: the
    white-noise input texture must be identical across renders of the same
    scenario at different frames, not reseeded per call."""
    noise_before = panel._noise_cache[0].copy()
    panel._current_index = 5
    panel._render()
    noise_after = panel._noise_cache[0]
    assert np.array_equal(noise_before, noise_after)


def test_different_scenarios_get_different_noise(panel):
    panel._render()  # scenario 0 (default combo selection)
    noise_0 = panel._noise_cache[0].copy()
    idx = panel.scenario_combo.findData(1)
    panel.scenario_combo.setCurrentIndex(idx)
    panel._render()
    noise_1 = panel._noise_cache[1]
    assert not np.array_equal(noise_0, noise_1)


def test_rerendering_same_frame_is_pixel_identical(panel):
    """No hidden per-call randomness anywhere in the pipeline: rendering
    the exact same scenario/frame twice must produce the exact same
    texture, not just a similar one."""
    ax = panel.canvas.fig.axes[0]
    texture_before = np.array(ax.get_images()[1].get_array()).copy()
    panel._render()
    ax2 = panel.canvas.fig.axes[0]
    texture_after = np.array(ax2.get_images()[1].get_array())
    assert np.array_equal(texture_before, texture_after)


def test_axis_limits_match_the_vector_field_extent(panel):
    ax = panel.canvas.fig.axes[0]
    x0, x1, z0, z1 = EXTENT
    assert ax.get_xlim() == pytest.approx((x0, x1))
    assert ax.get_ylim() == pytest.approx((z0, z1))


def test_gated_velocity_shows_honest_message_not_a_fabricated_plot(qapp, manifest):
    gated_provider = FakeProvider(gated_vector_cases=(0,))
    p = LICFlowPanel(gated_provider, manifest, fps=4)
    p.ensure_loaded()
    ax = p.canvas.fig.axes[0]
    assert len(ax.get_images()) == 0
    assert p.status.text() != ""


def test_gated_temperature_shows_honest_message_not_a_fabricated_plot(qapp, manifest):
    gated_provider = FakeProvider(gated_temp_cases=(0,))
    p = LICFlowPanel(gated_provider, manifest, fps=4)
    p.ensure_loaded()
    ax = p.canvas.fig.axes[0]
    assert len(ax.get_images()) == 0
    assert p.status.text() != ""


def test_switching_scenario_recomputes_the_fields(panel):
    """Different scenarios have different swirl scale/hot-spot magnitude --
    confirms a real per-scenario recompute, not a cached/stuck first
    scenario."""
    field0 = panel._ensure_field(0)
    field1 = panel._ensure_field(1)
    assert not np.allclose(field0.speed, field1.speed)
    temp0 = panel._ensure_temperature(0)
    temp1 = panel._ensure_temperature(1)
    assert not np.allclose(temp0, temp1)


# --------------------------------------------------------- compute_lic_texture

def test_compute_lic_texture_matches_input_shape_and_has_no_nans():
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, (N_Z, N_X))
    w = rng.uniform(-1, 1, (N_Z, N_X))
    noise = _generate_noise((N_Z, N_X), seed=1)
    texture = compute_lic_texture(u, w, noise, EXTENT)
    assert texture.shape == (N_Z, N_X)
    assert not np.any(np.isnan(texture))


def test_compute_lic_texture_is_deterministic():
    u = np.full((N_Z, N_X), 0.3)
    w = np.full((N_Z, N_X), -0.2)
    noise = _generate_noise((N_Z, N_X), seed=42)
    t1 = compute_lic_texture(u, w, noise, EXTENT)
    t2 = compute_lic_texture(u, w, noise, EXTENT)
    assert np.array_equal(t1, t2)


def test_compute_lic_texture_stagnant_field_just_resamples_its_own_texel():
    """Zero velocity everywhere means no advection: every texel's LIC value
    should equal the (weighted average of a texel resampling itself
    repeatedly, which collapses to) the noise's own value at that texel."""
    u = np.zeros((N_Z, N_X))
    w = np.zeros((N_Z, N_X))
    noise = _generate_noise((N_Z, N_X), seed=7)
    texture = compute_lic_texture(u, w, noise, EXTENT)
    assert np.allclose(texture, noise)


def test_contrast_stretch_maps_into_unit_range():
    rng = np.random.default_rng(3)
    texture = rng.normal(loc=0.5, scale=0.05, size=(N_Z, N_X))
    stretched = _contrast_stretch(texture)
    assert stretched.min() >= 0.0
    assert stretched.max() <= 1.0
    assert stretched.max() - stretched.min() > texture.max() - texture.min()
