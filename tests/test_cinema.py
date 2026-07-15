"""Unit tests for cinema/luts.py + cinema/pipeline.py (FireLab roadmap
Phase 2, tasks 1-4: FireLUT + alpha + filmic tone map + auto-exposure,
bloom + HRR-driven flicker + sub-frame interpolation, smoke tiers 1-2,
heat shimmer + ember particles). No Qt/matplotlib canvas involved -- pure
array math, see test_views.py's TestSliceViewCinematicMode for the
SliceView/scatter-artist integration."""

import numpy as np

from cinema.bloom import apply_bloom
from cinema.interp import lerp_frames
from cinema.luts import FIRE_RGBA_LUT
from cinema.particles import EmberParticles
from cinema.pipeline import AutoExposure, EffectsPipeline, filmic_tonemap
from cinema.shimmer import HeatShimmer
from cinema.smoke import SmokeSimulator, composite_over, smoke_rgba


class TestFireLUT:
    def test_shape_and_dtype(self):
        assert FIRE_RGBA_LUT.shape == (256, 4)
        assert FIRE_RGBA_LUT.dtype == np.uint8

    def test_ambient_end_is_transparent(self):
        assert FIRE_RGBA_LUT[0, 3] == 0

    def test_hot_end_is_opaque(self):
        assert FIRE_RGBA_LUT[-1, 3] == 255

    def test_alpha_is_monotonically_non_decreasing(self):
        alpha = FIRE_RGBA_LUT[:, 3].astype(np.int64)
        assert (np.diff(alpha) >= 0).all()


class TestFilmicTonemap:
    def test_zero_maps_to_zero(self):
        assert filmic_tonemap(np.array([0.0]))[0] == 0.0

    def test_never_exceeds_one_and_approaches_it(self):
        # A Reinhard-style shoulder rolls off toward 1.0 without ever
        # hard-clipping to it -- that's what avoids the "flat white blob"
        # look a plain Normalize gives flashover frames.
        t = np.linspace(0.0, 1.0, 50)
        out = filmic_tonemap(t)
        assert out[-1] < 1.0
        assert out[-1] > 0.75

    def test_compresses_relative_to_linear_in_upper_range(self):
        # Filmic curve should sit at-or-below the identity line for a
        # Reinhard-style shoulder to actually roll off highlights.
        t = np.linspace(0.0, 1.0, 50)
        assert (filmic_tonemap(t) <= t + 1e-9).all()

    def test_monotonically_increasing(self):
        t = np.linspace(0.0, 1.0, 100)
        assert (np.diff(filmic_tonemap(t)) >= 0).all()


class TestAutoExposure:
    def test_locked_never_moves(self):
        exp = AutoExposure(vmax_init=300.0)
        exp.locked = True
        exp.update(np.full((10, 10), 900.0))
        assert exp.vmax == 300.0

    def test_unlocked_tracks_toward_hotter_frames(self):
        exp = AutoExposure(vmax_init=300.0, tau_frames=4.0)
        vmax_before = exp.vmax
        for _ in range(50):
            exp.update(np.full((10, 10), 900.0))
        assert exp.vmax > vmax_before
        assert abs(exp.vmax - 900.0) < 1.0, "should converge close to the sustained percentile"


class TestBloom:
    def test_shape_and_dtype_preserved(self):
        rgba = FIRE_RGBA_LUT[np.full((20, 20), 255, dtype=np.uint8)]
        intensity = np.ones((20, 20), dtype=np.float32)
        out = apply_bloom(rgba, intensity)
        assert out.shape == rgba.shape
        assert out.dtype == np.uint8

    def test_hot_spot_raises_alpha_in_neighboring_ambient_pixels(self):
        """The glow's whole point: light spills into pixels that were
        fully transparent (ambient), not just brightens already-hot ones."""
        intensity = np.zeros((21, 21), dtype=np.float32)
        intensity[10, 10] = 1.0
        rgba = np.zeros((21, 21, 4), dtype=np.uint8)
        rgba[10, 10] = FIRE_RGBA_LUT[-1]
        out = apply_bloom(rgba, intensity, strength=1.0)
        assert out[10, 8, 3] > 0, "a nearby ambient pixel should pick up some halo alpha"

    def test_zero_intensity_no_glow(self):
        rgba = FIRE_RGBA_LUT[np.zeros((10, 10), dtype=np.uint8)]
        intensity = np.zeros((10, 10), dtype=np.float32)
        out = apply_bloom(rgba, intensity)
        assert (out == rgba).all()


class TestLerpFrames:
    def test_endpoints_and_midpoint(self):
        a = np.zeros((3, 3), dtype=np.float32)
        b = np.full((3, 3), 10.0, dtype=np.float32)
        assert (lerp_frames(a, b, 0.0) == a).all()
        assert (lerp_frames(a, b, 1.0) == b).all()
        assert (lerp_frames(a, b, 0.5) == 5.0).all()

    def test_clamps_out_of_range_t(self):
        a = np.zeros((2, 2), dtype=np.float32)
        b = np.full((2, 2), 10.0, dtype=np.float32)
        assert (lerp_frames(a, b, -1.0) == a).all()
        assert (lerp_frames(a, b, 2.0) == b).all()


class TestSmokeSimulator:
    def test_no_source_below_threshold_stays_empty(self):
        sim = SmokeSimulator((10, 10), ambient_c=20.0)
        ambient_frame = np.full((10, 10), 20.0, dtype=np.float32)
        for _ in range(5):
            density = sim.step(ambient_frame)
        assert (density == 0.0).all()

    def test_hot_frame_accumulates_then_decays(self):
        sim = SmokeSimulator((10, 10), ambient_c=20.0)
        hot_frame = np.full((10, 10), 300.0, dtype=np.float32)
        d1 = sim.step(hot_frame).copy()
        d2 = sim.step(hot_frame).copy()
        assert d2.sum() > d1.sum(), "sustained heat should keep building smoke density"
        ambient_frame = np.full((10, 10), 20.0, dtype=np.float32)
        for _ in range(50):
            after_decay = sim.step(ambient_frame)
        assert after_decay.sum() < d2.sum(), "removing the source should let density decay away"

    def test_tier2_moves_mass_toward_velocity_direction(self):
        """A point source with a strong velocity field should advect
        differently than Tier 1's fixed drift -- exercises the tier-2
        (velocity_frame given) code path without asserting exact physics."""
        shape = (21, 21)
        sim = SmokeSimulator(shape, ambient_c=20.0)
        frame = np.full(shape, 20.0, dtype=np.float32)
        frame[10, 10] = 400.0  # a single hot cell as the plume source
        velocity = np.full(shape, 3.0, dtype=np.float32)
        for _ in range(10):
            density = sim.step(frame, velocity_frame=velocity)
        assert density.sum() > 0.0
        assert np.isfinite(density).all()


class TestSmokeCompositing:
    def test_smoke_rgba_alpha_tracks_density(self):
        density = np.array([[0.0, 1.0]], dtype=np.float32)
        rgba = smoke_rgba(density)
        assert rgba[0, 0, 3] == 0
        assert rgba[0, 1, 3] > 0

    def test_opaque_top_fully_occludes_bottom(self):
        top = np.zeros((2, 2, 4), dtype=np.uint8)
        top[..., 0] = 200
        top[..., 3] = 255
        bottom = np.zeros((2, 2, 4), dtype=np.uint8)
        bottom[..., 1] = 200
        bottom[..., 3] = 255
        out = composite_over(top, bottom)
        assert (out == top).all()

    def test_transparent_top_shows_bottom(self):
        top = np.zeros((2, 2, 4), dtype=np.uint8)
        bottom = np.zeros((2, 2, 4), dtype=np.uint8)
        bottom[..., 1] = 200
        bottom[..., 3] = 255
        out = composite_over(top, bottom)
        assert (out == bottom).all()


class TestEffectsPipelineFlicker:
    def test_zero_hrr_intensity_gives_deterministic_repeated_output(self):
        """hrr_intensity=0 should mean no flicker modulation at all --
        with exposure locked (isolating flicker from auto-exposure's own
        frame-to-frame adaptation), a constant input frame should render
        identically every call."""
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=250.0)
        pipeline.exposure.locked = True
        frame = np.full((30, 30), 250.0, dtype=np.float32)
        first = pipeline.render(frame, hrr_intensity=0.0)
        second = pipeline.render(frame, hrr_intensity=0.0)
        assert (first == second).all()

    def test_nonzero_hrr_intensity_varies_frame_to_frame(self):
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=250.0)
        pipeline.exposure.locked = True
        frame = np.full((30, 30), 250.0, dtype=np.float32)
        renders = [pipeline.render(frame, hrr_intensity=1.0) for _ in range(5)]
        assert any(not (renders[0] == r).all() for r in renders[1:])


class TestEffectsPipeline:
    def test_render_output_shape_and_dtype(self):
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full((49, 101), 250.0, dtype=np.float32)
        rgba = pipeline.render(frame)
        assert rgba.shape == (49, 101, 4)
        assert rgba.dtype == np.uint8

    def test_ambient_frame_fully_transparent(self):
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.full((49, 101), 20.0, dtype=np.float32)
        rgba = pipeline.render(frame)
        assert (rgba[..., 3] == 0).all()

    def test_render_cost_within_budget(self):
        """DoD (ROADMAP-FIRELAB.md Phase 2 task 1): full chain should be
        well under the per-frame budget even before upsampling/bloom/smoke
        are added on top in later tasks."""
        pipeline = EffectsPipeline(vmin=20.0, vmax_init=300.0)
        frame = np.random.default_rng(0).uniform(20.0, 400.0, size=(49, 101)).astype(np.float32)
        for _ in range(20):
            pipeline.render(frame)
        assert pipeline.last_cost_ms < 4.0


class TestHeatShimmer:
    def test_ambient_frame_is_unwarped(self):
        shimmer = HeatShimmer()
        image = np.zeros((30, 30, 4), dtype=np.uint8)
        image[10, 10] = [255, 0, 0, 255]
        ambient = np.full((30, 30), 20.0, dtype=np.float32)
        out = shimmer.warp(image, ambient, ambient_c=20.0)
        assert (out == image).all(), "no heat above ambient should mean no displacement at all"

    def test_hot_frame_shape_and_dtype_preserved(self):
        shimmer = HeatShimmer()
        image = np.full((30, 30, 4), 128, dtype=np.uint8)
        hot = np.full((30, 30), 300.0, dtype=np.float32)
        out = shimmer.warp(image, hot, ambient_c=20.0)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_advances_over_time(self):
        """Successive calls should scroll the noise field, not repeat the
        exact same warp every frame."""
        shimmer = HeatShimmer()
        image = np.zeros((40, 40, 4), dtype=np.uint8)
        image[20, 20] = [255, 255, 0, 255]
        hot = np.full((40, 40), 300.0, dtype=np.float32)
        first = shimmer.warp(image.copy(), hot, ambient_c=20.0)
        second = shimmer.warp(image.copy(), hot, ambient_c=20.0)
        assert not (first == second).all()


class TestEmberParticles:
    def test_no_spawn_below_threshold(self):
        sim = EmberParticles((20, 20))
        cool_frame = np.full((20, 20), 100.0, dtype=np.float32)  # well under the 150C-above-ambient knee
        for _ in range(10):
            sim.step(cool_frame, ambient_c=20.0)
        assert len(sim.pos) == 0

    def test_hot_frame_spawns_and_caps_at_max(self):
        sim = EmberParticles((20, 20), max_particles=15)
        hot_frame = np.full((20, 20), 400.0, dtype=np.float32)
        for _ in range(60):
            sim.step(hot_frame, ambient_c=20.0)
        assert 0 < len(sim.pos) <= 15

    def test_particles_die_of_old_age(self):
        sim = EmberParticles((20, 20), max_particles=10)
        hot_frame = np.full((20, 20), 400.0, dtype=np.float32)
        sim.step(hot_frame, ambient_c=20.0)
        assert len(sim.pos) > 0
        cool_frame = np.full((20, 20), 20.0, dtype=np.float32)
        for _ in range(200):  # far past any particle's max lifetime, no new spawns
            sim.step(cool_frame, ambient_c=20.0)
        assert len(sim.pos) == 0

    def test_render_arrays_shapes_match_particle_count(self):
        sim = EmberParticles((20, 20), max_particles=10)
        hot_frame = np.full((20, 20), 400.0, dtype=np.float32)
        for _ in range(20):
            sim.step(hot_frame, ambient_c=20.0)
        offsets, sizes, colors = sim.render_arrays()
        n = len(sim.pos)
        assert offsets.shape == (n, 2)
        assert sizes.shape == (n,)
        assert colors.shape == (n, 4)
