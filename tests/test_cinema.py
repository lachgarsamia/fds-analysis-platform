"""Unit tests for cinema/luts.py + cinema/pipeline.py (FireLab roadmap
Phase 2, task 1: FireLUT + alpha + filmic tone map + auto-exposure). No
Qt/matplotlib canvas involved -- pure array math, see test_views.py's
TestSliceViewCinematicMode for the SliceView integration."""

import numpy as np

from cinema.luts import FIRE_RGBA_LUT
from cinema.pipeline import AutoExposure, EffectsPipeline, filmic_tonemap


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
