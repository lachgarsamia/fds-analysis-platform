import matplotlib as mpl
import numpy as np
import pytest

from colormaps import FIRE_NAME, FLOW_NAME, build_fire_colormap, build_flow_colormap


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class TestRegistration:
    def test_fire_and_flow_registered_with_matplotlib(self):
        assert FIRE_NAME in mpl.colormaps
        assert FLOW_NAME in mpl.colormaps

    def test_registered_by_name_matches_builder(self):
        by_name = mpl.colormaps[FIRE_NAME]
        built = build_fire_colormap()
        assert by_name(0.5) == pytest.approx(built(0.5))

    def test_reimporting_does_not_crash(self):
        # register_custom_colormaps() must be idempotent -- this module
        # (and therefore config.py, which imports it) can be imported
        # more than once per process across a test session.
        import colormaps
        colormaps.register_custom_colormaps()
        colormaps.register_custom_colormaps()


class TestFireColormap:
    def test_starts_black_ends_near_white_not_pure_white(self):
        cmap = build_fire_colormap()
        black = cmap(0.0)
        assert black[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        hottest = cmap(1.0)
        # Deliberately short of pure white (see colormaps.py's own
        # rationale: the plot canvas background is pure white, so the
        # hottest pixels must stay visually distinct from it) -- pale and
        # warm-toned (high luminance, but not neutral gray-white).
        assert hottest[:3] != (1.0, 1.0, 1.0)
        assert _luminance(hottest) > 0.85
        r, g, b = hottest[:3]
        assert r >= g >= b  # warm (red-leaning), not a neutral gray

    def test_luminance_increases_monotonically(self):
        # A legible sequential colormap should read as "getting brighter"
        # from cold to hot, not oscillate.
        cmap = build_fire_colormap()
        samples = np.linspace(0.0, 1.0, 20)
        luminances = [_luminance(cmap(s)) for s in samples]
        diffs = np.diff(luminances)
        assert (diffs >= -1e-9).all()  # non-decreasing, allowing float noise

    def test_hazard_band_calibration_points_are_ordered(self):
        """Real calibration check: at the normalized positions
        corresponding to the app's own TEMPERATURE hazard bands (20C
        ambient floor, 60/100/300C from config.ISOTHERM_LEVELS, all
        relative to vmin=20/default-vmax=300), color intensity must
        strictly increase -- these aren't arbitrary sample points, they're
        the exact values a user sees switching on the contour overlay."""
        cmap = build_fire_colormap()
        vmin, vmax = 20.0, 300.0
        temps_c = [20.0, 60.0, 100.0, 300.0]
        positions = [(t - vmin) / (vmax - vmin) for t in temps_c]
        assert positions[0] == pytest.approx(0.0)
        assert positions[-1] == pytest.approx(1.0)
        luminances = [_luminance(cmap(p)) for p in positions]
        diffs = np.diff(luminances)
        assert (diffs > 0).all(), f"luminance must strictly increase across hazard bands: {luminances}"


class TestFlowColormap:
    def test_slow_end_is_blue_dominant(self):
        cmap = build_flow_colormap()
        slow = cmap(0.0)
        r, g, b = slow[:3]
        assert b > r and b > g

    def test_fast_end_is_red_dominant(self):
        cmap = build_flow_colormap()
        fast = cmap(1.0)
        r, g, b = fast[:3]
        assert r > b

    def test_endpoints_are_distinct(self):
        cmap = build_flow_colormap()
        assert cmap(0.0) != cmap(1.0)
