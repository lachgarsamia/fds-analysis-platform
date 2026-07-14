"""Unit tests for widgets.py's MplCanvas (GUI modernization pass, item 7:
rendering-quality DPI bump + the always-white plot background)."""

from widgets import MplCanvas


class TestMplCanvasRenderingQuality:
    def test_default_dpi_is_bumped_above_matplotlibs_own_default(self, qapp):
        """matplotlib's own Figure default is 100 -- confirms this app
        deliberately raises it for crisper on-screen rendering, not just
        inheriting whatever matplotlib's default happens to be."""
        canvas = MplCanvas()
        assert canvas.fig.dpi > 100
        assert canvas.fig.dpi == MplCanvas.DEFAULT_DPI

    def test_dpi_still_overridable(self, qapp):
        canvas = MplCanvas(dpi=72)
        assert canvas.fig.dpi == 72

    def test_plot_background_is_explicit_white(self, qapp):
        canvas = MplCanvas()
        assert canvas.fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert MplCanvas.PLOT_BG == "#FFFFFF"
