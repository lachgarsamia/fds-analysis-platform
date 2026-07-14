"""Unit tests for widgets.py's MplCanvas (GUI modernization pass, item 7:
rendering-quality DPI bump + the always-white plot background) and
CollapsibleSection (Streamlit-style redesign pass: card, not divider)."""

from PyQt5 import QtWidgets

from widgets import CollapsibleSection, MplCanvas


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


class TestCollapsibleSectionCard:
    """Streamlit-style redesign: each section is a rounded 'card', not a
    title with a divider line drawn underneath it."""

    def test_has_a_named_card_frame_for_theme_qss_to_target(self, qapp):
        section = CollapsibleSection("Playback speed")
        assert isinstance(section.card, QtWidgets.QFrame)
        assert section.card.objectName() == "sectionCard"

    def test_no_longer_has_a_divider_line(self, qapp):
        section = CollapsibleSection("Playback speed")
        assert section.findChild(QtWidgets.QFrame, "divider") is None

    def test_added_rows_land_inside_the_card(self, qapp):
        section = CollapsibleSection("Playback speed")
        row = QtWidgets.QPushButton("1x")
        section.add_row(row)
        assert row.parentWidget() is section.card
