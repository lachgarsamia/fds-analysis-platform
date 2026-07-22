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

    def test_plot_background_follows_theme(self, qapp):
        # RC polish: plot chrome is theme-aware. Light -> white figure; dark ->
        # a dark figure. The scientific field colormaps are unaffected (not
        # tested here -- they are passed explicitly per imshow).
        import widgets
        from theme import LIGHT, DARK
        widgets.set_plot_theme(LIGHT)
        assert MplCanvas().fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        widgets.set_plot_theme(DARK)
        assert MplCanvas().fig.get_facecolor()[0] < 0.2   # dark background
        widgets.set_plot_theme(LIGHT)                      # restore for other tests


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


# ---------------------------------------------------------------- V2 M1.3
from PyQt5 import QtCore, QtTest  # noqa: E402

from widgets import EventMarkerBar, TimelineWidget  # noqa: E402


class TestEventMarkerBar:
    def test_set_markers_and_range(self, qapp):
        bar = EventMarkerBar()
        bar.set_range(100)
        bar.set_markers([(10, "First frame above 100 °C"), (50, "Peak")])
        assert bar.markers == [(10, "First frame above 100 °C"), (50, "Peak")]

    def test_click_near_marker_emits_frame(self, qapp):
        bar = EventMarkerBar()
        bar.resize(200, bar.BAR_HEIGHT)
        bar.set_range(101)
        bar.set_markers([(50, "Peak")])
        clicked = []
        bar.marker_clicked.connect(clicked.append)
        x = bar._x_for_frame(50)
        QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(x), 5))
        assert clicked == [50]

    def test_click_far_from_any_marker_emits_nothing(self, qapp):
        bar = EventMarkerBar()
        bar.resize(200, bar.BAR_HEIGHT)
        bar.set_range(101)
        bar.set_markers([(100, "Peak")])
        clicked = []
        bar.marker_clicked.connect(clicked.append)
        QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton, pos=QtCore.QPoint(2, 5))
        assert clicked == []


class TestTimelineEventMarkers:
    def test_marker_click_is_a_seek_request(self, qapp):
        timeline = TimelineWidget()
        timeline.set_range(100, fps=4)
        timeline.set_event_markers([(25, "event")])
        seeks = []
        timeline.seek_requested.connect(seeks.append)
        timeline.marker_bar.marker_clicked.emit(25)
        assert seeks == [25]

    def test_set_range_propagates_to_marker_bar(self, qapp):
        timeline = TimelineWidget()
        timeline.set_range(77, fps=4)
        assert timeline.marker_bar._n_frames == 77
