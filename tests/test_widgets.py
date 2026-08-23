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


class TestMplCanvasBlitCacheGuard:
    """Render-artifact investigation: blit_update() used to trust a cached
    background's size blindly -- restore_region() doesn't validate that
    itself, it just paints the stale/wrong-sized region into place, which
    reproduced as striped/misplaced content extending past the heatmap's
    real bounds whenever a resize slipped past resizeEvent's own cache
    invalidation. Tests the confirmed mechanism directly (a forced size
    mismatch), not the still-unpinned real-world trigger path."""

    def _canvas_with_image(self):
        canvas = MplCanvas()
        canvas.resize(200, 150)
        ax = canvas.fig.add_subplot(111)
        image = ax.imshow([[0, 1], [1, 0]])
        canvas.capture_background()
        return canvas, image

    def test_matched_size_blits_normally_no_warning(self, qapp, caplog):
        canvas, image = self._canvas_with_image()
        before = canvas._background
        with caplog.at_level("WARNING", logger="widgets"):
            canvas.blit_update(image)
        assert canvas._background is before  # unchanged -- no fallback triggered
        assert "cached background size" not in caplog.text

    def test_mismatched_size_falls_back_to_full_redraw_not_corruption(self, qapp, caplog):
        canvas, image = self._canvas_with_image()
        # Force the exact mechanism: a cached background whose size no
        # longer matches the figure's current size, without a matching
        # resizeEvent to invalidate it first (see widgets.py's own repro
        # for how this can happen on a resize resizeEvent doesn't catch).
        canvas.fig.set_size_inches(1.0, 1.0)
        canvas.draw()
        canvas._background = canvas.copy_from_bbox(canvas.fig.bbox)
        canvas.fig.set_size_inches(4.0, 3.0)
        canvas.draw()
        mismatched_bg = canvas._background

        with caplog.at_level("WARNING", logger="widgets"):
            canvas.blit_update(image)

        assert "cached background size" in caplog.text
        # The guard must have replaced the stale cache with a fresh one
        # matching the figure's real current size, not left it as-is.
        assert canvas._background is not mismatched_bg
        x0, y0, x1, y1 = canvas._background.get_extents()
        assert (x1 - x0, y1 - y0) == (canvas.fig.bbox.width, canvas.fig.bbox.height)

    def test_no_cached_background_falls_back_without_warning(self, qapp, caplog):
        """The pre-existing "no cache yet" fallback (first paint) is a
        distinct, expected case -- must not also log a spurious mismatch
        warning."""
        canvas, image = self._canvas_with_image()
        canvas._background = None
        with caplog.at_level("WARNING", logger="widgets"):
            canvas.blit_update(image)
        assert "cached background size" not in caplog.text
        assert canvas._background is not None


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
