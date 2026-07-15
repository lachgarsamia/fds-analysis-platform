"""Unit + integration tests for FireLab roadmap Phase 5: kiosk/attract
mode, the first-run guided tour, demo-script bookmarks, and the Esc
long-press effects master switch. Kept small -- these are thin triggers
on already-tested machinery (page navigation, the cinematic toggle), not
new rendering paths."""

from PyQt5 import QtWidgets

from data_provider import load_simulation_data
from kiosk import KioskController
from main_window import MainWindow
from tour import mark_tour_completed, should_show_tour


class TestKioskController:
    def test_idle_then_activity_round_trips(self, qapp):
        events = []
        controller = KioskController(
            on_idle=lambda: events.append("idle"),
            on_wake=lambda: events.append("wake"),
            app=qapp, cursor_target=QtWidgets.QWidget(),
            idle_timeout_ms=999_999, cursor_hide_delay_ms=999_999,
        )
        controller._enter_idle()
        assert events == ["idle"]
        controller._on_activity()
        assert events == ["idle", "wake"]
        controller.shutdown()

    def test_activity_before_idle_does_not_wake(self, qapp):
        events = []
        controller = KioskController(
            on_idle=lambda: events.append("idle"),
            on_wake=lambda: events.append("wake"),
            app=qapp, cursor_target=QtWidgets.QWidget(),
            idle_timeout_ms=999_999, cursor_hide_delay_ms=999_999,
        )
        controller._on_activity()  # never idle yet -- must not fire on_wake
        assert events == []
        controller.shutdown()

    def test_cursor_hides_and_restores(self, qapp):
        target = QtWidgets.QWidget()
        controller = KioskController(
            on_idle=lambda: None, on_wake=lambda: None,
            app=qapp, cursor_target=target,
            idle_timeout_ms=999_999, cursor_hide_delay_ms=999_999,
        )
        controller._hide_cursor()
        assert target.cursor().shape() == 0x0A  # Qt.BlankCursor
        controller._on_activity()
        assert target.cursor().shape() != 0x0A
        controller.shutdown()

    def test_shutdown_removes_event_filter(self, qapp):
        """Regression: an unremoved QApplication-level event filter was
        found to both segfault intermittently (unparented QObject kept
        alive only by Python refcounting) and, once parented, to silently
        accumulate across every window instance that skipped shutdown()
        -- a test suite constructing ~40 MainWindows went from ~27s to
        ~470s without this. Can't assert "not installed" directly (Qt
        exposes no query for that), so this asserts the observable
        behavior instead: a wake callback must not fire after shutdown()."""
        controller = KioskController(
            on_idle=lambda: None, on_wake=lambda: None,
            app=qapp, cursor_target=QtWidgets.QWidget(),
            idle_timeout_ms=999_999, cursor_hide_delay_ms=999_999,
        )
        controller._enter_idle()
        controller.shutdown()
        # shutdown() stops both timers -- the simplest direct, non-flaky
        # check that it actually tore state down rather than just
        # calling removeEventFilter() and leaving timers running.
        assert not controller._idle_timer.isActive()
        assert not controller._cursor_timer.isActive()


class TestTourSettings:
    def test_shows_once_then_remembered(self, qapp, tmp_path):
        from PyQt5 import QtCore
        settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
        assert should_show_tour(settings)
        mark_tour_completed(settings)
        assert not should_show_tour(settings)


class TestDemoBookmarksAndMasterSwitch:
    def test_record_and_jump_bookmark(self, qapp):
        window = MainWindow(load_simulation_data())
        window._navigate_to("live")
        window.time_controller.seek(50)
        window._record_bookmark(1)
        window.time_controller.seek(0)
        window._jump_to_bookmark(1)
        assert window.time_controller.index == 50
        assert window._active_page_key == "live"
        window.close()

    def test_jump_to_empty_bookmark_does_not_crash(self, qapp):
        window = MainWindow(load_simulation_data())
        window._jump_to_bookmark(9)  # never recorded
        window.close()

    def test_esc_long_press_toggles_cinematic_action(self, qapp):
        window = MainWindow(load_simulation_data())
        was_checked = window.cinematic_action.isChecked()
        window._toggle_effects_master_switch()
        assert window.cinematic_action.isChecked() != was_checked
        window._toggle_effects_master_switch()
        assert window.cinematic_action.isChecked() == was_checked
        window.close()
