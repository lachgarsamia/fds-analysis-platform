"""Unit tests for TimeController (M1.4.1): the pull-based playback clock."""

import time

from time_controller import TimeController


class TestTimeController:
    def test_starts_paused_at_index_zero(self, qapp):
        tc = TimeController(lambda: 100, 4)
        assert not tc.is_playing()
        assert tc.index == 0

    def test_play_pause_toggles_state_and_emits_signal(self, qapp):
        tc = TimeController(lambda: 100, 4)
        events = []
        tc.playing_changed.connect(events.append)
        tc.play()
        assert tc.is_playing()
        tc.pause()
        assert not tc.is_playing()
        assert events == [True, False]

    def test_play_is_idempotent(self, qapp):
        tc = TimeController(lambda: 100, 4)
        tc.play()
        tc.play()  # must not restart the timer or raise
        assert tc.is_playing()
        tc.pause()

    def test_seek_clamps_to_valid_range(self, qapp):
        tc = TimeController(lambda: 50, 4)
        tc.seek(1000)
        assert tc.index == 49
        tc.seek(-10)
        assert tc.index == 0
        tc.seek(25)
        assert tc.index == 25

    def test_seek_emits_time_changed(self, qapp):
        tc = TimeController(lambda: 50, 4)
        seen = []
        tc.time_changed.connect(seen.append)
        tc.seek(10)
        assert seen == [10]

    def test_step_moves_relative_to_current_index(self, qapp):
        tc = TimeController(lambda: 50, 4)
        tc.seek(10)
        tc.step(5)
        assert tc.index == 15
        tc.step(-3)
        assert tc.index == 12

    def test_set_speed_takes_effect_immediately_while_playing(self, qapp):
        """DoD: 'speed change takes effect immediately'. Restarting the
        QTimer with the new interval (rather than waiting for the current
        one to elapse) is what makes this true -- verify play() actually
        produces ticks at both speeds within a bounded wait."""
        tc = TimeController(lambda: 10_000, 4)  # long scenario, no wraparound noise
        ticks = []
        tc.time_changed.connect(ticks.append)

        tc.set_speed(1)
        tc.play()
        deadline = time.perf_counter() + 1.0
        while len(ticks) < 1 and time.perf_counter() < deadline:
            qapp.processEvents()
        assert len(ticks) >= 1, "expected at least one tick at 1x speed within 1s"

        n_before = len(ticks)
        tc.set_speed(3)  # 3x -> ~83ms/tick instead of ~250ms/tick at 4fps
        deadline = time.perf_counter() + 1.0
        while len(ticks) < n_before + 2 and time.perf_counter() < deadline:
            qapp.processEvents()
        tc.pause()
        assert len(ticks) >= n_before + 2, (
            f"expected several ticks within 1s at 3x speed, got {len(ticks) - n_before}"
        )

    def test_loop_wraps_to_zero(self, qapp):
        tc = TimeController(lambda: 5, 4)
        tc.set_loop(True)
        tc.seek(4)  # last valid index
        tc._tick()
        assert tc.index == 0

    def test_no_loop_stops_at_last_frame(self, qapp):
        tc = TimeController(lambda: 5, 4)
        tc.set_loop(False)
        tc.seek(3)  # not yet at the end, so play() won't trigger its own
        tc.play()   # separate "restart from top" convenience (tested below)
        tc._tick()  # 3 -> 4, still valid (last index)
        assert tc.index == 4
        assert tc.is_playing()
        tc._tick()  # would overflow past the last index -> stop instead
        assert tc.index == 4, "must stay on the last frame, not overflow"
        assert not tc.is_playing(), "must pause itself at the end when not looping"

    def test_play_restarts_from_top_if_sitting_at_the_end_without_loop(self, qapp):
        tc = TimeController(lambda: 5, 4)
        tc.set_loop(False)
        tc.seek(4)  # already at the last frame
        tc.play()   # pressing Play again should restart, not do nothing
        assert tc.index == 0
        assert tc.is_playing()
        tc.pause()

    def test_restart_seeks_to_zero_and_preserves_play_state(self, qapp):
        tc = TimeController(lambda: 50, 4)
        tc.seek(30)
        tc.play()
        tc.restart()
        assert tc.index == 0
        assert tc.is_playing()

        tc.pause()
        tc.seek(30)
        tc.restart()
        assert tc.index == 0
        assert not tc.is_playing()

    def test_zero_frame_count_tick_is_a_safe_noop(self, qapp):
        """No scenario data available yet shouldn't crash a tick."""
        tc = TimeController(lambda: 0, 4)
        tc.play()
        tc._tick()  # must not raise
        tc.pause()
