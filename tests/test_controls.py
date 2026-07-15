"""Unit tests for the physical-mirror control widgets + narration
templates (FireLab roadmap Phase 3). Focused on the drop-in contract
(value_changed/set_value/set_enabled_all) these widgets must honor to be
safe replacements for widgets.ToggleGroup -- not on pixel-level painting,
which the manual visual check already covered."""

import pytest

from auto_summary import narrate_frame
from controls.candle_card import CandleCard
from controls.door_widget import DoorWidget
from controls.vent_widget import VentWidget


class TestCandleCard:
    def test_default_index_selected(self, qapp):
        card = CandleCard([("1 candle", 0), ("2 candles", 1)], default_index=0)
        assert card.value == 0

    def test_click_emits_value_changed(self, qapp):
        card = CandleCard([("1 candle", 0), ("2 candles", 1)], default_index=0)
        received = []
        card.value_changed.connect(received.append)
        card._buttons[1].click()
        assert received == [1]

    def test_set_value_does_not_reemit(self, qapp):
        card = CandleCard([("1 candle", 0), ("2 candles", 1)], default_index=0)
        received = []
        card.value_changed.connect(received.append)
        card.set_value(1)
        assert card.value == 1
        assert received == []

    def test_set_enabled_all(self, qapp):
        card = CandleCard([("1 candle", 0), ("2 candles", 1)], default_index=0)
        card.set_enabled_all(False)
        assert all(not b.isEnabled() for b in card._buttons)


class TestDoorWidget:
    def test_click_emits_and_starts_swing(self, qapp):
        door = DoorWidget([("Wide open", 1), ("Narrow", 0)], default_index=0)
        received = []
        door.value_changed.connect(received.append)
        door._buttons[1].click()
        assert received == [0]
        assert door._anim_timer.isActive()

    def test_set_value_updates_without_reemitting(self, qapp):
        door = DoorWidget([("Wide open", 1), ("Narrow", 0)], default_index=0)
        received = []
        door.value_changed.connect(received.append)
        door.set_value(0)
        assert door.value == 0
        assert received == []


class TestVentWidget:
    VOD_STATES = {0: "open", 1: "closed", 2: "HVAC"}

    def test_click_emits_value_changed(self, qapp):
        vent = VentWidget([("Open", 0), ("Closed", 1), ("HVAC", 2)], state_labels=self.VOD_STATES, default_index=0)
        received = []
        vent.value_changed.connect(received.append)
        vent._buttons[2].click()
        assert received == [2]

    def test_unknown_value_falls_back_to_open_behavior(self, qapp):
        """A value missing from state_labels must not crash the flow
        animation -- falls back to "open" per the class's own docstring."""
        vent = VentWidget([("Mystery", 99)], state_labels={}, default_index=0)
        vent._tick()  # must not raise


class TestNarrateFrame:
    def test_near_ambient(self):
        text = narrate_frame(current_temp_c=25.0, peak_temp_c=25.0, ambient_c=20.0, door_wide_open=True)
        assert "starting temperature" in text

    def test_smoke_forming(self):
        text = narrate_frame(current_temp_c=100.0, peak_temp_c=200.0, ambient_c=20.0, door_wide_open=True)
        assert "smoke layer is forming" in text

    def test_hazardous(self):
        text = narrate_frame(current_temp_c=350.0, peak_temp_c=350.0, ambient_c=20.0, door_wide_open=True)
        assert "hazardous" in text

    def test_door_open_mentions_feeding_the_flame(self):
        text = narrate_frame(current_temp_c=150.0, peak_temp_c=200.0, ambient_c=20.0, door_wide_open=True)
        assert "feeding fresh air" in text

    def test_door_narrow_mentions_limiting_air(self):
        text = narrate_frame(current_temp_c=150.0, peak_temp_c=200.0, ambient_c=20.0, door_wide_open=False)
        assert "limiting" in text

    def test_peak_moment_noted(self):
        text = narrate_frame(current_temp_c=400.0, peak_temp_c=400.0, ambient_c=20.0, door_wide_open=True)
        assert "hottest point" in text

    def test_door_clause_absent_near_ambient(self):
        """Door influence is only mentioned once there's actually a fire
        producing airflow worth describing."""
        text = narrate_frame(current_temp_c=22.0, peak_temp_c=22.0, ambient_c=20.0, door_wide_open=True)
        assert "door" not in text.lower()
