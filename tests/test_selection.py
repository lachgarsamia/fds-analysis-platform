"""V5-M1 Step 1: the Shared Selection Model foundation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selection import Selection, SelectionContext, SelectionBus  # noqa: E402


class TestSelection:
    def test_defaults(self):
        s = Selection()
        assert s.scenario is None and s.quantity == "TEMPERATURE"
        assert s.point is None and s.interval is None and s.comparison is None

    def test_with_is_immutable_copy(self):
        a = Selection(scenario=0, time_s=8.0)
        b = a.with_(scenario=3, point=(0.9, 0.1))
        assert a.scenario == 0 and a.point is None      # original untouched
        assert b.scenario == 3 and b.point == (0.9, 0.1) and b.time_s == 8.0

    def test_equality_and_hashable(self):
        assert Selection(scenario=1) == Selection(scenario=1)
        assert Selection(scenario=1) != Selection(scenario=2)
        assert len({Selection(scenario=1), Selection(scenario=1)}) == 1  # hashable

    def test_frame_from_time(self):
        assert Selection(time_s=8.0).frame(4) == 32
        assert Selection().frame(4) is None


class TestSelectionContext:
    def test_accessors_and_helpers(self):
        s = Selection(scenario=2, quantity="VELOCITY", point=(0.9, 0.1),
                      interval=(10.0, 40.0), time_s=8.0, phase="growth")
        ctx = SelectionContext(s)
        assert ctx.scenario == 2 and ctx.quantity == "VELOCITY"
        assert ctx.point == (0.9, 0.1) and ctx.phase == "growth"
        assert ctx.has_point() and ctx.has_interval() and not ctx.has_region()
        assert ctx.frame(4) == 32
        assert ctx.selection is s

    def test_carries_provider(self):
        sentinel = object()
        ctx = SelectionContext(Selection(), provider=sentinel)
        assert ctx.provider is sentinel


class TestSelectionBus:
    def test_set_publishes_with_origin(self, qapp):
        bus = SelectionBus()
        seen = []
        bus.changed.connect(lambda sel, origin: seen.append((sel, origin)))
        origin = object()
        changed = bus.set(Selection(scenario=1), origin=origin)
        assert changed is True
        assert bus.current == Selection(scenario=1)
        assert len(seen) == 1 and seen[0][1] is origin

    def test_set_is_noop_when_unchanged(self, qapp):
        bus = SelectionBus(Selection(scenario=1))
        seen = []
        bus.changed.connect(lambda *a: seen.append(a))
        changed = bus.set(Selection(scenario=1))     # equal to current
        assert changed is False and seen == []       # no ricochet

    def test_update_partial(self, qapp):
        bus = SelectionBus(Selection(scenario=1, time_s=2.0))
        bus.update(scenario=1, time_s=5.0)
        assert bus.current == Selection(scenario=1, time_s=5.0)

    def test_context_reflects_current(self, qapp):
        bus = SelectionBus(Selection(scenario=7))
        assert bus.context().scenario == 7
