"""V6-M4 Investigation History: pure dataclass/logic tests, no Qt fixture."""

from __future__ import annotations

from history import InvestigationHistory
from selection import Selection


class TestRecord:
    def test_records_and_becomes_current(self):
        h = InvestigationHistory()
        h.record(Selection(scenario=0))
        assert h.current.selection == Selection(scenario=0)
        assert len(h) == 1

    def test_no_op_when_unchanged(self):
        h = InvestigationHistory()
        h.record(Selection(scenario=0))
        assert h.record(Selection(scenario=0)) is False
        assert len(h) == 1

    def test_distinct_selections_both_recorded(self):
        h = InvestigationHistory()
        h.record(Selection(scenario=0))
        h.record(Selection(scenario=1))
        assert len(h) == 2
        assert h.current.selection == Selection(scenario=1)


class TestBackForward:
    def test_back_and_forward_round_trip(self):
        h = InvestigationHistory()
        h.record(Selection(scenario=0))
        h.record(Selection(scenario=1))
        h.record(Selection(scenario=2))
        assert h.back() == Selection(scenario=1)
        assert h.back() == Selection(scenario=0)
        assert h.back() is None                    # already at the oldest entry
        assert h.forward() == Selection(scenario=1)
        assert h.forward() == Selection(scenario=2)
        assert h.forward() is None                  # already at the newest entry

    def test_can_back_can_forward_flags(self):
        h = InvestigationHistory()
        assert not h.can_back() and not h.can_forward()
        h.record(Selection(scenario=0))
        assert not h.can_back() and not h.can_forward()
        h.record(Selection(scenario=1))
        assert h.can_back() and not h.can_forward()
        h.back()
        assert not h.can_back() and h.can_forward()

    def test_record_after_back_truncates_forward_branch(self):
        h = InvestigationHistory()
        h.record(Selection(scenario=0))
        h.record(Selection(scenario=1))
        h.record(Selection(scenario=2))
        h.back()                                     # now at scenario=1
        h.record(Selection(scenario=9))               # branches off -- scenario=2 is gone
        assert not h.can_forward()
        assert [e.selection.scenario for e in h.entries] == [0, 1, 9]


class TestBounded:
    def test_oldest_entries_drop_when_over_max_len(self):
        h = InvestigationHistory(max_len=3)
        for i in range(5):
            h.record(Selection(scenario=i))
        assert len(h) == 3
        assert [e.selection.scenario for e in h.entries] == [2, 3, 4]
        assert h.current.selection == Selection(scenario=4)

    def test_back_still_works_after_overflow(self):
        h = InvestigationHistory(max_len=3)
        for i in range(5):
            h.record(Selection(scenario=i))
        assert h.back() == Selection(scenario=3)
        assert h.back() == Selection(scenario=2)
        assert h.back() is None


class TestLabel:
    def test_label_is_stored(self):
        h = InvestigationHistory()
        h.record(Selection(scenario=0), label="placed a device")
        assert h.current.label == "placed a device"
