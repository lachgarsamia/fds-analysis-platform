"""Investigation History (V6-M4): browser-style back/forward through the
Shared Selection Model, so a researcher can retrace an investigation
without re-deriving it.

A bounded log of `Selection`s. `record()` appends (no-op if the incoming
selection equals the current head -- the same "no-op when unchanged" rule
`SelectionBus.set` already applies, so a resend doesn't pollute the log).
`back()`/`forward()` move the cursor and return the `Selection` to replay;
recording a new entry after a `back()` truncates any forward branch (plain
browser-history semantics).

Pure, Qt-free -- MainWindow owns the Qt wiring (see main_window.py's
`_on_history_changed`, which guards against re-recording its own replay
with a dedicated sentinel origin).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from selection import Selection

_DEFAULT_MAX_LEN = 200


@dataclass(frozen=True)
class HistoryEntry:
    selection: Selection
    label: str = ""


class InvestigationHistory:
    def __init__(self, max_len: int = _DEFAULT_MAX_LEN):
        self._max_len = max(1, max_len)
        self._entries: list[HistoryEntry] = []
        self._cursor = -1   # index of the "current" entry in _entries

    def record(self, selection: Selection, label: str = "") -> bool:
        """Append `selection` as the new current entry, truncating any
        forward branch. No-op (returns False) if it equals the current
        entry's selection -- a resend/no-op selection must not grow the
        log. Returns True if an entry was recorded."""
        if self._cursor >= 0 and self._entries[self._cursor].selection == selection:
            return False
        del self._entries[self._cursor + 1:]
        self._entries.append(HistoryEntry(selection, label))
        if len(self._entries) > self._max_len:
            self._entries.pop(0)
        else:
            self._cursor += 1
        self._cursor = min(self._cursor, len(self._entries) - 1)
        return True

    def can_back(self) -> bool:
        return self._cursor > 0

    def can_forward(self) -> bool:
        return 0 <= self._cursor < len(self._entries) - 1

    def back(self) -> Optional[Selection]:
        if not self.can_back():
            return None
        self._cursor -= 1
        return self._entries[self._cursor].selection

    def forward(self) -> Optional[Selection]:
        if not self.can_forward():
            return None
        self._cursor += 1
        return self._entries[self._cursor].selection

    @property
    def current(self) -> Optional[HistoryEntry]:
        return self._entries[self._cursor] if 0 <= self._cursor < len(self._entries) else None

    @property
    def entries(self) -> list:
        """Every recorded entry, oldest first (for a recent-history list UI)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
