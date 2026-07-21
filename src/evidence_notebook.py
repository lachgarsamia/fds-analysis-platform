"""Evidence Notebook (V4-M2), the connective tissue of the analysis
environment.

Every measurement a researcher takes -- a height readout, a query answer,
a semantic-diff finding, a Fire-story event -- is produced as an
`Insight` (insight.py). The notebook makes those Insights *persistent*:
each saved Insight becomes a `NotebookEntry` the researcher can annotate
(free-text note) and tag, reorder, and remove. The notebook is saved in
the session (session.py schema v2) and flows into reports, so an analysis
that took real work is never lost between sittings.

Pure model + serialization, Qt-free. The dockable widget lives in
evidence_notebook_panel.py; the shared navigation stays the one
`insight_activated` interaction (main_window).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from insight import Insight


def insight_to_dict(ins: Insight) -> dict:
    """Serialize an Insight to JSON-safe plain values. time_s may be a
    tuple (interval); JSON turns tuples into lists, and Insight accepts a
    list back for the interval case, so we normalize on load, not here."""
    return {
        "statement": ins.statement,
        "category": ins.category,
        "quantity": ins.quantity,
        "time_s": list(ins.time_s) if isinstance(ins.time_s, tuple) else ins.time_s,
        "location": list(ins.location) if ins.location is not None else None,
        "region": list(ins.region) if ins.region is not None else None,
        "value": ins.value,
        "unit": ins.unit,
        "basis": ins.basis,
    }


def insight_from_dict(d: dict) -> Insight:
    """Rebuild an Insight from insight_to_dict output. Evidence
    sub-Insights are not persisted (they are recomputable context, not a
    saved measurement), so `evidence` stays empty."""
    time_s = d.get("time_s")
    if isinstance(time_s, list):
        time_s = tuple(time_s)
    loc = d.get("location")
    reg = d.get("region")
    return Insight(
        statement=d.get("statement", ""),
        category=d.get("category", "event"),
        quantity=d.get("quantity"),
        time_s=time_s,
        location=tuple(loc) if loc is not None else None,
        region=tuple(reg) if reg is not None else None,
        value=d.get("value"),
        unit=d.get("unit", ""),
        basis=d.get("basis", ""),
    )


@dataclass
class NotebookEntry:
    insight: Insight
    note: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"insight": insight_to_dict(self.insight),
                "note": self.note, "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, d: dict) -> "NotebookEntry":
        return cls(insight=insight_from_dict(d.get("insight", {})),
                   note=str(d.get("note", "")),
                   tags=[str(t) for t in d.get("tags", [])])


class EvidenceNotebook:
    """An ordered, annotatable collection of saved Insights. Model only:
    the widget observes it and rebuilds its rows on change."""

    def __init__(self, entries: List[NotebookEntry] | None = None):
        self._entries: List[NotebookEntry] = list(entries or [])

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[NotebookEntry]:
        return self._entries

    def is_empty(self) -> bool:
        return not self._entries

    def add(self, insight: Insight, note: str = "", tags: List[str] | None = None) -> NotebookEntry:
        entry = NotebookEntry(insight=insight, note=note, tags=list(tags or []))
        self._entries.append(entry)
        return entry

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._entries):
            del self._entries[index]

    def clear(self) -> None:
        self._entries.clear()

    def move(self, index: int, delta: int) -> int:
        """Reorder entry `index` by `delta` (+1 down, -1 up); returns its
        new index (clamped, a no-op at the ends)."""
        j = index + delta
        if 0 <= index < len(self._entries) and 0 <= j < len(self._entries):
            self._entries[index], self._entries[j] = self._entries[j], self._entries[index]
            return j
        return index

    def set_note(self, index: int, note: str) -> None:
        if 0 <= index < len(self._entries):
            self._entries[index].note = note

    def set_tags(self, index: int, tags: List[str]) -> None:
        if 0 <= index < len(self._entries):
            self._entries[index].tags = [t.strip() for t in tags if t.strip()]

    def to_list(self) -> list:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, data: list | None) -> "EvidenceNotebook":
        return cls([NotebookEntry.from_dict(d) for d in (data or [])
                    if isinstance(d, dict)])
