"""The Insight model and its navigation widget (V3 Phase 0, Fire
Intelligence Layer).

An `Insight` is the single output type of every interpretive feature in
V3 (events, semantic diff, query results, cause chains): a computed
statement bound to WHEN (time), WHERE (location/region), WHAT (quantity +
value), and WHY (its supporting evidence), plus a `basis` string naming
how it was computed. This is the connective tissue of the whole layer --
one type produced everywhere, one interaction consumed everywhere: click
the statement, jump to the time, highlight the location, show the field.

Honesty rule (inherited from auto_summary.py): a statement is a template
filled by computed values. Nothing here generates unsupported text; the
`basis` field records the computation so a claim is always traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Insight:
    statement: str                       # human-readable, templated from computed values
    category: str = "event"              # event | difference | query | cause
    quantity: Optional[str] = None       # which physical field, if applicable
    time_s: Union[float, tuple, None] = None   # instant, or (t0, t1) interval
    location: Optional[tuple] = None     # physical (x, z), if applicable
    region: Optional[tuple] = None       # physical (x0, x1, z0, z1), if applicable
    value: Optional[float] = None
    unit: str = ""
    basis: str = ""                      # how it was computed (traceability, no hallucination)
    evidence: tuple = field(default_factory=tuple)  # supporting sub-Insights

    def primary_time(self) -> Optional[float]:
        """The single time to seek to (interval start for an interval)."""
        if isinstance(self.time_s, tuple):
            return float(self.time_s[0]) if self.time_s else None
        return None if self.time_s is None else float(self.time_s)

    def frame_index(self, fps: int) -> Optional[int]:
        t = self.primary_time()
        return None if t is None else int(round(t * max(1, fps)))

    def to_selection(self, scenario=None):
        """This Insight as a Selection (V5-M1): an instant time_s maps to
        `time_s`, an interval to `interval`; location/region/quantity carry
        across. `scenario` is supplied by the caller (Insights don't hold
        one). Imported lazily so this module stays Qt-free to import."""
        from selection import Selection
        time_s = self.time_s if isinstance(self.time_s, (int, float)) else None
        interval = tuple(self.time_s) if isinstance(self.time_s, tuple) else None
        return Selection(scenario=scenario, quantity=self.quantity or "TEMPERATURE",
                         point=self.location, region=self.region,
                         time_s=time_s, interval=interval)


# ------------------------------------------------------------------ widget
from PyQt5 import QtCore, QtWidgets  # noqa: E402


class InsightList(QtWidgets.QListWidget):
    """A clickable list of Insights. Emits `insight_activated(Insight)`
    when a row is clicked or activated -- MainWindow wires that one signal
    to the shared navigation (seek to the insight's time, show its
    quantity, flash its location), so every V3 feature that produces
    Insights gets the same navigation for free."""

    insight_activated = QtCore.pyqtSignal(object)  # the Insight (navigate)
    insight_saved = QtCore.pyqtSignal(object)      # the Insight (-> Evidence Notebook, V4-M2)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("Insights list")
        self.setWordWrap(True)
        self.setAlternatingRowColors(True)
        self.itemClicked.connect(self._emit)
        self.itemActivated.connect(self._emit)
        # V4-M2: any measurement (an Insight, produced by every panel) can
        # be saved to the Evidence Notebook via a right-click, so the save
        # affordance is defined once and inherited everywhere.
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Save to Evidence Notebook")
        if menu.exec_(self.viewport().mapToGlobal(pos)) is act:
            ins = item.data(QtCore.Qt.UserRole)
            if ins is not None:
                self.insight_saved.emit(ins)

    def set_insights(self, insights: list) -> None:
        self.clear()
        for ins in insights:
            item = QtWidgets.QListWidgetItem(self._row_text(ins))
            item.setData(QtCore.Qt.UserRole, ins)
            tip = ins.basis or ins.statement
            item.setToolTip(tip)
            self.addItem(item)

    @staticmethod
    def _row_text(ins: Insight) -> str:
        t = ins.primary_time()
        prefix = f"t = {t:.1f} s   " if t is not None else ""
        return prefix + ins.statement

    def _emit(self, item) -> None:
        ins = item.data(QtCore.Qt.UserRole)
        if ins is not None:
            self.insight_activated.emit(ins)
