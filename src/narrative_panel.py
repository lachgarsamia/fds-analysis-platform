"""Scientific Narrative++ panel (V5-M5 / Phase 5), an Analysis-page tab.

Extends the V3 Fire Story from a flat list into an expandable, evidence-backed
event chain: ignition → fastest heating → threshold crossings → peak → layer
descent → stabilization. Each node is a detected event; expanding it shows the
computed evidence behind it (its value, its time, and the `basis` that names
how it was computed -- no invented prose). Activating a node publishes its
selection to the SelectionBus (M1), so the Live Viewer and every panel jump to
that moment.

Reuses descriptors + events; scenario_combo is bound to the bus by main_window.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from slice_key import SliceKey
from descriptors import compute_descriptors
from events import detect_events


class NarrativePanel(QtWidgets.QWidget):
    event_activated = QtCore.pyqtSignal(object)   # the event Insight (-> bus, seek)

    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Fire narrative")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Narrative scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "The detected event chain, in order. Expand a step for its computed "
            "evidence; click a step to jump the whole workspace to that moment.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setAccessibleName("Fire narrative tree")
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemClicked.connect(self._on_item)
        layout.addWidget(self.tree, 1)

        self.scenario_combo.currentIndexChanged.connect(self._reload)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _events(self, case_index):
        if case_index not in self._cache:
            data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
            extent = self._store.get_extent(case_index, SliceKey("TEMPERATURE"))
            try:
                desc = compute_descriptors(data, extent, self._fps)
                events = detect_events(desc)
            except Exception:
                events = []
            self._cache[case_index] = sorted(
                events, key=lambda e: (e.primary_time() is None, e.primary_time() or 0.0))
        return self._cache[case_index]

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        self.tree.clear()
        for i, ev in enumerate(self._events(case_index), 1):
            t = ev.primary_time()
            head = (f"{i}. " + (f"[{t:.1f} s] " if t is not None else "") + ev.statement)
            node = QtWidgets.QTreeWidgetItem([head])
            node.setData(0, QtCore.Qt.UserRole, ev)
            for label in self._evidence_lines(ev):
                node.addChild(QtWidgets.QTreeWidgetItem([label]))
            self.tree.addTopLevelItem(node)
        if self.tree.topLevelItemCount():
            self.tree.topLevelItem(0).setExpanded(True)

    @staticmethod
    def _evidence_lines(ev) -> list:
        lines = []
        t = ev.primary_time()
        if t is not None:
            lines.append(f"when: t = {t:.1f} s")
        if ev.value is not None:
            lines.append(f"value: {ev.value:.1f} {ev.unit}".rstrip())
        if getattr(ev, "location", None):
            lines.append(f"where: ({ev.location[0]:.2f}, {ev.location[1]:.2f}) m")
        if ev.basis:
            lines.append(f"basis: {ev.basis}")
        return lines or ["(no further evidence)"]

    def _on_item(self, item, _col) -> None:
        # a top-level (event) node carries the Insight; a child does not
        ev = item.data(0, QtCore.Qt.UserRole)
        if ev is None and item.parent() is not None:
            ev = item.parent().data(0, QtCore.Qt.UserRole)
        if ev is not None:
            self.event_activated.emit(ev)
