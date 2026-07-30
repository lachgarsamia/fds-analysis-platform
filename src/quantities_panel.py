"""Quantities reference panel (V4-M11), an Analysis-page tab.

A single, honest view of every physical quantity the app knows: the ones
readable now, the derived ones computed from them, and the target
quantities that are gated on the M-SIM cluster re-run. Each row shows unit,
kind, status, and a plain-language interpretation; gated rows show the
gate reason (non-breaking -- gated quantities never enter the tool combos,
which are data-driven). Derived quantities offer a live preview computed
on a real scenario, proving they are feasible today.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from registry import QUANTITY_REGISTRY, get_quantity, quantity_status
from slice_key import SliceKey
from analysis_panel_base import populate_scenario_combo
import derived_quantities as dq

_STATUS_LABEL = {"available": "Available", "derived": "Derived (computable)",
                 "gated": "Gated"}


class QuantitiesPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Quantities")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Preview scenario")
        populate_scenario_combo(self.scenario_combo, self._manifest)
        header.addWidget(QtWidgets.QLabel("Preview on:"))
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "Available quantities are read from the current run; derived ones are "
            "computed from them; gated ones await the M-SIM cluster re-run and are "
            "shown here for reference only (they never enter the analysis tools).")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.table = QtWidgets.QTableWidget()
        self.table.setAccessibleName("Quantities table")
        cols = ["Quantity", "Unit", "Kind", "Status"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.currentCellChanged.connect(lambda *_: self._on_row())
        layout.addWidget(self.table, 1)

        self.detail = QtWidgets.QLabel("Select a quantity to see its interpretation.")
        self.detail.setWordWrap(True)
        self.detail.setProperty("role", "value")
        self.detail.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.detail)

        self.preview_button = QtWidgets.QPushButton("Preview derived on scenario")
        self.preview_button.setAccessibleName("Preview derived quantity")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._preview_derived)
        layout.addWidget(self.preview_button)

        self._populate()

    def _populate(self) -> None:
        names = list(QUANTITY_REGISTRY.keys())
        self.table.setRowCount(len(names))
        for r, name in enumerate(names):
            q = QUANTITY_REGISTRY[name]
            status = quantity_status(name)
            cells = [q.label, q.unit, q.kind, _STATUS_LABEL[status]]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, name)
                if status == "gated":
                    item.setForeground(QtCore.Qt.gray)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        if names:
            self.table.selectRow(0)

    def _current_name(self):
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _on_row(self) -> None:
        name = self._current_name()
        if name is None:
            return
        q = get_quantity(name)
        status = quantity_status(name)
        text = f"<b>{q.label}</b> ({q.unit}) — {q.interpretation}"
        if status == "gated":
            text += f"<br><i>Gated:</i> {q.gate_reason}"
        elif status == "derived":
            src = dq.source_quantity(name)
            text += f"<br><i>Derived from {src}. Use the preview button below.</i>"
        self.detail.setText(text)
        self.preview_button.setEnabled(status == "derived"
                                       and self.scenario_combo.count() > 0)

    def _preview_derived(self) -> None:
        name = self._current_name()
        src = dq.source_quantity(name)
        case_index = self.scenario_combo.currentData()
        if src is None or case_index is None:
            return
        try:
            data = np.asarray(self._store.get(case_index, SliceKey(src)))
        except Exception:
            self.detail.setText(self.detail.text() + "<br><i>Source data unavailable.</i>")
            return
        frame = data[int(data.shape[0] * 0.6)]
        field = dq.derive(name, frame)
        q = get_quantity(name)
        self.detail.setText(
            f"<b>{q.label}</b> preview on {self.scenario_combo.currentText()}: "
            f"min {field.min():.2f}, mean {field.mean():.2f}, max {field.max():.2f} {q.unit} "
            f"(computed from {src}).")
