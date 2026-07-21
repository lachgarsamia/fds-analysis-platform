"""The dockable Evidence Notebook widget (V4-M2).

Hosts the `EvidenceNotebook` model and renders each saved measurement as
a row (statement, note, tags). Clicking a row re-navigates via the shared
`insight_activated` signal (same interaction as every Insight list);
per-row actions annotate, tag, reorder, and remove. main_window connects
each panel's `InsightList.insight_saved` to `add_insight` here, and reads
`self.notebook` when saving a session and calls `load_notebook` when
restoring one.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from insight import Insight
from evidence_notebook import EvidenceNotebook


class EvidenceNotebookDock(QtWidgets.QDockWidget):
    insight_activated = QtCore.pyqtSignal(object)  # reuse the shared navigation

    def __init__(self, parent=None):
        super().__init__("Evidence Notebook", parent)
        self.setObjectName("evidenceNotebookDock")
        self.notebook = EvidenceNotebook()

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.caption = QtWidgets.QLabel(
            "Right-click any measurement and choose \"Save to Evidence "
            "Notebook\". Saved findings are annotatable and stored in the "
            "session.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Evidence notebook entries")
        self.list.setWordWrap(True)
        self.list.setAlternatingRowColors(True)
        self.list.itemClicked.connect(self._emit_activated)
        self.list.itemDoubleClicked.connect(lambda _i: self._edit_note())
        layout.addWidget(self.list, 1)

        buttons = QtWidgets.QHBoxLayout()
        for text, slot, name in (
            ("Note", self._edit_note, "notebook-note"),
            ("Tag", self._edit_tags, "notebook-tag"),
            ("Up", lambda: self._move(-1), "notebook-up"),
            ("Down", lambda: self._move(1), "notebook-down"),
            ("Remove", self._remove, "notebook-remove"),
            ("Clear", self._clear, "notebook-clear"),
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setAccessibleName(name)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.setWidget(body)
        self._refresh()

    # ------------------------------------------------------------ public API
    def add_insight(self, insight: Insight) -> None:
        self.notebook.add(insight)
        self._refresh()
        self.list.setCurrentRow(len(self.notebook) - 1)
        if self.isHidden():
            self.show()

    def load_notebook(self, notebook: EvidenceNotebook) -> None:
        """Replace the contents (session restore)."""
        self.notebook = notebook
        self._refresh()

    # --------------------------------------------------------------- helpers
    def _current(self) -> int:
        return self.list.currentRow()

    @staticmethod
    def _row_text(entry) -> str:
        ins = entry.insight
        t = ins.primary_time()
        parts = [f"t = {t:.1f} s   " + ins.statement if t is not None else ins.statement]
        if entry.tags:
            parts.append("   [" + ", ".join(entry.tags) + "]")
        if entry.note:
            parts.append("\n    note: " + entry.note)
        return "".join(parts)

    def _refresh(self) -> None:
        row = self.list.currentRow()
        self.list.clear()
        for entry in self.notebook.entries:
            item = QtWidgets.QListWidgetItem(self._row_text(entry))
            item.setData(QtCore.Qt.UserRole, entry.insight)
            item.setToolTip(entry.insight.basis or entry.insight.statement)
            self.list.addItem(item)
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)
        self.caption.setVisible(self.notebook.is_empty())

    def _emit_activated(self, item) -> None:
        ins = item.data(QtCore.Qt.UserRole)
        if ins is not None:
            self.insight_activated.emit(ins)

    def _edit_note(self) -> None:
        i = self._current()
        if i < 0:
            return
        entry = self.notebook.entries[i]
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Annotate", "Note:", text=entry.note)
        if ok:
            self.notebook.set_note(i, text)
            self._refresh()

    def _edit_tags(self) -> None:
        i = self._current()
        if i < 0:
            return
        entry = self.notebook.entries[i]
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Tags", "Comma-separated tags:", text=", ".join(entry.tags))
        if ok:
            self.notebook.set_tags(i, text.split(","))
            self._refresh()

    def _move(self, delta: int) -> None:
        i = self._current()
        if i < 0:
            return
        j = self.notebook.move(i, delta)
        self._refresh()
        self.list.setCurrentRow(j)

    def _remove(self) -> None:
        i = self._current()
        if i >= 0:
            self.notebook.remove(i)
            self._refresh()

    def _clear(self) -> None:
        if self.notebook.is_empty():
            return
        if QtWidgets.QMessageBox.question(
                self, "Clear Notebook",
                "Remove all saved evidence?") == QtWidgets.QMessageBox.Yes:
            self.notebook.clear()
            self._refresh()
