"""Sessions management panel (V4-M6), an Analysis-page tab.

The library view over named analysis sessions: save the current
investigation under a name + intent, browse/search saved sessions with a
preview, and load, delete, or export any one to a report. The panel is
pure UI over session_store -- it emits intent signals and main_window does
the state collection/application (it owns the grid, notebook, zones, and
time-window), so this stays decoupled from the app internals.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


class SessionsPanel(QtWidgets.QWidget):
    save_requested = QtCore.pyqtSignal(str, str)   # name, intent
    load_requested = QtCore.pyqtSignal(str)        # path
    delete_requested = QtCore.pyqtSignal(str)      # path
    export_requested = QtCore.pyqtSignal(str)      # path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._infos = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Analysis sessions")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.save_button = QtWidgets.QPushButton("Save current as session…")
        self.save_button.setAccessibleName("Save analysis session")
        self.save_button.clicked.connect(self._on_save)
        header.addWidget(self.save_button)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "A session is a complete, reproducible snapshot: the grid and view, "
            "the Evidence Notebook, named zones, the time window, and browser "
            "filters. Load one to restore the whole investigation exactly.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search sessions by name or intent")
        self.search.setAccessibleName("Search sessions")
        self.search.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search)

        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Session list")
        self.list.setAlternatingRowColors(True)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemDoubleClicked.connect(lambda _i: self._on_load())
        layout.addWidget(self.list, 1)

        self.preview = QtWidgets.QLabel("Select a session to preview it.")
        self.preview.setWordWrap(True)
        self.preview.setProperty("role", "caption")
        layout.addWidget(self.preview)

        buttons = QtWidgets.QHBoxLayout()
        for text, slot, name in (("Load", self._on_load, "session-load"),
                                 ("Export report", self._on_export, "session-export"),
                                 ("Delete", self._on_delete, "session-delete")):
            btn = QtWidgets.QPushButton(text)
            btn.setAccessibleName(name)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    # -------------------------------------------------------------- populate
    def set_sessions(self, infos: list) -> None:
        """Replace the list with SessionInfo objects (session_store)."""
        self._infos = list(infos)
        self._apply_filter()

    def _apply_filter(self) -> None:
        term = self.search.text().strip().lower()
        self.list.clear()
        for info in self._infos:
            if term and term not in info.name.lower() and term not in (info.intent or "").lower():
                continue
            label = ("★ " if info.is_draft else "") + info.name
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, info)
            item.setToolTip(info.preview())
            self.list.addItem(item)
        self.preview.setText("Select a session to preview it." if self.list.count()
                             else "No saved sessions yet.")

    def _current_info(self):
        item = self.list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _on_row_changed(self, _row: int) -> None:
        info = self._current_info()
        if info is not None:
            self.preview.setText(f"<b>{info.name}</b><br>{info.preview()}")

    # ---------------------------------------------------------------- actions
    def _on_save(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save session", "Session name:")
        if not ok or not name.strip():
            return
        intent, ok = QtWidgets.QInputDialog.getText(
            self, "Save session", "Intent / description (optional):")
        if not ok:
            return
        self.save_requested.emit(name.strip(), intent.strip())

    def _on_load(self) -> None:
        info = self._current_info()
        if info is not None:
            self.load_requested.emit(info.path)

    def _on_export(self) -> None:
        info = self._current_info()
        if info is not None:
            self.export_requested.emit(info.path)

    def _on_delete(self) -> None:
        info = self._current_info()
        if info is None:
            return
        if QtWidgets.QMessageBox.question(
                self, "Delete session", f"Delete \"{info.name}\"?") == QtWidgets.QMessageBox.Yes:
            self.delete_requested.emit(info.path)
