"""Experiment Management panel (V4-M9), an Analysis-page tab.

A library over Experiments -- named, tagged batches of related scenarios
with a designated baseline and shared parameters. Create an experiment,
check to include scenarios, pick a baseline, check availability, and hand
a baseline-vs-scenario pair to the Advanced Comparison workflow (V4-M8).

Self-contained CRUD (save/list/load/delete/export via experiment.py); the
only thing it delegates is the comparison hand-off (main_window owns the
Compare-axes panel and page navigation), emitted as compare_requested.

Honesty: scenarios are pre-computed cluster runs, so "Check availability"
validates and loads existing data (ready/missing), never launches a solver.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

import experiment as ex
from report_builder import build_experiment_report, write_report


class ExperimentsPanel(QtWidgets.QWidget):
    compare_requested = QtCore.pyqtSignal(str, str)   # baseline folder, other folder

    def __init__(self, store, manifest: list, experiments_dir: str, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._folders = [e.folder for e in self._manifest]
        self._dir = experiments_dir
        self._current = ex.Experiment(name="")
        self._status = None
        self._current_path = None

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # --- library (left) ------------------------------------------------
        left = QtWidgets.QVBoxLayout()
        lib_header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Experiments")
        title.setProperty("role", "section-title")
        lib_header.addWidget(title)
        lib_header.addStretch(1)
        self.new_button = QtWidgets.QPushButton("New")
        self.new_button.setAccessibleName("New experiment")
        self.new_button.clicked.connect(self._new)
        lib_header.addWidget(self.new_button)
        left.addLayout(lib_header)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search experiments")
        self.search.textChanged.connect(self._refresh_library)
        left.addWidget(self.search)
        self.library = QtWidgets.QListWidget()
        self.library.setAccessibleName("Experiment library")
        self.library.itemClicked.connect(self._on_library_click)
        left.addWidget(self.library, 1)
        lib_btns = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.clicked.connect(self._load_selected)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected)
        lib_btns.addWidget(self.load_button); lib_btns.addWidget(self.delete_button)
        left.addLayout(lib_btns)
        root.addLayout(left, 2)

        # --- editor (right) ------------------------------------------------
        right = QtWidgets.QVBoxLayout()
        self.name_label = QtWidgets.QLabel("New experiment")
        self.name_label.setProperty("role", "section-title")
        right.addWidget(self.name_label)
        form = QtWidgets.QFormLayout()
        self.desc_edit = QtWidgets.QLineEdit()
        self.desc_edit.setPlaceholderText("Description / purpose")
        self.tags_edit = QtWidgets.QLineEdit()
        self.tags_edit.setPlaceholderText("comma-separated tags")
        form.addRow("Description", self.desc_edit)
        form.addRow("Tags", self.tags_edit)
        right.addLayout(form)

        right.addWidget(QtWidgets.QLabel("Scenarios (check to include):"))
        self.scenario_list = QtWidgets.QListWidget()
        self.scenario_list.setAccessibleName("Experiment scenarios")
        for folder in self._folders:
            item = QtWidgets.QListWidgetItem(folder)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.scenario_list.addItem(item)
        self.scenario_list.itemChanged.connect(self._on_scenario_toggled)
        right.addWidget(self.scenario_list, 1)

        baseline_row = QtWidgets.QHBoxLayout()
        baseline_row.addWidget(QtWidgets.QLabel("Baseline:"))
        self.baseline_combo = QtWidgets.QComboBox()
        self.baseline_combo.setAccessibleName("Experiment baseline")
        baseline_row.addWidget(self.baseline_combo, 1)
        right.addLayout(baseline_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setProperty("role", "caption")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        for text, slot, name in (("Save", self._save, "experiment-save"),
                                 ("Check availability", self._check, "experiment-check"),
                                 ("Compare baseline vs selected", self._compare, "experiment-compare"),
                                 ("Export summary", self._export, "experiment-export")):
            b = QtWidgets.QPushButton(text)
            b.setAccessibleName(name)
            b.clicked.connect(slot)
            actions.addWidget(b)
        right.addLayout(actions)
        root.addLayout(right, 3)

        self._refresh_library()
        self._sync_editor_from_model()

    # ---------------------------------------------------------------- library
    def refresh(self) -> None:
        self._refresh_library()

    def _refresh_library(self) -> None:
        term = self.search.text().strip().lower()
        self.library.clear()
        for info in ex.list_experiments(self._dir):
            if term and term not in info.name.lower() and term not in info.description.lower():
                continue
            item = QtWidgets.QListWidgetItem(info.name)
            item.setData(QtCore.Qt.UserRole, info)
            item.setToolTip(info.preview())
            self.library.addItem(item)

    def _selected_info(self):
        item = self.library.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _on_library_click(self, item) -> None:
        info = item.data(QtCore.Qt.UserRole)
        if info is not None:
            self.status_label.setText(info.preview())

    # ----------------------------------------------------------------- editor
    def _new(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New experiment", "Name:")
        if not ok or not name.strip():
            return
        self._current = ex.Experiment(name=name.strip())
        self._current_path = None
        self._status = None
        self._sync_editor_from_model()

    def _sync_editor_from_model(self) -> None:
        e = self._current
        self.name_label.setText(e.name or "New experiment")
        self.desc_edit.setText(e.description)
        self.tags_edit.setText(", ".join(e.tags))
        self.scenario_list.blockSignals(True)
        for i in range(self.scenario_list.count()):
            item = self.scenario_list.item(i)
            item.setCheckState(QtCore.Qt.Checked if item.text() in e.scenarios
                               else QtCore.Qt.Unchecked)
        self.scenario_list.blockSignals(False)
        self._refresh_baseline_combo()
        self._update_status_label()

    def _checked_folders(self) -> list:
        return [self.scenario_list.item(i).text()
                for i in range(self.scenario_list.count())
                if self.scenario_list.item(i).checkState() == QtCore.Qt.Checked]

    def _on_scenario_toggled(self, _item) -> None:
        self._current.scenarios = self._checked_folders()
        self._refresh_baseline_combo()
        self._status = None
        self._update_status_label()

    def _refresh_baseline_combo(self) -> None:
        current = self._current.baseline
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.clear()
        self.baseline_combo.addItem("(none)", "")
        for folder in self._current.scenarios:
            self.baseline_combo.addItem(folder, folder)
        idx = self.baseline_combo.findData(current)
        self.baseline_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.baseline_combo.blockSignals(False)

    def _collect(self) -> ex.Experiment:
        self._current.description = self.desc_edit.text().strip()
        self._current.tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        self._current.scenarios = self._checked_folders()
        self._current.baseline = self.baseline_combo.currentData() or ""
        return self._current

    # ---------------------------------------------------------------- actions
    def _save(self) -> None:
        exp = self._collect()
        if not exp.name:
            name, ok = QtWidgets.QInputDialog.getText(self, "Save experiment", "Name:")
            if not ok or not name.strip():
                return
            exp.name = name.strip()
        try:
            self._current_path = ex.save_experiment(self._dir, exp, path=self._current_path)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Save experiment", f"Could not save: {e}")
            return
        self.name_label.setText(exp.name)
        self._refresh_library()

    def _load_selected(self) -> None:
        info = self._selected_info()
        if info is None:
            return
        self._current = ex.load_experiment(info.path)
        self._current_path = info.path
        self._status = None
        self._sync_editor_from_model()

    def _delete_selected(self) -> None:
        info = self._selected_info()
        if info is None:
            return
        if QtWidgets.QMessageBox.question(
                self, "Delete experiment", f"Delete \"{info.name}\"?") == QtWidgets.QMessageBox.Yes:
            ex.delete_experiment(info.path)
            if info.path == self._current_path:
                self._current_path = None
            self._refresh_library()

    def _available_folders(self) -> set:
        """Folders whose data the store can actually load (real check)."""
        from slice_key import SliceKey
        ready = set()
        by_folder = {e.folder: e.case_index for e in self._manifest}
        for folder in self._collect().scenarios:
            ci = by_folder.get(folder)
            if ci is None:
                continue
            try:
                self._store.get(ci, SliceKey("TEMPERATURE"))
                ready.add(folder)
            except Exception:
                pass
        return ready

    def _check(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self._status = ex.experiment_status(self._collect(), self._available_folders())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._update_status_label()

    def _update_status_label(self) -> None:
        exp = self._current
        if self._status is None:
            self.status_label.setText(
                f"{len(exp.scenarios)} scenario(s). Click \"Check availability\" for status.")
            return
        s = self._status
        missing = [k for k, v in s["statuses"].items() if v == "missing"]
        text = (f"{s['ready']}/{s['total']} ready ({s['completion'] * 100:.0f}%).")
        if missing:
            text += "  Missing: " + ", ".join(missing)
        self.status_label.setText(text)

    def _compare(self) -> None:
        exp = self._collect()
        baseline = exp.baseline
        item = self.scenario_list.currentItem()
        other = item.text() if item is not None else None
        if not baseline or other is None or other == baseline:
            QtWidgets.QMessageBox.information(
                self, "Compare",
                "Pick a baseline and select a different scenario in the list, "
                "then compare.")
            return
        self.compare_requested.emit(baseline, other)

    def _export(self) -> None:
        exp = self._collect()
        if not exp.scenarios:
            QtWidgets.QMessageBox.information(self, "Export", "Add scenarios first.")
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export experiment summary", "experiment_summary.html", "HTML (*.html)")
        if not out:
            return
        if not out.lower().endswith(".html"):
            out += ".html"
        try:
            write_report(out, build_experiment_report(exp.to_dict(), self._status))
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Export", f"Could not write: {e}")
