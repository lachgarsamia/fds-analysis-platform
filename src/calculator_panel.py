"""Field Calculator panel (V6-M1), an Analysis-page tab.

Create deterministic derived quantities from existing fields with a safe
expression (field_calculator). Type an expression and a name, see live
validation and a preview on a real scenario, save it, and it registers as a
first-class quantity the QuantityProvider computes. Created fields are listed,
deletable, and persisted in the session.

Minimal UI following the existing panel patterns; no analysis-page redesign.
Reuses the QuantityProvider (compute), field_calculator (validate/register),
and the registry.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, plot_fg_color
from slice_key import SliceKey
from registry import get_quantity
import field_calculator as fc
from analysis_panel_base import populate_scenario_combo


class CalculatorPanel(QtWidgets.QWidget):
    fields_changed = QtCore.pyqtSignal()   # a field was created/deleted (session dirty)

    def __init__(self, provider, manifest: list, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Field calculator")
        title.setProperty("role", "section-title")
        layout.addWidget(title)
        self.caption = QtWidgets.QLabel(
            "Define a quantity from existing fields, e.g. \"Temperature - 20\", "
            "\"gradient(Temperature)\", \"rate(Temperature)\". Functions: abs, sqrt, "
            "clip, log, exp, min, max, where, gradient, rate. Every field stores its "
            "expression as its basis — deterministic and reproducible.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Temperature Rise")
        self.expr_edit = QtWidgets.QLineEdit()
        self.expr_edit.setPlaceholderText("e.g. Temperature - 20")
        self.expr_edit.textChanged.connect(self._on_expr_changed)
        self.unit_edit = QtWidgets.QLineEdit()
        self.unit_edit.setPlaceholderText("optional — inferred if blank")
        form.addRow("Name", self.name_edit)
        form.addRow("Expression", self.expr_edit)
        form.addRow("Unit", self.unit_edit)
        layout.addLayout(form)

        controls = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "caption")
        controls.addWidget(self.status, 1)
        controls.addWidget(QtWidgets.QLabel("Preview on:"))
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Calculator preview scenario")
        populate_scenario_combo(self.scenario_combo, self._manifest)
        controls.addWidget(self.scenario_combo)
        self.save_button = QtWidgets.QPushButton("Save field")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        controls.addWidget(self.save_button)
        layout.addLayout(controls)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Calculated field preview")
        self.canvas.setMinimumHeight(160)
        layout.addWidget(self.canvas, 1)

        list_row = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setAccessibleName("Calculated fields")
        self.list.setMaximumHeight(120)
        list_row.addWidget(self.list, 1)
        btns = QtWidgets.QVBoxLayout()
        self.export_button = QtWidgets.QPushButton("Export CSV")
        self.export_button.clicked.connect(self._export)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete)
        btns.addWidget(self.export_button)
        btns.addWidget(self.delete_button)
        btns.addStretch(1)
        list_row.addLayout(btns)
        layout.addLayout(list_row)

        self.scenario_combo.currentIndexChanged.connect(lambda _i: self._preview())
        self._refresh_list()

    # ---------------------------------------------------------- validation
    def _on_expr_changed(self, _text) -> None:
        expr = self.expr_edit.text().strip()
        if not expr:
            self.status.setText("")
            self.save_button.setEnabled(False)
            return
        try:
            fc.validate(expr)
        except fc.CalculatorError as e:
            self.status.setText(f"✗ {e}")
            self.save_button.setEnabled(False)
            return
        deps = ", ".join(fc.dependencies(expr))
        self.status.setText(f"✓ valid · depends on {deps} · unit {fc.infer_unit(expr)}")
        self.save_button.setEnabled(True)
        self._preview()

    def _preview(self) -> None:
        expr = self.expr_edit.text().strip()
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        case = self.scenario_combo.currentData()
        try:
            fc.validate(expr)
            data = fc.evaluate(
                expr,
                lambda key: np.asarray(self._provider.get(case, SliceKey(key))),
                self._provider._fps)
            frame = data[int(data.shape[0] * 0.6)]
            im = ax.imshow(frame, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"preview · min {data.min():.1f} / mean {data.mean():.1f} / "
                         f"max {data.max():.1f}", fontsize=8)
        except Exception as e:  # noqa: BLE001 - preview must never crash the panel
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8,
                    wrap=True, transform=ax.transAxes, color=plot_fg_color())
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    # --------------------------------------------------------------- actions
    def _save(self) -> None:
        try:
            field = fc.make_field(self.name_edit.text(), self.expr_edit.text(),
                                  self.unit_edit.text())
        except fc.CalculatorError as e:
            self.status.setText(f"✗ {e}")
            return
        fc.register(field)
        self.name_edit.clear(); self.expr_edit.clear(); self.unit_edit.clear()
        self._refresh_list()
        self.fields_changed.emit()

    def _refresh_list(self) -> None:
        self.list.clear()
        for f in fc.all_fields():
            item = QtWidgets.QListWidgetItem(f"{f.name} = {f.expression}  ({f.unit})")
            item.setData(QtCore.Qt.UserRole, f.name)
            item.setToolTip(f.basis)
            self.list.addItem(item)

    def _current_field_name(self):
        item = self.list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _delete(self) -> None:
        name = self._current_field_name()
        if name is not None:
            fc.unregister(name)
            self._refresh_list()
            self.fields_changed.emit()

    def _export(self) -> None:
        name = self._current_field_name()
        if name is None:
            return
        case = self.scenario_combo.currentData()
        try:
            data = np.asarray(self._provider.get(case, SliceKey(name)))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export", f"Could not compute: {e}")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export calculated field (per-frame mean)",
            f"{name}.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        from timeseries import write_series_csv
        times = np.arange(data.shape[0]) / self._provider._fps
        unit = get_quantity(name).unit
        write_series_csv(path, "time_s", times,
                         [(f"{name}_mean ({unit})", data.mean(axis=(1, 2))),
                          (f"{name}_max ({unit})", data.max(axis=(1, 2)))])
