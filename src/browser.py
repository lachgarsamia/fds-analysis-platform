"""Experiment browser dock for scenario summaries (M2.5)."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


FACTOR_LABELS = {
    "candles": {0: "1 candle", 1: "2 candles"},
    "door": {0: "Narrow", 1: "Wide"},
    "vod": {0: "Vent 1 open", 1: "Vent 1 closed", 2: "Vent 1 HVAC"},
    "voc": {0: "Vent 2 open", 1: "Vent 2 closed"},
}


class SummaryTableModel(QtCore.QAbstractTableModel):
    COLUMNS = (
        ("folder", "Scenario"),
        ("candles", "Candles"),
        ("door", "Door"),
        ("vod", "Vent 1"),
        ("voc", "Vent 2"),
        ("max_temp_c", "Peak T (C)"),
        ("time_to_100c_s", "T>100C (s)"),
        ("time_to_300c_s", "T>300C (s)"),
        ("time_to_600c_s", "T>600C (s)"),
        ("mean_upper_temp_c", "Mean upper T (C)"),
        ("peak_hrr_kw", "Peak HRR (kW)"),
        ("total_energy_kj", "Energy (kJ)"),
    )

    def __init__(self, summaries: list, parent=None):
        super().__init__(parent)
        self._summaries = list(summaries)

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._summaries)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section: int, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.COLUMNS[section][1]
        return section + 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        summary = self._summaries[index.row()]
        key = self.COLUMNS[index.column()][0]
        value = getattr(summary, key)

        if role == QtCore.Qt.UserRole:
            return summary
        if role == QtCore.Qt.UserRole + 1:
            return value
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            return self._format_value(key, value)
        if role == QtCore.Qt.TextAlignmentRole and isinstance(value, (int, float)):
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    @staticmethod
    def _format_value(key: str, value):
        if value is None:
            return "n/a"
        if key in FACTOR_LABELS:
            return FACTOR_LABELS[key].get(value, str(value))
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)


class SummaryFilterProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._factor_filters = {}

    def set_text_filter(self, text: str):
        self._text = text.strip().lower()
        self.invalidateFilter()

    def set_factor_filter(self, factor: str, value):
        if value is None:
            self._factor_filters.pop(factor, None)
        else:
            self._factor_filters[factor] = value
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        model = self.sourceModel()
        idx = model.index(source_row, 0, source_parent)
        summary = model.data(idx, QtCore.Qt.UserRole)
        for factor, value in self._factor_filters.items():
            if getattr(summary, factor) != value:
                return False
        if not self._text:
            return True
        haystack = " ".join([
            summary.folder,
            *(SummaryTableModel._format_value(f, getattr(summary, f)) for f in FACTOR_LABELS),
        ]).lower()
        return self._text in haystack

    def lessThan(self, left, right) -> bool:
        model = self.sourceModel()
        lv = model.data(left, QtCore.Qt.UserRole + 1)
        rv = model.data(right, QtCore.Qt.UserRole + 1)
        if lv is None:
            return False
        if rv is None:
            return True
        return lv < rv


class ExperimentBrowserDock(QtWidgets.QDockWidget):
    scenario_activated = QtCore.pyqtSignal(int)
    open_grid_requested = QtCore.pyqtSignal(list)
    open_ensemble_requested = QtCore.pyqtSignal(list)

    def __init__(self, summaries: list, parent=None):
        super().__init__("Experiment Browser", parent)
        self.setObjectName("experimentBrowserDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        filter_row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Filter scenarios")
        self.search_edit.setAccessibleName("Filter experiment browser")
        filter_row.addWidget(self.search_edit, 1)
        layout.addLayout(filter_row)

        factor_row = QtWidgets.QHBoxLayout()
        self.factor_combos = {}
        for factor, labels in FACTOR_LABELS.items():
            combo = QtWidgets.QComboBox()
            combo.setAccessibleName(f"Filter by {factor}")
            combo.addItem("All", None)
            for value, label in labels.items():
                combo.addItem(label, value)
            combo.currentIndexChanged.connect(
                lambda _idx, f=factor, c=combo: self.proxy.set_factor_filter(f, c.currentData())
            )
            self.factor_combos[factor] = combo
            factor_row.addWidget(combo)
        layout.addLayout(factor_row)

        self.model = SummaryTableModel(summaries, self)
        self.proxy = SummaryFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.table = QtWidgets.QTableView()
        self.table.setAccessibleName("Experiment summary table")
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self.table, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.open_grid_button = QtWidgets.QPushButton("Open as grid")
        self.open_grid_button.setToolTip("Open the selected scenarios in the visible grid cells")
        self.open_grid_button.clicked.connect(lambda: self.open_grid_requested.emit(self.selected_case_indices()))
        self.open_ensemble_button = QtWidgets.QPushButton("Open as ensemble")
        self.open_ensemble_button.setToolTip("Open the selected scenarios as an ensemble statistic")
        self.open_ensemble_button.clicked.connect(lambda: self.open_ensemble_requested.emit(self.selected_case_indices()))
        button_row.addWidget(self.open_grid_button)
        button_row.addWidget(self.open_ensemble_button)
        layout.addLayout(button_row)

        self.search_edit.textChanged.connect(self.proxy.set_text_filter)
        self.setWidget(root)

    def selected_case_indices(self) -> list[int]:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        cases = []
        for proxy_row in rows:
            source_index = self.proxy.mapToSource(self.proxy.index(proxy_row, 0))
            summary = self.model.data(source_index, QtCore.Qt.UserRole)
            cases.append(summary.case_index)
        return cases

    def _on_double_clicked(self, proxy_index):
        source_index = self.proxy.mapToSource(proxy_index)
        summary = self.model.data(source_index, QtCore.Qt.UserRole)
        self.scenario_activated.emit(summary.case_index)
