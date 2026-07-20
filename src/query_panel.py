"""Physics Query panel (V3-M4), an Analysis-page tab.

A query bar over the scenario data: type a physical question (or pick an
example), and the engine (query_engine.py) answers it deterministically,
showing the answer as a navigable Insight and marking it on the field at
the moment it happens. A question that matches no pattern in the closed
grammar returns "not understood" rather than a made-up answer.

Static/lazy, same Analysis-panel convention.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity
from slice_key import SliceKey
from insight import InsightList
import query_engine as qe


class QueryPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._ax = None
        self._image = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Ask")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Query scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        bar = QtWidgets.QHBoxLayout()
        self.examples = QtWidgets.QComboBox()
        self.examples.setAccessibleName("Example queries")
        self.examples.addItem("Examples…", None)
        for ex in qe.EXAMPLE_QUERIES:
            self.examples.addItem(ex, ex)
        bar.addWidget(self.examples)
        self.query_edit = QtWidgets.QLineEdit()
        self.query_edit.setPlaceholderText("e.g. first time temperature exceeds 300 near the candle")
        self.query_edit.setAccessibleName("Physics query")
        bar.addWidget(self.query_edit, 1)
        self.run_button = QtWidgets.QPushButton("Ask")
        self.run_button.setObjectName("primaryButton")
        bar.addWidget(self.run_button)
        layout.addLayout(bar)

        self.answer_label = QtWidgets.QLabel(
            "Ask a physical question. Answers are computed from the data, never generated.")
        self.answer_label.setWordWrap(True)
        self.answer_label.setProperty("role", "caption")
        layout.addWidget(self.answer_label)

        body = QtWidgets.QSplitter()
        self.results = InsightList()
        body.addWidget(self.results)
        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Query answer map")
        body.addWidget(self.canvas)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        self.run_button.clicked.connect(self._run)
        self.query_edit.returnPressed.connect(self._run)
        self.examples.currentIndexChanged.connect(self._on_example)
        self.results.insight_activated.connect(self._show_answer)

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

    def _on_example(self, idx: int) -> None:
        ex = self.examples.currentData()
        if ex:
            self.query_edit.setText(ex)
            self._run()

    def _run(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        text = self.query_edit.text()
        query = qe.parse(text)
        if query is None:
            self.answer_label.setText(
                "Not understood. Try an example, or ask about a temperature/velocity "
                "threshold, the hottest region, the plume height, or ventilation.")
            self.results.set_insights([])
            return
        key = SliceKey(query.quantity)
        data = self._store.get(case_index, key)
        extent = self._store.get_extent(case_index, key)
        answers = qe.execute(query, data, extent, self._fps)
        self.results.set_insights(answers)
        self.answer_label.setText(answers[0].statement if answers else "No answer.")
        if answers:
            self._show_answer(answers[0])

    def _show_answer(self, insight) -> None:
        case_index = self.scenario_combo.currentData()
        key = SliceKey(insight.quantity or "TEMPERATURE")
        data = np.asarray(self._store.get(case_index, key))
        extent = self._store.get_extent(case_index, key)
        display = get_quantity(key.quantity)
        idx = insight.frame_index(self._fps)
        idx = min(idx if idx is not None else data.shape[0] - 1, data.shape[0] - 1)

        fig = self.canvas.fig
        fig.clear()
        self._ax = fig.add_subplot(111)
        self._image = self._ax.imshow(data[idx], cmap=display.cmap, vmin=display.vmin,
                                       vmax=display.slider_default, aspect="auto",
                                       extent=extent if extent else None)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax.set_title(f"{display.label} at t = {idx / self._fps:.1f} s", fontsize=9)
        if insight.location is not None and extent is not None:
            self._ax.plot(insight.location[0], insight.location[1], "o", markersize=13,
                          markerfacecolor="none", markeredgecolor="#00E5FF", markeredgewidth=2)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.98)
        self.canvas.draw_idle()
