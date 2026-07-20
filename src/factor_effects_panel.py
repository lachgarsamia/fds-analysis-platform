"""Factor-Effect Field Maps panel (V2 roadmap M3.1, feature F2 flagship).

An Analysis-page tab: pick a factor and quantity, see the playable
main-effect diverging field (RdBu_r, symmetric scale) with a frame
slider, plus an ANOVA-style table ranking every factor by its
space-time-integrated effect magnitude on the selected quantity. An
optional second factor switches to the 2-factor interaction field.

Static/playback-independent (its own frame slider, not TimeController),
same convention as analytics_panel.py/energy_panel.py. Heavy first
compute (streams all scenarios once per quantity) is deferred to the
first time the tab is actually shown (showEvent) so opening the Analysis
page stays fast; results are cached per quantity, so switching factors is
instant.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from config import QUANTITY_DISPLAY
from slice_key import SOOT_QUANTITY
import factor_effects as fx


class FactorEffectsPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, quantity_options: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        # Factor effects run over the .sf quantities only (M3.1 scope);
        # running them over the 24 .s3d SOOT planes would decode the whole
        # volumetric dataset.
        self._quantity_options = [(label, key) for label, key in quantity_options
                                  if key.quantity != SOOT_QUANTITY]
        self._fps = max(1, fps)
        self._loaded = False
        # quantity -> {factor -> effect field series}; cached so factor
        # switches don't reload. One field is ~9.5 MB here.
        self._fields: dict = {}
        self._current_field = None

        # Only factors with at least two levels present are meaningful.
        self._factors = [f for f in fx.FACTORS if len(fx.factor_levels(self._manifest, f)) >= 2]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Factor effects")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)

        self.factor_combo = QtWidgets.QComboBox()
        self.factor_combo.setAccessibleName("Factor")
        self.factor_combo.setToolTip("Which design factor's main effect to show")
        for f in self._factors:
            self.factor_combo.addItem(fx.FACTOR_LABELS[f], f)
        header.addWidget(self.factor_combo)

        self.interaction_combo = QtWidgets.QComboBox()
        self.interaction_combo.setAccessibleName("Interaction factor")
        self.interaction_combo.setToolTip("Optionally cross with a second factor to show their interaction")
        self.interaction_combo.addItem("(main effect)", None)
        for f in self._factors:
            self.interaction_combo.addItem("× " + fx.FACTOR_LABELS[f], f)
        header.addWidget(self.interaction_combo)

        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Factor-effect quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "The map shows where the high factor level runs hotter (red) or cooler (blue) "
            "than the low level, averaged over every other factor.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Factor-effect field map")
        body.addWidget(self.canvas)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Factor", "Mean |effect|", "Peak |effect|"])
        self.table.setToolTip("Every factor ranked by its space-time-integrated effect on this quantity")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        body.addWidget(self.table)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        layout.addWidget(body, 1)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Effect-field frame")
        self.frame_slider.setToolTip("Scrub through the effect field over time")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        self._ax = None
        self._image = None

        self.factor_combo.currentIndexChanged.connect(self._refresh_field)
        self.interaction_combo.currentIndexChanged.connect(self._refresh_field)
        self.quantity_combo.currentIndexChanged.connect(self._on_quantity_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    @property
    def _current_key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    def _display(self):
        return QUANTITY_DISPLAY[self._current_key.quantity]

    def ensure_loaded(self) -> None:
        """First-use compute for the current quantity: streams all
        scenarios once, builds and caches every factor's main-effect
        field, fills the ANOVA-style table, and draws the first field.
        Idempotent per quantity."""
        if self._loaded or not self._factors or not self._quantity_options:
            return
        self._loaded = True
        self._compute_quantity_fields()
        self._refresh_field()

    def _compute_quantity_fields(self) -> None:
        key = self._current_key
        if key.quantity in self._fields:
            return
        fields = {}
        for f in self._factors:
            field = fx.main_effect_series(self._store, self._manifest, f, key)
            if field is not None:
                fields[f] = field
        self._fields[key.quantity] = fields
        self._fill_table(fields)

    def _fill_table(self, fields: dict) -> None:
        unit = self._display()['unit']
        rows = sorted(fields.items(), key=lambda kv: fx.effect_magnitude(kv[1]), reverse=True)
        self.table.setRowCount(len(rows))
        for r, (factor, field) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(fx.FACTOR_LABELS[factor]))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{fx.effect_magnitude(field):.2f} {unit}"))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{fx.effect_peak(field):.1f} {unit}"))

    def _on_quantity_changed(self, _idx: int) -> None:
        self._compute_quantity_fields()
        self._refresh_field()

    def _selected_field(self) -> np.ndarray:
        """The field to show: a 2-factor interaction if a second factor is
        picked, else the selected factor's cached main effect."""
        key = self._current_key
        factor = self.factor_combo.currentData()
        other = self.interaction_combo.currentData()
        if other is not None and other != factor:
            return fx.interaction_series(self._store, self._manifest, factor, other, key)
        return self._fields.get(key.quantity, {}).get(factor)

    def _refresh_field(self) -> None:
        if not self._loaded:
            return
        field = self._selected_field()
        self._current_field = field
        fig = self.canvas.fig
        fig.clear()
        self._ax = fig.add_subplot(111)
        if field is None:
            self._ax.text(0.5, 0.5, "Not enough factor levels for this combination",
                           ha="center", va="center", fontsize=9)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._image = None
            self.canvas.draw_idle()
            return
        vmax = fx.symmetric_vmax(field)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, field.shape[0] - 1)
        self.frame_slider.blockSignals(False)
        idx = min(self.frame_slider.value(), field.shape[0] - 1)
        self._image = self._ax.imshow(field[idx], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        other = self.interaction_combo.currentData()
        factor = self.factor_combo.currentData()
        which = (f"{fx.FACTOR_LABELS[factor]} × {fx.FACTOR_LABELS[other]} interaction"
                 if other is not None and other != factor
                 else f"{fx.FACTOR_LABELS[factor]} main effect")
        self._ax.set_title(which, fontsize=9, fontweight="bold")
        cbar = fig.colorbar(self._image, ax=self._ax, fraction=0.046, pad=0.02)
        cbar.set_label(f"Δ{self._display()['label']} ({self._display()['unit']})", fontsize=8)
        fig.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.95)
        self.canvas.draw_idle()
        self._update_frame_label(idx)

    def _on_frame_changed(self, value: int) -> None:
        self._update_frame_label(value)
        if self._image is not None and self._current_field is not None:
            idx = min(value, self._current_field.shape[0] - 1)
            self._image.set_data(self._current_field[idx])
            self.canvas.draw_idle()

    def _update_frame_label(self, index: int) -> None:
        self.frame_label.setText(f"t = {index / self._fps:.1f} s")
