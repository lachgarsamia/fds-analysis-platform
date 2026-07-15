"""Home page (FireLab roadmap Phase 4): a landing page with a "Start the
fire" CTA into Live Viewer and three real stat tiles. The full-bleed
looping hero video and idle attract-mode return are explicitly Phase 5
scope per ROADMAP-FIRELAB.md -- not built here.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets

from pages.base import Page


class HomePage(Page):
    title = "Home"

    def __init__(self, on_start: Optional[Callable[[], None]] = None, parent=None):
        super().__init__(parent)
        self._on_start = on_start

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(16)
        layout.addStretch(1)

        title = QtWidgets.QLabel("FireLab Digital Twin")
        title.setProperty("role", "display")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        tagline = QtWidgets.QLabel("A live, cinematic view into a parametric fire simulation ensemble.")
        tagline.setAlignment(QtCore.Qt.AlignCenter)
        tagline.setWordWrap(True)
        layout.addWidget(tagline)

        self.start_button = QtWidgets.QPushButton("Start the fire →")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setFixedWidth(220)
        self.start_button.clicked.connect(self._on_start_clicked)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.start_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.stats_row = QtWidgets.QHBoxLayout()
        self.stats_row.setSpacing(24)
        layout.addLayout(self.stats_row)

        layout.addStretch(1)

    def set_stats(self, n_experiments: int, n_timesteps: int, n_quantities: int) -> None:
        while self.stats_row.count():
            item = self.stats_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for value, label in (
            (n_experiments, "experiments"),
            (n_timesteps, "time steps"),
            (n_quantities, "physics fields"),
        ):
            self.stats_row.addWidget(self._stat_tile(value, label))

    def _stat_tile(self, value: int, label: str) -> QtWidgets.QWidget:
        tile = QtWidgets.QWidget()
        tile_layout = QtWidgets.QVBoxLayout(tile)
        value_label = QtWidgets.QLabel(str(value))
        value_label.setProperty("role", "title")
        value_label.setAlignment(QtCore.Qt.AlignCenter)
        caption = QtWidgets.QLabel(label)
        caption.setAlignment(QtCore.Qt.AlignCenter)
        tile_layout.addWidget(value_label)
        tile_layout.addWidget(caption)
        return tile

    def _on_start_clicked(self) -> None:
        if self._on_start is not None:
            self._on_start()
