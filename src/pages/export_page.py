"""Export page (FireLab roadmap Phase 4): re-hosts the M1.5 animation
exporter's entry point, plus a one-click "demo postcard" -- the active
cell's current frame, saved as a PNG with a simple FireLab title-card
overlay.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtWidgets

from pages.base import Page


class ExportPage(Page):
    title = "Export"

    def __init__(self, on_export_animation: Optional[Callable[[], None]] = None,
                 on_export_postcard: Optional[Callable[[], None]] = None, parent=None):
        super().__init__(parent)
        self._on_export_animation = on_export_animation
        self._on_export_postcard = on_export_postcard

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Export")
        title.setProperty("role", "title")
        layout.addWidget(title)

        animation_button = QtWidgets.QPushButton("Export animation (MP4/GIF)…")
        animation_button.setObjectName("primaryButton")
        animation_button.clicked.connect(self._export_animation)
        layout.addWidget(animation_button)

        postcard_button = QtWidgets.QPushButton("Save demo postcard (PNG)…")
        postcard_button.clicked.connect(self._export_postcard)
        layout.addWidget(postcard_button)

        layout.addStretch(1)

    def _export_animation(self) -> None:
        if self._on_export_animation is not None:
            self._on_export_animation()

    def _export_postcard(self) -> None:
        if self._on_export_postcard is not None:
            self._on_export_postcard()
