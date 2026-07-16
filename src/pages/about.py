"""About page (UI/UX modernization Phase 5): static project/institute
info, replacing the Phase 1 PlaceholderPage stub. Pure presentation --
no live state, no signals in.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from branding import build_logo_widget
from pages.base import Page
from widgets import CollapsibleSection

_INFO_ROWS = (
    ("Purpose", "A live, cinematic view into a parametric fire simulation "
                "ensemble -- 24 FDS scenarios varying candle count, door "
                "width, and vent configuration."),
    ("Data source", "Fire Dynamics Simulator (FDS) SLCF slice output, parsed "
                     "and cached locally (temperature and velocity fields)."),
    ("Analysis", "PCA + clustering over per-scenario feature curves, and a "
                 "Fourier Neural Operator model for short-horizon rollout "
                 "prediction."),
    ("Built at", "Forschungszentrum Jülich"),
)


class AboutPage(Page):
    title = "About"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._built = False

    def on_enter(self) -> None:
        if self._built:
            return
        self._built = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(16)
        layout.addStretch(1)

        logo_row = QtWidgets.QHBoxLayout()
        logo_row.addStretch(1)
        logo_row.addWidget(build_logo_widget(72))
        logo_row.addStretch(1)
        layout.addLayout(logo_row)

        title = QtWidgets.QLabel("FireLab Digital Twin")
        title.setProperty("role", "display")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("FDS SLCF Visualizer -- research platform")
        subtitle.setProperty("role", "caption")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(subtitle)

        card_row = QtWidgets.QHBoxLayout()
        card_row.addStretch(1)
        info_card = CollapsibleSection("Project")
        info_card.setMaximumWidth(560)
        for label, value in _INFO_ROWS:
            row = QtWidgets.QLabel(f"<b>{label}:</b> {value}")
            row.setWordWrap(True)
            info_card.add_row(row)
        card_row.addWidget(info_card)
        card_row.addStretch(1)
        layout.addLayout(card_row)

        layout.addStretch(1)
