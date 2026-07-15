"""First-run guided tour overlay (FireLab roadmap Phase 5): 4 short coach
marks shown once on the Live page, dismissible, remembered via QSettings
so a returning user never sees it again.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

STEPS = [
    ("Welcome to FireLab", "This is a live, cinematic view of a real fire simulation ensemble."),
    ("Control the room", "Use the panel on the left to change candles, doors, and vents."),
    ("Watch the story", "The Inspector on the right narrates what's happening as it happens."),
    ("Explore more", "Use the rail on the left to compare scenarios, browse the dataset, and more."),
]

SETTINGS_KEY = "tour/completed"


def should_show_tour(settings: QtCore.QSettings) -> bool:
    return not settings.value(SETTINGS_KEY, False, type=bool)


def mark_tour_completed(settings: QtCore.QSettings) -> None:
    settings.setValue(SETTINGS_KEY, True)


class TourOverlay(QtWidgets.QWidget):
    """A dark scrim + centered card, sized to cover `parent` at
    construction. Not resize-tracking (a one-time first-run overlay
    doesn't need to survive a window resize gracefully)."""

    finished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self._step = 0
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet("TourOverlay { background-color: rgba(0, 0, 0, 160); }")
        self.setGeometry(parent.rect())

        card = QtWidgets.QWidget(self)
        card.setObjectName("tourCard")
        card.setStyleSheet(
            "#tourCard { background-color: #14171B; border-radius: 14px; }"
            " #tourCard QLabel { color: #EDEEF2; }"
        )
        card.setFixedWidth(360)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        self._title_label = QtWidgets.QLabel()
        self._title_label.setProperty("role", "title")
        self._body_label = QtWidgets.QLabel()
        self._body_label.setWordWrap(True)
        card_layout.addWidget(self._title_label)
        card_layout.addWidget(self._body_label)

        button_row = QtWidgets.QHBoxLayout()
        self._skip_button = QtWidgets.QPushButton("Skip")
        self._skip_button.clicked.connect(self._finish)
        self._next_button = QtWidgets.QPushButton("Next")
        self._next_button.setObjectName("primaryButton")
        self._next_button.clicked.connect(self._advance)
        button_row.addWidget(self._skip_button)
        button_row.addStretch(1)
        button_row.addWidget(self._next_button)
        card_layout.addLayout(button_row)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addStretch(1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)

        self._render_step()
        self.show()
        self.raise_()

    def _render_step(self) -> None:
        title, body = STEPS[self._step]
        self._title_label.setText(title)
        self._body_label.setText(body)
        self._next_button.setText("Done" if self._step == len(STEPS) - 1 else "Next")

    def _advance(self) -> None:
        if self._step >= len(STEPS) - 1:
            self._finish()
            return
        self._step += 1
        self._render_step()

    def _finish(self) -> None:
        self.finished.emit()
        self.hide()
        self.deleteLater()
