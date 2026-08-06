"""First-run guided tour overlay (FireLab roadmap Phase 5): short coach
marks shown once on a page, dismissible, remembered via QSettings so a
returning user never sees it again.

Generalized (Analysis roadmap A4) to carry more than one tour: STEPS/
SETTINGS_KEY below are still the Live page's own (unchanged, still the
default for should_show_tour()/mark_tour_completed()/TourOverlay so the
Live page's existing calls need no changes), and ANALYSIS_STEPS/
ANALYSIS_SETTINGS_KEY are the Analysis page's own -- a distinct QSettings
key so dismissing one tour never marks the other seen. Same functions,
same overlay widget, no new mechanism.
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

# Analysis page's own coach-mark (roadmap A4): 2 steps, pointing at the
# group tabs (the page's main organizing structure -- 6 groups by research
# question) and the collapsed Experimental group specifically, since that
# one's a different interaction pattern (a toggle button, not a tab) from
# everything else on the page and easy to miss entirely.
ANALYSIS_STEPS = [
    ("Six ways to investigate", "Tabs are grouped by the question they answer -- what's "
     "happening, how scenarios compare, what drives the response, how a quantity evolves "
     "over time and space, and reference tools."),
    ("More tools, tucked away", "\"Experimental\" holds exploratory tools -- heuristic "
     "saliency, causal tracing, forecasting -- collapsed by default so they don't compete "
     "with the core workflow. Click it to expand."),
]
ANALYSIS_SETTINGS_KEY = "tour/analysis_completed"


def should_show_tour(settings: QtCore.QSettings, key: str = SETTINGS_KEY) -> bool:
    return not settings.value(key, False, type=bool)


def mark_tour_completed(settings: QtCore.QSettings, key: str = SETTINGS_KEY) -> None:
    settings.setValue(key, True)


class TourOverlay(QtWidgets.QWidget):
    """A dark scrim + centered card, sized to cover `parent` at
    construction. Not resize-tracking (a one-time first-run overlay
    doesn't need to survive a window resize gracefully)."""

    finished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget, steps: list = None):
        super().__init__(parent)
        self._steps = steps if steps is not None else STEPS
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
        title, body = self._steps[self._step]
        self._title_label.setText(title)
        self._body_label.setText(body)
        self._next_button.setText("Done" if self._step == len(self._steps) - 1 else "Next")

    def _advance(self) -> None:
        if self._step >= len(self._steps) - 1:
            self._finish()
            return
        self._step += 1
        self._render_step()

    def _finish(self) -> None:
        self.finished.emit()
        self.hide()
        self.deleteLater()
