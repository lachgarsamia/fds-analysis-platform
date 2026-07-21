"""Forschungszentrum Jülich logo widget -- shared by the top bar (menu
bar corner) and the Home page header. Falls back to a plain text label
if the SVG asset isn't found, rather than failing to build the window.
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtWidgets

_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "banner", "img", "logo_fzj.svg")
_ASPECT = 246.61 / 136.06  # the SVG's own viewBox width/height


def build_logo_widget(height: int = 28) -> QtWidgets.QWidget:
    path = os.path.normpath(_LOGO_PATH)
    if os.path.exists(path):
        try:
            from PyQt5 import QtSvg

            class _FixedSizeSvgWidget(QtSvg.QSvgWidget):
                """QSvgWidget.sizeHint() always reports the SVG's own
                intrinsic viewBox size, ignoring setFixedSize() -- and
                QMainWindow's menu bar sizes its row from the corner
                widget's sizeHint(), not its constrained size. Left
                un-overridden, a 24px-tall logo silently forces a
                136px-tall menu bar (the SVG's native height) with a
                huge dead gap above every page."""

                def sizeHint(self) -> QtCore.QSize:
                    return self.size()

            widget = _FixedSizeSvgWidget(path)
            widget.setFixedSize(round(height * _ASPECT), height)
            widget.setAccessibleName("Forschungszentrum Jülich logo")
            return widget
        except ImportError:
            pass
    # Placeholder: no SVG asset/QtSvg available.
    label = QtWidgets.QLabel("Forschungszentrum Jülich")
    label.setProperty("role", "caption")
    label.setAccessibleName("Forschungszentrum Jülich (logo unavailable)")
    return label
