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
            widget = QtSvg.QSvgWidget(path)
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
