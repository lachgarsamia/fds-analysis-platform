"""Base class for FireLab's navigable pages (FireLab roadmap Phase 1).

A Page is a plain QWidget with two lifecycle hooks the page host
(MainWindow's QStackedWidget navigation) calls on every switch:
on_enter() (a page that hasn't built its content yet does so here --
the same lazy-on-first-show pattern M3.1's docked panels used to fix a
startup-latency regression, see ROADMAP.md) and on_leave() (e.g. pause
playback). Neither is required -- the no-op defaults are correct for a
page with nothing special to do on switch.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class Page(QtWidgets.QWidget):
    """title is shown in the nav rail; subclasses set it as a class attr."""

    title: str = ""

    def on_enter(self) -> None:
        """Called every time this page becomes the visible page."""

    def on_leave(self) -> None:
        """Called every time this page stops being the visible page."""
