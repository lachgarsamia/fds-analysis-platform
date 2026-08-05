"""Forschungszentrum Jülich + Bergische Universität Wuppertal logo widgets
-- shared by the top bar (menu bar corner) and the Home page header. Each
falls back to a plain text label if its asset isn't found, rather than
failing to build the window.

Both marks are pre-cropped PNGs (banner/img/), each with a black-ink and a
white-ink variant for theme switching -- see _LogoLabel. FZJ's original
asset was banner/img/logo_fzj.svg, whose declared viewBox has enormous
built-in padding (measured directly: the visible circle+wordmark occupies
only ~33% of the viewBox's own height, worse than Wuppertal's ~74%) --
sizing both logos to a shared nominal height/aspect from their raw canvases
therefore made FZJ's *actual visible mark* much smaller than Wuppertal's
even when the maths said the boxes were area-matched (live-testing
feedback: "one bigger than the other" persisted through two rounds of
ratio tuning that never touched this). logo_fzj_cropped.png is that SVG
rendered at high resolution and cropped to its real (alpha-detected)
content bounding box + a small margin, once, offline -- not something this
module recomputes at runtime -- so its own aspect ratio is now
representative of what's actually drawn, the same way Wuppertal's already
was.
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "banner", "img")

_FZJ_PATH = os.path.join(_ASSET_DIR, "logo_fzj_cropped.png")
_FZJ_PATH_WHITE = os.path.join(_ASSET_DIR, "logo_fzj_cropped_white.png")
_WUPPERTAL_PATH = os.path.join(_ASSET_DIR, "logo_wuppertal.png")
_WUPPERTAL_PATH_WHITE = os.path.join(_ASSET_DIR, "logo_wuppertal_white.png")


class _LogoLabel(QtWidgets.QLabel):
    """A QLabel showing one of two PNG variants (black/white ink) at a
    fixed height, swapped via set_dark() -- same "reflects the currently
    applied theme, doesn't own the choice itself" convention as
    NavRail.set_dark() for the theme toggle button, so both logos can be
    driven by the exact same call site. Both FZJ and Wuppertal need this:
    FZJ's navy (#023d6b) measures a ~1.8:1 contrast ratio against the dark
    theme's near-black bg_sunken (#07080A) -- well under WCAG's 3:1 floor
    for graphics, almost as illegible as Wuppertal's flat black was."""

    def __init__(self, black_path: str, white_path: str, height: int,
                 accessible_name: str, parent=None):
        super().__init__(parent)
        self._pixmaps = {}
        self._aspect = None
        for is_dark, path in ((False, black_path), (True, white_path)):
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                pix = QtGui.QPixmap(normalized)
                if self._aspect is None and pix.height() > 0:
                    self._aspect = pix.width() / pix.height()
                self._pixmaps[is_dark] = pix.scaledToHeight(height, QtCore.Qt.SmoothTransformation)
        if self._aspect is not None:
            self.setFixedSize(round(height * self._aspect), height)
        self.setAccessibleName(accessible_name)
        self.set_dark(False)

    @property
    def aspect(self) -> float:
        return self._aspect or 1.0

    def set_dark(self, is_dark: bool) -> None:
        pix = self._pixmaps.get(is_dark) or self._pixmaps.get(not is_dark)
        if pix is not None:
            self.setPixmap(pix)


def _build_logo(black_path: str, white_path: str, height: int,
                 accessible_name: str, fallback_text: str) -> QtWidgets.QWidget:
    if os.path.exists(os.path.normpath(black_path)):
        return _LogoLabel(black_path, white_path, height, accessible_name)
    label = QtWidgets.QLabel(fallback_text)
    label.setProperty("role", "caption")
    label.setAccessibleName(f"{fallback_text} (logo unavailable)")
    return label


def build_logo_widget(height: int = 28) -> QtWidgets.QWidget:
    return _build_logo(_FZJ_PATH, _FZJ_PATH_WHITE, height,
                        "Forschungszentrum Jülich logo", "Forschungszentrum Jülich")


def build_wuppertal_logo_widget(height: int = 28) -> QtWidgets.QWidget:
    return _build_logo(_WUPPERTAL_PATH, _WUPPERTAL_PATH_WHITE, height,
                        "Bergische Universität Wuppertal logo", "Bergische Universität Wuppertal")


class PartnerLogosWidget(QtWidgets.QWidget):
    """FZJ above Wuppertal, one accessible unit -- both partner institutions
    read as a pair wherever the app already showed just FZJ (nav rail, Home
    header), not as two unrelated logos that happen to be near each other.
    Stacked, not side by side: at a shared height tall enough to read, the
    two logos' combined width overran the ~340px nav rail and visibly
    overlapped -- a vertical stack only grows the (plentiful) rail height,
    never its (scarce, user-resizable-but-bounded) width.

    Split by *area*, not by equal height: the two marks have different
    content aspect ratios even now that both are tightly cropped (FZJ
    ~2.92:1, Wuppertal ~2.37:1) -- equal height alone would still leave the
    wider one (FZJ, now) looking heavier. h_a/h_b = sqrt(aspect_b/aspect_a)
    equalizes total ink area instead, computed from each _LogoLabel's own
    measured aspect (module-level constants would drift the moment either
    asset is re-cropped)."""

    def __init__(self, total_height: int = 28, parent=None):
        super().__init__(parent)
        gap = max(round(total_height * 0.08), 4)
        budget = max(total_height - gap, 28)

        # Build once at a nominal size just to read each asset's own aspect
        # ratio, then rebuild at the real, area-matched heights below --
        # cheap (label construction, not disk I/O twice: QPixmap is loaded
        # once per path per build_..._widget call either way).
        probe_fzj = build_logo_widget(100)
        probe_wuppertal = build_wuppertal_logo_widget(100)
        aspect_fzj = getattr(probe_fzj, "aspect", 1.0)
        aspect_wuppertal = getattr(probe_wuppertal, "aspect", 1.0)
        probe_fzj.deleteLater()
        probe_wuppertal.deleteLater()

        ratio = (aspect_wuppertal / aspect_fzj) ** 0.5  # h_fzj / h_wuppertal for equal area
        h_wuppertal = max(round(budget / (1 + ratio)), 12)
        h_fzj = max(budget - h_wuppertal, 12)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(gap)
        self._fzj = build_logo_widget(h_fzj)
        self._wuppertal = build_wuppertal_logo_widget(h_wuppertal)
        layout.addWidget(self._fzj, 0, QtCore.Qt.AlignHCenter)
        layout.addWidget(self._wuppertal, 0, QtCore.Qt.AlignHCenter)

    def set_dark(self, is_dark: bool) -> None:
        for widget in (self._fzj, self._wuppertal):
            set_dark = getattr(widget, "set_dark", None)
            if callable(set_dark):
                set_dark(is_dark)


def build_partner_logos_widget(total_height: int = 28) -> QtWidgets.QWidget:
    return PartnerLogosWidget(total_height)
