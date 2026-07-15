"""
theme.py
--------
A small design-token system for the FDS SLCF Visualizer.

Instead of a single hardcoded QSS string, we define named tokens (spacing,
type scale, color roles, radii) for both a light and dark palette, and a
function that renders a QSS stylesheet from whichever token set is active.

Why this exists:
    - One dark stylesheet baked into the code can't be swapped or audited.
    - Tokens give every widget a single source of truth for color/spacing,
      so accessibility fixes (contrast, focus rings) happen in one place.
    - Runtime theme switching (light/dark) becomes a one-line call.
"""

from dataclasses import dataclass
from typing import Dict

from PyQt5 import QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Spacing / radius / type scale (shared across themes)
# ---------------------------------------------------------------------------

SPACE = {
    "xs": 6,
    "sm": 10,
    "md": 16,
    "lg": 24,
    "xl": 32,
    "xxl": 40,
}

RADIUS = {
    "sm": 8,
    "md": 10,
    "lg": 14,
}

TYPE_SCALE = {
    "caption": 11,
    "body": 13,
    "label": 13,
    "subtitle": 16,
    "title": 20,
    "display": 26,
}

FONT_FAMILY = '"Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif'


@dataclass
class Palette:
    """Color roles. Every widget should reference a role, never a raw hex."""
    name: str
    bg_base: str
    bg_elevated: str
    bg_sunken: str
    surface: str
    border: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_disabled: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str          # text color drawn on top of `accent`
    focus_ring: str
    success: str
    danger: str
    warning: str


# WCAG AA target: body text vs bg_base and vs surface should be >= 4.5:1.
# accent_text vs accent should be >= 4.5:1 as well. Every pair below is
# checked computationally (relative luminance, WCAG's own formula), not
# eyeballed -- see the design-pass notes in ROADMAP.md for the exact
# ratios measured before these values were picked.
#
# Design direction (GUI modernization pass): cool, desaturated neutrals
# (not the previous slightly-warm grays) with a single crisp accent blue,
# closer to current-generation dashboard tools (Linear/Vercel/Grafana)
# than a stock Qt app -- bg_base a hair off pure black/white rather than
# a mid gray, tighter border colors, and a more saturated/legible accent.

LIGHT = Palette(
    name="light",
    bg_base="#F7F8FA",
    bg_elevated="#FFFFFF",
    bg_sunken="#EEF0F3",
    surface="#FFFFFF",
    border="#E2E5EA",
    border_strong="#C7CCD4",
    text_primary="#14171F",
    text_secondary="#5B6472",
    text_disabled="#9AA1AC",
    accent="#2563EB",       # crisp, modern blue -- passes AA on white with margin
    accent_hover="#3B76F0",
    accent_pressed="#1D4FC4",
    accent_text="#FFFFFF",
    focus_ring="#2563EB",
    success="#16A34A",
    danger="#DC2626",
    warning="#B45309",
)

DARK = Palette(
    name="dark",
    bg_base="#0B0D10",
    bg_elevated="#14171B",
    bg_sunken="#07080A",
    surface="#14171B",
    border="#23272E",
    border_strong="#363B44",
    text_primary="#EDEEF2",
    text_secondary="#9CA3AF",
    text_disabled="#5B6270",
    accent="#4C8DFF",       # lighter, more saturated blue for AA contrast on near-black
    accent_hover="#6FA3FF",
    accent_pressed="#3568D4",
    accent_text="#06101F",
    focus_ring="#4C8DFF",
    success="#3DD68C",
    danger="#F1594F",
    warning="#F5A623",
)


# FireLab roadmap Phase 1: the demo default. Near-black surfaces (deeper
# than DARK's own near-black) so the cinematic fire palette (cinema/luts.py's
# FireLUT) reads as the only bright thing on screen -- fire only looks hot
# against real darkness. Ember-orange accent instead of DARK's blue, to
# stay visually consistent with the fire theme rather than competing with
# it. Contrast ratios follow the same AA-target discipline as LIGHT/DARK
# above, just not re-derived here since this reuses DARK's text/border
# roles verbatim (only bg/accent shift).
THEATRE = Palette(
    name="theatre",
    bg_base="#08090C",
    bg_elevated="#101318",
    bg_sunken="#050506",
    surface="#101318",
    border="#1C2027",
    border_strong="#2E333C",
    text_primary="#EDEEF2",
    text_secondary="#9CA3AF",
    text_disabled="#5B6270",
    accent="#FF6B35",       # ember orange -- matches the FireLUT's palette family
    accent_hover="#FF8B5C",
    accent_pressed="#E05A28",
    accent_text="#0B0500",
    focus_ring="#FF6B35",
    success="#3DD68C",
    danger="#F1594F",
    warning="#F5A623",
)


def apply_card_shadow(widget: QtWidgets.QWidget, p: Palette) -> None:
    """Attach a soft drop shadow standing in for a hard border -- the
    "card" look of the Streamlit-style redesign. QSS has no box-shadow
    equivalent, so this is done in Python via QGraphicsDropShadowEffect.

    Deliberately reserved for widgets that are static once built (sidebar
    section cards, the analytics figure's outer frame) -- NOT for anything
    that wraps a frequently-repainting child (grid cells wrapping the
    blitted MplCanvas during playback, or a scrolling QTableView). Qt has
    to re-render a QGraphicsEffect's whole offscreen buffer whenever any
    descendant repaints, so putting one on an animated container would
    reintroduce the exact per-frame jank this same redesign pass was
    asked to check for. Those widgets get a QSS-only "soft card" look
    (rounded corners + a subtle border) instead -- see MplCanvas's QSS
    rule and QTableView's QSS rule below.

    Call again after a theme switch: light/dark need different shadow
    opacity (a dark shadow barely reads on an already-dark background).
    """
    effect = QtWidgets.QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(28)
    effect.setXOffset(0)
    effect.setYOffset(4)
    color = QtGui.QColor("#000000")
    color.setAlpha(50 if p.name == "light" else 110)
    effect.setColor(color)
    widget.setGraphicsEffect(effect)


def build_qss(p: Palette, ui_scale: float = 1.0) -> str:
    """Render a QSS stylesheet from a palette + a user-adjustable UI scale.

    ui_scale multiplies the type scale and spacing so users who need larger
    text/touch targets can bump it up (see MainWindow's "UI scale" control).
    """

    def px(v: float) -> str:
        return f"{round(v * ui_scale)}px"

    body = px(TYPE_SCALE["body"])
    label = px(TYPE_SCALE["label"])
    subtitle = px(TYPE_SCALE["subtitle"])
    title = px(TYPE_SCALE["title"])
    caption = px(TYPE_SCALE["caption"])

    pad_xs = px(SPACE["xs"])
    pad_sm = px(SPACE["sm"])
    pad_md = px(SPACE["md"])
    pad_lg = px(SPACE["lg"])

    r_sm = px(RADIUS["sm"])
    r_md = px(RADIUS["md"])
    r_lg = px(RADIUS["lg"])

    return f"""
    * {{
        font-family: {FONT_FAMILY};
        font-size: {body};
        color: {p.text_primary};
        outline: none;
    }}

    QMainWindow, QWidget#centralWidget {{
        background-color: {p.bg_base};
    }}

    /* Grid cells (views.py's GridCell -- each holds one plot) are cards
    that sit on that muted backdrop, same pattern as the sidebar's
    sectionCard below: a flat page background, distinct-toned cards on
    top. Without this, GridCell (a plain unstyled QWidget) falls back to
    Qt's own default palette background, which stays off-white/light even
    under the dark theme -- a stray pale rectangle behind the plot. */
    GridCell {{
        background-color: {p.surface};
    }}

    /* Streamlit's defining trait: a sidebar in a flat, slightly muted
    tone next to a brighter main content area -- no hard dividing line,
    the tone difference alone reads as a separate region. Section "cards"
    inside the sidebar (CollapsibleSection, widgets.py) then sit on top in
    `surface` white so they pop against the muted sidebar background. */
    QWidget#controlPanel {{
        background-color: {p.bg_base};
        border: none;
    }}

    QLabel {{
        color: {p.text_secondary};
        background: transparent;
    }}

    QLabel[role="section-title"] {{
        color: {p.text_primary};
        font-size: {subtitle};
        font-weight: 700;
        padding-top: 0;
    }}

    QLabel[role="title"] {{
        color: {p.text_primary};
        font-size: {title};
        font-weight: 700;
    }}

    QLabel[role="value"] {{
        color: {p.text_primary};
        font-size: {label};
        font-weight: 600;
    }}

    QLabel[role="caption"] {{
        color: {p.text_secondary};
        font-size: {caption};
    }}

    QFrame#divider {{
        background-color: {p.border};
        max-height: 1px;
        min-height: 1px;
        border: none;
    }}

    /* Sidebar section "card" (widgets.py's CollapsibleSection) -- replaces
    the old title+divider grouping. Real drop shadow applied in Python via
    theme.apply_card_shadow (QSS has no box-shadow); this widget never
    hosts frequently-repainting content, so the shadow's extra compositing
    pass is safe here in a way it wouldn't be on a grid cell. */
    QFrame#sectionCard {{
        background-color: {p.surface};
        border: none;
        border-radius: {r_lg};
    }}

    /* The plot canvas itself is always white (widgets.py's MplCanvas.PLOT_BG
    -- standard practice so colormaps read true colors, unaffected by the
    app's own theme), which would otherwise look like a stray white hole
    punched into a dark UI; a soft, subtle frame reads as an intentional
    "card" without a hard line. No QGraphicsDropShadowEffect here -- this
    widget repaints every playback frame via blitting, and a real drop
    shadow would force Qt to recomposite on every frame (see
    apply_card_shadow's docstring). */
    MplCanvas {{
        border: 1px solid {p.border};
        border-radius: {r_lg};
    }}

    QPushButton {{
        background-color: {p.bg_sunken};
        border: none;
        border-radius: {r_md};
        padding: {pad_sm} {pad_lg};
        color: {p.text_primary};
        font-weight: 500;
        min-height: 26px;
    }}

    QPushButton:hover {{
        background-color: {p.border};
    }}

    QPushButton:pressed {{
        background-color: {p.border_strong};
    }}

    QPushButton:disabled {{
        color: {p.text_disabled};
        background-color: {p.bg_sunken};
    }}

    QPushButton:focus {{
        border: 2px solid {p.focus_ring};
        padding: {px(SPACE["sm"] - 1)} {px(SPACE["lg"] - 1)};
    }}

    /* Toggle-style buttons (checkable) used in ToggleGroup */
    QPushButton[toggle="true"]:checked {{
        background-color: {p.accent};
        color: {p.accent_text};
        font-weight: 600;
    }}

    QPushButton[toggle="true"]:checked:hover {{
        background-color: {p.accent_hover};
    }}

    QPushButton[toggle="true"]:checked:disabled {{
        background-color: {p.accent};
        color: {p.accent_text};
        opacity: 0.6;
    }}

    /* Primary actions: solid accent fill. Secondary/default QPushButton
    above stays a minimal, ghost-style subtle tint. */
    QPushButton#primaryButton {{
        background-color: {p.accent};
        color: {p.accent_text};
        font-weight: 600;
    }}

    QPushButton#primaryButton:hover {{
        background-color: {p.accent_hover};
    }}

    QPushButton#primaryButton:pressed {{
        background-color: {p.accent_pressed};
    }}

    QPushButton#primaryButton:disabled {{
        background-color: {p.bg_sunken};
        color: {p.text_disabled};
    }}

    /* FireLab roadmap Phase 1: left navigation rail. */
    QWidget#navRail {{
        background-color: {p.bg_sunken};
        border-right: 1px solid {p.border};
    }}

    QPushButton#navButton {{
        background-color: transparent;
        border: none;
        border-radius: {r_md};
        text-align: left;
        padding: {pad_sm} {pad_md};
        color: {p.text_secondary};
        font-weight: 500;
    }}

    QPushButton#navButton:hover {{
        background-color: {p.bg_elevated};
        color: {p.text_primary};
    }}

    QPushButton#navButton:checked {{
        background-color: {p.accent};
        color: {p.accent_text};
        font-weight: 600;
    }}

    QPushButton#navCollapseButton {{
        background-color: transparent;
        border: none;
        color: {p.text_secondary};
    }}

    QPushButton#navCollapseButton:hover {{
        color: {p.text_primary};
    }}

    QSlider::groove:horizontal {{
        height: 6px;
        background: {p.bg_sunken};
        border-radius: 3px;
    }}

    QSlider::sub-page:horizontal {{
        background: {p.accent};
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        background: {p.accent};
        width: {px(18)};
        height: {px(18)};
        margin: -6px 0;
        border-radius: {px(9)};
        border: 2px solid {p.bg_elevated};
    }}

    QSlider:focus::handle:horizontal {{
        border: 2px solid {p.focus_ring};
    }}

    QProgressBar {{
        border: 1px solid {p.border};
        border-radius: {r_sm};
        background: {p.bg_sunken};
        text-align: center;
        color: {p.text_primary};
        min-height: 16px;
    }}

    QProgressBar::chunk {{
        background-color: {p.accent};
        border-radius: {r_sm};
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        border-radius: 5px;
        min-height: 24px;
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QSplitter::handle {{
        background-color: {p.border};
    }}

    QSplitter::handle:hover {{
        background-color: {p.accent};
    }}

    QStatusBar {{
        background-color: {p.bg_elevated};
        border-top: 1px solid {p.border};
        color: {p.text_secondary};
    }}

    QToolTip {{
        background-color: {p.bg_elevated};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        padding: {pad_sm};
        border-radius: {r_sm};
    }}

    QComboBox {{
        background-color: {p.bg_sunken};
        border: none;
        border-radius: {r_md};
        padding: {pad_sm} {pad_md};
        min-height: 26px;
    }}

    QComboBox:hover {{
        background-color: {p.border};
    }}

    QComboBox:focus {{
        border: 2px solid {p.focus_ring};
    }}

    /* Without an explicit :disabled rule here, macOS's native style takes
    over painting a disabled QComboBox and ignores this stylesheet entirely
    -- shows up as a stray white pill in dark mode (e.g. a per-cell quantity
    combo when only one quantity is available to pick). QPushButton below
    already has the same :disabled override for the same reason. */
    QComboBox:disabled {{
        color: {p.text_disabled};
        background-color: {p.bg_sunken};
    }}

    QComboBox QAbstractItemView {{
        background-color: {p.bg_elevated};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: {r_md};
        selection-background-color: {p.accent};
        selection-color: {p.accent_text};
        outline: none;
    }}

    NavigationToolbar2QT {{
        background-color: {p.bg_elevated};
        border-bottom: 1px solid {p.border};
    }}

    /* Docked panels (experiment browser, analytics) previously fell back
    to Qt's native/system palette here -- e.g. QTableView followed the
    OS's own light/dark appearance rather than this app's chosen theme,
    which could show a dark table under a light app theme (or vice
    versa) regardless of which palette is active.

    QSS-only "soft card" (rounded corners, subtle border, no real
    QGraphicsDropShadowEffect) -- this view repaints on every scroll, and
    a real shadow effect would recomposite the whole table on each one;
    see apply_card_shadow's docstring in this file for the full reasoning. */
    QTableView {{
        background-color: {p.surface};
        alternate-background-color: {p.bg_sunken};
        color: {p.text_primary};
        gridline-color: {p.border};
        border: 1px solid {p.border};
        border-radius: {r_lg};
        selection-background-color: {p.accent};
        selection-color: {p.accent_text};
    }}

    QTableView::item {{
        padding: {pad_xs};
    }}

    QHeaderView::section {{
        background-color: {p.bg_sunken};
        color: {p.text_secondary};
        padding: {pad_sm};
        border: none;
        border-bottom: 1px solid {p.border};
        font-weight: 600;
    }}

    QLineEdit {{
        background-color: {p.bg_sunken};
        border: none;
        border-radius: {r_md};
        padding: {pad_sm} {pad_md};
        color: {p.text_primary};
        min-height: 26px;
    }}

    QLineEdit:focus {{
        border: 2px solid {p.focus_ring};
    }}

    QDockWidget {{
        color: {p.text_primary};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}

    /* Same QSS-only soft-card treatment as QTableView above -- dock
    content can scroll/animate, so no real drop shadow. */
    QDockWidget::title {{
        background-color: {p.bg_sunken};
        color: {p.text_secondary};
        padding: {pad_md};
        border: none;
        border-top-left-radius: {r_lg};
        border-top-right-radius: {r_lg};
        font-weight: 700;
        font-size: {label};
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {p.text_secondary};
        border: none;
        border-top-left-radius: {r_md};
        border-top-right-radius: {r_md};
        padding: {pad_sm} {pad_lg};
    }}

    QTabBar::tab:selected {{
        background-color: {p.bg_sunken};
        color: {p.text_primary};
        border-bottom: 2px solid {p.accent};
        font-weight: 600;
    }}

    QTabBar::tab:hover:!selected {{
        color: {p.text_primary};
        background-color: {p.bg_sunken};
    }}

    QMenuBar {{
        background-color: {p.bg_base};
        color: {p.text_primary};
    }}

    QMenuBar::item:selected {{
        background-color: {p.bg_sunken};
        border-radius: {r_sm};
    }}

    QMenu {{
        background-color: {p.bg_elevated};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: {r_lg};
        padding: {pad_xs};
    }}

    QMenu::item {{
        padding: {pad_sm} {pad_lg};
        border-radius: {r_sm};
    }}

    QMenu::item:selected {{
        background-color: {p.accent};
        color: {p.accent_text};
    }}

    QMenu::separator {{
        height: 1px;
        background: {p.border};
        margin: {px(4)} {pad_sm};
    }}

    QCheckBox::indicator, QMenu::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {p.border_strong};
        border-radius: {px(3)};
        background-color: {p.bg_sunken};
    }}

    QCheckBox::indicator:checked, QMenu::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
    }}
    """


THEMES: Dict[str, Palette] = {
    "light": LIGHT,
    "dark": DARK,
    "theatre": THEATRE,
}
