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


# ---------------------------------------------------------------------------
# Spacing / radius / type scale (shared across themes)
# ---------------------------------------------------------------------------

SPACE = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
}

RADIUS = {
    "sm": 4,
    "md": 6,
    "lg": 10,
}

TYPE_SCALE = {
    "caption": 11,
    "body": 13,
    "label": 13,
    "subtitle": 15,
    "title": 18,
    "display": 22,
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
# accent_text vs accent should be >= 4.5:1 as well.

LIGHT = Palette(
    name="light",
    bg_base="#F5F6F8",
    bg_elevated="#FFFFFF",
    bg_sunken="#E9EBEF",
    surface="#FFFFFF",
    border="#D4D7DD",
    border_strong="#AEB3BD",
    text_primary="#1B1E24",
    text_secondary="#4B5160",
    text_disabled="#9AA0AC",
    accent="#0B5FA5",       # darker blue than typical UI blue -> passes AA on white
    accent_hover="#0E6FBE",
    accent_pressed="#094A80",
    accent_text="#FFFFFF",
    focus_ring="#0B5FA5",
    success="#1E7A46",
    danger="#B3261E",
    warning="#8A5B00",
)

DARK = Palette(
    name="dark",
    bg_base="#1A1C20",
    bg_elevated="#24272C",
    bg_sunken="#141518",
    surface="#24272C",
    border="#3A3E45",
    border_strong="#52565F",
    text_primary="#EDEEF0",
    text_secondary="#B3B8C2",
    text_disabled="#6B707A",
    accent="#5AA6E0",       # lighter blue for AA contrast on dark bg
    accent_hover="#78B7E8",
    accent_pressed="#3E8CC9",
    accent_text="#0B1420",
    focus_ring="#5AA6E0",
    success="#4CAE7C",
    danger="#E5695F",
    warning="#D8A93B",
)


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

    pad_sm = px(SPACE["sm"])
    pad_md = px(SPACE["md"])
    pad_lg = px(SPACE["lg"])

    r_sm = px(RADIUS["sm"])
    r_md = px(RADIUS["md"])

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

    QWidget#controlPanel {{
        background-color: {p.bg_elevated};
        border-right: 1px solid {p.border};
    }}

    QLabel {{
        color: {p.text_secondary};
        background: transparent;
    }}

    QLabel[role="section-title"] {{
        color: {p.text_primary};
        font-size: {subtitle};
        font-weight: 600;
        padding-top: {pad_md};
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

    QPushButton {{
        background-color: {p.bg_sunken};
        border: 1px solid {p.border};
        border-radius: {r_md};
        padding: {pad_sm} {pad_md};
        color: {p.text_primary};
        min-height: 22px;
    }}

    QPushButton:hover {{
        border-color: {p.border_strong};
    }}

    QPushButton:pressed {{
        background-color: {p.bg_sunken};
    }}

    QPushButton:disabled {{
        color: {p.text_disabled};
        background-color: {p.bg_sunken};
        border-color: {p.border};
    }}

    QPushButton:focus {{
        border: 2px solid {p.focus_ring};
        padding: {px(SPACE["sm"] - 1)} {px(SPACE["md"] - 1)};
    }}

    /* Toggle-style buttons (checkable) used in ToggleGroup */
    QPushButton[toggle="true"]:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
        color: {p.accent_text};
        font-weight: 600;
    }}

    QPushButton[toggle="true"]:checked:disabled {{
        background-color: {p.accent};
        color: {p.accent_text};
        opacity: 0.6;
    }}

    QPushButton#primaryButton {{
        background-color: {p.accent};
        border-color: {p.accent};
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
        border-color: {p.border};
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
        border: 1px solid {p.border};
        border-radius: {r_md};
        padding: {pad_sm};
        min-height: 22px;
    }}

    QComboBox:focus {{
        border: 2px solid {p.focus_ring};
    }}

    NavigationToolbar2QT {{
        background-color: {p.bg_elevated};
        border-bottom: 1px solid {p.border};
    }}
    """


THEMES: Dict[str, Palette] = {
    "light": LIGHT,
    "dark": DARK,
}
