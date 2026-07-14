"""Unit tests for theme.py's Streamlit-style redesign pass: generous
spacing/radius tokens and the QGraphicsDropShadowEffect card-shadow helper.
"""

from PyQt5 import QtWidgets

from theme import DARK, LIGHT, RADIUS, SPACE, apply_card_shadow, build_qss


class TestTokens:
    def test_space_tokens_are_generous(self):
        """Streamlit-style breathing room: at least 16px around panels,
        8px+ between related controls (redesign spec item 1)."""
        assert SPACE["md"] >= 16
        assert SPACE["sm"] >= 8

    def test_radius_tokens_are_rounded(self):
        """10-14px radius on cards (redesign spec item 2)."""
        assert RADIUS["md"] >= 10
        assert RADIUS["lg"] >= 14


class TestApplyCardShadow:
    def test_attaches_a_graphics_effect(self, qapp):
        widget = QtWidgets.QWidget()
        assert widget.graphicsEffect() is None
        apply_card_shadow(widget, LIGHT)
        assert isinstance(widget.graphicsEffect(), QtWidgets.QGraphicsDropShadowEffect)

    def test_dark_theme_shadow_is_more_opaque_than_light(self, qapp):
        """A dark shadow barely reads against an already-dark background,
        so dark mode needs a higher-alpha shadow to stay visible."""
        light_widget = QtWidgets.QWidget()
        dark_widget = QtWidgets.QWidget()
        apply_card_shadow(light_widget, LIGHT)
        apply_card_shadow(dark_widget, DARK)
        light_alpha = light_widget.graphicsEffect().color().alpha()
        dark_alpha = dark_widget.graphicsEffect().color().alpha()
        assert dark_alpha > light_alpha

    def test_reapplying_after_theme_switch_replaces_the_effect(self, qapp):
        """Called again on theme switch (main_window.py's _apply_theme) --
        must not stack multiple effects on the same widget."""
        widget = QtWidgets.QWidget()
        apply_card_shadow(widget, LIGHT)
        apply_card_shadow(widget, DARK)
        effect = widget.graphicsEffect()
        assert isinstance(effect, QtWidgets.QGraphicsDropShadowEffect)
        assert effect.color().alpha() == 110


class TestBuildQssCardRules:
    def test_qss_defines_a_borderless_rounded_section_card(self):
        qss = build_qss(LIGHT)
        assert "QFrame#sectionCard" in qss

    def test_default_buttons_are_borderless(self):
        """Flat, no border by default (redesign spec item 3)."""
        qss = build_qss(LIGHT)
        button_rule_start = qss.index("QPushButton {")
        button_rule = qss[button_rule_start:qss.index("}", button_rule_start)]
        assert "border: none" in button_rule

    def test_primary_button_keeps_solid_accent_fill(self):
        qss = build_qss(LIGHT)
        assert "QPushButton#primaryButton {" in qss
        assert LIGHT.accent in qss
