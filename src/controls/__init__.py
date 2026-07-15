"""Physical-mirror control widgets (FireLab roadmap Phase 3): drop-in
replacements for widgets.ToggleGroup with the same value_changed/set_value/
set_enabled_all contract, so main_window.py's existing wiring and every
test that drives these controls via that contract keeps working
unchanged -- only the visual presentation (animated candle/door/vent
icons) is new.
"""
