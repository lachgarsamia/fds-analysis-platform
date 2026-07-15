"""Cinematic rendering effects (FireLab roadmap, Phase 2).

Turns a raw TEMPERATURE frame into an RGBA image via a black-body LUT with
an alpha ramp, a filmic tone curve, and an auto-exposure EMA -- instead of
matplotlib's Normalize+cmap drawing the data directly. See
ROADMAP-FIRELAB.md Phase 2 for the full effects-pipeline design; this
package currently implements the first slice of it (FireLUT + tone map +
auto-exposure), not the later bloom/smoke/shimmer/particle passes.
"""
