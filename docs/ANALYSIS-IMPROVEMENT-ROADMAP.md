# Analysis Section Improvement Roadmap

Derived directly from the Analysis-section usefulness audit (34 panels
reviewed). No new heavy subsystems — every item reuses an existing engine
(SelectionBus, QuantityProvider, Knowledge Graph, descriptors, zone_stats,
report_builder).

## Phase A — Quick Wins (this session)
1. Merge Semantic Diff into Compare Axes (confirmed structural duplicate:
   same two-scenario-combo + quantity + physics-difference shape).
2. Fold Inspect Moment into Dashboard ("Jump to peak moment") — same four
   numbers (temp/HRR/smoke-layer/velocity) Dashboard's cards already show.
3. Wire Calculator's preview combo to the SelectionBus.
4. Make Ensemble Analytics' PCA scatter clicks publish to the SelectionBus.

## Phase B — Structural Cleanup (next)
1. Merge Tenability + Hazard into one "Hazard & Tenability" tab with a mode
   toggle (same CO gate, same classification territory).
2. Fold Factor Effects into Study as a sub-tab (Study already reuses its
   convention).
3. Fold Energy Budget into Dashboard's HRR card as an expandable detail.
4. Merge Ask into Assistant (both Q&A-style, overlapping purpose).

## Phase C — High-Impact Reuse Features
1. Cross-scenario comparison for Devices (reuse Zones' existing pattern).
2. "Pin what-if to Knowledge Graph" from Sensitivity.
3. Synthesized point-story in Context panel (Narrative + Cause + local
   reading combined via `context.gather_context`).
4. One-click "Add comparison to session report" from Compare Axes (reuse
   `report_builder.py`).

## Phase D — Layout & Navigation
1. Re-group Analysis tabs: Core Investigation / Study-Level / Comparison /
   Interpretation & Communication / Experimental.
2. Experimental section (Fire MRI, Energy Budget, Attention, Why is it
   hot?, Forecasting) collapsed by default.

Full audit: see the "Analysis Section — Usefulness Audit" report (this
session's conversation history).
