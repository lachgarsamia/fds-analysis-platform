# Changelog

## v4.0.0 — Researcher-Centered Interactive Analysis Environment

V4 turns the FDS slice viewer into an interactive scientific *analysis
environment*: researchers explore, measure, compare, annotate, and publish
without leaving the app. Twelve milestones, all honesty-gated where data is
unavailable; the full test suite is green (734 tests).

### Highlights

- **Height-Aware Analysis Workspace (M1)** — pick a vertical line and read the
  fire's vertical behaviour: the temperature-vs-height profile, smoke-layer,
  plume, and ceiling-jet over time, as analysed curves.
- **Evidence Notebook (M2)** — every measurement is a saveable, annotatable,
  taggable Insight; a dockable notebook that persists in the session and flows
  into reports. Defined once on the shared Insight list, inherited everywhere.
- **Linked Multi-Quantity Inspection (M3)** — one moment across the physics:
  temperature field, HRR, smoke layer, and velocity under a shared time cursor.
- **Named Zone Statistics (M4)** — draw a named zone; get mean/max, time-to-
  threshold, thermal dose, hazard duration, affected fraction, and an energy
  proxy, per scenario and compared across scenarios.
- **Time-Window & Interval Analysis (M5)** — time as a selectable dimension:
  interval stats, before/after split, and one-click detected-phase windows.
- **Reproducible Named Sessions (M6)** — save/browse/reload the whole
  investigation (grid, notebook, zones, interval, filters) with metadata and a
  data-run fingerprint; export to a report.
- **Measurement Tools (M7)** — distance, path, rectangle (area + stats), and a
  bilinear probe, at an instant or averaged over an interval; saved in sessions.
- **Advanced Comparison Workflows (M8)** — temporal (danger cross-over), spatial
  (per-region difference), and physics (ranked associated drivers, explicitly
  association-not-causation) axes.
- **Experiment Management (M9)** — named, tagged batches of scenarios with a
  baseline and availability status; one-click bulk comparison hand-off.
- **Publication Workflow Completion (M10)** — journal/slide export presets,
  figure export from any analysis panel, and an Evidence-Notebook report.
- **Quantity-Framework Breadth (M11, partly gated)** — pressure, visibility,
  heat flux, CO, soot-mass, U/W-velocity registered as first-class quantities
  (gated on the M-SIM cluster re-run); two derived quantities (temperature
  rise, dynamic pressure) shipped now.
- **Safe Assistant (M12)** — a bounded, deterministic organizer of computed
  evidence (summaries, findings, outlines, comparisons, captions). It never
  asserts a physical cause; "why" requests are refused by design.

### Principles carried throughout

- Every stated conclusion is a template filled from computed values with a
  traceable `basis`; no invented physics.
- Honesty gates enforced in code and tests: partial-FED (temperature-only),
  heuristic-saliency, association-not-causation, and the M-SIM data gate for
  quantities the current run cannot supply.
- All prior behaviour and the cinematic Live viewer preserved.

### Built on

V4 sits on the V2 Research Platform and V3 Fire Intelligence Layer, both merged
into `main` as part of this release (the branch stack had lagged `main`).
