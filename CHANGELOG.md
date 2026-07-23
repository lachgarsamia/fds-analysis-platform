# Changelog

## v6.0.0 — Gated Capabilities Realized

V6 turns V5 Phase 7's prepared-but-gated seams into working features, wherever
the current FDS output actually supports them -- and keeps every capability the
data doesn't yet support cleanly, honestly gated (`GatedQuantityError`, never a
fabricated value) rather than silently faked. The suite is green throughout
(1004 tests at release).

### Highlights

- **M1 — Field Calculator** — a safe, deterministic expression engine
  (Python `ast`, whitelisted operators/functions only, never `eval`/`exec`) so
  a researcher can define derived quantities like `Temperature - 20` or
  `gradient(Temperature)`; every field records its expression as its basis.
- **M1.5 — Calculated fields as first-class visual quantities** — a calculated
  field appears in the Live Viewer quantity combo, renders as a heatmap, and
  updates during playback, routed through `QuantityProvider` with no changes
  to the cinematic pipeline, cache, or `TimeController`.
- **M2 — Virtual Device Network** — place virtual thermocouples, heat
  detectors, and RTI sprinklers at a point; each computes its time series once
  (cached) and never recomputes per playback tick. Session/report/CSV
  round-trip.
- **M3 — True 3D velocity: streamlines / quiver** — the in-plane (U, W)
  vector field, gated on `U-VELOCITY`/`W-VELOCITY` (absent from this dataset,
  so exercised via synthetic providers); RK2/RK4/Euler streamline
  integration, adaptive quiver density, per-frame-memoized so playback never
  re-integrates.
- **M4 — Unified scientific workspace** — a Context Panel connecting devices,
  vector probes, notebook entries, zones, measurements, the Knowledge Graph,
  and saved sessions to whatever is currently selected; one `_reveal()`
  cross-navigation primitive reused everywhere; an Investigation History
  (back/forward through recorded selections) that ignores per-tick playback
  noise.
- **M5 — Multi-plane cross-sections (plane-gating foundation)** — `SliceKey`/
  `QuantityProvider` extended to any (direction, offset) plane, checked
  against the real `.smv` inventory before ever touching the store (a plane
  that's merely *declared* but not actually loadable is caught the same way);
  devices and the Space-Time Cube both became plane-aware.
- **M6 — Full FED / CO / smoke toxicity** — the standard ISO 13571 / SFPE
  Handbook (Purser) dose equations: toxic-gas dose + convected-heat dose:
  full FED. Tenability and Hazard Spaces automatically upgrade from the
  temperature-only partial screen to full FED wherever `CARBON MONOXIDE
  VOLUME FRACTION` is available (gated today); thermocouples report FED
  alongside temperature.
- **M7 — Multi-plane linked cross-sections + true 3D velocity** — a
  simultaneous XY/XZ/YZ view synced via the `SelectionBus` (a new
  `Selection.depth` field carries the third spatial coordinate); an optional
  `V-VELOCITY` third component for a true 3D vector reading, layered onto M3
  without changing its existing 2D behaviour.

### Honesty, unchanged from V5

Every capability above is additive at a seam V5 Phase 7 already prepared.
Nothing fabricates data the current FDS output doesn't have: absent
quantities raise `GatedQuantityError` with a clear reason
(`docs/msim-preparation.md`), never a guessed value. **V6-4 (validation
toolkit: simulation vs. experiment) was not started** and remains open.

## v5.0.0 — Computational Fire Scientist (connected analysis environment)

V5's theme is **connection**: the panels (live), the artifacts (memory), and the
scenarios (study-level science) are unified through one shared selection. Phases
0–7, all honesty-gated where data is unavailable; the suite is green.

### Highlights

- **Phase 0 — Architecture Stabilization** — a three-layer model (Data /
  Selection / Presentation); the `SelectionModel` (immutable), `SelectionContext`
  (façade), and `SelectionBus` (origin-guarded, no feedback loops); an additive
  `AnalysisPanelBase` + binder; and the `QuantityProvider` computation layer that
  wraps the store and resolves raw *and* derived quantities.
- **M1 — Shared Selection Model** — one selection (scenario · point · region ·
  height · time · interval · phase · quantity · comparison) broadcast to every
  panel; the Insight becomes a first-class selection payload; the Live Viewer
  participates, migrated last. Quantity is a shared field too.
- **M2 — Study-Level Analytics** — the 24-run factorial as a designed experiment:
  parallel coordinates, factor influence, correlation, outliers, statistics.
- **M3 — Sensitivity Explorer** — "what-if" by multilinear interpolation across
  the existing runs (response surfaces, tornado), labelled *Estimated from
  Existing Scenarios*, never a new simulation.
- **Phase 4 — Research Workspace** — Hazard Spaces (Safe/Warning/Critical/
  Untenable + flashover *indicator*), a live Mission-Control Dashboard, fuller
  Adaptive Workspace presets (tab + quantity focus), and a lightweight
  Space-Time Cube.
- **Phase 5 — Scientific Communication** — Fire Narrative++ (expandable
  evidence-backed event chain), Publication Mode (one-click journal-styled figure
  bundle + manifest), Assistant++ (bounded experiment search over the closed
  grammar), and Ensemble Spread (parametric min/mean/max envelopes).
- **Phase 6 — Research Knowledge Graph** — the laboratory memory: experiments →
  scenarios → sessions → insights → zones → measurements → events → tags, all
  navigable; click a node to jump the workspace, click a tag to surface
  everything connected to it.
- **Phase 7 — V6 Preparation** — gated interfaces prepared (not implemented):
  `QuantityProvider.get_vector` + `GatedQuantityError`, a `validation.py` stub,
  and panel/registry seams for 3D velocity, multi-plane cross-sections, full FED,
  and validation. See `ROADMAP-V6.md`.

### Principles

Everything reads from Layer 2, never from another panel. Every estimate carries
its honesty label (*Estimated from Existing Scenarios*, *parametric ensemble
spread*, temperature-only partial FED, association-not-causation, the M-SIM
gate). All V2/V3/V4 behaviour and the cinematic Live viewer preserved.

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
