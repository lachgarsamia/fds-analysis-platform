# ROADMAP V5 — From Interactive Analysis Environment to Computational Fire Scientist

> **The theme is not more analyses. It is connection.** V4 shipped a dozen
> analyses that do not talk to each other, and a 24-scenario factorial still
> explored one run at a time. V5's question is the one that matters after V4:
> *what does a researcher still do by hand?* — switch panels, re-find context,
> compare scenarios manually. V5 removes that by connecting the panels (live),
> the artifacts (memory), and the scenarios (study-level science).

## 0. Organizing principle: connection at three levels, over three layers

Connection: **live** (one selection drives every panel), **memory** (a knowledge
graph links every artifact), **study** (the factorial becomes a designed
experiment).

Architecture (full design in `docs/architecture-selection-model.md`):

```
Layer 1 — Data          ScenarioStore · Descriptors · Signatures · Derived Quantities
Layer 2 — Selection     SelectionModel · SelectionContext · SelectionBus   ◄── UI depends only on this
Layer 3 — Presentation  Panels · Plots · Dashboard · Notebook · Assistant · Reports · Comparison · Browser
```

Every UI component depends only on Layer 2 — it publishes and reacts to a
`Selection`, never references another panel. The `SelectionModel` is the
**canonical interaction object**: every subsystem (Insight, Measurement,
Notebook entry, report link, semantic diff, assistant answer) can produce and
consume one, so every interaction reduces to
`object → SelectionModel → SelectionBus → everything updates`.

**Project rule:** *M1 establishes the architecture; M2–M6 consume it.* A
successful M1 proves the foundation on a couple of panels; it does not finish the
migration.

Everything here is buildable on **current** data (2D TEMPERATURE +
VELOCITY-magnitude slices, `.s3d` soot volume, HRR CSV, 24-scenario candle
factorial). Nothing waits on the M-SIM re-run.

---

## Phase 0 — Architecture Stabilization

**Objectives:** remove duplicated selection state; introduce `SelectionModel`,
`SelectionContext`, and `SelectionBus`; introduce `AnalysisPanelBase`; introduce
the `QuantityProvider` computation layer + Derived Quantities Framework; preserve
all V4 behaviour.

**Derived Quantities Framework (infrastructure, not a milestone).** A
`QuantityProvider` wraps `ScenarioStore` (which is never modified) and resolves
raw *and* derived quantities behind one call, so a derived quantity is just a
registry entry + a function (`dT/dt`, temperature gradient, thermal dose, heat
accumulation, hot-layer thickness, a combined hazard index). Because it flows
through the same quantity id, **every** V5 feature — plots, notebook, dashboard,
analytics, reports, comparison, sensitivity — supports it for free. Each derived
entry carries the M11 honesty metadata (unit, interpretation, `basis`).

**Deliverables:** the architecture note (done); pure unit tests for
`SelectionModel`/`SelectionContext`/`SelectionBus` and the Insight/Session
adapters; the `QuantityProvider` with one generalized derived quantity end to
end; zero UI regressions (full suite green); the foundation for M1–M6.
**Priority: prerequisite.**

---

## Phase 1 — Unified Interaction Layer

### M1 — Shared Selection Model ⭐ (flagship)
Build the central event bus + selection model. One selection synchronizes
scenario, point, region, height, time, interval, phase, quantity, and comparison
state. Every existing panel *subscribes* to it instead of holding local state;
the single existing cross-panel link (`insight_activated → seek`) is subsumed
into the general mechanism. **M1 scope:** the model/context/bus, the
Insight/Session adapters, the `QuantityProvider` proof, `AnalysisPanelBase`, and
**two** migrated panels (Height, then Linked Inspection for bidirectional sync) —
not the full migration. **Reuse:** every panel's seek/marker code, `insight.py`,
`session.py`. **Difficulty:** medium. **Priority: highest** — the foundation for
everything else.

---

## Phase 2 — Study Intelligence

### M2 — Study-Level Analytics
Turn an experiment into a first-class object: parallel coordinates, response
surfaces, factor-interaction visualization, ANOVA/effect summaries, a correlation
explorer, clustering, outlier detection, and study statistics across all 24
scenarios. This expands the software from simulation analysis to **experiment
analysis**. **Reuse:** `factor_effects`, `state_space`, `summary_stats`,
`descriptors`; consumes M1. **Difficulty:** medium. **Priority: high.**

---

## Phase 3 — Computational Exploration

### M3 — Sensitivity Explorer
Estimate parameter influence from the *existing* simulations: ventilation / HRR /
door-width sensitivity, response surfaces, local interpolation across factor
levels, and "what-if" exploration. **Honesty gate (restated in code):** every
readout is labelled **"Estimated from Existing Scenarios"** — never implying a new
simulation was run. **Reuse:** M2's response surfaces, the factor axes.
**Difficulty:** medium. **Priority: medium–high.**

---

## Phase 4 — Research Workspace

Now that interaction (M1) and analytics (M2–M3) exist, these become far more
powerful.

- **Hazard Spaces** — dynamic classification (Safe / Warning / Critical /
  Untenable / flashover-*indicator*) as a per-frame map and a hazard-state
  timeline, from temperature thresholds + exposure time (no combustion model;
  flashover flagged as an indicator).
- **Mission-Control Dashboard** — one synchronized board reading M1's selection:
  current phase, HRR, layer height, max hazard, critical locations, door status,
  current insight.
- **Adaptive Workspace** — task presets (ventilation / smoke / temperature study)
  that arrange the relevant panels automatically.
- **Space-Time Cube (optional, if stable)** — the 2D slice stack as an (x, z, t)
  volume: scrub, select intervals, read propagation directly.

**Reuse:** `registry`, `tenability`, `events`, the inspector, the page/tab system.
**Difficulty:** low–medium. **Priority: medium.**

---

## Phase 5 — Scientific Communication

Focus on communicating computed evidence.

- **Narrative++** — the V3 Fire Story as an expandable, evidence-backed event
  chain; each node links (via M1) to the moment that produced it.
- **Publication Mode** — one-click journal-styled figure bundles building on the
  V4-M10 presets: fonts, Nature/Elsevier sizing, panel labels, colourblind
  palettes, scale bars, captions, metadata.
- **Assistant++** — extend the bounded V4-M12 assistant: a natural-language
  front-end over the **closed** query grammar (parse → `Query`, never an answer)
  and experiment search. **Hard rule (in code):** physics conclusions come only
  from computed evidence with a `basis`.
- **Ensemble Spread** — spread *across the factorial* (min/mean/max envelopes,
  per-cell variability, ensemble prediction intervals), labelled **parametric
  ensemble spread**, never calibrated stochastic UQ.

**Reuse:** `events`, `figure_export`, `report_builder`, `query_engine`,
`assistant`, the ensemble machinery. **Difficulty:** low–medium. **Priority:
medium.**

---

## Phase 6 — Knowledge Layer

### Research Knowledge Graph ⭐
Now that interaction, analytics, and communication exist, the graph has enough
node types to be genuinely useful (not just linking a handful of entities). It
connects: Experiment → Scenario → Selection → Measurement → Region → Insight →
Notebook Entry → Figure → Report → Publication → Session → Comparison →
Sensitivity Analysis → Narrative → Derived Evidence → Export. A graph-browser
panel: click a node (e.g. "flashover") → every connected scenario, figure, note,
peak, and report; jump from any node into the view that produced it (via M1). The
app becomes laboratory memory. **Reuse:** `experiment.py`, `session_store.py`,
`evidence_notebook.py`, `events.py`, `report_builder`; consumes M1's universal
`Selection`. **Difficulty:** medium–high. **Priority: high (placed late so it
graphs a full ecosystem, not a stub).**

---

## Phase 7 — V6 Preparation (gated — prepare interfaces, do not implement)

Prepare the interfaces (registry entries, provider slots, panel hooks) for the
data-gated capabilities, without implementing them, so they drop in when the data
lands (`docs/msim-preparation.md`):

| Deferred capability | Blocked on |
|---|---|
| 3D streamlines / quiver / volumetric analysis | U/W-VELOCITY components (M-SIM) |
| Multi-plane linked XY / XZ / YZ cross-sections | multi-plane slice output (M-SIM) |
| Full FED / CO / smoke-toxicity hazard | CO output (M-SIM; today temperature-only partial) |
| Validation datasets (sim vs experiment, RMSE, arrival times) | experimental sensor measurements (none exist) |

Ungated optional now: **soot iso-surfaces / volume clipping** on the existing
`.s3d` field (soot only).

---

## Prioritization summary

| Phase | Item | Sci. value | Novelty | Feasible now | Priority |
|---|---|---|---|---|---|
| 0 | Architecture Stabilization | — | — | yes | **prerequisite** |
| 1 | Shared Selection Model (M1) | 5 | 4 | yes | **P1** |
| 2 | Study-Level Analytics (M2) | 5 | 4 | yes | **P1** |
| 3 | Sensitivity Explorer (M3) | 5 | 4 | yes | **P2** |
| 4 | Hazard / Dashboard / Adaptive / Cube | 4 | 3 | yes | **P2** |
| 5 | Narrative / Publication / Assistant++ / Ensemble | 3 | 3 | yes | **P3** |
| 6 | Research Knowledge Graph ⭐ | 5 | 5 | yes | **P1 (late)** |
| 7 | V6 preparation (gated) | — | — | interface only | **doc** |

## Non-negotiable principles (carried from V2/V3/V4)

- Every stated conclusion is a template filled from computed values with a
  traceable `basis`; no invented physics, no asserted causes.
- Honesty gates stay enforced in code and tests: partial-FED, heuristic-
  saliency, association-not-causation, the M-SIM data gate, and V5's new
  **"Estimated from Existing Scenarios"** (sensitivity) and **parametric
  ensemble spread** (uncertainty) labels.
- Every new measurement is deterministic, real-data cross-validated, pinned in
  tests; the suite stays green and the app runnable after every milestone.
- The cinematic Live viewer and all V2/V3/V4 functionality are preserved; the
  Live viewer is migrated to the selection model **last** (highest risk).

**The V5 test:** a researcher clicks one point and the whole study answers — the
moment across every quantity, the scenarios that share it, the factors that move
it, and the evidence already collected — without touching another control.
