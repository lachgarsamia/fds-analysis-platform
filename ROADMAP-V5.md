# ROADMAP V5 — From Interactive Analysis Environment to Computational Fire Scientist

> **The theme is not more analyses. It is connection.** V4 shipped a dozen
> analyses that do not talk to each other, and a 24-scenario factorial that is
> still explored one run at a time. V5's question is the one that matters after
> V4: *what does a researcher still do by hand?* The answer is switch panels,
> re-find context, and compare scenarios manually. V5 removes that work by
> connecting three things — the panels (live), the artifacts (memory), and the
> scenarios (study-level science).

## 0. Organizing principle: Connection at three levels

1. **Live** — one *selection* (scenario · point (x,z) · time · phase · quantity)
   is broadcast to every panel. Click the doorway once; the heatmap marker,
   the temperature curve, the HRR cursor, the layer height, the Fire-MRI
   signature, the timeline, the notebook, and the comparison view all update
   together. No manual switching.
2. **Memory** — every artifact V4 produces (experiment → scenario → event →
   measurement → insight → figure → report) becomes a node in a navigable
   **knowledge graph**. Click "flashover" → every scenario, figure, and note
   connected to it. The app becomes laboratory memory.
3. **Study** — the candle factorial (candles × door × vod × voc) is finally
   used as a *designed experiment*: factor influence, response surfaces,
   parameter relationships, outliers, and honest ensemble spread.

Everything in V5 is buildable on the **current** data (2D TEMPERATURE +
VELOCITY-magnitude slices, `.s3d` soot volume, HRR CSV, 24 scenarios). Nothing
here waits on the M-SIM re-run.

---

## 1. The V5 spine (flagships)

- **Shared Selection Model (M1)** — the connective tissue for *live* linking.
- **Research Knowledge Graph (M2)** — the connective tissue for *memory*.
- **Study-Level Analytics (M3) + Sensitivity Explorer (M4)** — the connective
  tissue for *the study*.

Ship those four and the platform crosses from "an environment with many tools"
to "one instrument that reasons about a whole study."

---

## 2. Milestones

### V5-M1 — Shared Selection Model ★ (flagship, live linking)
A single `Selection` (scenario, point, time, phase, quantity) with a broadcast
bus; panels subscribe and react (highlight the point, seek the time, mark the
phase). A selection is itself savable to the notebook and restorable from a
session. **Reuse:** every existing panel's seek/marker code, `insight.py`,
`session.py`. **Difficulty:** medium (wiring, one shared model). **Priority:
highest** — it multiplies the value of every V4 panel.

### V5-M2 — Research Knowledge Graph ★ (flagship, memory)
A deterministic graph over objects that already exist: Experiment → Scenario →
Event → Measurement → Insight → Figure → Report, plus tags (e.g. "flashover",
"door-open"). A graph-browser panel: click a node → its neighbours; filter by
tag; jump from any node into the view that produced it (via M1's selection).
**Reuse:** `experiment.py`, `session_store.py`, `evidence_notebook.py`,
`events.py`, `report_builder`. **Difficulty:** medium. **Priority: high** — it
makes every prior and future milestone compound.

### V5-M3 — Study-Level Analytics (the factorial as a designed experiment)
Treat all 24 scenarios at once: a parameter/response table, **factor influence**
(extend `factor_effects`), **parallel coordinates**, a **correlation explorer**,
and **outlier detection** on scenario descriptors. **Reuse:** `factor_effects`,
`state_space`, `summary_stats`, `descriptors`. **Difficulty:** medium.
**Priority: high.**

### V5-M4 — Sensitivity Explorer
"What if ventilation changed / the door opened earlier?" — estimated from the
*existing* runs, never a new simulation: local response surfaces over factor
levels, parameter sliders that interpolate across the factorial, and factor-
influence rankings. **Honesty gate:** every readout is labelled *estimated from
existing scenarios by interpolation*, not a simulated result. **Reuse:** M3's
response surfaces, the factor axes. **Difficulty:** medium. **Priority:
medium–high.**

### V5-M5 — Hazard Spaces
Replace bare isotherms with a dynamic hazard classification —
Safe / Warning / Critical / Untenable / (flashover-indicator) — as a per-frame
map and a hazard-state timeline, from temperature thresholds and exposure time.
**Honesty:** flashover is an *indicator* from temperature criteria, flagged as
such (no combustion model). **Reuse:** `registry` hazard bands, `tenability`,
`events`. **Difficulty:** low–medium. **Priority: medium–high.**

### V5-M6 — Scientific Dashboard (mission control)
One synchronized screen reading from M1's selection: current phase, HRR, layer
height, maximum hazard, critical locations, door status, and the current
insight — for the selected scenario/time. Not dozens of windows; one board.
**Reuse:** the inspector, descriptors, M1, M5. **Difficulty:** low–medium.
**Priority: medium.**

### V5-M7 — Space-Time Cube (optional, feasible)
The 2D slice stack as an (x, z, time) volume: scrub, select time intervals, and
read propagation directly. **Reuse:** the slice store, `time_window`.
**Difficulty:** medium. **Priority: medium, optional.**

### V5-M8 — Scientific Narrative++
Extend the V3 Fire Story into an expandable, evidence-backed chain (ignition →
ceiling jet → layer descent → untenable → peak HRR → cooling); each node links
to its evidence and, via M1, to the moment that produced it. **Reuse:**
`events`, `insight`, M1, M2. **Difficulty:** low–medium. **Priority: medium.**

### V5-M9 — Adaptive Workspace
Task presets — ventilation / smoke / temperature study — that arrange the
relevant panels automatically (velocity+pressure-proxy+door profiles;
layer+visibility+MRI; thermal dose+isotherms+MRI). **Reuse:** the page/tab
system, sessions. **Difficulty:** low. **Priority: medium.**

### V5-M10 — Publication Mode
One-click journal-styled figure bundles building on V4-M10 presets: consistent
fonts, Nature/Elsevier sizing, panel labels, colourblind palettes, scale bars,
captions, and metadata. **Reuse:** `figure_export`, `report_builder`.
**Difficulty:** low. **Priority: medium.**

### V5-M11 — Assistant++ (still bounded)
Extend the V4-M12 assistant: a natural-language front-end over the **closed**
query grammar (parse → `Query`, never an answer) and **experiment search**
("scenarios where smoke descended below 2 m"). **Hard rule (restated in code):**
physics conclusions come only from computed evidence with a `basis`. **Reuse:**
`query_engine`, `assistant`. **Difficulty:** medium. **Priority: medium.**

### V5-M12 — Ensemble Spread (honest uncertainty)
Spread *across the factorial*, not stochastic UQ: min/mean/max envelopes and
per-cell variability across scenario groups, and prediction-style intervals from
the ensemble. **Honesty gate:** labelled *parametric ensemble spread across the
designed scenarios*, never a calibrated stochastic uncertainty. **Reuse:** the
ensemble machinery, `summary_stats`. **Difficulty:** medium. **Priority: low–
medium.**

---

## 3. Deferred to V6 (data-gated — documented, not built)

These are valuable but **cannot be honestly built on the current output**; they
are recorded here with their data prerequisite so they are ready when the data
lands (see `docs/msim-preparation.md`).

| Deferred theme | Blocked on |
|---|---|
| True 3D streamlines / quiver / volume rendering | U/W-VELOCITY components (M-SIM) |
| Linked XY / XZ / YZ cross-sections | multi-plane slice output (M-SIM) |
| Validation toolkit (sim vs experiment, RMSE, arrival times) | experimental sensor measurements (none exist) |
| Full-FED hazard spaces (CO/CO₂) | CO output (M-SIM; today: temperature-only partial) |

Ungated 3D that *may* ship as optional V5 work: **soot iso-surfaces / volume
clipping** on the existing `.s3d` field (soot only).

---

## 4. Prioritization

Order: **Shared Selection (M1) → Knowledge Graph (M2) → Study-Level Analytics
(M3) → Sensitivity (M4) → Hazard Spaces (M5) → Dashboard (M6)**, then narrative,
adaptive workspace, publication mode, assistant++, ensemble spread, and the
optional space-time cube. *Rationale:* lead with the two connective flagships so
every existing and future analysis immediately compounds; then make the 24-run
factorial a real designed experiment; then layer interpretation (hazard,
dashboard) on top of the now-shared state.

| Feature | Sci. value | Novelty vs Smokeview | Feasible now | Priority |
|---|---|---|---|---|
| Shared Selection Model (M1) | 5 | 4 | yes | **P1** |
| Knowledge Graph (M2) | 5 | 5 | yes | **P1** |
| Study-Level Analytics (M3) | 5 | 4 | yes | **P1** |
| Sensitivity Explorer (M4) | 5 | 4 | yes | **P2** |
| Hazard Spaces (M5) | 4 | 3 | yes | **P2** |
| Scientific Dashboard (M6) | 4 | 3 | yes | **P2** |
| Narrative++ / Adaptive / Pub / Assistant++ | 3 | 3 | yes | **P3** |
| Space-Time Cube (M7) | 3 | 4 | yes | **P3, opt** |
| Ensemble Spread (M12) | 3 | 3 | yes | **P3** |

## 5. Non-negotiable principles (carried from V2/V3/V4)

- Every stated conclusion is a template filled from computed values with a
  traceable `basis`; no invented physics, no asserted causes.
- Honesty gates stay enforced in code and tests: partial-FED, heuristic-
  saliency, association-not-causation, the M-SIM data gate, and V5's new
  *estimated-from-existing-scenarios* (sensitivity) and *parametric-ensemble-
  spread* (uncertainty) labels.
- Every new measurement is deterministic, real-data cross-validated, and pinned
  in tests; the suite stays green and the app runnable after every milestone.
- The cinematic Live viewer and all V2/V3/V4 functionality are preserved.

**The V5 test:** a researcher clicks one point and the whole study answers — the
moment across every quantity, the scenarios that share it, the factors that move
it, and the evidence already collected — without touching another control.
