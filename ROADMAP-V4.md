# FDS Visualizer V4 — Researcher-Centered Interactive Analysis Environment

**Audience:** executor agents, milestone by milestone. **Consumes:** the shipped V1/FireLab/V2/V3 codebase, `ROADMAP-V2.md`, `ROADMAP-V3.md`, `docs/*`.
**Mission:** the platform can already *show* and *interpret* a simulation. V4 closes the last gap — from *interpretation* to *investigation*. Make every view a place a researcher can **measure, mark, and carry forward as reproducible evidence**, so the tool answers not just "what happened" but "here is the evidence, saved and shareable."

> **This roadmap is deliberately not "more visualizations."** V3 was visualization-and-interpretation heavy. V4 pivots to the *interactive analysis loop*: point/region/height measurement, time as an investigable dimension, linked multi-quantity inspection, persistent annotations, and reproducible analysis sessions. Where a milestone renders something new, it earns its place only by making an *interaction* possible.

---

## 0. The organizing principle: Location → Time → Quantity → Evidence

Every interaction in V4 produces the same four-part object and connects them:

- **Location** — a point, a line, a height, or a named region.
- **Time** — an instant, an interval, or a detected phase.
- **Quantity** — temperature, velocity, HRR, smoke, a derived measure.
- **Evidence** — the computed number/curve/field, plus a *basis* (how it was computed) and a place it can be re-opened.

V3 already built the atom of this: the **`Insight`** (`insight.py`) — a statement bound to when/where/what/why. V4's central move is to make the Insight **persistent, annotatable, and session-backed**: every measurement a researcher takes becomes a saved Insight in an **Evidence Notebook** that travels with the study, exports into reports, and reopens exactly. This is the spine that turns a viewer into an analysis environment.

---

## 1. What already exists (do not rebuild)

V4 stands on a mature platform. Reuse relentlessly; several "asks" in the brief are already shipped:

| Capability area | Already built |
|---|---|
| Playback as analysis | seek/step/speed/loop, synchronized multi-view grid, event markers on the scrubber |
| Point/line/region over time | Time-Series Workspace (`timeseries.py`): click a point/segment/rectangle → curve over time or profile over distance, multi-scenario overlay, CSV export |
| Time phases | Event engine + Fire Story (`events.py`, V3-M2): ignition → growth → fastest-heating → peak → layer-descent → stabilization, click-to-seek |
| Comparison | difference / ensemble / difference-over-time / **Semantic Diff** (physics-aware, ranked, navigable, V3-M3) |
| Ask questions safely | **Physics Query Engine** (V3-M4): NL → *closed deterministic grammar*, never a generated answer |
| Temporal signatures | **Fire MRI** (V3-M1), attention map, cause explorer, state-space + genome |
| Publication + reproducibility | publication figures with provenance, per-scenario & A-vs-B **reports**, headless **CLI**, **session** save/load (layout+cells+time) |
| Quantity framework | the **registry** (`registry.py`): per-quantity units/colormap/scale/hazard-levels/kind |
| Fire-science measures | tenability screening, energy budget, factor-effect maps, smoke-layer height |

**The safe-AI ask (brief §9) is already satisfied** by the query engine's closed grammar; V4 only extends its vocabulary, never its authority.

## 2. What is genuinely missing (V4's real work)

Ranked by the gap between the brief's vision and today:

1. **Height-aware analysis** — *the brief's explicit priority, and the biggest scientific gap.* The app has one derived layer-height series and no interactive vertical tooling. Fire dynamics is vertical (stratification, ceiling jet, plume, doorway flow); a 2D heatmap cannot express it. **This is V4's flagship.**
2. **Persistent, named regions/zones** — region-mean today is a transient time-series probe. Researchers need *named* zones ("doorway", "upper layer", "occupant head height") with a full statistics bundle, reusable across scenarios.
3. **Annotations / bookmarks / notes / tags** — none exist. No way to mark a moment, note an observation, tag a scenario, or carry any of it into a session or report.
4. **Linked multi-quantity inspection** — clicking a temperature peak does not yet show HRR, layer height, and velocity *at that instant* in linked plots. The "inspect this moment across quantities" loop is missing.
5. **Time-window / interval analysis** — no way to select a time window and get interval statistics or a before/after comparison.
6. **Measurement tools** — probe reports coordinate+value, but there is no distance/height measurement or contour extraction on the field.
7. **Experiment management** — the browser sorts by factors but has no tags, groups, notes, or free metadata.
8. **Named, reproducible analysis sessions** — sessions restore a grid but carry no annotations, no named study intent, no saved comparisons/filters.
9. **Quantity-framework breadth** — only TEMPERATURE / VELOCITY / SOOT are registered; pressure, visibility, heat flux, CO, etc. are absent (partly M-SIM-gated on new FDS output, but the framework can be prepared).

---

## 3. The V4 features (investigation → concrete plan)

### 3.1 Height-Aware Analysis Workspace ★ (flagship — brief §2 "prioritize height")

The scientific interpretation a 2D map cannot give. A dedicated workspace where the researcher picks a **vertical line** (an x-column, or drags one) and gets, live and time-scrubbable:

- **Vertical temperature profile** T(z) at the chosen x, at the current time — the canonical stratification plot.
- **Smoke-layer height over time** (reuse `layer_height.py`) with the two-zone interface drawn on the profile.
- **Ceiling-jet analysis** — the near-ceiling temperature band and its horizontal spread over time.
- **Plume-height tracking** — the height the hot core reaches over time (reuse the query engine's plume-height computation).
- **Doorway / opening profile** — T(z) and (when velocity is available) flow direction across a vertical opening: the in/out neutral-plane a ventilation study lives on.
- **Interface / neutral-plane detection** — the height where upper hot layer meets lower cool layer, as a time series.

Each readout is an `Insight` (height + time + quantity + value), click-to-seek, savable to the Evidence Notebook. **Reuse:** `layer_height`, `descriptors`, `signatures`, the extent/coordinate conventions. **Why it beats Smokeview:** Smokeview shows the field; it does not hand you T(z), layer height, and neutral-plane as *analysed, exportable curves*. **Difficulty:** medium. **Priority: highest.**

### 3.2 Named Region / Zone Statistics (brief §2 region-based)

Persistent, named rectangular (later polygon) zones with a full bundle: mean/max temperature, time-to-threshold, heat-exposure integral (thermal dose), smoke accumulation, energy proxy, **hazard duration**, affected-cell fraction — each as a scalar *and* a curve over time, per scenario and **compared across scenarios**. Zones are saved with the session and reused ("apply the doorway zone to Case A and Case B"). **Reuse:** `timeseries` region math, `signatures`, `tenability`. **Difficulty:** medium. **Priority: high.**

### 3.3 Linked Multi-Quantity Inspection (brief §4)

Make a moment inspectable *across quantities at once*. Selecting a time (or clicking a Fire-Story event, or a curve peak) opens a linked panel showing, at that instant: the temperature field, HRR value, smoke-layer height, velocity, and the active-region readout — with a shared time cursor across every plot. The brief's exact example ("a temperature peak → immediately inspect HRR, smoke layer, velocity, affected region") is this feature. **Reuse:** the inspector, `descriptors`, the registry (drives which quantities appear, their units/thresholds). **Difficulty:** medium. **Priority: high.**

### 3.4 Time-Window & Interval Analysis (brief §3)

Promote time to a first-class *selectable* dimension: drag a window on the timeline → interval statistics (mean/peak/integral of any quantity in the window), **before/after** split at a chosen instant, and temporal-trend summaries. The Fire Story's detected phases (already computed) become selectable windows ("analyse the growth phase"). **Reuse:** `events` (phase boundaries), `descriptors`, the timeline widget. **Difficulty:** low–medium. **Priority: high.**

### 3.5 Evidence Notebook + Annotations + Bookmarks (brief §1, §5)

The connective tissue. A dockable notebook where every measurement (point history, region stat, height readout, query answer, diff finding) is captured as a saved `Insight` the researcher can **annotate** (free-text note), **tag**, and reorder. Plus **bookmarks** (named Location+Time+Quantity states) and **event/observation markers** on the timeline. The notebook is saved *in the session* and flows into reports. This delivers the brief's "annotate observations / create analysis checkpoints / bookmarking important moments." **Reuse:** `insight.py`, `session.py` (extend the schema), `report_builder`. **Difficulty:** medium. **Priority: high** (it is what makes analysis *stick*).

### 3.6 Reproducible Named Sessions (brief §5)

Extend sessions from "grid state" to a full **analysis session**: a name and intent ("Door ventilation study — Case A vs B — 120 s"), the Evidence Notebook, named zones, saved comparisons and browser filters, and the time window — all reopenable exactly, and one-click into a **report**. **Reuse:** `session.py` (schema v2), the browser filters, `report_builder`. **Difficulty:** low–medium (mostly serialization). **Priority: high.**

### 3.7 Measurement Tools (brief §6)

On-field scientific measurement: **distance** and **height** measurement (drag a ruler, read Δx/Δz in metres), **coordinate + value inspection** (have it; formalize), **threshold/contour extraction** (isotherms exist; add "extract this contour as coordinates/length"), and region **volume/area statistics** (2D area of the affected zone; true volume gated on `.s3d`). **Valuable-for-FDS verdict:** distance/height and contour extraction are directly useful (flame width, layer depth, plume reach); generic 3D volume waits on the `.s3d` backbone. **Reuse:** `views.py` overlays, extent mapping. **Difficulty:** low–medium. **Priority: medium.**

### 3.8 Advanced Comparison Workflows (brief §7)

Extend Semantic Diff (V3-M3) into the three comparison axes the brief names, all producing navigable `Insight`s:
- **Temporal** — "when did Case B become more dangerous than Case A?" (cross-over time of a hazard metric).
- **Spatial** — "where did heat accumulate differently?" (already the spatial diff Insight; surface it per-region).
- **Physics** — "why higher temperatures?" (rank the driving descriptors: HRR, ventilation, growth rate).
Comparison dimensions: peak T, HRR evolution, smoke height, hazard duration, affected area, fire-growth rate. **Reuse:** `semantic_diff`, `descriptors`, `events`. **Difficulty:** medium. **Priority: medium–high.**

### 3.9 Experiment Management (brief §5)

Add to the browser: free-text **notes**, **tags**, and **groups** per scenario, persisted per study; filter/sort by tag; parameter tracking already exists (the factor columns). **Reuse:** `browser.py`, per-study `.cache` for the sidecar metadata. **Difficulty:** low. **Priority: medium.**

### 3.10 Publication Workflow Completion (brief §8)

Mostly built (figures + provenance + reports + CLI). Add: **export presets** (journal/slide), a **comparison-report** that assembles the Evidence Notebook's findings + figures + provenance, and figure export directly from any analysis panel (height profile, zone stats, linked inspection). **Reuse:** `figure_export`, `report_builder`. **Difficulty:** low. **Priority: medium.**

### 3.11 Quantity-Framework Breadth (brief §4) — partly gated

Register pressure, visibility, heat flux, CO, soot-mass as first-class quantities (units/colormap/thresholds/interpretation) in the registry, so every tool above works on them for free. **Gate:** most require FDS to output them — the **M-SIM** re-run (`docs/msim-preparation.md`) that also unblocks real velocity components and full FED. Prepare the registry entries and the derived-measure definitions now; wire data when M-SIM lands. **Difficulty:** low (framework) + gated (data). **Priority: medium, gated.**

### 3.12 Safe AI Assistant (brief §9) — bounded

Already safe by construction via the query engine. Optional extensions, each of which may only ever *emit a deterministic query or organize computed evidence*, never assert physics: a natural-language front-end over the closed grammar (parse → `Query`, never an answer), **experiment search** ("find scenarios where the doorway exceeded 100 °C"), and **report auto-organization** (order the Evidence Notebook into sections). **Hard rule (restate in code):** physics conclusions come only from computed evidence with a `basis`. **Difficulty:** medium. **Priority: low–medium, optional.**

---

## 4. Researcher-Centered Interactive Workflow *(required section)*

### How researchers analyse FDS today
Run FDS → open Smokeview to watch fields → export frames/slices → write throwaway Python (fdsreader/pandas) to pull curves at points, integrate HRR, estimate layer height → paste numbers and screenshots into a document → repeat per scenario, by hand, per comparison. The interpretation lives in scripts and the researcher's head, not the tool.

### Current pain points
- **The vertical dimension is invisible** in 2D views, yet fire behaviour is vertical — layer height, stratification, ceiling jet, neutral plane all require hand-rolled analysis.
- **Measurements don't persist** — a value read off a plot is gone; there is no record of *where/when/why* it was taken.
- **Comparison is manual and qualitative** — "B looks hotter" instead of "B crossed 100 °C 4 s sooner at the doorway."
- **Nothing is reproducible** — reopening yesterday's analysis means re-deriving it.
- **The output→interpretation gap is a scripting gap** — every question becomes code.

### How this platform improves the workflow
Every question is answered *inside* the tool, as a saved `Insight` bound to Location→Time→Quantity→Evidence: click a height → get the stratification curve; drag a zone → get its hazard duration; select the growth phase → get its interval stats; ask a question → get a computed, located answer; diff two runs → get ranked physics differences. Each is annotatable, bookmarkable, session-saved, and one click from a provenance-stamped figure or report. The researcher moves *simulation → analysis → figure → paper without leaving the app*.

### The most valuable interactive features
1. **Height-Aware Analysis Workspace** — the scientific interpretation Smokeview can't hand you.
2. **Evidence Notebook + annotations** — makes analysis persistent and reproducible.
3. **Linked multi-quantity inspection** — understand a moment, not just see it.
4. **Named zone statistics** — the room/region questions researchers actually ask.
5. **Time-window / phase analysis** — treat time as investigable, not as an animation.

### Recommended implementation order
Height Workspace → Evidence Notebook/annotations → Linked inspection → Named zones → Time-window analysis → Reproducible named sessions → Measurement tools → Advanced comparison → Experiment management → Publication completion → (gated) quantity breadth & AI extensions. *Rationale:* lead with the biggest scientific gap (height), immediately give it persistence (notebook/sessions) so the value compounds, then broaden the interaction surface; leave gated/optional work last.

---

## 5. Prioritization (ranked by all six criteria)

Scores 1–5 (5 = best; difficulty inverted so 5 = easiest).

| Feature | Sci. value | Workflow | Novelty vs Smokeview | Ease | Demo impact | Deps | Verdict |
|---|---|---|---|---|---|---|---|
| **Height-Aware Workspace** (3.1) | 5 | 5 | 5 | 3 | 5 | layer_height/descriptors | **P1 flagship** |
| **Evidence Notebook + annotations** (3.5) | 4 | 5 | 5 | 3 | 4 | insight/session | **P1** |
| **Linked multi-quantity inspection** (3.3) | 5 | 5 | 4 | 3 | 5 | inspector/registry | **P1** |
| **Named zone statistics** (3.2) | 5 | 5 | 4 | 3 | 4 | timeseries/tenability | **P1** |
| **Time-window / phase analysis** (3.4) | 4 | 5 | 3 | 4 | 4 | events/timeline | **P2** |
| **Reproducible named sessions** (3.6) | 3 | 5 | 4 | 4 | 3 | session | **P2** |
| **Measurement tools** (3.7) | 3 | 4 | 2 | 4 | 3 | views/extent | **P2** |
| **Advanced comparison** (3.8) | 4 | 4 | 4 | 3 | 4 | semantic_diff | **P2** |
| **Experiment management** (3.9) | 2 | 4 | 2 | 5 | 2 | browser | **P3** |
| **Publication completion** (3.10) | 3 | 4 | 3 | 5 | 3 | report/figure | **P3** |
| **Quantity breadth** (3.11) | 4 | 3 | 3 | 4 | 2 | **M-SIM gate** | **P3 gated** |
| **Safe-AI extensions** (3.12) | 2 | 3 | 4 | 3 | 4 | query_engine | **P4 optional** |

**Key tradeoffs.** *Height before breadth:* one deep vertical workspace on the quantities we have beats registering quantities we can't yet output. *Notebook before more analyses:* persistence multiplies the value of every existing and future measurement — build it early. *Comparison depth over new views:* extend Semantic Diff rather than add another static map. *Gated work last:* quantity breadth and full FED wait on the M-SIM re-run, already prepared in `docs/msim-preparation.md`.

**The features that make a researcher say "this is not just another FDS viewer — this helps me understand my simulations":** the **Height-Aware Workspace** (interpretation a 2D viewer can't give), the **Evidence Notebook** (analysis that persists and reproduces), and **Linked multi-quantity inspection** (understanding a moment across the physics at once). Ship those three and the platform crosses from viewer to analysis environment.

---

## 6. Architecture impact & non-negotiables

**Extend, do not fork:** `insight.py` (persistence + annotation), `session.py` (schema v2: notebook, zones, filters, window, name), `registry.py` (new quantities/derived measures), `browser.py` (tags/notes), `report_builder`/`figure_export` (notebook → report), `timeseries`/`views` (height + zone + measurement overlays), `semantic_diff` (three comparison axes). New surfaces are Analysis-page panels + a dockable Evidence Notebook, never replacements for the Live viewer.

**Keep frozen:** the validated parser core, `ScenarioStore` LRU/threading (threading changes → report-first), the cinema pipeline, the pages shell, `ml/` isolation.

**Non-negotiable principles (carried from V2/V3):** every measurement is deterministic and physics-based; every stated conclusion carries a computed `basis` (no invented explanations — the `auto_summary`/query-engine rule); every honesty gate (partial-FED, heuristic-saliency, association-not-causation, M-SIM data gate) stays enforced in code and tests; suite green and app runnable after every milestone; real-data cross-validation pinned for every new measurement (the M1.3s/M2.3 precedent).

**Definition of V4 success:** a researcher opens a study, drags a vertical line and reads the stratification and neutral plane, marks the flashover moment with a note, drags a zone over the doorway and gets its hazard duration for Case A *and* Case B, asks when B overtook A, and exports the whole thing — every number computed, located, and reproducible — as a provenance-stamped report, **without writing a line of code or leaving the application.**
