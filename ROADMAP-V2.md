# FDS Visualizer V2 — Research Platform Roadmap (Blueprint for Executor)

**Prepared:** 2026-07-17 · **Author role:** planning agent (no code in this pass) · **Consumes:** `ROADMAP.md` (V1, all core milestones ✅), `ROADMAP-FIRELAB.md` (Phases 1–5 ✅), `docs/spike-parser-validation.md`, `docs/spike-s3d.md`, `docs/decisions.md`
**Mission:** evolve the app from *validated single-study visualizer + cinematic demo* into a **research-grade FDS analysis platform** — the feeling of Smokeview/ParaView/Tecplot, specialized for FDS ensembles. Scientific interpretation > aesthetics. Evolution, not rewrite.

---

## 1. Current State Assessment

### 1.1 Strengths (verified, do not re-litigate)
- **Trusted, fast data spine:** vectorized `.sf` parser cross-validated against `fdsreader` (interior agreement <4 °C max / 0.007 °C mean; per-frame max exact), `.npy` disk cache (cold ≈55–82 ms, warm ≈2–6 ms/scenario), thread-safe LRU keyed by `(scenario, SliceKey)`, extent-aware geometry (`get_extent`, probe verified at all 4 corners).
- **Comparison machinery that matches the dataset's identity:** `ViewGrid` up to 3×3, `DifferenceView` (+ min/max/mean/RMS stats), `EnsembleView` (mean/σ/min/max), linked color scales, synchronized pull-based playback (~57.7 fps real-display on 2×2).
- **Real analysis content, honestly evaluated:** summary-stats index (peak T, time-to-thresholds, peak HRR/energy), PCA/KMeans (candle-count alignment 83.3% — pinned as regression test), deterministic auto-summaries, FNO forecasting that beats persistence at every lead 1–8 with in-app truth/prediction/error view.
- **Two coexisting personalities behind one `PlotView` seam:** science mode (pixel-regression-protected) and cinema mode (FireLUT/bloom/smoke/shimmer). The seam means V2 can invest in science without touching the demo.
- **Engineering discipline as an asset:** 220+ tests, benchmark scripts, decision docs, blit rendering, lazy-loading conventions (the M3.1 startup-regression lesson), demo fallback paths.

### 1.2 Limitations a researcher hits today
1. **Everything is a heatmap.** There is no XY-plot workspace: no T(t) at a probed point, no line profiles, no region-mean-vs-time, no overlaid curves across scenarios. Pro tools treat "plot over time / plot over line" as table stakes; here the only time-series is the inspector's peak-T sparkline.
2. **One study, hardwired.** `manifest.py` parses only `c<n>_d<n>_vod<n>_voc<n>`; browser columns, controls (candle cards/door/vents), schematic, and compare presets all assume the candle factorial. `banner/fds/line_burner.fds` proves a second study is already arriving — today the app cannot open it at all.
3. **One fixed plane.** The y=0 slice is the app's world. The second on-disk plane (y=−0.005, cell-centered) is never surfaced; the `.s3d` full-3D fields (SOOT DENSITY, HRRPUV, TEMPERATURE, CO₂ density — spike: conditional GO) are unread by science mode.
4. **HRR data 90% unused.** `*_hrr.csv` has 12 columns (HRR, Q_RADI, Q_CONV, Q_COND, Q_DIFF, Q_PRES, Q_PART, Q_ENTH, Q_TOTAL, MLR_FUEL, MLR_TOTAL); only peak HRR and total energy are extracted. Radiative fraction, energy budget closure, MLR — core fire-science quantities — are one CSV-read away.
5. **No reproducibility layer.** No session files (view layout/scenario/time/annotations lost on close), no provenance (FDS version from `.out`, input-file hash) attached to exports, no publication-grade figure export (current export = screen PNG / MP4/GIF animation).
6. **Velocity is magnitude-only.** `&SLCF QUANTITY='VELOCITY'` stores speed; streamlines/quiver from real u,w components are impossible without M-SIM re-runs. (Cinema's ∇T pseudo-advection is explicitly not science-grade — keep it out of science mode.)

### 1.3 Technical constraints that shape V2
- **Solo developer; threading rule stands:** changes to `simulation_controller.py`/`scenario_store.py` threading → report first, wait (standing instruction).
- **Matplotlib, not GL:** fine for 2D (57.7 fps measured); any true 3D rendering means adopting a GL stack (pyqtgraph GL/VTK) — a gated decision, not a default.
- **Memory:** 2D dataset ≈230 MB total; **one** `.s3d` quantity for **one** scenario ≈327 MB as float32 but ≈82 MB as stored uint8-levels + per-frame bounds. 3D work must keep the uint8+bounds representation and decode-on-demand, or it breaks the laptop budget.
- **Known defects to clear before publication claims:** edge-column discrepancy vs `fdsreader` (unadjudicated, §0.1 of V1 roadmap); `_pending_load_case` tracks case but not key (bounded race).

---

## 2. Missing Capability Analysis & Professional Tool Investigation

Filtered to what FDS researchers actually use in Smokeview/ParaView/Tecplot/VisIt, mapped to this app. Rejected items listed at the end with reasons.

### 2.1 Scientific visualization
| Capability | Pro-tool analogue | Why it matters for FDS here | Verdict |
|---|---|---|---|
| **Plot-over-time at probe points/regions** | Tecplot time series, ParaView "Plot Data Over Time" | Fire analysis is fundamentally temporal: T(t) at a sensor location, ceiling-jet temperature, upper-layer mean. The dataset (481 frames, tiny arrays) makes this nearly free to compute. | **V2 core (Phase 1)** |
| **Plot-over-line profiles** | ParaView "Plot Over Line" | Vertical T profiles are how fire scientists read stratification/layer height; horizontal profiles show doorway flows. | **V2 core (Phase 1)** |
| **Arbitrary axis-aligned slicing** | Smokeview slice menu, ParaView slice filter | `.s3d` carries the full 3D field; extracting any x/y/z plane turns a fixed-plane viewer into a spatial instrument. Vertical y-slices through the doorway are the money view for vent studies. | **Phase 2, gated on `.s3d` backbone** |
| Isosurfaces / volume rendering | Smokeview 3D smoke, ParaView contour/volume | High interpretive value but requires GL stack + 3D data path. Do the data path first (slicing), defer rendering. | **Phase 4 (gated)** |
| Streamlines / vector glyphs | Tecplot/ParaView vectors | Only honest with true u,w components — needs M-SIM template edit (`U-VELOCITY`,`W-VELOCITY` SLCF) + cluster re-run. Do not fake from ∇T in science mode. | **Phase 3, gated on M-SIM** |
| Measurement + annotation | Tecplot geometry annotations | Distance measure on a slice (plume width, layer depth), text/arrow annotations saved with session → directly reusable in figures. | **Phase 1 (light), session-backed Phase 2** |
| Labeled contours | All | Isotherms exist (M2.6); adding inline °C labels + configurable levels is cheap and reads as "professional" instantly. | **Phase 1 quick win** |

### 2.2 Fire-dynamics analysis
| Capability | Why | Verdict |
|---|---|---|
| **Energy-budget panel** (all 12 `_hrr.csv` columns: HRR, Q_RADI/Q_CONV/... , MLR) | Radiative fraction and budget closure are how modelers sanity-check a run; MLR ties to fuel consumption. Zero new data needed. | **Phase 1, low cost** |
| **Smoke-layer height (N-percent / integral method) as a derived time series** | THE canonical zone-model quantity; computable from existing vertical T columns of the 2D slice per frame. Bridges CFD output to fire-protection-engineering vocabulary. | **Phase 1–2, high science value** |
| **Event timeline** (auto-detected markers on the scrubber: ignition, threshold crossings, peak HRR, steady state) | Researchers scrub to "the important moments"; the stats already exist in `summary_stats.py` — surfacing them on the `TimelineWidget` converts stats into navigation. | **Phase 1 quick win** |
| **Tenability screening** (temperature criteria now; CO₂ from `.s3d` later; honest about missing CO) | Hazard interpretation is the point of these simulations. Partial FED is publishable if the limitation is stated; the app must not imply full FED without CO. | **Phase 3** |
| Fire-growth characterization (αt² fit on HRR curve) | Standard descriptor; trivial fit; feeds auto-summary + browser column. | **Phase 1, opportunistic** |

### 2.3 Simulation comparison (this dataset's identity)
| Capability | Why | Verdict |
|---|---|---|
| **Factor-effect field maps** (main-effect / interaction maps over the 2×2×3×2 factorial: e.g. mean field difference "wide − narrow door" averaged over all other factors, per frame) | This is ANOVA lifted to field level — the single analysis this *factorial* dataset was built for, and something none of the generic tools give you out of the box. M2.3's finding (door effect visible in VELOCITY, not TEMPERATURE) becomes systematic instead of anecdotal. | **Phase 3 flagship** |
| **Ensemble curve overlays** ("spaghetti plots": peak-T(t) or layer-height(t) for N scenarios, colored by factor) | Cheapest possible uncertainty/spread visualization for a deterministic ensemble; complements the existing field-level σ view. | **Phase 1** |
| Difference time-series (RMS/max Δ vs t, curve not just per-frame number) | M2.3's difference stats exist per frame; plotting them over time shows *when* scenarios diverge — often the actual research question. | **Phase 1, extends existing code** |
| Scenario ranking | Already effectively exists (sortable browser); add derived columns (growth rate, layer height minimum, time-to-untenable). | **Phase 1, opportunistic** |
| Forecast-uncertainty (deep ensemble / MC dropout on FNO) | Required for the ML story's credibility in a paper; not needed for the platform itself. | **Phase 4 / research track** |

### 2.4 Research workflow
| Capability | Why | Verdict |
|---|---|---|
| **Publication figure export** (SVG/PDF vector, journal-preset sizes, font control, colorbar labels, provenance caption) | The gap between "screenshot" and "Figure 3"; matplotlib already renders everything, so this is configuration + a dialog, not new rendering. | **Phase 1 core** |
| **Session files** (JSON: layout, per-cell scenario/quantity/type, time index, annotations, link state) | Reproducibility of *views* — reopening yesterday's comparison exactly. Also the substrate for "story presets" already hand-coded in Compare. | **Phase 2 core** |
| **Provenance layer** (parse `.out` for FDS version/runtime; hash input `.fds`; stamp exports + reports) | Reviewer question #1 is "which FDS version?" — the answer is already on disk, unparsed. | **Phase 1, low cost** |
| **Multi-study workspace** (open any FDS output dir; per-study manifest schema; line-burner case as the forcing function) | Without it, V2 is a candle-study app with a roadmap. With it, it's a lab instrument. | **Phase 0 groundwork, Phase 2 completion** |
| **Automated analysis report** (HTML/PDF per scenario or comparison: figures + stats + auto-summary sentences + provenance) | Turns an afternoon of figure assembly into a click; `auto_summary.py` + `summary_stats.py` + the figure exporter are 80% of the parts. | **Phase 3** |
| **Headless CLI / Python API** (`python -m fdsviz export|stats|report ...`) | ParaView's pvpython is why it's scriptable lab infrastructure; the layered architecture (store/analytics importable without Qt) makes this cheap. | **Phase 3** |

### 2.5 Explicitly rejected
- **Full volume rendering in matplotlib** — wrong tool; only via a GL stack, only after `.s3d` backbone proves demand (Phase 4 gate).
- **Zarr/SQLite/out-of-core now** — 2D data is 230 MB; `.s3d` handled by uint8+bounds + per-scenario LRU. Revisit only if multi-study 3D breaks RAM.
- **Generic dashboarding / KPI tiles** — explicitly not the identity.
- **PINNs, diffusion, RL, generative anything** — V1 analysis stands (24 runs; no fit).
- **Plugin API** — still no second contributor; keep the registry-dict pattern.
- **Streamlines from |v| or ∇T in science mode** — would be fabricated data in a research tool; hard no until M-SIM.

---

## 3. New Feature Discovery (beyond the existing list)

| # | Feature | Scientific problem solved | Complexity | Depends on | Priority |
|---|---|---|---|---|---|
| F1 | **Time-Series Workspace** — point/line/region probes producing linked XY plots over time; multi-scenario overlay; CSV export | Temporal analysis (the core of fire dynamics) currently impossible in-app | Med | DataKey generalization (Phase 0) | **Must Have** |
| F2 | **Factor-Effect Field Maps** — per-factor main-effect and 2-factor interaction fields across the factorial, playable like any slice | Systematic answer to "what does each design variable do, where, and when"; the dataset's raison d'être | Med | F7 derived-field pipeline | **Must Have** |
| F3 | **Event Timeline on scrubber** — auto-detected markers (thresholds, peak HRR, growth phases) clickable to seek | Finding "the important moments" in 481 frames × 24 scenarios | Low | summary_stats (exists) | **High Value** |
| F4 | **Energy-Budget & MLR panel** — full `_hrr.csv` visualization, radiative fraction, budget-closure check | Run sanity-checking and heat-transfer interpretation; data already on disk | Low | none | **High Value** |
| F5 | **Smoke-Layer Height derived series** (+ optional overlay line on the slice) | Zone-model-vocabulary output from CFD data; hazard interpretation | Low–Med | F1 plotting | **High Value** |
| F6 | **Publication Figure Exporter** — vector SVG/PDF, journal presets, provenance-stamped | Figures for papers without leaving the app | Low–Med | provenance parse (cheap) | **Must Have** |
| F7 | **Derived-Field Pipeline** — derived quantities (Δ fields, factor effects, time-aggregates, layer height) as first-class cache keys flowing through the existing `ScenarioStore` machinery | One abstraction powering F2/F5/§2.3 without new storage code | Med | Phase 0 DataKey | **Must Have (enabler)** |
| F8 | **`.s3d` Science Backbone** — purpose-built regular-grid stitcher (spike showed `fdsreader.to_global()`'s 1.7–5 s is its general-case machinery, not the decode), uint8+bounds memory model, quantity registry entries for SOOT/HRRPUV/CO₂ | Unlocks 3 unused physical fields + any-plane slicing; the app's biggest data expansion at zero simulation cost | Med–High | spike (done, conditional GO) | **Must Have** |
| F9 | **Any-Plane Slice Extraction** — choose axis + offset from the 3D field; vertical doorway slices | Spatial investigation instead of a fixed window | Med | F8 | **High Value** |
| F10 | **Session Files** — save/restore full workspace state; presets become shareable files | Reproducibility of analyses across days/machines/colleagues | Med | none | **High Value** |
| F11 | **Comparison Report Builder** — one-click HTML/PDF: figures, stats tables, auto-summary prose, provenance | Scientific communication; assembles existing parts | Med | F6, F1 | **High Value** |
| F12 | **Headless CLI/API** — batch stats/export/report without GUI | Lab automation, cluster post-processing, CI of simulations | Low–Med | F11 helps | **High Value** |
| F13 | **Multi-Study Workspace** — open arbitrary FDS output dirs; pluggable manifest schema; line-burner as first guest study | Generalization from "candle app" to "FDS platform" | Med–High | Phase 0 groundwork | **Must Have (strategic)** |
| F14 | **True Vector Fields + Streamlines** — after M-SIM adds U/W slices on cluster re-run | Convective-structure analysis done honestly | Med (+sim time) | M-SIM (gated) | **Future Research** |
| F15 | **3D View (GL)** — isosurfaces/volume of `.s3d` fields | Full spatial context | High | F8 + GL-stack gate | **Future Research** |
| F16 | **Forecast uncertainty + error-vs-lead-time panel** | ML credibility for publication | Med | ml/ (exists) | **Future Research** |

Decorative candidates considered and rejected: animated transitions in science mode, more cinema effects, 3D-looking 2D embellishments — all fail the "improves interpretation/analysis/comparison/reproducibility/communication" test.

---

## 4. Future Product Vision

**Identity:** *the lab-standard interactive instrument for parametric FDS studies* — ParaView-lite specialized for FDS ensembles, with an integrated, honestly-evaluated ML loop, and a cinematic public mode kept one keystroke away for outreach. Not a Smokeview replacement for arbitrary geometry walkthroughs; the differentiator is **ensemble comparison + factorial analysis + reproducible reporting**, which no mainstream tool does for FDS.

**The V2 researcher workflow:**
open study (any FDS output dir) → browser shows computed summaries → load scenario(s) into grid → scrub via event markers → probe points/lines, watch linked time-series → switch plane/quantity (incl. 3D-derived) → build factor-effect or difference views → annotate → export publication figures / one-click report → save session; rerun any of it headless from the CLI.

**Primary use cases:** ventilation-effect quantification (factor maps + doorway slices), hazard/tenability screening, simulation QA (energy budget, provenance, parser-validated data), ensemble papers (figures + reports), ML-forecasting research (existing FNO loop), second-study onboarding (line burner).

---

## 5. Prioritized Feature Matrix

Scores 1–5 (5 = best/highest). Complexity inverted (5 = cheapest).

| Feature | Science impact | User value | Cheapness | Dependency risk | Priority |
|---|---|---|---|---|---|
| F7 Derived-field pipeline | 4 | 3 | 3 | low | **P0 (enabler)** |
| F1 Time-series workspace | 5 | 5 | 3 | low | **P1** |
| F4 Energy-budget panel | 4 | 4 | 5 | none | **P1** |
| F3 Event timeline | 3 | 5 | 5 | none | **P1** |
| F6 Publication exporter | 4 | 5 | 4 | none | **P1** |
| F5 Layer height | 4 | 4 | 4 | F1 | **P1–2** |
| F8 `.s3d` backbone | 5 | 4 | 2 | med (perf/memory) | **P2** |
| F9 Any-plane slicing | 4 | 4 | 3 | F8 | **P2** |
| F10 Session files | 3 | 4 | 3 | low | **P2** |
| F13 Multi-study | 4 | 4 | 2 | med (touches many files) | **P0 groundwork, P2 completion** |
| F2 Factor-effect maps | 5 | 4 | 3 | F7 | **P3 flagship** |
| F11 Report builder | 4 | 4 | 3 | F6 | **P3** |
| F12 CLI/API | 3 | 4 | 4 | low | **P3** |
| Tenability module | 4 | 3 | 3 | F8 (CO₂); honesty constraint | **P3** |
| F14 Vectors/streamlines | 4 | 3 | 2 | **M-SIM gate** | **P3–4 (gated)** |
| F15 3D GL view | 3 | 4 | 1 | F8 + GL gate | **P4 (gated)** |
| F16 Forecast uncertainty | 3 | 2 | 3 | low | **P4** |

**Key tradeoffs:**
- *F8 before F2?* No — factor maps need only 2D slices; F2 is deliberately not blocked on 3D work.
- *Multi-study now or later?* Split: Phase 0 removes new hardcoding cheaply (interfaces stop assuming the factorial); the full workspace waits until Phase 2 so Phase 1's science wins land first. Cost of splitting is low; cost of retrofitting after Phases 1–3 would be high.
- *GL stack:* adopt only when (a) F8 has shipped, (b) a concrete 3D question exists that slices can't answer. Same discipline as the M2.4 pyqtgraph gate — and that gate's outcome (don't migrate) is precedent, not prejudice.

---

## 6. Phased Development Roadmap

Effort assumes the same solo developer + agent execution style as V1. Phases are sequential; tasks within a phase can interleave.

### Phase 0 — Foundation (≈1 wk)
**Goal:** clear correctness debt and put in the two abstractions everything else keys on. No visible features; everything after gets cheaper.
- **V2-M0.1 Edge-column adjudication** (timeboxed spike, 1 d): resolve V1 defect §0.1 (our parser vs `fdsreader` boundary column) using FDS file-format docs and/or Smokeview if installable; fix whichever side is wrong; re-run cross-validation. *Gate: publication-facing features (F6, F11) cite this.* Also fix the `_pending_load_case` case-vs-key race (bounded, known, cheap).
- **V2-M0.2 Quantity Registry + DataKey**: formalize `config.QUANTITY_DISPLAY` into a registry keyed by FDS quantity name (units, colormap, clim policy, hazard levels, kind: slice2d | series | derived | volume). Generalize `SliceKey` to a `DataKey` union so derived fields (F7) and 3D extracts (F9) flow through `ScenarioStore`'s existing LRU/disk-cache **without new storage code**. Mechanical, backward-compatible parameter threading — same pattern as M2.1; the threading-rule (report-first) applies to any lock changes.
- **V2-M0.3 Study abstraction groundwork**: introduce a `Study` object (root dir, manifest strategy, geometry) owned by the shell; route `manifest.py`/`data_provider.py`/browser through it. The candle factorial becomes *a* `FactorialStudy` implementation, not *the* app. Do **not** build the open-study UI yet.
- **Risks:** M0.2 touches the store's key type — keep the default-key parity trick from M2.1 (all old call sites work unchanged). M0.3 must not disturb the 220-test suite; extract mechanically.

### Phase 1 — Scientific Usability (≈2.5 wks) — low risk, highest researcher value per day
**Goal:** a researcher can *measure and plot*, not just look.
- **V2-M1.1 Time-Series Workspace (F1)**: new `TimeSeriesView` implementing `PlotView` (the protocol's first non-heatmap member — proves the seam). Click-to-place probes on any slice cell → linked T(t)/|v|(t) curves; line-profile tool (drag a segment → profile plot with frame scrub); region-mean rectangles. Multi-scenario overlay (spaghetti, colored by factor). CSV export of any curve. *This is the phase centerpiece.*
- **V2-M1.2 Energy-Budget panel (F4)**: full `_hrr.csv` parse into `summary_stats`' cache pattern; stacked Q_* budget plot + MLR + radiative-fraction; αt² growth fit as a browser column and auto-summary sentence.
- **V2-M1.3 Event Timeline (F3)**: markers on `TimelineWidget` from existing stats (time-to-100/300/600 °C, peak HRR, growth-phase boundaries from the αt² fit); click = seek. Registry-driven per quantity.
- **V2-M1.4 Publication Exporter (F6)**: export dialog producing vector SVG/PDF at journal presets (single/double column), font/DPI control, colorbar + axis labels from the registry, optional provenance footer (FDS version from `.out`, scenario id, git-ignored-data hash). Includes labeled isotherm contours (cheap upgrade to M2.6's overlay).
- **V2-M1.5 Difference-over-time + ranking columns**: plot M2.3's per-frame RMS/max Δ as curves; add derived columns (growth rate, layer-height min once F5 lands) to the browser.
- **Why now:** all of it rides on cached 2D arrays + matplotlib; nothing is gated; every item converts an existing computation into researcher-visible capability.
- **Risks:** UI sprawl on the Analysis/Live pages — anchor the workspace on the Analysis page, keep Live lean.

### Phase 2 — Advanced Visualization Workspace (≈3 wks)
**Goal:** escape the fixed plane; make analyses reproducible artifacts.
- **V2-M2.1 `.s3d` Science Backbone (F8)**: purpose-built regular-grid stitcher (the spike showed decode is ~64 ms/submesh-series; the 1.7–5 s cost was `fdsreader`'s general-case stitching — ours is a known-regular 24-block grid). Memory model: keep uint8 levels + per-frame `.sz` bounds (≈82 MB/quantity/scenario), rescale on extraction only; active-scenario-only LRU. Registry entries for SOOT DENSITY, HRRPUV, CO₂. **Verification bar (M1.3s standard):** cross-check the extracted y=0 TEMPERATURE plane from `.s3d` against the trusted `.sf` slice — a built-in ground truth no other feature has; investigate the spike's open flag (SMOKG3D temperature bounds all-zero for one scenario) before trusting that channel.
- **V2-M2.2 Any-plane slicing (F9)**: axis+offset selector per cell; vertical doorway slice as the demo case; probe/isotherms/extents work unchanged through the same view code.
- **V2-M2.3 Smoke-layer height (F5)**: derived series via F7 pipeline (N-percent rule on vertical T columns); overlay line on slices; browser column; auto-summary sentence.
- **V2-M2.4 Session files (F10)**: JSON schema (layout, per-cell DataKey + view type, time, annotations, link state, study ref); File→Save/Load Session; Compare-page "story presets" become session files on disk instead of code.
- **V2-M2.5 Multi-study completion (F13)**: Open Study dialog; per-study cache/summary dirs; browser adapts columns to the study's manifest schema; line-burner study loads end-to-end (single scenario, no factors — degenerate manifest must be first-class, not an error).
- **Risks:** F8 performance on first full-domain decode (mitigate: background decode + progress, per-plane extraction laziness); multi-study touching browser/controls (mitigate: candle study remains default; guest study is read-only viz first — no schematic/controls required for it).

### Phase 3 — Advanced CFD Analysis (≈3 wks)
**Goal:** analyses that produce paper-ready claims.
- **V2-M3.1 Factor-Effect Field Maps (F2)**: main-effect fields per factor (mean over all other factors, per frame, per quantity) + optional 2-factor interaction; rendered as a playable diverging-map view via F7 keys; companion ANOVA-style table (effect magnitudes, integrated over time/space) in the browser/Analysis page. Validate against M2.3's pinned findings (door → VELOCITY effect).
- **V2-M3.2 Tenability screening**: temperature tenability criteria (configurable thresholds/heights) + CO₂ from F8; explicit "no CO output in this dataset — partial FED" labeling; time-to-untenable map + browser column. *Feeds M-SIM wishlist: add CO/soot slices.*
- **V2-M3.3 Report Builder (F11)**: per-scenario and A-vs-B HTML (PDF via print stylesheet) assembling F6 figures, stats tables, auto-summary prose, provenance block.
- **V2-M3.4 Headless CLI (F12)**: `python -m fdsviz stats|export|report|session-render` against a study dir; reuses everything; smoke-tested in CI-style pytest.
- **V2-M3.5 (gated) M-SIM execution**: cluster re-run with U/W-VELOCITY (+CO, soot slices) per the V1 gate (parser validation is solid → gate condition met once M0.1 closes); then **F14 streamlines/quiver** on real components.
- **Risks:** M3.1 statistical framing (keep it descriptive — effect fields + magnitudes, not p-values on n=1 deterministic runs); M3.5 external dependency (cluster access) — everything else in the phase is independent of it.

### Phase 4 — Research Platform (post-V2, directional)
- 3D GL view (isosurface/volume of F8 fields) behind an M2.4-style timeboxed gate; pyqtgraph-GL first candidate.
- Forecast uncertainty (deep ensemble) + error-vs-lead-time panel; FNO on new M-SIM quantities.
- Continuous-parameter sweeps → surrogates/inverse design (needs new simulation campaigns).
- Reader registry promotion (fdsreader as alternate backend), web companion viewer, collaboration features — only with a second contributor.

---

## 7. Architecture Impact Analysis

**Keep unchanged (non-negotiable):** `fds/slice/slice.py` parse core (post-M0.1 fix), `scenario_store.py` LRU/disk-cache mechanics, `time_controller.py`, blit path, `pages/` shell + nav, cinema pipeline (frozen — demo mode is done), `ml/` isolation (torch never an app dep), test/benchmark discipline.

**Extend:**
- `views.py`: first non-heatmap `PlotView` (`TimeSeriesView`), plane selector on cells, overlay hooks (layer-height line). The protocol was built for exactly this.
- `scenario_store.py`/`load_data.py`: key type widened (`DataKey`), new loaders registered — **no changes to locking/threading semantics** (report-first rule otherwise).
- `summary_stats.py`: full HRR columns, αt² fit, layer-height stats — same cache/invalidations pattern.
- `browser.py`: schema-driven columns (study-provided), new derived columns.
- `export.py`: figure-spec export alongside animation export.
- `auto_summary.py`: sentences for budget/growth/layer/tenability — same deterministic-template rule (zero free generation).

**New abstractions (the complete list — resist inventing more):**
1. `DataKey` + quantity **Registry** (M0.2) — one identity system for measured, derived, and 3D-extracted fields.
2. **Derived-field pipeline** (F7) — pure functions `(inputs: [DataKey], params) → array`, cached through the store.
3. `Study` (M0.3/M2.5) — dataset root + manifest strategy + geometry provider.
4. `fds/s3d/` reader + regular-grid stitcher (F8) — sibling of `fds/slice/`, same validation standard.
5. `Session` schema (F10) and `ReportBuilder` (F11) — serialization layers over existing state, no new runtime state.

---

## 8. Risks and Tradeoffs
1. **`.s3d` memory/perf** is the only high-complexity engineering in V2 — contained by uint8+bounds, active-scenario-only caching, and the `.sf` cross-check giving an unambiguous correctness oracle. Timebox the stitcher like M2.4; fallback is shipping SOOT/HRRPUV at y=0 only (still 3 new quantities).
2. **Scientific-honesty hazards** (the expensive kind of bug): no streamlines before real u,w; partial-FED labeling; factor maps described as descriptive effects of a deterministic ensemble; adjudicate the edge column before provenance-stamped figures. Each is encoded as a gate above, not a norm to remember.
3. **Multi-study scope creep** — guest studies get visualization + browser only; controls/schematic/cinema stay candle-study features. Say this in the UI ("study features unavailable for this dataset") rather than generalizing three more subsystems.
4. **Solo-dev bandwidth** — Phase 1 is deliberately cuttable-from-the-bottom (M1.5 first, then M1.3); Phase 2's M2.5 can slip to Phase 3 without breaking dependencies; F2 must not slip (it's the scientific headline).
5. **Two-personality drift** — cinema mode is feature-frozen in V2; any demo request lands in a `ROADMAP-FIRELAB` addendum, not in these phases.

---

## 9. Executor Agent Brief

**Mission:** ship Phases 0–3 above, in order, keeping the app runnable and the suite green at every commit, evolving the validated V1/FireLab codebase into a research-grade FDS ensemble-analysis platform.

**Non-negotiable constraints:**
- Never rewrite: parser core, store LRU/threading, TimeController, pages shell, cinema pipeline. Extend through the seams named in §7.
- `simulation_controller.py`/`scenario_store.py` threading issues: **report first and wait — even mid-task, even for a small confident fix** (standing instruction).
- Branch before the first edit of each milestone (`feat/v2-m<phase>.<n>-<slug>`); commit `ROADMAP-V2.md` updates in the same session they're made.
- Every new data path gets M1.3s-standard validation (independent cross-check, decision doc if a spike) before UI wiring; every claim in a DoD is measured, not assumed — real-display numbers for anything performance-related (the M2.4 lesson).
- Real-data findings that contradict the plan's examples get reported honestly and pinned as tests (the M2.3/M3.1 precedent).
- `ml/` deps never leak into the app's requirements.

**Implementation order (exact):** V2-M0.1 → M0.2 → M0.3 → M1.1 → M1.2 → M1.3 → M1.4 → M1.5 → M2.1 → M2.2 → M2.3 → M2.4 → M2.5 → M3.1 → M3.2 → M3.3 → M3.4 → (M3.5 when cluster access lands — may run in parallel from Phase 2 onward once M0.1 closes the validation gate).

**First three milestones in detail:**
1. **V2-M0.1** (1–1.5 d): adjudicate + fix the edge-column defect; fix `_pending_load_case` key race; re-run fdsreader cross-validation; update `docs/spike-parser-validation.md` with the resolution.
2. **V2-M0.2** (2 d): quantity Registry + `DataKey`; default-key parity so all existing call sites/tests pass unchanged; disk-cache filenames extended the same way M2.1 did.
3. **V2-M1.1** (4–5 d): `TimeSeriesView` + probe/line/region tools + multi-scenario overlays + CSV export, hosted on the Analysis page; first non-heatmap `PlotView` proves the protocol.

**Validation criteria for "V2 succeeded":**
- A researcher can: open the line-burner study and view its slices; probe any point and export T(t) to CSV; read the full energy budget of any candle scenario; jump between auto-detected events; produce a provenance-stamped vector figure; extract a vertical doorway slice from `.s3d` data that cross-checks against the `.sf` ground truth; render the door-width main-effect field and see the VELOCITY-borne effect M2.3 pinned; save a session, reopen it identically, and regenerate its report from the CLI.
- Suite green throughout; no startup-time regression (M3.1 lesson: everything new is lazy/background); science mode's existing pixel-regression tests never change.
