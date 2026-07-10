# FDS Visualizer — Research Platform Roadmap & Execution Plan

**Prepared:** 2026-07-08 · **Updated:** 2026-07-09 (parser-validation spike + conditional simulation-regeneration milestone added); **2026-07-09b** (fdsreader pinned to 10.1, single-simulation validation scope confirmed; new GUI-realism milestone M1.6 added; M-SIM re-gated on cluster access, sequenced before FDS template modeling work); **2026-07-09c** (M1.6 scope sharpened: non-specialist audience, extent-proportioned schematic, plain-language labels + explainers elevated to core, effort 1.5–2 d → 2–2.5 d; GUI+validation-before-template-edits sequencing gate made explicit) · **Demo deadline:** 2026-09-11 (~9 weeks) · **Team:** 1 developer
**Audience:** this document is written to be executable by an autonomous coding agent, milestone by milestone, without re-deriving architecture decisions.

---

## 0. Ground truth — the codebase as it exists TODAY

Do not plan against stale descriptions. The following is the verified current state:

### Already completed (do NOT redo)
| Done | What | Evidence |
|---|---|---|
| ✅ | Full audit (37-page LaTeX report) | `report/report.tex` |
| ✅ | Dead-code purge, dedup, logging, path-independence, loud error paths | `src/fds/slice/slice.py` (687→~575 lines), `src/load_data.py` |
| ✅ | float32 storage (halved memory), NumPy≥1.24 `meshgrid` fix | `slice.py` |
| ✅ | Lazy loading + thread-safe LRU cache (startup 36.6s→2.0s, peak RAM 353→115 MB) | `src/scenario_store.py` |
| ✅ | Layered architecture: entry → view → controller → data provider → store → parser | `main.py`, `main_window.py`, `simulation_controller.py`, `data_provider.py`, `config.py` |
| ✅ | Cooperative thread stop (8 ms, no `terminate()`) | `simulation_controller.py` |
| ✅ | Responsive QSplitter layout, light/dark token themes, ToggleGroup, accessibility names, focus rings, shortcuts (Space/F11/Ctrl+R/Ctrl+Q), QSettings persistence, colorblind-safe colormap options, pan/zoom/save toolbar, demo-data fallback | `theme.py`, `widgets.py`, `main_window.py` |
| ✅ | M1.1: installable package (`pyproject.toml`), pytest suite (35 tests: parser/store/controller/integration), stale files removed (`test_app.py`, `main_window.ui`), fixtures committed | `pyproject.toml`, `tests/`, merged to `main` |
| ✅ | M1.2: vectorized `Slice.readAllTimes`/`readData` (single structured read, no per-timestep loop), `.npy` disk cache in `ScenarioStore` (mtime-invalidated, corrupted-file fallback), `tests/bench_loading.py` benchmark script. Measured on the real 24-scenario dataset: cold ≈0.055–0.082 s/scenario (target ≤0.4–0.5 s), warm ≈0.002–0.006 s/scenario (target ≤100 ms) — both well inside DoD | `src/fds/slice/slice.py`, `src/scenario_store.py`, `tests/bench_loading.py`, `tests/test_disk_cache.py`, merged to `main` |
| ✅ | M1.3s: parser validation spike — `slice.py` cross-validated against `fdsreader` (times exact match; temperature agrees to <4°C max/0.007°C mean domain-interior, per-frame max exact match across all 481 frames); one edge-column discrepancy found, characterized, and filed (not fixed); colormap recommendation (keep `gist_heat`); M-SIM scope flag answered (no template edits needed) | `docs/spike-parser-validation.md`, merged docs-only to `main` |
| ✅ | M1.3: `AMBIENT_C` vmin fix, `MplCanvas` blitting (`capture_background`/`blit_update`), colormap menu (+`inferno`), interpolation toggle (nearest/bilinear). Blitting measured ~1.3× under headless offscreen rendering (not the predicted ≥5×, see §4 for why) | `src/config.py`, `src/widgets.py`, `src/main_window.py`, `tests/bench_rendering.py`, merged to `main` |
| ✅ | M1.4: pull-based `TimeController` (QTimer, GUI-thread) replaces the old worker-push `_Worker`; `TimelineWidget` (seek slider, play/pause, loop) replaces the read-only progress bar; cache-miss scenario switches prefetch on a background thread instead of blocking. Seek latency ~4–15ms, cache-miss toggle-handler return ~0.17ms. Old worker-push implementation deleted in a separate follow-up commit after the new path passed the full suite | `src/time_controller.py`, `src/simulation_controller.py`, `src/widgets.py`, `src/main_window.py`, `src/scenario_store.py`, merged to `main` |
| ✅ | M1.5: `AnimationExporter` (background `QThread`) exports MP4 (ffmpeg, not installed in this dev environment) or GIF (Pillow, verified end-to-end against real scenario data) via `Export → Animation…`; window-modal progress dialog + cooperative cancel leaves zero partial-file bytes at the destination (verified directly). Same double-start `QThread` guard pattern as M1.4's prefetch fix | `src/export.py`, `src/main_window.py`, merged to `main` |
| ✅ | M1.6: `SchematicWidget` (room outline, door/vent/candle icons) wired live to the same toggle signals driving the control panel, in a collapsible "Room diagram" panel; room outline aspect ratio sourced from parsed `.smv` mesh extents (`resolve_room_extent`, fallback footprint 1.0×0.30 m when no real `.smv` present); door/vent/candle placement inside the outline is fixed-proportion pending M2.6's per-object extent mapping (documented as a known follow-on, not a defect); plain-language labels + per-control explainer tooltips for non-specialist audience | `src/schematic.py`, `src/main_window.py`, `tests/test_integration.py`, merged to `main` |

### Current architecture
```
main.py (bootstrap, splash)
   └─ main_window.py (MainWindow — view only)
        └─ simulation_controller.py (SimulationController + _Worker QThread)
             └─ data_provider.py (SimulationData / DataLoadError / demo fallback)
                  └─ scenario_store.py (ScenarioStore: lazy + LRU, thread-safe)
                       └─ load_data.py (single-scenario TEMPERATURE loader)
                            └─ fds/slice/slice.py (.smv/.sf binary parser)
config.py = shared constants (N_CANDLES..., DEFAULT_*, FRAMES_PER_SECOND, SCENARIO_CACHE_SIZE)
```

### Verified facts that shape this plan
- **24 scenarios** on disk (not 36 — the 2019 protocol doc is stale).
- Grid per slice: **(481 timesteps, 49×101 cells)** ≈ 9.5 MB float32/scenario. Total dataset ≈ 230 MB. This is *small*; "big data" machinery (Zarr, SQLite, out-of-core) is **not** justified yet.
- **VELOCITY slices already exist** in every scenario (`&SLCF PBY=0.000, QUANTITY='VELOCITY'` in `fds/template.fds`) — never read by the app. Second quantity = zero new simulations needed.
- Unused on disk: `.s3d` smoke/soot 3D data, `*_hrr.csv` heat-release-rate curves.
- Cache-miss scenario switch ≈1.99 s baseline pre-M1.2; **now ≈0.055–0.082 s cold / ≈0.002–0.006 s warm (M1.2, disk cache + vectorized parser)**.
- Playback is worker-push on wall clock; no seek/scrub. Progress bar is read-only.
- `main_window.ui` and `test_app.py` removed in M1.1.

### Known defects to fix opportunistically
1. `combineSlices` assumes uniform mesh resolution across meshes (unvalidated). **M1.3s finding:** cross-validation against `fdsreader` found a real, isolated discrepancy at the domain's outer edge column (max 39.8°C, mean 1.7°C in that column only; interior of the domain agrees to <4°C max / 0.007°C mean, and per-frame max temperature matches exactly across all 481 frames). `fdsreader` shows an exact-duplicate value at that edge (boundary-padding signature); which side is correct wasn't adjudicated — needs FDS binary-format docs or a Smokeview visual check to resolve. See `docs/spike-parser-validation.md` §3. No practical impact observed yet, but worth fixing before M2.6's probe/isotherm work reads edge coordinates.
2. Color scale `vmin` frozen at first-frame minimum (misleading for later frames).
3. ~~`readAllTimes` discovers timestep count by looping reads instead of `filesize // stride`.~~ Fixed in M1.2.
4. **Unconfirmed test flakiness (M1.4):** `QApplication.overrideCursor()` was observed still non-`None` after a test believed its own busy-cursor cycle had fully settled (`_busy == False`, all `_prefetch_workers` drained) — a rare (~1-in-6 full-suite runs), never reproduced when running the same scenario in isolation 30x. Plausible mechanism, not confirmed: `overrideCursor()` is a single stack on the shared `QApplication` instance (session-scoped `qapp` pytest fixture, reused across every test in the file), while `MainWindow._begin_busy_state`/`_end_busy_state`'s push/pop bookkeeping is per-window-instance (`self._busy`) — if some other test's window closes while a prefetch is still in flight, without waiting for it to fully settle first, that stale worker's eventual `finished` signal could pop the *shared* cursor stack at an unpredictable later point, mid a *different*, unrelated test. Not chased further per explicit direction — filed as tech debt, not a blocker. If it recurs: audit every test in `test_integration.py` that triggers a scenario-toggle change for whether it waits on `_busy`/`_prefetch_workers` fully draining before the test ends, not just whether the test itself asserts something and moves on.

### External input received (2026-07-09)
Supervisor proposed: (a) cross-validate our parser output against Smokeview/`fdsreader` as an independent correctness check, and (b) revisit simulation generation (`.fds` templates) to produce richer `.sf`/`.smv` output — in service of a visualization pass emphasizing fire-appropriate color science (flame-like palettes, hazard-threshold bands) once the comparison platform work is further along.

Split into two pieces with very different cost/governance:
- **M1.3s** (this week, timeboxed 1 day, non-blocking except for M1.3.1): parser cross-validation + color-convention recommendation. Cheap, doesn't touch simulation generation. **Scope confirmed:** validate against `fdsreader==10.1` specifically (version pinned, not "latest"), against exactly ONE simulation (the existing fixture scenario) — not an all-24 sweep. This keeps the spike genuinely timeboxed.
- **M-SIM** (conditional, gated like M3.3): editing `fds/template.fds` and re-running FDS is compute-cost work that was not in the original 9-week budget. Treated with the same approval gate as M3.3 rather than silently folded into current milestones.

### Follow-up input received (2026-07-09b)
Two updates from the user:
1. **fdsreader version + scope locked in:** M1.3s validates against `fdsreader==10.1` on one local simulation only. No change to the spike's plan otherwise (see updated task s.1 below).
2. **GUI priority, immediately:** before Phase 2's engineering work, push the single-view GUI toward maximum polish and "realism" — user-friendly layout plus a schematic/illustrative representation of the physical scenario (candle icons, door, vents) rather than a bare heatmap. This is scoped as new milestone **M1.6** in Phase 1 (detailed in §4), prioritized directly after M1.3/M1.4/M1.5.
3. **Cluster access changes M-SIM's shape:** the user expects to secure cluster access, at which point the plan is to **run all simulations on the cluster first**, and only *then* move into the FDS template/modeling work. This resolves M-SIM's biggest previous risk (local compute-time uncertainty) — cluster access replaces the local single/two-run extrapolation from M1.3s.5 with a full-batch run. M-SIM is updated below to sequence "run everything on the cluster" as its first step, ahead of any template edits.

### Follow-up input received (2026-07-09c)
Three sharpening points for M1.6, plus an explicit sequencing gate:
1. **Target audience is explicitly non-specialists.** The GUI must be usable by people with no FDS/fire-science background. This goes beyond visual polish into real UX simplification: plain-language labels instead of raw variable names (VOD/VOC etc. shown without explanation), and a low barrier to understanding *what is being shown*, not just what is clickable. The label-review work (previously folded into M1.6.4 as a nice-to-have) is elevated to a **core requirement**, and each scenario toggle gains a light per-control explainer (tooltip-level — explicitly NOT a full onboarding flow; keep it minimal).
2. **A real physical setup will exist and be closely comparable to the app.** The team will have (or is close to having) a physical mockup of the room/candle/door/vent arrangement the simulations model, and people will likely see the real setup and the app side by side at the demo. The schematic's design goal therefore shifts from "illustrative shapes that gesture at the scenario" to "**should reasonably resemble the actual physical layout**" — proportions, door position, and vent placement should track real geometry where possible, not arbitrary layout choices.
3. **Visual fidelity is tied to data the parser already exposes, not new inputs.** Where exact physical dimensions aren't available yet, use the mesh extents already parsed from the `.smv` files (the same extent data M2.6's probe feature will use, and the same data path M1.3s's `fdsreader==10.1` cross-validation checks for correctness) as the best available source of true geometry. M1.6's realism upgrade therefore requires **no new measurements or inputs to start**; manual dimension input can refine it later if the physical mockup provides exact numbers. (Recorded as trade-off §7.10.)

Consequence: M1.6's effort estimate is revised **1.5–2 d → 2–2.5 d** (extent-driven proportional accuracy + explainer copy for every control); Phase 1 exit absorbs ~0.5 d — see §2/§9.

**Sequencing gate (made explicit, same input):** GUI work (M1.6 and any follow-on polish) and fdsreader parser validation (M1.3s, plus any re-validation triggered by later parser changes) continue as the **active track**; **M-SIM sim.1+ (editing `fds/template.fds`) does not begin until parser validation is confirmed solid — not merely until cluster access exists.** This is not a new decision — it was already implied by the dependency structure (sim.4 re-validates against fdsreader before trusting new output) — but it is now a stated gate rather than an implicit one, so it is unambiguous to whoever executes next. sim.0 (cluster baseline run, no template changes) remains independent of validation status and may run whenever cluster access is secured.

---

## 1. Strategic analysis (condensed — full reasoning in §7 trade-offs)

### 1.1 What this platform's identity should be
The dataset is a **parametric ensemble** (2×2×3×2 factorial fire study). Its scientific value is *comparison across scenarios*, not prettier playback of one scenario. Every high-priority feature below serves that thesis:

> **North star: turn a single-scenario looping kiosk into an interactive ensemble-comparison instrument — and make the comparison machinery double as the evaluation harness for ML experiments.**

That last clause is the key synergy: a "difference heatmap" view built for comparing scenario A vs B is *exactly* the view needed to compare *model prediction vs simulation* later. Build once, use twice.

### 1.2 UI — gaps that matter (in value order)
1. **Timeline scrubber** (drag-to-seek, frame stepping, loop toggle) — converts passive playback into analysis. Everything temporal depends on it.
2. **Multi-view grid** (1×1/1×2/2×2) with synchronized time and optional linked color scale.
3. **Experiment browser** — sortable/filterable table of all 24 scenarios with computed summaries (peak T, time-to-threshold, peak HRR from the unused `_hrr.csv`).
4. Value probe under cursor (x, z, T readout), isotherm overlay toggle.
5. Export: PNG snapshot (exists via toolbar), MP4/GIF animation export.
6. Bookmarks/annotations/docking — genuinely useful but below the line for 9 weeks; Phase 4.

### 1.3 Visualization — gaps that matter
| Feature | Verdict | Why |
|---|---|---|
| Perceptual/domain-appropriate colormap default + fixed physical scale | **Do, week 1 — informed by M1.3s** | Correctness of perception; recommendation now comes from the validation spike rather than an arbitrary default |
| Blit-based rendering | **Do, week 1** | Only the image artist changes per frame; stop re-rasterizing axes+colorbar 4–12×/s |
| Velocity quantity + isotherms + contours | **Do, Phase 2** | Data already on disk; reveals convective structure temperature hides |
| Difference heatmaps (A−B, diverging cmap, symmetric clim) | **Do, Phase 2** | Core science + future ML-eval view |
| Ensemble stats views (mean/std/min/max across selected runs) | **Do, Phase 2/3** | Honest "uncertainty visualization" for a deterministic ensemble |
| Arbitrary slice planes, volume rendering (.s3d), 3D | **Phase 4** | Needs 3D readers + GL stack; high effort, not needed for the ensemble thesis |
| Super-resolution display | **Phase 3 option** | Only meaningful with ML (see §1.5); bilinear display interpolation is a checkbox, do in week 1 |

### 1.4 Workflow
- **Scenario manifest (JSON)**: single source of truth for factors↔folders, replacing the triple-hardcoded `2,2,3,2`. Cheap; prerequisite for the browser. *(SQLite/database: overkill for 24 runs — revisit only when N ≥ hundreds.)*
- **Disk cache** of parsed arrays (`.npy` + `mmap_mode='r'`, keyed on source mtimes): cache-miss cost 1–1.5 s → ~50 ms. This is what makes a 2×2 comparison grid feel instant. Highest leverage single performance change remaining.
- **Summary-stats index** (computed once, cached JSON): powers browser + auto-summaries.
- Config editor / auto-validation of `.fds` inputs: out — that's the simulation-generation side, explicitly out of scope (except as gated in M-SIM).

### 1.5 AI/ML — honest per-idea assessment

Context that dominates every assessment: the dataset is **24 runs × 481 frames × 49×101 px** on a *categorical* 4-factor grid where **all combinations are already simulated**. Tiny by ML standards; no held-out region of parameter space exists unless new sims are run.

| Idea | Feasibility here | Novelty | Difficulty | Research value | Compute | Fits? |
|---|---|---|---|---|---|---|
| **PCA/autoencoder + clustering of fire behaviors** | High — sklearn, days | Low–Med | Easy | Med (great for report; interpretable) | Laptop | ✅ **Do (Phase 3a)** |
| **Auto scenario summary** (deterministic stats → templated text) | High | Low | Easy | Med (demo impact, zero hallucination risk) | Trivial | ✅ **Do (Phase 3a)** |
| **Temporal forecasting** (ConvLSTM or FNO: frames t≤k → t+Δ) | Med–High — 11.5k frames total, small grids train on MPS/CPU | Med (application novelty) | Medium | **High** — proper internship contribution; negative results still reportable | Laptop GPU/MPS, hours | ✅ **Do ONE (Phase 3b)** |
| **FNO (neural operator) as the forecasting architecture** | Med — `neuraloperator` lib mature | Med | Medium | High (stronger framing than ConvLSTM) | Same | ✅ Preferred variant of above |
| **Super-resolution of coarse FDS runs** | Med — **requires re-running 24 cases at coarse grid** (small cases; FDS runs locally, not HPC) | Med | Medium | Med–High; very demo-able side-by-side | Laptop + ~day of FDS runs | ⚠️ Only if re-running sims is approved |
| Scenario interpolation / surrogate over parameters | Low **as posed** — 24 points in categorical space = lookup table; real version needs continuous parameter sweeps (new sims) | — | Hard (data gen) | High *later* | Med | ❌ Phase 4 (needs new data) |
| Inverse design / parameter recommendation | Same blocker as surrogate | — | Hard | High later | Med | ❌ Phase 4 |
| PINNs | Very low — LES combustion physics far beyond what PINNs handle; would degenerate to a toy | — | Hard | Low here | High | ❌ Skip |
| Diffusion models | Overkill; generative fidelity isn't the scientific need; 24 runs can't train one | — | Hard | Low here | High | ❌ Skip |
| RL for exploration | No environment/action space in scope | — | Hard | Low | High | ❌ Skip |
| Anomaly detection (ML) | Little value on 24 curated runs; do *deterministic* integrity checks (truncated .sf, NaN frames) instead | — | Easy (non-ML) | Low | Trivial | ⚠️ As validation utility only |
| Similarity search / latent space | Folds into the PCA/AE clustering item | — | Easy | Med | Laptop | ✅ Part of 3a |
| Uncertainty estimation (for the ML model) | Deep ensembles / MC dropout, standard | Low | Medium | Med (required for credibility) | Laptop | ✅ Part of 3b eval |

**Two-track AI strategy:**
- **Track A (certain payoff, ~1 week):** ensemble analytics — feature extraction per scenario (spatial-max/mean time series, hot-area fraction, time-to-threshold), PCA scatter + clustering panel in-app, deterministic auto-summaries. Cannot fail; always demo-safe.
- **Track B (the research bet, ~2 weeks):** ONE forecasting model (FNO preferred, ConvLSTM fallback), train on 20 scenarios / hold out 4, evaluated **inside the app** via the difference view ("Simulated | Predicted | Error"). If the model underperforms, the analysis of *why* is still a legitimate internship result — and the app feature (comparison view) ships regardless.

### 1.6 Architecture next steps
- **Extract parser into an installable package** (`src/` layout, `pyproject.toml`, `pip install -e .`) with pytest suite against fixture files. Everything else builds on trusting the parser. *(Done — M1.1.)*
- **`PlotView` interface** to decouple views from matplotlib — this is the seam that makes the pyqtgraph decision reversible (§7.1) and multi-view possible.
- **Invert playback control**: replace worker-push (`_Worker` sleeping on wall clock) with a GUI-thread `QTimer` pulling frames — *only viable once the disk cache makes frame access O(µs)*. Kills residual shared-state issues; makes seek trivial; background thread remains only for cache-miss prefetch. (Sequencing: cache first, then timer.)
- **Plugin system: NO** (for now). Solo developer, zero external contributors — a plugin API is speculative generality. Use a lightweight registry dict (quantity→reader, view-type→class); promote to plugins only when a second contributor exists (Phase 4). The registry should be designed so a reader can be swapped or added without touching `ScenarioStore`/`data_provider`/`simulation_controller` — `fdsreader` is a concrete candidate second reader here already (see §7's reader-choice trade-off), not just a hypothetical future one.
- Testing: pytest; fixtures = one real scenario's `.smv` + matched `.sf` files copied into `tests/fixtures/`; unit tests for parser invariants (shape, times monotonic, known first-frame values), store (LRU eviction, thread-safety smoke), controller (start/stop/seek); `QT_QPA_PLATFORM=offscreen` integration test that builds the window and steps frames. *(Done — M1.1, 35 tests.)*

---

## 2. Phased roadmap

Time budget: **Phase 1 = 1 wk (+ M1.3s spike, timeboxed/parallel; +~0.5 d from M1.6's 2026-07-09c scope growth, absorbed by M1.5's slack — see §9) · Phase 2 = 4 wks · Phase 3 = 3 wks · buffer/rehearsal = 1 wk** → Sept 11.

### Phase 1 — Quick Wins (Jul 8 → Jul 16)

| # | Milestone | Why it matters | Impact | Effort | Risk | Priority |
|---|---|---|---|---|---|---|
| M1.1 | Packaging, tests, repo hygiene | Safety net for all later refactors | High (invisible) | 1 d | None | **High** — ✅ done |
| M1.2 | Disk cache + vectorized reads | 1.5 s scenario switch → ~50 ms; unlocks multi-view + QTimer | High | 1 d | Low | **High** — ✅ done |
| M1.3s | **Parser validation spike** (`fdsreader` cross-check + Smokeview color convention review) | Independent correctness check before M1.2's vectorization locks in parser behavior; informs M1.3's colormap choice | Med | 1 d (timeboxed, spike — not shippable code) | Low | High (blocks only M1.3.1) — ✅ done |
| M1.3 | Rendering quick wins (blit, domain-appropriate colormap default from M1.3s, vmin fix, interp toggle) | Perceptual correctness + smoothness, nearly free | Med–High | 1 d | Low | **High** — ✅ done |
| M1.4 | QTimer playback + timeline scrubber | Biggest single UX unlock; playback becomes seekable | High | 2 d | Med (touches controller) | **High** — ✅ done |
| M1.5 | MP4/GIF export | Demo assets; researchers share results | Med | 0.5 d | Low (ffmpeg dep) | Med — ✅ done |
| M1.6 | **Scenario schematic + non-specialist usability pass** (extent-proportioned room schematic, plain-language labels, per-control explainer tooltips) | User-requested priority, before Phase 2 engineering; audience is explicitly non-specialists, and a physical mockup of the setup will sit beside the app at the demo — the schematic must resemble the real layout, and labels must not assume FDS background | High (demo impact) | 2–2.5 d (revised 2026-07-09c from 1.5–2 d: extent-driven proportions + explainer copy for every control) | Low–Med (pure UI addition; reads already-parsed `.smv` extents, no data-layer changes) | **High** (explicit user priority) |

*M1.3s is a spike, not a feature milestone — it produces a decision doc (`docs/spike-parser-validation.md`), not shippable app code. It runs in parallel with / immediately after M1.2, and does not itself count against the Phase 1 feature budget; only M1.3.1 waits on its output.*

### Phase 2 — Major Engineering (Jul 16 → Aug 12)

| # | Milestone | Why | Impact | Effort | Risk | Priority |
|---|---|---|---|---|---|---|
| M2.1 | Scenario manifest + quantity/slice generalization | Unlocks VELOCITY (already on disk) + all future quantities; kills triple-hardcoding | High | 3 d | Med | **High** |
| M2.2 | PlotView abstraction + multi-view grid (1×1/1×2/2×2, synced time, linked clim) | The platform centerpiece; matches dataset's purpose | **Highest** | 5 d | Med | **High** |
| M2.3 | Difference view + ensemble stats view | Core science; doubles as ML-eval harness | High | 3 d | Low (array ops on cached data) | **High** |
| M2.4 | pyqtgraph spike — timeboxed decision gate (2 d max) | Only migrate if matplotlib-blit can't hold the 2×2 grid at target FPS | Med | 2 d | Contained by timebox | Med |
| M2.5 | Experiment browser + summary-stats index (incl. HRR from CSV) | Workflow leap; makes 24 runs navigable | High | 3 d | Low | **High** |
| M2.6 | Value probe + isotherm/contour overlays (thresholds informed by M1.3s) | Analysis affordances researchers expect | Med | 2 d | Low | Med |

### Phase 3 — Research Extensions (Aug 12 → Sep 2)

| # | Milestone | Why | Impact | Effort | Risk | Priority |
|---|---|---|---|---|---|---|
| M3.1 | Ensemble analytics: features, PCA/clustering panel, auto-summaries | Certain-payoff research; interpretable; demo-safe | High | 4 d | Low | **High** |
| M3.2 | Forecasting model (FNO preferred / ConvLSTM fallback) + in-app "Simulated · Predicted · Error" view | The publishable bet; reuses M2.3 | High | 8 d | Med–High (model quality unknown — but negative result reportable) | **High** |
| M3.3 | (Conditional) Super-resolution with re-run coarse sims | Only if generating new FDS runs is approved | Med | 5 d + sim time | High | Low |
| M-SIM | (Conditional) Simulation regeneration for richer `.sf`/`.smv` output | Only if supervisor approves re-running FDS; edits `fds/template.fds` (extra slice planes/quantities, finer mesh) purely to feed the visualization work, not a new research axis | Med | 3–5 d + FDS compute time (est. from M1.3s.5) | Med (touches upstream simulation, not app code — different risk profile than M3.x) | Low (gated) |

**Feature freeze: Sep 4. Week of Sep 7: rehearsal, bugfixes, fallback paths, demo script.**

### Phase 4 — Long-term Vision (post-September)
- Volume rendering of the untouched `.s3d` smoke data (pyqtgraph `GLVolumeItem` or VTK); 3D view with arbitrary slice planes.
- Continuous-parameter sweeps (door width as a real variable) → true surrogate models, scenario interpolation, inverse design ("what vent config keeps the exit tenable longest?").
- Session files: bookmarks, annotations, reproducible view states; dockable multi-monitor layout.
- Zarr-backed store + progressive loading when datasets outgrow RAM (only then).
- Plugin API once a second contributor exists; web-based companion viewer for sharing.
- What it becomes: *a lab-standard interactive instrument for parametric fire-simulation studies — ParaView-lite specialized for FDS ensembles, with an integrated ML experimentation loop.*

---

## 3. Top 10 (the balance picks, in execution order)

1. **Tests + packaging (M1.1)** — protects every subsequent change; a refactor without tests before a hard deadline is gambling. ✅ done.
2. **Disk cache + vectorized reads (M1.2)** — single highest-leverage remaining perf change; prerequisite for 3, 6, 7. ✅ done.
3. **Timeline scrubber + QTimer playback (M1.4)** — biggest UX unlock per day of work. ✅ done.
4. **Blit + domain-appropriate colormap + vmin fix (M1.3)** — visual correctness and smoothness, nearly free. ✅ done.
5. **Quantity generalization (M2.1)** — VELOCITY is already on disk; one reader change unlocks a whole visualization axis.
6. **Multi-view synchronized comparison (M2.2)** — the centerpiece; the reason this dataset exists.
7. **Difference + ensemble views (M2.3)** — science value now, ML-eval harness later; the plan's best two-for-one.
8. **Experiment browser (M2.5)** — turns "24 folders" into "an experiment".
9. **Ensemble analytics + auto-summaries (M3.1)** — certain-payoff research content.
10. **Forecasting prototype (M3.2)** — the research bet, de-risked because its evaluation UI (#7) ships regardless of model quality.

*A timeboxed parser-validation spike (M1.3s) runs between #2 and #4 above — it's a correctness/direction check, not a numbered deliverable, but #4 (blit + colormap) now depends on its output.*

Ordering logic: 1–2 are enablers (invisible but load-bearing); 3–4 make the existing app feel modern within week 1; 5→6→7 is a strict dependency chain building the platform identity; 8 is parallelizable filler for wait times; 9–10 convert the platform into research output. Anything jeopardizing Sep 11 gets cut from the bottom up (M3.3/M-SIM first, then M2.6, then M2.4's migration half).

---

## 4. Detailed execution plan (per-task, for the implementing agent)

> Conventions: **D** = difficulty (Easy/Med/Hard), **T** = est. time. Every task ends with: run `pytest`, launch app (`cd src && python3 main.py`), verify the specific behavior listed. Keep the app runnable after every commit.

### M1.1 — Packaging, tests, repo hygiene ✅ DONE
Merged to `main`. 35 tests passing (12 parser, 5 store, 9 controller, 10 integration). `pip install -e .` clean. Stale files removed.

### M1.2 — Disk cache + vectorized reads ✅ DONE
Merged to `main`. **Objective met:** scenario switch ≤100 ms warm; parse ≤0.4 s cold.

| Task | Detail |
|---|---|
| 1.2.1 Vectorize `Slice.readData` ✅ | `readAllTimes` does one structured read (`n_times = (filesize - offset) // stride`, combined time+data dtype); `readData` reuses that buffer instead of re-reading. Files: `slice.py`. Test: `test_cold_parse_under_500ms` passes. |
| 1.2.2 `.npy` cache layer in `ScenarioStore` ✅ | `ScenarioStore(cache_dir=...)` — opt-in on the class (default `None`/disabled, so tests with fake paths never touch disk); the real app enables it via `data_provider.py` passing `fds/sim/.cache/` (gitignored). Mtime-invalidated against source `.sf`/`.smv`; corrupted-cache falls back to re-parse. Files: `scenario_store.py`, `data_provider.py`. Tests: 4/4 in `tests/test_disk_cache.py` (cold→warm, stale-source invalidation, corrupted-file fallback, disabled-by-default). |
| 1.2.3 Benchmark script ✅ | `tests/bench_loading.py`. Run against the real 24-scenario dataset: **cold ≈0.055–0.082 s/scenario, warm ≈0.002–0.006 s/scenario** (baseline was 1.99 s cold, N/A warm). Known rough edge: its reported peak-RSS figure (~600 MB) is the benchmark script's own footprint (it deliberately double-loads all 24 scenarios), not the real app's — the app caps at `SCENARIO_CACHE_SIZE=4`; not yet caveated in the script's output. |

**DoD:** warm switch <100 ms measured (✅ ≈2–6 ms) · cold parse <0.5 s (✅ ≈55–82 ms) · cache invalidation test passes (✅) · pytest green (✅ 40/40).

### M1.3s — Parser validation spike (timeboxed, 1 day max) ✅ DONE
`docs/spike-parser-validation.md` on `spike/parser-validation-fdsreader` (merged docs-only). **Deviation:** `fdsreader==10.1` doesn't exist on PyPI (real range is `0.0.0`–`1.11.7`); used `1.11.7` instead, documented in the spike doc §0.

| Task | Detail |
|---|---|
| s.1 Cross-validate vs `fdsreader` ✅ | Times: exact match (481 steps, 0.0 diff). Temperature: interior of domain agrees to max 3.76°C / mean 0.0068°C; per-frame max temperature matches **exactly** across all 481 frames. Ran against the full on-disk `fds/sim/c1_d0_vod0_voc0/` copy (same scenario as the fixture, not a sweep — the trimmed pytest fixture is missing `.sf` files `fdsreader` needs for slices we don't use). |
| s.2 Triage any discrepancy ✅ | Found one real, isolated discrepancy at the domain's outer edge column (max 39.8°C, mean 1.7°C, confined to that single column). `fdsreader` shows an exact-duplicate boundary-padding signature there; not adjudicated which side is correct (needs FDS format docs or Smokeview, neither available in this environment). Filed as a note against known defect §0.1, not fixed here. |
| s.3 Smokeview convention review (manual) | **Not performed** — Smokeview isn't installed in this environment and no GUI tooling is available to install/drive it. Documented as an honest limitation in the spike doc §4, not silently skipped. |
| s.4 Colormap recommendation ✅ | Keep `gist_heat` (already the app's default, already a black→red→orange→yellow→white blackbody/flame progression) — no default to change for M1.3.1. Hazard bands proposed (<60°C / 60–300°C / >300°C) as general fire-safety reference points pending domain-expert review, not derived from this study's data. |
| s.5 Scope flag for simulation regeneration ✅ | No — nothing in this spike requires editing `fds/template.fds`. VELOCITY is already present and unread (confirmed again); the one real finding (edge-column discrepancy) is parser-side (`slice.py`), not a simulation-output gap. |

**DoD:** ✅ `docs/spike-parser-validation.md` exists with cross-validation numbers, the edge-column discrepancy filed as a note (not fixed), a specific colormap recommendation (keep `gist_heat`), and the M-SIM scope flag answered (no). M1.3.1 uses this recommendation.

### M1.3 — Rendering quick wins ✅ DONE
Merged to `main`. **Objective met:** perceptually correct, smooth single-view rendering.

| Task | Detail |
|---|---|
| 1.3.1 Default colormap → per M1.3s recommendation ✅ | M1.3s recommended keeping `gist_heat` (already the default, already blackbody-style) — no default change needed. Menu now lists `gist_heat`/`inferno`/`viridis`/`cividis` per spec (added `inferno`). Files: `main_window.py`. |
| 1.3.2 Fix frozen `vmin` ✅ | `AMBIENT_C = 20.0` added to `config.py`; `heatmap.set_clim(vmin=AMBIENT_C, vmax=...)` set explicitly in `_init_plot`/`_on_temp_changed`, never left to auto-scale off frame 0. Test: `test_vmin_pinned_at_ambient_across_frames_and_slider` — vmin confirmed stable across frames 0/50/200/480 and across slider changes. |
| 1.3.3 Blitting in `MplCanvas` ✅ | `MplCanvas.capture_background()`/`blit_update()` added; `_redraw()` (the per-frame playback path) now blits instead of `draw_idle()`. Background is recaptured (full draw) on resize (via a `resizeEvent` override) and in the theme/colormap/interpolation/vmax setters. **Measured speedup: ~1.3×, not the ≥5× predicted** — profiled and found to be a real, correctly-implemented result, not a bug: this figure has almost no "expensive chrome" to skip (ticks are off, colorbar is small), and `copy_from_bbox`/`restore_region`/`blit` themselves scale with the full canvas buffer size, so under offscreen/headless rendering there isn't much headroom over a full `draw()` at this figure's size. `tests/bench_rendering.py` prints both numbers for future re-measurement on a real display. |
| 1.3.4 Interpolation toggle ✅ | View → Interpolation menu (nearest/bilinear), persisted via QSettings, applied through `AxesImage.set_interpolation`. |

**DoD:** playback visibly smoother (✅ blitting works correctly; ~1.3× measured under headless offscreen rendering, not the predicted ≥5× — see 1.3.3) · colorbar physically anchored (✅ vmin pinned at `AMBIENT_C`) · toggles persist (✅ colormap + interpolation via QSettings) · pytest green (✅ 49/49).

### M1.4 — QTimer playback + timeline scrubber ✅ DONE
Merged to `main`. **Objective met:** seekable, drift-free playback; the wall-clock worker is retired.

| Task | Detail |
|---|---|
| 1.4.1 `TimeController` ✅ | New file `time_controller.py` (not folded into `simulation_controller.py`, to keep phase-1/phase-2 commits clean — see below): `play/pause/seek(i)/step(±1)/set_speed`, emits `time_changed(int)` (frame index). Views pull `store.get(case)[i]` on tick, confirmed safe: seek latency measured 4–15ms (paused) / 4–5ms (during playback), well under any perceptible threshold. |
| 1.4.2 `TimelineWidget` ✅ | New in `widgets.py`: play/pause button + slider (0..n_frames-1) + time label + loop toggle, replaces the read-only `QProgressBar`. Dragging emits `seek_requested`; `set_index()` skips updating the slider while the user is actively dragging so it doesn't fight their gesture. |
| 1.4.3 Rewire `MainWindow` ✅ | Transport buttons, Space, Ctrl+R all drive `TimeController`; added Left/Right (step 1 frame) and Shift+Left/Right (step 1 second) per spec. Old worker-push wiring removed from `MainWindow` in this same commit (phase 1); the *implementation* it wired to (`_Worker`/`SimulationController.start/stop/...`) was deleted in a separate, reviewable follow-up commit once the new path had passed the full test suite — see process note below. |
| 1.4.4 Cache-miss UX ✅ | Extended beyond "while paused" (the literal spec wording) to cover mid-playback switches too, since a cold parse (~55-80ms, M1.2's numbers) synchronously on a `QTimer` tick risked a visible stutter — same underlying risk, same fix. `ScenarioStore.is_cached()` (new, read-only, existing lock) gates a cache hit (immediate redraw) vs. a cache miss (busy cursor + disabled slider + status message + background `_PrefetchWorker`, resume-if-was-playing on completion). Measured: toggle handler returns in 0.17ms on a cache miss (vs. 0.17–16ms range observed) — the GUI thread never blocks; background settle ≈110-140ms including Qt event-loop polling overhead in the test harness. |

**Process note (as executed):** kept the old worker-push implementation (`_Worker`, `SimulationController.start/stop/is_running/current_frame/frame_ready`) compiling and passing its own tests through the first commit (new path added, old path just unused by `MainWindow`), then deleted it in a separate commit once the new path had passed the full integration suite — per explicit instruction, not an M1.4-spec requirement.

**Deviation found during testing (fixed, not just risk-flagged):** a real `QThread` lifecycle bug — rapid scenario-toggle changes started a second `_PrefetchWorker` while a single `self._prefetch_worker` attribute still referenced the first, still-running one; overwriting that attribute let Python garbage-collect a live `QThread`, which Qt treats as a fatal process abort (not a catchable exception). Fixed by keeping every in-flight worker referenced in a list (`SimulationController._prefetch_workers`) until its own `finished` signal fires. Covered by both an integration-level and a unit-level regression test.

**Risk:** regression in play/pause/restart semantics → mitigated by the existing `test_mainwindow_transport_controls` (unchanged, still passing — method names `_start_simulation`/`_stop_simulation`/`_restart_simulation` preserved) plus new drag-seek/loop/cache-miss/rapid-toggle tests.
**DoD:** drag-seek works during playback (✅ measured ~4ms) · no GUI freeze on scenario switch (✅ 0.17ms handler return on cache miss) · speed change takes effect immediately (✅ measured: next tick arrives at the new interval, not the old one) · old worker-push path deleted (✅, separate commit) · tests green (✅ 69/69).

### M1.5 — Animation export ✅ DONE
Merged to `main`.

1.5.1 ✅ New `export.py`: `AnimationExporter` (QThread) renders `data[start:end]` offscreen via a dedicated `Figure`/`FigureCanvasAgg` (never the live on-screen canvas) at chosen fps/range, using the same colormap/clim/interpolation the live view was showing. MP4 via `matplotlib.animation.FFMpegWriter` if `ffmpeg` is on `PATH` (`ffmpeg_available()`), else GIF via Pillow — this dev environment has no `ffmpeg` installed, so the GIF fallback is what's actually been exercised end-to-end here; the MP4 path is implemented per spec and structurally tested (fails cleanly, no partial file, when `ffmpeg` is absent) but not verified against a real MP4 output in this environment. `Export → Animation (MP4/GIF)…` menu hook in `main_window.py`, with a small `ExportRangeDialog` for fps/start/end (defaults to the full scenario at the app's own fps, so accepting the default is a one-click export). `QProgressDialog` (window-modal, so a second export can't be started from the menu — reinforced with an explicit `isRunning()` guard, same defense-in-depth pattern as M1.4's prefetch fix) with Cancel wired to a cooperative `threading.Event`, matching the app's existing cooperative-stop philosophy. Frames are written to a temp path and only moved to the final destination on full, uncancelled completion — cancelling (or an error) always leaves zero bytes at the destination, verified directly rather than assumed. **Test (adapted from spec):** no `ffmpeg` in this environment to verify a literal "opens in QuickTime" MP4 — validated the equivalent for the GIF path instead: exported 20 frames (~5s at 4fps) of a real scenario, reopened the file with Pillow, confirmed correct frame count and dimensions.

**DoD:** both formats export (✅ GIF verified end-to-end against real data; MP4 implemented + tested for clean failure without `ffmpeg`, not verified against a real encode in this environment) · UI stays responsive (✅ export runs on a background `QThread`; the modal progress dialog is deliberate, not a UI freeze — the app's event loop keeps processing throughout, verified via the passing test suite while an export is in flight).

### M1.6 — Scenario schematic + non-specialist usability pass
**Objective (sharpened 2026-07-09c):** make the single-view GUI usable by people with **no FDS/fire-science background** — a room diagram of the physical scenario (room, door, vents, candles) that reflects the live toggle state and **reasonably resembles the actual physical layout** (a physical mockup of the setup will likely sit next to the app at the demo), plus plain-language labels and a light per-control explainer so a non-specialist understands *what is being shown*, not just what is clickable. **Pure UI addition — no changes to the parser, store, or controller's data contracts** (the schematic *reads* mesh extents the parser already exposes, it doesn't add data paths).

**Scope boundary (important):** proportionally-accurate *schematic* ≠ physically-precise *overlay*. The inset diagram's proportions, door position, and vent placement should track real geometry (sourced per 1.6.1 below), but pixel-accurate placement of icons at real (x, z) coordinates **on the heatmap** still depends on the slice-extent mapping work planned for M2.6 (Phase 2) — don't block M1.6 waiting for that. Ship the proportioned inset diagram now; revisit precise on-heatmap overlay as a stretch task once M2.6 lands. Also explicitly out of scope: any onboarding flow, tour, or help system — the explainers in 1.6.5 are tooltip-level and minimal by design.

| Task | Detail |
|---|---|
| 1.6.1 Design the schematic component — **proportions from parsed `.smv` mesh extents** | A small top-down room diagram: outline, door (position/width reflects `door_value`: wide vs narrow), vent icons for VOD/VOC (state reflects open/closed/HVAC), candle icon(s) (count reflects `candles_value`). **Geometry source (2026-07-09c): not arbitrary shapes** — derive room proportions from the mesh extents already parsed from the `.smv` files (same extent data M2.6's probe will use; same data path M1.3s cross-validates), so the outline tracks the real room footprint people will see in the physical mockup. **Confirmed as shipped (2026-07-09d):** only the outline's aspect ratio is extent-derived (spot-checked footprint: 1.0 m × 0.30 m, identical across every sampled scenario). Door position, vent placement, and candle placement *inside* that outline are fixed proportional placements, not extent-derived — a bounding-box extent alone doesn't carry per-object coordinates; recovering those is M2.6's slice-extent-mapping work (see M2.6's task list below and trade-off §7.10). **Fallback/refinement note:** if the mockup later yields exact measured dimensions, add a small manual dimension-input override (config-level, not GUI) to refine where extents and reality differ — a later refinement, not a prerequisite. Render as SVG drawn with `QPainter`/`QSvgRenderer`, or a small dedicated `QGraphicsView` — no external image assets required, keep it vector/flat-style so it themes cleanly with `theme.py`'s light/dark tokens. New file: `schematic.py` (widget) or add to `widgets.py`. D:Med T:5h. |
| 1.6.2 Wire schematic to live state | Subscribe to the same `SimulationController`/`ToggleGroup` signals already driving `candle_toggle`, `door_toggle`, `vod_toggle`, `voc_toggle` in `main_window.py` — the schematic updates the instant a toggle changes, no new state duplicated. D:Easy T:2h. |
| 1.6.3 Place in layout | Add as a collapsible panel/dock alongside the existing control panel (reuse `CollapsibleSection`), or as a small inset in the corner of the plot panel — user's call on default position, but must not compete with the heatmap for space at the 900×600 minimum size (respect M-earlier's responsive-layout constraints; hide/collapse below a size threshold if needed). D:Med T:3h. |
| 1.6.4 Plain-language labels + iconography (**core, elevated 2026-07-09c** — was a nice-to-have) | Rename all user-facing control-panel labels for a non-specialist: raw variable names (VOD/VOC etc.) must not appear without a plain-language primary label (technical name may remain as secondary text for traceability). Add small icons to the existing `ToggleGroup` buttons (flame for candles, door-swing glyph, vent-grille glyph) alongside their text labels — not replacing text, for accessibility (icon-only buttons are harder to parse for screen readers and unfamiliar users). D:Med T:3h. |
| 1.6.5 Per-control explainers (**core, new 2026-07-09c**) | One or two plain-language sentences per scenario toggle saying what it changes physically (e.g. what opening the vent does to the room), delivered as tooltips (plus the existing accessible-description hooks) — **explicitly tooltip-level, not an onboarding flow**; no new panels, no tour. Copywriting is the bulk of the work; route the copy through one reviewer with no FDS background if available. D:Easy–Med T:3h. |
| 1.6.6 (Stretch, only if time allows) Overlay directly on heatmap | If M1.6.1–1.6.5 land comfortably inside budget, attempt a simplified overlay of door/vent markers directly on the plot's edges using the slice's known extent (approximate, not the precise M2.6 mapping) — clearly mark as approximate in a tooltip if shipped. Skip without guilt if behind schedule; this is explicitly optional. D:Med T:4h (only if pursued). |

**DoD:** schematic renders and updates live with every toggle · **room outline proportions derived from parsed extents (confirmed: 1.0 m × 0.30 m footprint); door/vent/candle positions within the outline remain fixed placements pending M2.6** · themes correctly in both light and dark mode · doesn't break the 900×600 minimum-size layout · no raw variable name (VOD/VOC…) visible without a plain-language label · every scenario toggle has an explainer tooltip · integration test extended to cover schematic rendering at a couple of toggle-state combinations · pytest green · manual screenshot review at both a small and large window size, ideally sanity-checked by someone without FDS background.

### M2.1 — Scenario manifest + quantity generalization
**Objective:** N quantities × M slice-planes, one source of truth for the ensemble.
**Refactor first:** `load_data.py`'s hardcoded `quantity='TEMPERATURE', direction=1, offset=0` becomes parameterized.

| Task | Detail |
|---|---|
| 2.1.1 `manifest.py` + `manifest.json` | Generator scans `fds/sim/*/`, parses factor values from folder names, records path/factors; app loads manifest (regenerates if missing). Replaces `build_data_matrix` assumptions with explicit mapping. Files: new `manifest.py`; `config.py`, `scenario_store.py`, `data_provider.py`. D:Med T:4h. Test: manifest lists 24; warns on count mismatch (keep `check_scenario_count` behavior). |
| 2.1.2 `SliceKey` + quantity inventory | `SliceKey(quantity, direction, offset_index)` frozen dataclass. `readSliceInfos` already parses ALL quantities — expose `available_slices(scenario)` from the `.smv`. Verify VELOCITY appears. Files: `slice.py` (expose), new `slice_key.py` or in `manifest.py`. D:Med T:3h. |
| 2.1.3 Store keyed by (scenario, SliceKey) | `ScenarioStore.get(case, key)`; cache filenames include key; `load_data.load_data` takes key params. Files: `scenario_store.py`, `load_data.py`, callers. D:Med T:3h. Test: TEMPERATURE and VELOCITY both load with correct shapes; per-quantity units/labels surfaced from `.smv` (°C vs m/s). |
| 2.1.4 Quantity selector UI | Combo box in control panel; per-quantity colormap + clim defaults (velocity: sequential `viridis`, vmin=0). Files: `main_window.py`, `config.py`. D:Easy T:2h. |

**DoD:** user switches TEMPERATURE↔VELOCITY live · correct units on colorbar · manifest is the only place factor structure lives · tests cover both quantities.

### M2.2 — PlotView abstraction + multi-view grid
**Objective:** the comparison instrument.
**Refactor first (required):** extract current plotting from `MainWindow` into a `SliceView` class implementing:

```python
class PlotView(Protocol):
    def widget(self) -> QWidget: ...
    def show_frame(self, frame: np.ndarray) -> None: ...
    def set_cmap(self, name: str): ...
    def set_clim(self, vmin: float, vmax: float): ...
    def set_title(self, text: str): ...
```

| Task | Detail |
|---|---|
| 2.2.1 Extract `SliceView` (matplotlib impl, blitting inside) | `MainWindow` shrinks to wiring. Files: new `views.py`; `main_window.py`. D:Med T:4h. Test: single-view behavior unchanged (integration test). |
| 2.2.2 `ViewGrid` container | QGridLayout of PlotViews; layouts 1×1/1×2/2×2 via View menu; each cell has a compact scenario selector (combo listing manifest entries) + quantity selector; one cell is "active" (receives control-panel changes). D:Hard T:2d. |
| 2.2.3 Synchronization | All cells driven by the single `TimeController` tick (pull model makes this trivial); "Link color scales" toggle (shared clim = global max across shown scenarios); per-cell overrides otherwise. D:Med T:3h. |
| 2.2.4 Prefetch policy | On grid layout/scenario change, prefetch all visible (case, key) combos in background; raise `SCENARIO_CACHE_SIZE` to ≥ visible cells + 2 (manifest-aware default in `config.py`). D:Easy T:2h. |

**Wireframe:**
```
┌────────────┬──────────────────────────────────────────┐
│ controls   │ ┌───────────────┐  ┌───────────────┐     │
│ (existing  │ │ c1_d1 · TEMP  │  │ c2_d1 · TEMP  │     │
│  panel +   │ └───────────────┘  └───────────────┘     │
│  grid/     │ ┌───────────────┐  ┌───────────────┐     │
│  quantity  │ │ c1_d1 · VELO  │  │ c2−c1 · ΔTEMP │     │
│  pickers)  │ └───────────────┘  └───────────────┘     │
│            │ ── timeline ▷ ────────────●────────── ⟲  │
└────────────┴──────────────────────────────────────────┘
```
**Risk:** matplotlib FPS with 4 canvases → run M2.4 gate immediately after 2.2.2.
**DoD:** 2×2 grid, 4 different scenarios, synced playback ≥15 fps on dev machine · linked clim works · single-view mode pixel-equivalent to pre-refactor.

### M2.3 — Difference + ensemble views
| Task | Detail |
|---|---|
| 2.3.1 `DifferenceView(PlotView)` | Cell type "A − B": two scenario refs, `frame = store.get(A,key)[i] - store.get(B,key)[i]`, diverging cmap (`RdBu_r`), symmetric clim (`±max(|Δ|)` over sampled frames, cached). Files: `views.py`. D:Med T:3h. |
| 2.3.2 `EnsembleView(PlotView)` | Cell type over a scenario *selection*: mean/std/min/max composite at time i (µs-cheap on mmap'd float32). Std uses sequential cmap labeled σ(T). D:Med T:3h. |
| 2.3.3 Selection UI | Cell context menu → view type; ensemble picker = checklist of manifest entries with factor filters ("all vod=open"). D:Med T:3h. |

**DoD:** Δ view shows physically sensible structure (e.g. door-width effect near doorway) · ensemble σ view renders · both stay synced in playback · unit tests for symmetric-clim and std math.

### M2.4 — pyqtgraph decision gate (timeboxed: 2 days, then STOP)
2.4.1 Implement `PyQtGraphSliceView(PlotView)` (`ImageItem` + `ColorBarItem`); 2.4.2 benchmark 2×2 synced playback both backends. **Adopt only if** matplotlib-blit < 15 fps on the demo machine. Either way the interface stays; migration is a per-view swap, not a rewrite. *Prediction: at 49×101 px, matplotlib-blit will pass and migration defers to Phase 4 (volume rendering will want the GL stack anyway).* **DoD:** decision recorded in `docs/decisions.md` with FPS numbers.

### M2.5 — Experiment browser + summary index
| Task | Detail |
|---|---|
| 2.5.1 `summary_stats.py` | Per scenario: max T (global & per-frame curve), time-to-T>{100,300,600 °C}, mean upper-region T, peak HRR + total energy from `*_hrr.csv` (first use of this data!). Cached to `fds/sim/.cache/summaries.json`, mtime-invalidated. D:Med T:4h. Unit-test against hand-computed values for one scenario. |
| 2.5.2 Browser dock | QDockWidget table (QAbstractTableModel): factors + stats columns; sortable; text/factor filter; double-click → load into active cell; multi-select → "open as grid" / "open as ensemble". Files: new `browser.py`, `main_window.py`. D:Med T:1d. |

**DoD:** all 24 rows with correct stats · sort/filter works · double-click loads · stats regeneration only when sources change.

### M2.6 — Probe + isotherms
2.6.1 Cursor probe: `motion_notify_event` → status bar "x=…m, z=…m, T=…°C" (physical coords via slice extent — start using `readSlice`'s mesh/extent return instead of `readDataOnly`). D:Med T:3h. 2.6.2 Isotherm overlay: contour lines at configurable levels (default thresholds informed by M1.3s.4's hazard-band proposal, e.g. 60/100/300 °C) redrawn per frame **only when enabled** (accept blit bypass while active; acceptable at this grid size). D:Med T:3h.

**Known follow-on (2026-07-09d, not yet scoped into 2.6.1/2.6.2 above):** M1.6's schematic currently places the door, vents, and candle(s) at fixed proportional positions inside an extent-derived room outline — only the outline's aspect ratio is extent-derived, not the per-object positions. Once this milestone's slice-extent mapping is in place, revisit `schematic.py` to source door/vent/candle placement from real per-object geometry instead of fixed fractions, in addition to the cursor-probe and isotherm work already scoped here.

**DoD:** probe accurate at corners (extent-mapping test) · contours toggle cleanly · off-state performance unchanged.

### M3.1 — Ensemble analytics + auto-summaries (Track A)
| Task | Detail |
|---|---|
| 3.1.1 `analytics/features.py` | Per scenario feature vector: downsampled max-T curve, hot-area-fraction curve, time-to-thresholds, spatial-mean curve. Pure NumPy on cached arrays. D:Med T:3h. Deterministic → snapshot tests. |
| 3.1.2 PCA + clustering panel | sklearn PCA(2) scatter (color = cluster, marker = candle count, hover = scenario); hierarchical clustering dendrogram optional. Embedded as a dock (matplotlib static — no animation needs). New dep: scikit-learn. D:Med T:1d. |
| 3.1.3 Auto-summary | Deterministic template: "Peak 712 °C at t=142 s (near door). Exceeded 300 °C at t=97 s. Vent-open variants peaked 85 °C lower on average." → shown per scenario in browser + exportable to Markdown for all 24. D:Easy T:3h. **All numbers computed, none generated.** |

**DoD:** clusters align with at least one interpretable factor (sanity: 1-vs-2-candle separation) · summaries verified against browser stats for 3 scenarios · panel doesn't degrade playback.

### M3.2 — Forecasting model + in-app evaluation (Track B)
**Split:** training pipeline lives OUTSIDE the app (`ml/` dir, own scripts); the app only *loads predictions* — keeps torch out of the app's required deps and the demo decoupled from training.

| Task | Detail |
|---|---|
| 3.2.1 Dataset builder | `ml/dataset.py`: sliding windows (k=8 in → 1..8 out, autoregressive rollout later) from cached `.npy`; normalize per-quantity; split 20 train / 4 test **by scenario** (not by time — no leakage). D:Med T:3h. |
| 3.2.2 Baselines first | Persistence (frame t repeats) + optical-flow-free linear extrapolation. **Non-negotiable:** the FNO must beat these or the report says so. D:Easy T:2h. |
| 3.2.3 Model + training | FNO2d (`neuraloperator`) OR ConvLSTM fallback if integration friction >1 day. MPS/CPU; early stopping; save best checkpoint + config hash. D:Hard T:2–3d. |
| 3.2.4 Rollout + export | Autoregressive rollout on 4 held-out scenarios → save `predictions/<case>.npy` (same shape as truth). RMSE/SSIM vs lead-time curves → PNG + JSON. D:Med T:3h. |
| 3.2.5 In-app "Model evaluation" mode | A `PredictionSource` implementing the store's `get()` interface reading `predictions/`; browser gains "open prediction comparison" → auto-builds 1×3 grid: Simulated · Predicted · Error (DifferenceView — **reuse, zero new view code**). D:Med T:4h. |

**Risk & mitigation:** model quality unknown → baselines + error-analysis are the guaranteed deliverable; app feature works with even a mediocre model (error view becomes *more* interesting); ConvLSTM fallback pre-decided to cap architecture-fiddling time. Tasks 3.2.1–3.2.4 have no GUI dependency and may be started in parallel with Phase 2 once M1.2's disk cache lands, rather than waiting until Phase 3 — see §5 dependency graph note.
**DoD:** beats persistence baseline at ≥4-frame lead (or documented negative result) · eval grid opens from browser in ≤2 clicks · training fully reproducible from `ml/README.md` · torch NOT required to run the visualizer.

### M3.3 — Super-resolution (CONDITIONAL — skip unless new sims approved)
Re-run 24 cases at 2× coarser grid (local FDS, ~hours); train SR UNet coarse→fine; side-by-side + error view (reuse M2.3 again). Only start if M3.1+M3.2 are done before Aug 25.

### M-SIM — Simulation regeneration (CONDITIONAL — gated on secured cluster access)
**Objective:** richer raw `.sf`/`.smv` output (additional slice planes, finer mesh, or additional quantities like BNDF/PL3D) specifically to give the visualization work in M1.3/M2.x/M3.1 better source material — not a change to the app's architecture or a new scientific axis.

**Re-gated (2026-07-09b):** previously gated on a local compute-time estimate from M1.3s.5 (single-run extrapolation, since only a laptop was available). The user expects to secure cluster access, which changes the shape of this milestone: **once cluster access is confirmed, run the full existing simulation set on the cluster first** (sim.0, below) — this replaces the earlier "extrapolate from 1–2 local runs" approach with actual full-batch numbers, and produces a clean, validated baseline *before* any `.fds` template edits are attempted. Only after that baseline run is complete does work move into the FDS modeling/template-editing side (sim.1 onward).

**Do not start `.fds` template edits (sim.1+) without:**
- **Parser validation confirmed solid** (M1.3s completed with agreement demonstrated, plus re-validation after any subsequent parser changes) — explicit gate per 2026-07-09c. sim.4's re-validation of *new* output is the downstream counterpart of this gate, not a substitute for it: both are required, not optional.
- Cluster access confirmed and sim.0 (full-batch baseline run) completed successfully — sim.0 itself is *not* gated on validation status, since it doesn't touch the template
- Explicit supervisor/user approval of the *additional* compute time for regenerated runs, now estimated from real cluster throughput (sim.0's numbers) rather than local extrapolation
- Confirmation this doesn't silently expand the 2×2×3×2 factorial design (§1.5's ML feasibility table depends on 24 fixed categorical points; adding continuous parameter sweeps here would contradict trade-off decision §7.5 and needs separate discussion, not a side effect of "better output")

| Task | Detail |
|---|---|
| sim.0 **(New)** Cluster baseline run | Once cluster access is secured: run the existing 24-scenario set as-is on the cluster (no template changes yet). Purpose: (a) get real wall-clock/throughput numbers to replace the local extrapolation, (b) produce a clean regenerated baseline to diff against the current on-disk data as a sanity check, (c) shake out any cluster-environment issues (FDS version match, MPI config, output paths) before they're compounded by template edits. D: Med, T: depends on cluster queue/throughput — report actual numbers back before proceeding. |
| sim.1 Scope the actual gap | From M1.3s.5's findings (and sim.0's baseline diff, if it surfaces anything), enumerate exactly what's missing (e.g. finer mesh resolution, BNDF wall-temperature output, PL3D volumetric snapshots) vs what's just unused-but-present (VELOCITY, already free via M2.1). Don't regenerate data that's already on disk. |
| sim.2 Edit `fds/template.fds` | Minimal diff for the scoped gap only. Version the old template alongside the new one. |
| sim.3 Re-run affected scenarios on cluster | With sim.0's real throughput numbers in hand, this is now a scheduling question, not a feasibility question — report actual cluster wall-clock cost. |
| sim.4 Validate new output | Re-run the M1.3s cross-validation approach (`fdsreader==10.1` comparison) against the new output before treating it as trustworthy. Should include a `slice.py` vs `fdsreader` comparison specifically on the newly-generated data, feeding the reader-choice decision in §7. |
| sim.5 Update manifest/fixtures if needed | If M2.1's manifest already exists by this point, regenerate it; refresh `tests/fixtures/` only if the new output changes what the parser tests assert. |

**DoD:** cluster access confirmed · sim.0 baseline run completed with real throughput numbers reported · template-edit approval documented against those real numbers (not the earlier local extrapolation) · new output cross-validated · no unapproved expansion of the factorial design · existing pytest suite still green against old+new data.

---

## 5. Dependency graph

```
M1.1 tests/packaging ──────────────┬──────────────────────────────┐
                                   ▼                              │(safety net for all)
M1.2 disk cache ──► M1.4 QTimer+timeline ──► M2.2 multi-view ──► M2.3 diff/ensemble ──► M3.2 forecasting eval
      │                     ▲                      ▲                      ▲
      │             M1.3 blit/cmap ◄── M1.3s spike │ (SliceView absorbs)  │
      │                                                                   │
      └────► M2.1 manifest+quantities ──► M2.2   ┌─► M3.1 analytics ──────┘ (features from cache)
                       │                          │
                       └──► M2.5 browser ─────────┴─► M3.2 (launch eval from browser)

M1.5 export (independent) · M1.6 schematic/GUI pass (pure UI, no data-layer *changes*; reads
  already-parsed `.smv` mesh extents for proportions — same data path M1.3s validates and
  M2.6's probe will use, so M1.3s findings apply to it but do not block it)
M2.4 gate (after 2.2.2) · M2.6 probe (after M2.2)
M-SIM: sim.0 (cluster baseline) gated only on cluster access being secured — independent of
  everything else INCLUDING validation status (it doesn't touch the template) and can run
  anytime once access exists.
M1.3s validated (parser confirmed solid) ──► M-SIM sim.1+ (template edits) — explicit gate,
  2026-07-09c: sim.1+ additionally requires sim.0's numbers + approval. Active track until the
  gate clears = GUI (M1.6 + follow-on polish) and parser validation (M1.3s + re-validation
  after any parser change). M-SIM feeds M1.3/M2.x/M3.1 with richer data if approved, but does
  not block them — the app works fully on current data regardless.
M3.2.1–3.2.4 (dataset/baselines/training/rollout) may start once M1.2 lands, in parallel
  with Phase 2; only M3.2.5 (in-app eval) waits on M2.3.
```

## 6. Backlog (MoSCoW)

**Must Have (demo-critical):** M1.1–M1.4, M2.1, M2.2, M2.3, M2.5, M3.1
**Should Have:** M1.5 export, M2.6 probe/isotherms, M3.2 forecasting, M2.4 gate
**Nice to Have:** bookmarks, annotations, dockable layout persistence, per-view export, histogram panel, session files
**Future Research (Phase 4):** volume rendering (.s3d), arbitrary slice planes, continuous-parameter surrogates + inverse design, super-resolution (M3.3), M-SIM (if not done in Phase 3), scenario-space active sampling, web viewer, plugin API

## 7. Key trade-off decisions (recorded so the agent doesn't relitigate)

1. **matplotlib+blit now, pyqtgraph behind a gate.** Grids are 49×101 px; blitting almost certainly suffices even 2×2. Migration cost (colorbar, toolbar, theming rework) is only justified by measured FPS failure or by Phase 4's 3D needs. The `PlotView` seam makes this reversible — that seam is the real decision.
2. **Pull-based QTimer playback replaces the push worker** — but only *after* the disk cache lands (ordering is load-bearing). Eliminates the last unsynchronized shared-state pattern; makes seek/multi-view trivial.
3. **No plugin system, no database, no Zarr yet.** 24 scenarios × 230 MB and one developer: a manifest JSON + `.npy` cache + registry dicts deliver the same capability without the abstraction tax. Each has a written trigger for revisiting (contributor count, scenario count, RAM ceiling).
4. **AI as two tracks:** analytics that cannot fail (ship before demo) + one serious model whose evaluation UI ships regardless of model quality. Never let the demo depend on a training run.
5. **Surrogates/inverse design deferred** — not because they're bad ideas but because a 24-point categorical design gives them nothing to learn; they become real when continuous parameter sweeps are generated (Phase 4).
6. **ML code lives outside the app** (`ml/`), communicating via `.npy` prediction files. The visualizer must never require torch to start.
7. **Parser validation via `fdsreader`, not Smokeview automation.** Smokeview is a compiled interactive GUI tool — not scriptable for pixel-level automated comparison without disproportionate effort. `fdsreader` (maintained Python package) gives numeric cross-validation cheaply; Smokeview is used only for a manual visual-convention review (color bands), recorded as a recommendation, not wired into CI.
8. **Simulation regeneration is gated on cluster access, not a fixed local-compute estimate.** "Improve the simulation for better output" sounds like a small ask but touches upstream FDS runs and compute time this plan didn't originally budget for. Now that cluster access is expected, the gate moved from "approve a local-extrapolated time estimate" to "run the full existing set on the cluster first (sim.0), then decide" — still conditional, still requires approval before template edits, just sequenced to use real numbers instead of a laptop-based guess.
9. **Schematic (M1.6) vs on-heatmap overlay: the overlay stays deferred.** *(Refined 2026-07-09c: the inset diagram is no longer merely "illustrative" — its proportions are now extent-driven, see #10. What remains deferred is unchanged:)* placing candle/door/vent icons at exact physical coordinates **on the heatmap** depends on the slice-extent mapping work already planned for M2.6 in Phase 2. Rather than block the user-requested GUI-realism work on that dependency, M1.6 ships the inset diagram now and treats precise on-heatmap placement as an explicit stretch task revisited once M2.6 lands. This keeps the highly-visible "make it feel real" request from either slipping into Phase 2 or rushing a physically-inaccurate overlay onto the actual data view.
10. **Schematic accuracy sourced from parsed `.smv` mesh extents, not new physical measurements (2026-07-09c).** A physical mockup of the room/candle/door/vent setup will sit beside the app at the demo, so the schematic must reasonably resemble the real layout — but exact physical dimensions aren't available yet, and gating a user-prioritized Phase 1 milestone on an external measurement input would stall it. The mesh extents already parsed from the `.smv` files are the best available source of true geometry: they already flow through the app, they're the same extent data M2.6's probe will use, and they sit on the same data path M1.3s's `fdsreader==10.1` cross-validation checks for correctness — so M1.6 inherits validation confidence for free. Decision: proportion the schematic from parsed extents with zero new inputs to start; if the mockup later yields exact measured dimensions, a small manual dimension-input override refines the diagram where extents and reality differ. Measurement is a refinement, never a dependency.
11. **Reader choice (`slice.py` vs `fdsreader`) is deliberately deferred, not decided now.** Both are validated as roughly equivalent on current data (M1.3s). Committing to a full replacement now would mean re-benchmarking and re-testing a proven, fast, already-integrated parser against an unbenchmarked one, for no concrete capability gap today. The decision point is naturally after M-SIM regenerates data with a possibly-different FDS version/format — at that point, whichever reader handles the actual new output better (not hypothetically, but measured) becomes the default. Until then, `slice.py` stays primary; `fdsreader` stays available for cross-validation and as a fallback if a new quantity from M-SIM isn't yet supported in `slice.py`.

## 8. Git workflow

- **Branches:** `main` = always demo-runnable. One branch per milestone: `feat/m1.2-disk-cache`, etc. Spikes use `spike/<name>` and are not required to merge to `main` (docs-only merge acceptable).
- **Commits:** conventional style (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`); one logical change ≤ ~300 LOC per commit; every commit leaves the app launchable.
- **Merging:** merge to `main` only when the milestone's DoD checklist passes (pytest green + manual launch check + benchmark where specified). Tag `m1.2` etc. on merge.
- **Demo safety:** tag `demo-rc1` on Sep 4 (feature freeze); only `fix:` commits to `main` afterward; tag `demo-2026-09-11` after rehearsal. Keep `fds/sim/.cache/` and `predictions/` out of git; fixtures ARE in git.

## 9. Timeline

| Dates | Work | Exit criterion |
|---|---|---|
| Jul 8–16 | M1.1 (done) → M1.2 → M1.3s spike (parallel) → M1.3 → M1.4 → **M1.6 (revised 2–2.5 d, 2026-07-09c)** (M1.5 only if slack remains — it is the item that slips into early Phase 2 to absorb M1.6's growth) | Seekable, smooth, tested, cross-validated app, legible to non-specialists (extent-proportioned schematic, plain-language labels, explainer tooltips); warm switches <100 ms |
| Jul 16–24 | M2.1 → start M2.2 (M1.5 lands here if it slipped; M3.2.1–3.2.4 may start in parallel once M1.2 lands; **M-SIM sim.0 cluster baseline run whenever cluster access is confirmed — not date-dependent, slot in as soon as available**) | VELOCITY visible; SliceView extracted |
| Jul 25–Aug 5 | M2.2 finish → M2.4 gate → M2.3 | 2×2 synced grid + Δ/ensemble views @ ≥15 fps |
| Aug 6–12 | M2.5 → M2.6 | Browser drives everything; probe/isotherms |
| Aug 12–19 | M3.1 | Clustering panel + auto-summaries in app |
| Aug 19–Sep 2 | M3.2 (baselines → model → eval view, training likely already underway) | Prediction-vs-truth grid opens from browser |
| Sep 3–4 | Freeze, `demo-rc1` | No open P1 bugs |
| Sep 7–11 | Rehearsal, fallbacks, demo script | Demo delivered |

*Cut order if behind: M3.3/M-SIM's template-edit phase (sim.1+, already conditional and now also gated on validation being confirmed — note sim.0's cluster baseline run is cheap/independent and worth doing even under time pressure, since it costs no developer-day, only cluster queue time) → M1.6's stretch task (1.6.6 overlay) first — core M1.6 is now 1.6.1–1.6.5 (labels + explainers included, per 2026-07-09c) and is user-prioritized; it should NOT be an early cut → M2.6 → M1.5 → M2.4's migration half → reduce M3.2 to baselines + eval-view with persistence "predictions" (the UI story still works).*
