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
| ✅ | M2.1: `manifest.py` (`manifest.json`, derived — not assumed — candles/door/vod/voc factor indices from actual `fds/sim/*` folder names; byte-for-byte identical to the old hardcoded `build_data_matrix(2,2,3,2)` on the real 24-scenario set, now unconditional on that assumption holding); `slice_key.py` (`SliceKey`, `available_slices()` — confirms VELOCITY is present at the same slice plane TEMPERATURE uses); `ScenarioStore`/`load_data` keyed by `(scenario, SliceKey)`, disk-cache filenames include the key; quantity combo box (Temperature / Air speed) with per-quantity colormap+clim+label defaults (`config.QUANTITY_DISPLAY`), wired through the same M1.4.4 busy-cursor/prefetch machinery, now key-aware | `src/manifest.py`, `src/slice_key.py`, `src/scenario_store.py`, `src/load_data.py`, `src/simulation_controller.py`, `src/main_window.py`, `src/config.py`, `tests/test_manifest.py`, `tests/test_slice_key.py`, merged to `main` |
| ✅ | M2.2: `views.py` — `PlotView` protocol + `SliceView` (matplotlib, extracted whole from `MainWindow`) + `GridCell`/`ViewGrid` (1×1/1×2/2×2 via View→Grid Layout menu, per-cell scenario+quantity combos, one active cell driven by the control panel, others independent). Single-view mode is the grid at its 1×1 default, not a separate path — pixel-equivalent to pre-refactor by construction (confirmed: all ~30 pre-existing single-view tests pass unchanged). Playback loops every visible cell each tick; "Link color scales" shares a data-derived vmax per same-quantity group. Non-active cells' own combo picks prefetch via the existing M1.4.4 worker-list machinery (no new threading code); `SCENARIO_CACHE_SIZE` raised 4→6. Measured 2×2 synced-playback FPS: ~247 fps offscreen, **~57.7 fps on a real display** (M2.4 re-measured this on-screen, see below — real number is the one that matters), both well past the ≥15 fps DoD target | `src/views.py`, `src/main_window.py`, `src/config.py`, `tests/test_views.py`, `tests/bench_grid_fps.py`, `tests/test_integration.py`, merged to `main` |
| ✅ | M2.4: pyqtgraph migration gate — **decision: no migration.** matplotlib-blit measured directly at ~247 fps offscreen / ~57.7 fps real-display (native `cocoa` backend, actual on-screen window) on the same 2×2/4-scenario grid, both clearing the ≥15 fps bar with wide margin; `PyQtGraphSliceView` deliberately not built (would be speculative work once the gate's own condition wasn't met). Real-display number confirms the offscreen figure alone would have overstated performance ~4×, same class of gap M1.3.3 first found | `docs/decisions.md`, `ROADMAP.md` §4 M2.4, no code changes |
| ✅ | M2.3: `DifferenceView`/`EnsembleView` (both compose a `SliceView` internally) + a per-cell right-click context menu (Slice/Difference/Ensemble) with a new `EnsemblePickerDialog` (checklist + quick factor filters) for building an ensemble selection. Verified `DifferenceView` against real data before any UI wiring, per explicit instruction: for TEMPERATURE the door-width DoD example's dominant signal is actually near the candle/plume, not the doorway — VELOCITY shows the expected door effect instead; both findings pinned as permanent tests, not just a one-off check | `src/views.py`, `src/main_window.py`, `tests/test_views.py`, `tests/test_integration.py`, merged to `main` |
| ✅ | M2.5: `summary_stats.py` builds a cached per-scenario summary index, including global/per-frame peak temperature, time-to-thresholds, mean upper-region temperature, and first use of `*_hrr.csv` for peak HRR + total energy; `browser.py` adds a docked sortable/filterable `QTableView` with factor filters, double-click-to-load active cell, and multi-select open-as-grid/open-as-ensemble actions. Real-data sanity check: 24 summary rows generated; first/last scenarios show nonzero HRR-derived stats (`c1_d0_vod0_voc0`: peak HRR 0.08 kW, 9.15 kJ; `c2_d1_vod2_voc1`: peak HRR 0.16 kW, 18.29 kJ) | `src/summary_stats.py`, `src/browser.py`, `src/main_window.py`, `tests/test_summary_stats.py`, `tests/test_integration.py`, merged to `main` |
| ✅ | M2.6: parser geometry metadata (`readSliceGeometry`/`combineSliceGeometry`, `ScenarioStore.get_extent`) + physical-extent-aware `SliceView`/`DifferenceView`/`EnsembleView` (verified against real data: a known M2.3 reference pixel round-trips through the new `value_at()` to the same physical coordinates; all 4 corners match exactly) + cursor probe (status bar shows x/z/value in physical units, wired for every grid cell) + isotherm overlay (View-menu toggle, M1.3s.4 hazard-band defaults, applies grid-wide, zero cost while off — verified, not assumed). M1.6's per-object door/vent/candle placement follow-on investigated and re-confirmed deferred (third mention) — `.smv` `OBST`/`VENT` records exist but are mesh-relative cell-index geometry, a materially different parsing task than the room-extent bounding box this milestone reads | `src/fds/slice/slice.py`, `src/load_data.py`, `src/scenario_store.py`, `src/data_provider.py`, `src/views.py`, `src/main_window.py`, `src/config.py`, `tests/test_views.py`, `tests/test_integration.py`, merged to `main` |

### Current architecture
```
main.py (bootstrap, splash)
   └─ main_window.py (MainWindow — view only)
        └─ views.py (PlotView protocol, SliceView, GridCell, ViewGrid — M2.2;
                     DifferenceView, EnsembleView, EnsemblePickerDialog — M2.3;
                     physical extent + isotherms + cursor probe on SliceView — M2.6)
        └─ browser.py (docked experiment browser QTableView — M2.5)
        └─ simulation_controller.py (SimulationController + _Worker QThread)
             └─ data_provider.py (SimulationData / DataLoadError / demo fallback)
                  └─ manifest.py (ScenarioEntry list ← fds/sim/* folder names, manifest.json)
                  └─ summary_stats.py (cached per-scenario stats incl. HRR — M2.5)
                  └─ scenario_store.py (ScenarioStore: lazy + LRU, thread-safe, keyed by
                                        (scenario, SliceKey); get_extent() geometry cache — M2.6)
                       └─ load_data.py (single-scenario, key-aware slice + geometry loader)
                            └─ fds/slice/slice.py (.smv/.sf binary parser)
slice_key.py = SliceKey (quantity/direction/offset) + available_slices() (M2.1)
config.py = shared constants (N_CANDLES..., DEFAULT_*, FRAMES_PER_SECOND, SCENARIO_CACHE_SIZE, QUANTITY_DISPLAY)
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
| M2.1 | Scenario manifest + quantity/slice generalization | Unlocks VELOCITY (already on disk) + all future quantities; kills triple-hardcoding | High | 3 d | Med | **High** — ✅ done |
| M2.2 | PlotView abstraction + multi-view grid (1×1/1×2/2×2, synced time, linked clim) | The platform centerpiece; matches dataset's purpose | **Highest** | 5 d | Med | **High** — ✅ done |
| M2.3 | Difference view + ensemble stats view | Core science; doubles as ML-eval harness | High | 3 d | Low (array ops on cached data) | **High** — ✅ done |
| M2.4 | pyqtgraph spike — timeboxed decision gate (2 d max) | Only migrate if matplotlib-blit can't hold the 2×2 grid at target FPS | Med | 2 d | Contained by timebox | Med — ✅ done (no migration) |
| M2.5 | Experiment browser + summary-stats index (incl. HRR from CSV) | Workflow leap; makes 24 runs navigable | High | 3 d | Low | **High** — ✅ done |
| M2.6 | Value probe + isotherm/contour overlays (thresholds informed by M1.3s) | Analysis affordances researchers expect | Med | 2 d | Low | Med — ✅ done |

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
8. **Experiment browser (M2.5)** — turns "24 folders" into "an experiment". ✅ done.
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

### M2.1 — Scenario manifest + quantity generalization ✅ DONE
Merged to `main`. **Objective met:** VELOCITY is reachable from the UI; the candles/door/vod/voc→case_index mapping is now derived from real folder names instead of an unverified nested-loop assumption.

| Task | Detail |
|---|---|
| 2.1.1 `manifest.py` + `manifest.json` ✅ | `scan_scenarios()` parses `c<n>_d<n>_vod<n>_voc<n>` from each `fds/sim/*/` folder name; factor indices are ranked from the *actual* distinct raw values seen per factor (not an assumed count), so a factor with only 1 level on disk gets index 0 for everyone rather than silently matching a hardcoded `N_DOORS=2`. `get_manifest()` loads `fds/sim/manifest.json` if present, else scans and writes it (gitignored alongside `fds/sim/`, same convention as the `.cache/` disk cache). Malformed folder names are skipped with a warning, not fatal. **Verified**: `data_matrix_from_manifest()` on the real 24-scenario set is byte-for-byte identical to the old hardcoded `build_data_matrix(2,2,3,2)` — confirms the replacement preserves current behavior while removing the assumption. Files: `manifest.py`, `data_provider.py` (real-data path now sources `folders`/`data_matrix` from the manifest; demo-data path unchanged, still uses `config.N_CANDLES` etc. since there's no real `.smv` to derive from). Tests: `tests/test_manifest.py` (11 tests: case_index ordering, factor-index derivation, malformed-name skipping, save/load round-trip, `get_manifest` cache/regenerate/corrupted-file paths, parity with old `build_data_matrix`) — synthetic folders under `tmp_path`, not the real dataset (matches this suite's existing convention of not depending on the gitignored `fds/sim/`). |
| 2.1.2 `SliceKey` + quantity inventory ✅ | New `slice_key.py`: `SliceKey(quantity, direction, offset)` frozen dataclass (hashable — used directly as a cache key); `available_slices(root_dir)` reads a scenario's `.smv` via the existing `readSliceInfos`/`scanDirectory` and returns deduplicated `SliceInfo(key, label, units)` entries. Confirmed against the real dataset: TEMPERATURE and VELOCITY both appear at the slice plane the app already renders (direction=1, offset=0), plus a second offset=15 plane not currently surfaced in the UI (out of scope here — M2.1 only generalizes *which quantity*, not *which plane*; see 2.1.4). Files: `slice_key.py`. Tests: `tests/test_slice_key.py` (5 tests, run against `tests/fixtures/c1_d0_vod0_voc0` — its `.smv` describes VELOCITY even though the fixture's `.sf` data was trimmed to TEMPERATURE-only, since `available_slices()` only reads `.smv` metadata). |
| 2.1.3 Store keyed by (scenario, SliceKey) ✅ | `ScenarioStore.get(scenario_index, key=DEFAULT_SLICE_KEY)`/`is_cached(...)`: in-memory LRU cache is now keyed on `(scenario_index, key)` tuples instead of bare `scenario_index`; disk-cache filenames include the key's quantity/direction/offset so two quantities for the same scenario never collide on one `.npy` file. `key` defaults to the original TEMPERATURE/dir1/offset0 combo, so every pre-M2.1 single-arg call site (`main.py`, several in `main_window.py`) kept working unchanged without being touched. `load_data.load_data(root_dir, key=DEFAULT_SLICE_KEY)` takes the same key. **Also extended `simulation_controller.py`'s `_PrefetchWorker`/`SimulationController.prefetch()`/`is_cached()` with the same optional `key` param** — flagged to the user before touching that file per the standing threading rule; approved as a mechanical, backward-compatible parameter-threading change with no changes to locking or the M1.4.4 worker-lifecycle-list mechanism. Files: `scenario_store.py`, `load_data.py`, `simulation_controller.py`. Tests: 3 new in `tests/test_scenario_store.py` (default-key parity, independent per-key caching, per-key `is_cached`), 1 new in `tests/test_disk_cache.py` (distinct `.npy` filenames per key), 2 new in `tests/test_simulation_controller.py` (key propagation through `prefetch`/`is_cached`) — all against real data shapes, verified end-to-end against the real dataset interactively (cold TEMPERATURE/VELOCITY loads, warm hits, disk-cache round trip) though not as committed pytest tests (the real `fds/sim/` isn't part of the committed fixture). |
| 2.1.4 Quantity selector UI ✅ | Combo box ("Temperature" / "Air speed" — plain-language, no bare `VELOCITY`) in a new "Data shown" control-panel section, populated from `available_slices()` filtered to the app's existing slice plane; falls back to a single disabled "Temperature" entry in demo mode (no real `.smv` to discover from). Per-quantity colormap/vmin/slider-range/label/unit defaults live in `config.QUANTITY_DISPLAY` (`TEMPERATURE`: `gist_heat`, vmin=20°C i.e. `AMBIENT_C`, 50–1000°C slider; `VELOCITY`: `viridis`, vmin=0, 1–10 m/s slider — an engineering estimate from observed on-disk magnitudes ~0–4 m/s, not a physical bound, adjustable via the existing slider same as TEMPERATURE). Switching quantity reuses the exact M1.4.4 busy-cursor/prefetch machinery (now key-aware per 2.1.3) rather than a separate code path. Files: `main_window.py`, `config.py`. Tests: 3 new in `tests/test_integration.py` (combo lists both quantities against real data, switching updates heatmap/colorbar/clim, demo-mode combo is disabled with one entry). |

**Known limitation (not fixed, documented rather than silently shipped):** `_pending_load_case` (the busy-state tracker `main_window.py` already had from M1.4.4) tracks `case_idx` only, not `(case_idx, key)`. A scenario toggle and a quantity switch that both resolve to the same `case_idx` while both are cache misses can race — whichever background prefetch finishes first ends the busy state, even if it wasn't for the currently-selected quantity. Bounded, self-correcting effect (one extra synchronous-parse hitch in `_sync_current_scenario`, not a crash); fixing it properly would mean widening `prefetch_finished`/`prefetch_error`'s signal signatures to carry the key too, which is a larger threading-adjacent change than the one already approved for this milestone — left for a follow-up if it proves noticeable in practice.

**Opportunistic fix (found while making the store interface key-aware, per ROADMAP's standing "known defects" convention):** `DemoScenarioStore` never implemented `is_cached()` — any scenario-param toggle in demo mode raised `AttributeError`. Fixed (`data_provider.py`); regression test added (`test_demo_mode_scenario_toggle_does_not_crash`).

**Pre-existing test flakiness, now reproducing deterministically in this environment (not caused by M2.1):** the cursor-stack flakiness already filed under known defects (§0.4) — `tests/test_integration.py::test_scenario_switch_cache_miss_does_not_block_gui_thread` and `::test_stale_prefetch_error_does_not_discard_newer_success` — was previously estimated at ~1-in-6 full-suite runs. In this dev environment it now reproduces on every full-suite run (both tests always fail together, always pass individually), confirmed via `git stash` to be identical on pre-M2.1 `main` — not a regression from this milestone. Left untouched per the standing threading/`simulation_controller.py` stop-and-ask rule; worth re-investigating given the higher, now-deterministic reproduction rate.

**DoD:** user switches TEMPERATURE↔VELOCITY live (✅) · correct units on colorbar (✅ "Temperature (°C)" / "Air speed (m/s)") · manifest is the only place factor structure lives for real data (✅ `data_provider.py`'s real-data path; demo-data path still uses `config.N_CANDLES` etc. by necessity — no real folders to derive a manifest from) · tests cover both quantities (✅ 21 new tests across `test_manifest.py`, `test_slice_key.py`, `test_scenario_store.py`, `test_disk_cache.py`, `test_simulation_controller.py`, `test_integration.py`) · pytest green (✅ 105/107; the 2 known-flaky, pre-existing, unrelated failures above).

### M2.2 — PlotView abstraction + multi-view grid ✅ DONE
Merged to `main`. **Objective met:** the comparison instrument works — a 2×2 grid of 4 independently-selectable scenarios, synced playback, linked color scales, single-view mode unchanged.

| Task | Detail |
|---|---|
| 2.2.1 Extract `SliceView` ✅ | `PlotView` Protocol + matplotlib `SliceView` (axes/heatmap/colorbar/blitting) extracted whole from `MainWindow` into new `views.py`. `MainWindow` keeps thin `heatmap`/`colorbar`/`canvas`/`ax` properties delegating to the active cell's view, so ~30 pre-existing single-view tests (vmin pinning, blit playback, colormap/interpolation toggles, temperature slider, etc.) needed zero changes and all still pass — the actual, verified form of "single-view behavior unchanged." Files: `views.py`, `main_window.py`. |
| 2.2.2 `ViewGrid` container ✅ | `GridCell` (compact scenario combo — manifest-backed, folder-name labels — + quantity combo, above a `SliceView`) + `ViewGrid` (QGridLayout, 1×1/1×2/2×2 via new View→Grid Layout menu). Growing the grid creates new cells (each initialized with real data via `cell_created`); shrinking hides cells without destroying them, so per-cell state survives a shrink/regrow cycle. Clicking a cell makes it "active"; **design decision confirmed with the user before implementing:** the control panel (candles/door/vod/voc toggles, quantity combo, colormap menu, display-scale slider) edits the active cell only — other visible cells keep their own last-set scenario/quantity/clim/colormap independently. Single-view mode *is* the grid at its 1×1 default (not a separate code path), which is what makes 2.2.1's pixel-parity claim automatic rather than something to separately re-verify. The pan/zoom/save toolbar stays bound to the first cell's canvas for its whole lifetime (`NavigationToolbar2QT` isn't meant to be rebound at runtime) and is shown only in 1×1 mode — pan/zoom is ambiguous across an ensemble grid where every cell should show the same framing. Files: `views.py`, `main_window.py`. |
| 2.2.3 Synchronization ✅ | `_on_time_changed` loops every visible cell each tick instead of redrawing a single view (pull model, no per-cell timers). "Link color scales" (View menu checkbox) groups visible cells by quantity (mixing vmax across different quantities/units wouldn't be physically meaningful) and sets each group's vmax to the real data max across that group, recomputed on structural changes (layout/scenario/quantity change) — not every tick, which would be wasteful. Files: `main_window.py`. |
| 2.2.4 Prefetch policy ✅ | A non-active cell's own combo picking an uncached (case, key) prefetches in the background instead of blocking the GUI thread — **reuses `SimulationController.prefetch()`/the M1.4.4 worker-list machinery as-is** (a second pair of slots on the same `prefetch_finished`/`prefetch_error` signals, independent of the active-cell busy-state bookkeeping) rather than writing new threading code, since that machinery already does exactly "warm a (case, key) in the background." **Deliberate scope line:** a *brand-new* cell's very first render (when the grid grows) still does a synchronous init — making that non-blocking too would need a "loading" placeholder frame state, judged out of scope for this task's D:Easy/T:2h sizing; that cold-parse hitch stays the same bounded/self-correcting shape already characterized for the M2.1 quantity-switch race. `SCENARIO_CACHE_SIZE` raised 4→6 (`config.py`) — the cache now keys on (scenario, quantity) pairs since M2.1, and a 2×2 grid can hold up to 4 distinct combos at once. Files: `config.py`, `main_window.py`. |

**Deviation from the spec's literal task list (not a gap, a design decision made explicit before coding):** the spec didn't say where the shared color-scale slider/colormap menu apply when unlinked; asked the user rather than guessing — confirmed "active cell only," which is what 2.2.2/2.2.3 above implement.

**Tests:** 22 unit tests in `tests/test_views.py` (SliceView init/setters; GridCell combo signals + `set_*_silently` non-emitting updates + active-border styling; ViewGrid layout switching, cell creation/preservation across shrink-regrow, active-cell tracking, signal relay, accent propagation) · 10 integration tests in `tests/test_integration.py` (menu-driven layout switch incl. demo-mode fallback, toolbar visibility, control-panel-edits-active-cell-only, click-to-activate syncs the control panel from the clicked cell, linked vs unlinked clim, playback ticks every visible cell, non-active cell scenario change prefetches without blocking, grid state survives shrink/regrow) · `tests/bench_grid_fps.py` (new, not pytest-collected, matches the `bench_rendering.py`/`bench_loading.py` convention) measuring the real 2×2 synced-playback path.

**DoD:** 2×2 grid, 4 different scenarios, synced playback ≥15 fps on dev machine (✅ measured ~247 fps offscreen — confirms §2.4's prediction that matplotlib-blit holds at this grid size; **M2.4's pyqtgraph gate is not urgent** based on this number, though the gate itself is still a separate, not-yet-run milestone) · linked clim works (✅ verified: same-quantity visible cells share an identical, data-derived vmax) · single-view mode pixel-equivalent to pre-refactor (✅ by construction — see 2.2.1 — and confirmed by the full pre-existing single-view test suite passing unchanged, 105/107 full-suite total; the 2 failures are the pre-existing, already-documented, unrelated cursor-stack flake).

### M2.3 — Difference + ensemble views ✅ DONE
Merged to `main`. **Objective met:** both new cell types render, verified against real data, and are reachable from the grid UI via a per-cell context menu.

| Task | Detail |
|---|---|
| 2.3.1 `DifferenceView(PlotView)` ✅ | Composes a `SliceView` internally (same rendering/blitting; only frame data + display defaults differ): `frame = store.get(A,key)[i] - store.get(B,key)[i]`, diverging `RdBu_r`, symmetric clim `±max(|Δ|)` sampled over up to 20 frames and cached per `(case_a, case_b, key)`. **Verified against real data before being wired into any UI**, per explicit instruction — see "Real-data finding" below. Files: `views.py`. |
| 2.3.2 `EnsembleView(PlotView)` ✅ | Same compose-a-`SliceView` split. `mean`/`min`/`max` keep the quantity's own absolute-value display conventions (still readings of that quantity); `std` gets its own sequential `viridis` cmap (always ≥0, no natural quantity floor) and a data-derived vmax (`std_vmax`, sampled + cached like `symmetric_clim`), colorbar labeled `σ(<quantity>)`. Visual spot-check against the same 4-scenario set used for 2.3.1's verification: std concentrates exactly at the fire source, where the selected scenarios actually differ (candle count, door width) — physically sensible. Files: `views.py`. |
| 2.3.3 Selection UI ✅ | `GridCell` gained a right-click context menu (Slice/Difference/Ensemble) that swaps its `PlotView` instance and rebuilds its header per type: difference mode → two scenario combos ("A − B"); ensemble mode → a "Select scenarios…" button opening a new `EnsemblePickerDialog` (checklist of manifest entries + quick factor-filter buttons — "Wide door", "2 candles", etc. — that bulk-check every matching scenario, the spec's "checklist … with factor filters" ask) plus a mean/std/min/max stat combo. `DifferenceView`/`EnsembleView` gained `heatmap`/`colorbar`/`canvas`/`ax` passthrough properties so they duck-type as `SliceView`-shaped for `MainWindow`'s existing delegating properties if a non-slice cell becomes the grid's active cell. `MainWindow._on_time_changed` dispatches per `cell_type` each tick (`_frame_for_cell`); `_render_difference_cell`/`_render_ensemble_cell` handle (re)compute-and-draw on type/scenario/quantity changes, synchronously — same deliberate no-prefetch scope decision M2.2.4 made for non-active-cell combo changes, for the same reason (bounded, self-correcting hitch, not worth a loading-placeholder state for this milestone's sizing). `_apply_link_clim` explicitly skips non-slice cells — their clim conventions (symmetric-around-zero, sigma-scale) aren't the "biggest value across cells" notion linking was built for. Files: `views.py`, `main_window.py`. |

**Real-data finding (2.3.1's required verification, not skipped):** checked the exact scenario pair ROADMAP's own DoD names — `c1_d0_vod0_voc0` (door height 0.05 m) vs `c1_d1_vod0_voc0` (door height 0.15 m), confirmed via the `.fds` files to differ in *only* that one `&HOLE` z-extent. For **TEMPERATURE** (the quantity the diff view defaults to), the dominant |Δ| signal is near the **candle/plume** (x≈0.90–0.96 m), not the doorway (x≈0.25–0.29 m) — roughly 15–30× larger there across sampled frames, and visually a coherent plume-shaped structure, not noise (screenshot-confirmed). The door band's own TEMPERATURE signal is actually *smaller* than an arbitrary control band elsewhere in the room — real but too small to separate from ambient room-wide variation at this coarse a check. For **VELOCITY** (a direct airflow measure), the door band *does* exceed the control band on both a per-frame-max and a time-averaged basis — this is where the ventilation effect the DoD's example was describing actually shows up. Both results are physically sensible and spatially coherent; the DoD's illustrative example just wasn't dominant for the quantity it named it for. Pinned as permanent regression tests (`TestDifferenceViewRealData`, `tests/test_views.py`, auto-skipped if `fds/sim/` isn't present) rather than left as a one-off finding.

**Tests:** unit tests for symmetric-clim math (`TestDifferenceView`: max-abs-over-samples, symmetry, caching, per-cache-key independence) and std math (`TestEnsembleView`: mean/min/max/std correctness, per-frame-index variation, `cmap_for`/`label_for`, `std_vmax` caching) — the DoD's explicit ask. Plus `TestDifferenceViewRealData` (4 tests, real data, see finding above), `TestEnsembleViewRealData` (2 tests, min≤mean≤max ordering + std≥0 on a real 4-scenario selection), `TestGridCellTypeSwitching` (12 tests: context-menu type swap, header rebuild, signal contracts, picker wiring), `TestEnsemblePickerDialog` (5 tests: checklist, factor filters — including that filters are additive, not exclusive — select-all/none, initial-selection pre-check), and 7 `MainWindow`-level integration tests (diff renders immediately with a symmetric clim, scenario-combo change recomputes and redraws, an ensemble cell stays blank until scenarios are picked, `std` stat uses a zero floor + `σ` label, playback ticks redraw both non-slice cell types, linking ignores them). 63 new tests in `test_views.py`, 7 in `test_integration.py`. Full suite: 185/188 (the 3 pre-existing, already-documented, unrelated cursor-stack-flake failures).

**DoD:** Δ view shows physically sensible structure (✅ — verified against real data, with the honest correction above rather than the literal "near doorway" claim) · ensemble σ view renders (✅ — screenshot + tests) · both stay synced in playback (✅ `_on_time_changed` dispatches every visible cell by type each tick) · unit tests for symmetric-clim and std math (✅).

### M2.4 — pyqtgraph decision gate (timeboxed: 2 days, then STOP) ✅ DONE — no migration
**Decision: matplotlib-blit stays; `PyQtGraphSliceView` was not implemented.** Full reasoning + numbers in `docs/decisions.md`. Measured the existing matplotlib-blit backend directly (2×2 grid, 4 distinct real scenarios, `_on_time_changed`'s real per-tick path):

| Rendering path | fps | vs. ≥15 fps DoD |
|---|---|---|
| Offscreen (`tests/bench_grid_fps.py`) | ~247 fps | 16× |
| **Real display** (native `cocoa` backend, actual on-screen window) | **~57.7 fps** | 3.8× |

Both clear the gate's "adopt only if matplotlib-blit < 15 fps" condition by a wide margin — the 2026-07-09 prediction ("at 49×101 px, matplotlib-blit will pass") held. Per spec, 2.4.1 (`PyQtGraphSliceView` implementation) was the mechanism for producing a second data point to compare against, not a goal in itself — since matplotlib already clears the bar this comfortably on the number that matters (real display, not offscreen), building a second backend purely to benchmark it would be exactly the kind of speculative work §7.1/§7.3/§1.6 already argue against elsewhere in this document (abstraction cost paid before it's earned). `PlotView` (M2.2) is still the seam that makes this reversible later, so nothing about deferring the build is a one-way door.

**Caveat carried into `docs/decisions.md` explicitly, not left implicit:** the offscreen number alone overstates real performance (~4× here, confirmed by comparing against the real-display run) — same class of gap M1.3.3's own benchmark first surfaced. The real-display figure came from a single spot-check in an environment that happened to have an actual attached display (verified via `QApplication.platformName() == "cocoa"`, not assumed); flagged in `docs/decisions.md` for a secondary spot-check on different hardware before the demo if the opportunity arises, though not because there's a specific reason to doubt the passing result.

### M2.5 — Experiment browser + summary index ✅ DONE
Implemented on `feat/m2.5-experiment-browser`. **Objective met:** all 24 real scenarios are indexed in a docked browser with sortable/filterable factors + summary statistics; browser selections drive the existing active-cell/grid/ensemble paths.

| Task | Detail |
|---|---|
| 2.5.1 `summary_stats.py` ✅ | Per scenario: max T (global & per-frame curve), time-to-T>{100,300,600 °C}, mean upper-region T (upper half of the displayed slice array), peak HRR + total energy from `*_hrr.csv` (first use of this data). Cached to `fds/sim/.cache/summaries.json`, mtime-invalidated against `.sf`, `.smv`, and `*_hrr.csv` sources, with manifest case-index validation before reuse. Unit tests cover hand-computed temperature thresholds, upper-region mean, trapezoidal HRR energy, cache reuse, and HRR-driven invalidation. |
| 2.5.2 Browser dock ✅ | New `browser.py`: `ExperimentBrowserDock` (`QDockWidget`) with `SummaryTableModel` + `SummaryFilterProxyModel`, factor/text filters, sortable stats columns, double-click → active cell, multi-select → open as grid / open as ensemble. Demo mode omits the browser because there is no real manifest/HRR source. Integration tests cover the 24-row real browser, HRR columns, filtering/sorting, double-click load, grid open, and ensemble open. |

**DoD:** all 24 rows with correct stats (✅ real-data check generated 24 rows; spot-check: `c1_d0_vod0_voc0` peak HRR 0.08 kW / 9.15 kJ, `c2_d1_vod2_voc1` peak HRR 0.16 kW / 18.29 kJ) · sort/filter works (✅) · double-click loads (✅) · stats regeneration only when sources change (✅ unit-tested HRR mtime invalidation).

### M2.6 — Probe + isotherms ✅ DONE
Merged to `main`. **Objective met:** physical-coordinate probing and a toggleable hazard-band contour overlay both work, on real data, across every cell in the grid (not just a single view).

Built in three layers (session interrupted between layer 1 and layers 2–3 by a usage-limit reset; layer 1's WIP was committed as its own checkpoint mid-session rather than risked, per standing practice):

| Layer | Detail |
|---|---|
| 1. Parser geometry metadata | `fds/slice/slice.py`: `readSliceGeometry()` (parses `.smv` mesh/slice metadata only, skips the `.sf` binary payload — a second full data-parse was never needed just to learn the axes) + `combineSliceGeometry()` (mesh-stitching logic factored out of the existing `combineSlices`, reused rather than duplicated). `load_data.load_slice_geometry()`, `ScenarioStore.get_extent(scenario_index, key)` (own cache, separate from the frame-data cache — a probe/isotherm redraw never forces a cold `.sf` read). `data_provider.py`: `ScenarioSource` protocol gains `get_extent`; `DemoScenarioStore.get_extent()` returns a stable synthetic footprint so probing still works, un-real-but-labeled, in demo mode. |
| 2. Physical extent + isotherms (`views.py`) | `SliceView.init_plot(..., extent=(x0,x1,z0,z1))` passes extent straight to `imshow()`. **Verified against real data, not just reasoned about:** matplotlib's default `origin='upper'` places row 0 at z1 (top) — which lines up for free with `load_data.py`'s existing `np.flip(axis=1)`, since that flip already makes row 0 the physical ceiling before any view sees the array. A known peak-difference pixel from M2.3's own investigation round-trips through the new `value_at()` to the same physical (x, z) independently derived there; all four corners of a real frame match exactly. Isotherms: `set_isotherm_levels()`/`set_isotherms_enabled()` draw/clear an `ax.contour()`; `show_frame()` redraws it fresh each call while enabled and drops to the ordinary blit path with zero contour cost while off (confirmed by a dedicated test, not assumed). `DifferenceView`/`EnsembleView` got matching passthroughs plus `extent` in their own `init_plot()`. |
| 3. `MainWindow` wiring | Every cell gets its extent from the new `get_extent()` at `init_plot()` time, fixed for that view instance's lifetime (documented simplification — every scenario in this dataset shares one room footprint, per M2.1/M2.3's own findings; a dataset with per-scenario geometry would need this revisited). Cursor probe: each cell's `SliceView` reports mouse position via `enable_probe()`; the status bar shows `"x = …m, z = …m, value = …<unit>"`, resetting to "Ready." off-hover. Isotherm overlay: a new View-menu checkbox applies `config.ISOTHERM_LEVELS` (the M1.3s.4 hazard bands, 60/100/300 °C) to **every visible cell**, not just the active one — a deliberate departure from M2.2's "active cell only" precedent for clim/colormap, since an overlay like this is a shared read-aid a user comparing cells wants uniformly on, not a per-cell data-scale choice. Isotherm state re-syncs on every quantity change (levels are quantity-keyed; a cell's quantity can change without its view being recreated). |

**M1.6 follow-on, re-confirmed deferred (third mention — not dropped):** investigated folding in real per-object door/vent/candle placement for `schematic.py` (the note filed at M1.6 and repeated at M2.6's original spec). Finding: the `.smv` file *does* carry `OBST`/`VENT` records with real geometry, but in mesh-relative cell-index form (index ranges + a per-mesh `PDIM`/`TRN` coordinate transform), not the simple physical-XB triples `readSliceGeometry()` reads for slice extents — a materially different, unverified parsing task, not a small extension of this milestone's work. Given the risk of a subtly-wrong door/vent position being *worse* than the current honestly-labeled fixed-proportion placement (M1.6's schematic sits next to a real physical mockup at the demo), this stays deferred rather than being rushed in under this milestone's remaining time. Filed again, explicitly, rather than silently dropped: real per-object schematic geometry needs its own scoped milestone to parse and verify the `.smv` `OBST`/`VENT` record format properly.

**Tests:** 18 view-layer unit tests (`test_views.py`: corner/known-pixel `value_at()` accuracy including 2 real-data cross-checks against the M2.3 reference pixel, synthetic motion-event callback contract, isotherm enable/disable/redraw-per-frame/zero-cost-while-off) + 11 `MainWindow` integration tests (`test_integration.py`: extent reaches the active cell and demo mode, probe status-bar text and reset-on-leave, probe wiring survives grid growth, isotherm toggle applies/clears grid-wide, redraws on ticks, off-state touches no contour artist, quantity switches update levels). Full suite 218/221 (bench scripts excluded; the 3 failures are the pre-existing, already-documented, unrelated cursor-stack flake).

**DoD:** probe accurate at corners (✅ exact match at all 4 corners + the M2.3 reference pixel, on real data) · contours toggle cleanly (✅ enable/disable/redraw-per-tick all verified) · off-state performance unchanged (✅ verified directly — off-state never touches the contour artist, stays on the existing blit path).

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
