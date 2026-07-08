# FDS Visualizer — Research Platform Roadmap & Execution Plan

**Prepared:** 2026-07-08 · **Demo deadline:** 2026-09-11 (~9 weeks) · **Team:** 1 developer
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
- Cache-miss scenario switch ≈ 1–1.5 s (binary parse). No disk cache yet.
- `Slice.readData` still does one `np.fromfile` per timestep (481 syscalls; vectorizable to 1).
- Playback is worker-push on wall clock; no seek/scrub. Progress bar is read-only.
- Git repo exists (branch `milestone/phase2-membership`); **no pytest suite, no pyproject.toml, no CI**. `src/test_app.py` is stale (imports removed `load_all_data`) — replace, don't keep.
- `main_window.ui` is on disk but unused by the new entry point — delete in M1.1.

### Known defects to fix opportunistically
1. `combineSlices` assumes uniform mesh resolution across meshes (unvalidated).
2. Color scale `vmin` frozen at first-frame minimum (misleading for later frames).
3. `readAllTimes` discovers timestep count by looping reads instead of `filesize // stride`.

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
| Perceptual colormap default (inferno) + fixed physical scale | **Do, week 1** | Correctness of perception; 1-line default change (options already exist) |
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
- Config editor / auto-validation of `.fds` inputs: out — that's the simulation-generation side, explicitly out of scope.

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
- **Extract parser into an installable package** (`src/` layout, `pyproject.toml`, `pip install -e .`) with pytest suite against fixture files. Everything else builds on trusting the parser.
- **`PlotView` interface** to decouple views from matplotlib — this is the seam that makes the pyqtgraph decision reversible (§7.1) and multi-view possible.
- **Invert playback control**: replace worker-push (`_Worker` sleeping on wall clock) with a GUI-thread `QTimer` pulling frames — *only viable once the disk cache makes frame access O(µs)*. Kills residual shared-state issues; makes seek trivial; background thread remains only for cache-miss prefetch. (Sequencing: cache first, then timer.)
- **Plugin system: NO** (for now). Solo developer, zero external contributors — a plugin API is speculative generality. Use a lightweight registry dict (quantity→reader, view-type→class); promote to plugins only when a second contributor exists (Phase 4).
- Testing: pytest; fixtures = one real scenario's `.smv` + smallest `.sf` copied into `tests/fixtures/`; unit tests for parser invariants (shape, times monotonic, known first-frame values), store (LRU eviction, thread-safety smoke), controller (start/stop/seek); `QT_QPA_PLATFORM=offscreen` integration test that builds the window and steps 10 frames.

---

## 2. Phased roadmap

Time budget: **Phase 1 = 1 wk · Phase 2 = 4 wks · Phase 3 = 3 wks · buffer/rehearsal = 1 wk** → Sept 11.

### Phase 1 — Quick Wins (Jul 8 → Jul 15)

| # | Milestone | Why it matters | Impact | Effort | Risk | Priority |
|---|---|---|---|---|---|---|
| M1.1 | Packaging, tests, repo hygiene | Safety net for all later refactors | High (invisible) | 1 d | None | **High** |
| M1.2 | Disk cache + vectorized reads | 1.5 s scenario switch → ~50 ms; unlocks multi-view + QTimer | High | 1 d | Low | **High** |
| M1.3 | Rendering quick wins (blit, inferno default, vmin fix, interp toggle) | Perceptual correctness + smoothness, nearly free | Med–High | 1 d | Low | **High** |
| M1.4 | QTimer playback + timeline scrubber | Biggest single UX unlock; playback becomes seekable | High | 2 d | Med (touches controller) | **High** |
| M1.5 | MP4/GIF export | Demo assets; researchers share results | Med | 0.5 d | Low (ffmpeg dep) | Med |

### Phase 2 — Major Engineering (Jul 16 → Aug 12)

| # | Milestone | Why | Impact | Effort | Risk | Priority |
|---|---|---|---|---|---|---|
| M2.1 | Scenario manifest + quantity/slice generalization | Unlocks VELOCITY (already on disk) + all future quantities; kills triple-hardcoding | High | 3 d | Med | **High** |
| M2.2 | PlotView abstraction + multi-view grid (1×1/1×2/2×2, synced time, linked clim) | The platform centerpiece; matches dataset's purpose | **Highest** | 5 d | Med | **High** |
| M2.3 | Difference view + ensemble stats view | Core science; doubles as ML-eval harness | High | 3 d | Low (array ops on cached data) | **High** |
| M2.4 | pyqtgraph spike — timeboxed decision gate (2 d max) | Only migrate if matplotlib-blit can't hold the 2×2 grid at target FPS | Med | 2 d | Contained by timebox | Med |
| M2.5 | Experiment browser + summary-stats index (incl. HRR from CSV) | Workflow leap; makes 24 runs navigable | High | 3 d | Low | **High** |
| M2.6 | Value probe + isotherm/contour overlays | Analysis affordances researchers expect | Med | 2 d | Low | Med |

### Phase 3 — Research Extensions (Aug 12 → Sep 2)

| # | Milestone | Why | Impact | Effort | Risk | Priority |
|---|---|---|---|---|---|---|
| M3.1 | Ensemble analytics: features, PCA/clustering panel, auto-summaries | Certain-payoff research; interpretable; demo-safe | High | 4 d | Low | **High** |
| M3.2 | Forecasting model (FNO preferred / ConvLSTM fallback) + in-app "Simulated · Predicted · Error" view | The publishable bet; reuses M2.3 | High | 8 d | Med–High (model quality unknown — but negative result reportable) | **High** |
| M3.3 | (Conditional) Super-resolution with re-run coarse sims | Only if generating new FDS runs is approved | Med | 5 d + sim time | High | Low |

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

1. **Tests + packaging (M1.1)** — protects every subsequent change; a refactor without tests before a hard deadline is gambling.
2. **Disk cache + vectorized reads (M1.2)** — single highest-leverage remaining perf change; prerequisite for 3, 6, 7.
3. **Timeline scrubber + QTimer playback (M1.4)** — biggest UX unlock per day of work.
4. **Blit + perceptual colormap + vmin fix (M1.3)** — visual correctness and smoothness, nearly free.
5. **Quantity generalization (M2.1)** — VELOCITY is already on disk; one reader change unlocks a whole visualization axis.
6. **Multi-view synchronized comparison (M2.2)** — the centerpiece; the reason this dataset exists.
7. **Difference + ensemble views (M2.3)** — science value now, ML-eval harness later; the plan's best two-for-one.
8. **Experiment browser (M2.5)** — turns "24 folders" into "an experiment".
9. **Ensemble analytics + auto-summaries (M3.1)** — certain-payoff research content.
10. **Forecasting prototype (M3.2)** — the research bet, de-risked because its evaluation UI (#7) ships regardless of model quality.

Ordering logic: 1–2 are enablers (invisible but load-bearing); 3–4 make the existing app feel modern within week 1; 5→6→7 is a strict dependency chain building the platform identity; 8 is parallelizable filler for wait times; 9–10 convert the platform into research output. Anything jeopardizing Sep 11 gets cut from the bottom up (M3.3 first, then M2.6, then M2.4's migration half).

---

## 4. Detailed execution plan (per-task, for the implementing agent)

> Conventions: **D** = difficulty (Easy/Med/Hard), **T** = est. time. Every task ends with: run `pytest`, launch app (`cd src && python3 main.py`), verify the specific behavior listed. Keep the app runnable after every commit.

### M1.1 — Packaging, tests, repo hygiene
**Objective:** installable package, pytest suite, clean repo. **Refactor first:** none.

| Task | Detail |
|---|---|
| 1.1.1 Create `pyproject.toml` | Root; pin PyQt5==5.15.*, matplotlib≥3.8, numpy≥1.26, pytest; `[project] name="fdsvis"`. D:Easy T:1h. Test: `pip install -e .` succeeds. |
| 1.1.2 `.gitignore` + repo cleanup | Ignore `__pycache__`, `.DS_Store`, `*.pyc`, `.cache/`; **delete** stale `src/test_app.py` and unused `src/main_window.ui`; commit report/ and ROADMAP.md. D:Easy T:30m. |
| 1.1.3 Test fixtures | `tests/fixtures/`: copy ONE scenario's `.smv` + its 4 `.sf` files (~3.5 MB) — small enough to commit. D:Easy T:30m. |
| 1.1.4 Parser unit tests | `tests/test_slice_parser.py`: shapes (481,49,101 after combine), `times` strictly increasing, dtype float32, known-value spot check (frame 0 ≈ ambient ~20 °C), `readSlice(...) is None`-free for TEMPERATURE. D:Med T:2h. |
| 1.1.5 Store/controller tests | LRU eviction order; thread-safety smoke (2 threads hammering `get`); controller start/stop leaves no running worker. Use `QT_QPA_PLATFORM=offscreen`. D:Med T:2h. |
| 1.1.6 Offscreen integration test | Build `MainWindow` with demo data, step frames, switch theme/colormap, resize to min. (Port of the ad-hoc checks already used in this project's history.) D:Easy T:1h. |

**DoD:** `pip install -e .` works · `pytest` green (≥10 tests) · app launches · stale files gone · all committed.

### M1.2 — Disk cache + vectorized reads
**Objective:** scenario switch ≤100 ms warm; parse ≤0.4 s cold. **Refactor first:** none.

| Task | Detail |
|---|---|
| 1.2.1 Vectorize `Slice.readData` | Replace 481-iteration `fromfile` loop with ONE structured read: dtype = time-record + data-record combined, `count=n_times`, then slice fields. Same for `readAllTimes` → `n_times = (filesize - header_offset) // stride`. Files: `slice.py`. D:Med T:3h. Test: parser tests still green; add timing assertion (<0.5 s/scenario). |
| 1.2.2 `.npy` cache layer in `ScenarioStore` | On miss: parse → `np.save(cache_dir/<case>_<quantity>_<slicekey>.npy)`; on hit: `np.load(..., mmap_mode='r')`. Invalidate if any source `.sf`/`.smv` mtime > cache mtime. Cache dir: `fds/sim/.cache/` (gitignored). Files: `scenario_store.py`. D:Med T:3h. Test: cold vs warm timing; corrupted-cache file falls back to re-parse. |
| 1.2.3 Benchmark script | `tests/bench_loading.py` printing cold/warm/RAM numbers (baseline: 1.99 s cold, N/A warm). D:Easy T:1h. |

**DoD:** warm switch <100 ms measured · cold parse <0.5 s · cache invalidation test passes · pytest green.

### M1.3 — Rendering quick wins
**Objective:** perceptually correct, smooth single-view rendering.

| Task | Detail |
|---|---|
| 1.3.1 Default colormap → `inferno` | Keep gist_heat as menu option; persist via existing QSettings. Files: `main_window.py` (COLORMAPS order, default). D:Easy T:15m. |
| 1.3.2 Fix frozen `vmin` | Explicit `vmin=AMBIENT_C` (20.0, in `config.py`); slider continues to drive vmax. D:Easy T:30m. Test: colorbar lower bound stable across frames. |
| 1.3.3 Blitting in `MplCanvas` | Cache background via `copy_from_bbox` after first draw; per frame: `set_data` → `restore_region` → `draw_artist(image)` → `blit`. Invalidate background on resize/theme/colormap change (connect to `resizeEvent` + the existing setters). Files: `widgets.py`, `main_window.py::_redraw`. D:Med T:3h. Test: visual check + FPS print in bench script (expect ≥5× frame-draw speedup). |
| 1.3.4 Interpolation toggle | View menu: nearest / bilinear (`AxesImage.set_interpolation`). D:Easy T:30m. |

**DoD:** playback visibly smoother · colorbar physically anchored · toggles persist · pytest green.

### M1.4 — QTimer playback + timeline scrubber
**Objective:** seekable, drift-free playback; retire the wall-clock worker.
**Refactor first (required):** M1.2 must be merged — pull-based playback is only correct when frame access is near-instant.

| Task | Detail |
|---|---|
| 1.4.1 `TimeController` | New class in `simulation_controller.py` (or `time_controller.py`): QTimer at `1000/(fps*speed)` ms; `play/pause/seek(i)/step(±1)/set_speed`; emits `time_changed(int)`. Views pull `store.get(case)[i]` on tick. Keep `_Worker` ONLY as background prefetcher for cache misses (`prefetch(case)` → thread → `store.get`). D:Med T:4h. |
| 1.4.2 `TimelineWidget` | New in `widgets.py`: QSlider(0..n_frames-1) + play/pause btn + time label ("t = 12.5 s / 120 s") + loop toggle; replaces QProgressBar. Dragging → `seek`; plays → slider follows. D:Med T:3h. |
| 1.4.3 Rewire `MainWindow` | Transport buttons → TimeController; Space toggles; ←/→ step 1 frame; Shift+←/→ step 1 s. Remove worker-push wiring. Files: `main_window.py`, `simulation_controller.py`. D:Med T:2h. |
| 1.4.4 Cache-miss UX | Scenario switch while paused: show busy cursor + status message, prefetch in background, redraw on ready (fixes the documented 1–1.5 s GUI freeze). D:Med T:2h. |

**Risk:** regression in play/pause/restart semantics → mitigated by integration test exercising all transport paths.
**DoD:** drag-seek works during playback · no GUI freeze on scenario switch · speed change takes effect immediately · old worker-push path deleted · tests green.

### M1.5 — Animation export
1.5.1 "Export → Animation (MP4/GIF)…": render frames offscreen via `FigureCanvasAgg` at chosen fps/range; MP4 via ffmpeg if present else GIF via Pillow; progress dialog + cancel. Files: new `export.py`, menu hook in `main_window.py`. D:Med T:4h. Test: 5-second export opens in QuickTime; cancel leaves no partial file.
**DoD:** both formats export; UI stays responsive.

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
2.6.1 Cursor probe: `motion_notify_event` → status bar "x=…m, z=…m, T=…°C" (physical coords via slice extent — start using `readSlice`'s mesh/extent return instead of `readDataOnly`). D:Med T:3h. 2.6.2 Isotherm overlay: contour lines at configurable levels (60/100/300 °C default) redrawn per frame **only when enabled** (accept blit bypass while active; acceptable at this grid size). D:Med T:3h.
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

**Risk & mitigation:** model quality unknown → baselines + error-analysis are the guaranteed deliverable; app feature works with even a mediocre model (error view becomes *more* interesting); ConvLSTM fallback pre-decided to cap architecture-fiddling time.
**DoD:** beats persistence baseline at ≥4-frame lead (or documented negative result) · eval grid opens from browser in ≤2 clicks · training fully reproducible from `ml/README.md` · torch NOT required to run the visualizer.

### M3.3 — Super-resolution (CONDITIONAL — skip unless new sims approved)
Re-run 24 cases at 2× coarser grid (local FDS, ~hours); train SR UNet coarse→fine; side-by-side + error view (reuse M2.3 again). Only start if M3.1+M3.2 are done before Aug 25.

---

## 5. Dependency graph

```
M1.1 tests/packaging ──────────────┬──────────────────────────────┐
                                   ▼                              │(safety net for all)
M1.2 disk cache ──► M1.4 QTimer+timeline ──► M2.2 multi-view ──► M2.3 diff/ensemble ──► M3.2 forecasting eval
      │                     ▲                      ▲                      ▲
      │             M1.3 blit/cmap ────────────────┘ (SliceView absorbs)  │
      │                                                                   │
      └────► M2.1 manifest+quantities ──► M2.2   ┌─► M3.1 analytics ──────┘ (features from cache)
                       │                          │
                       └──► M2.5 browser ─────────┴─► M3.2 (launch eval from browser)
M1.5 export (independent) · M2.4 gate (after 2.2.2) · M2.6 probe (after M2.2)
```

## 6. Backlog (MoSCoW)

**Must Have (demo-critical):** M1.1–M1.4, M2.1, M2.2, M2.3, M2.5, M3.1
**Should Have:** M1.5 export, M2.6 probe/isotherms, M3.2 forecasting, M2.4 gate
**Nice to Have:** bookmarks, annotations, dockable layout persistence, per-view export, histogram panel, session files
**Future Research (Phase 4):** volume rendering (.s3d), arbitrary slice planes, continuous-parameter surrogates + inverse design, super-resolution (M3.3), scenario-space active sampling, web viewer, plugin API

## 7. Key trade-off decisions (recorded so the agent doesn't relitigate)

1. **matplotlib+blit now, pyqtgraph behind a gate.** Grids are 49×101 px; blitting almost certainly suffices even 2×2. Migration cost (colorbar, toolbar, theming rework) is only justified by measured FPS failure or by Phase 4's 3D needs. The `PlotView` seam makes this reversible — that seam is the real decision.
2. **Pull-based QTimer playback replaces the push worker** — but only *after* the disk cache lands (ordering is load-bearing). Eliminates the last unsynchronized shared-state pattern; makes seek/multi-view trivial.
3. **No plugin system, no database, no Zarr yet.** 24 scenarios × 230 MB and one developer: a manifest JSON + `.npy` cache + registry dicts deliver the same capability without the abstraction tax. Each has a written trigger for revisiting (contributor count, scenario count, RAM ceiling).
4. **AI as two tracks:** analytics that cannot fail (ship before demo) + one serious model whose evaluation UI ships regardless of model quality. Never let the demo depend on a training run.
5. **Surrogates/inverse design deferred** — not because they're bad ideas but because a 24-point categorical design gives them nothing to learn; they become real when continuous parameter sweeps are generated (Phase 4).
6. **ML code lives outside the app** (`ml/`), communicating via `.npy` prediction files. The visualizer must never require torch to start.

## 8. Git workflow

- **Branches:** `main` = always demo-runnable. One branch per milestone: `feat/m1.2-disk-cache`, etc. Current work on `milestone/phase2-membership` should be merged or renamed to fit the scheme.
- **Commits:** conventional style (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`); one logical change ≤ ~300 LOC per commit; every commit leaves the app launchable.
- **Merging:** merge to `main` only when the milestone's DoD checklist passes (pytest green + manual launch check + benchmark where specified). Tag `m1.2` etc. on merge.
- **Demo safety:** tag `demo-rc1` on Sep 4 (feature freeze); only `fix:` commits to `main` afterward; tag `demo-2026-09-11` after rehearsal. Keep `fds/sim/.cache/` and `predictions/` out of git; fixtures ARE in git.

## 9. Timeline

| Dates | Work | Exit criterion |
|---|---|---|
| Jul 8–15 | M1.1 → M1.2 → M1.3 → M1.4 (+M1.5 if time) | Seekable, smooth, tested app; warm switches <100 ms |
| Jul 16–24 | M2.1 → start M2.2 | VELOCITY visible; SliceView extracted |
| Jul 25–Aug 5 | M2.2 finish → M2.4 gate → M2.3 | 2×2 synced grid + Δ/ensemble views @ ≥15 fps |
| Aug 6–12 | M2.5 → M2.6 | Browser drives everything; probe/isotherms |
| Aug 12–19 | M3.1 | Clustering panel + auto-summaries in app |
| Aug 19–Sep 2 | M3.2 (baselines → model → eval view) | Prediction-vs-truth grid opens from browser |
| Sep 3–4 | Freeze, `demo-rc1` | No open P1 bugs |
| Sep 7–11 | Rehearsal, fallbacks, demo script | Demo delivered |

*Cut order if behind: M3.3 (already conditional) → M2.6 → M1.5 → M2.4's migration half → reduce M3.2 to baselines + eval-view with persistence "predictions" (the UI story still works).*
