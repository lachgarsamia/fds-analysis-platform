# FireLab Digital Twin — Cinematic Demo Roadmap

**Prepared:** 2026-07-15 · **Audience:** Executor Advisor (autonomous coding agent), milestone by milestone
**Mission:** transform the FDS Visualizer from a research instrument into a breathtaking public-facing
"FireLab Digital Twin" that stands next to the *real physical chamber* (candles, door, ventilation) at the
demo — with a fire/smoke visualization so alive that visitors instinctively compare it to the flame in front of them.
**Mission:** transform the FDS Visualizer from a research instrument into a breathtaking public-facing
"FireLab Digital Twin" that stands next to the *real physical chamber* (candles, door, ventilation) at the
demo — with a fire/smoke visualization so alive that visitors instinctively compare it to the flame in front of them.

This roadmap **builds on**, and does not replace, `ROADMAP.md`. Everything listed as ✅ done there is the
foundation: lazy loading + LRU + disk cache, `TimeController` (QTimer pull, seek ~4–15 ms), blitting
`MplCanvas`, the `PlotView` protocol seam (`views.py`), `ViewGrid` (measured **~57.7 fps real-display**
on a 2×2 grid — our performance budget), `SchematicWidget`, extent-aware probe/isotherms, analytics,
FNO predictions, and the Fusion-style token theme (`theme.py`).

**Hard constraints inherited from ROADMAP.md (do not violate):**
- Keep the pytest suite green after every commit; the app stays runnable.
- Branch before the first edit of each milestone (`feat/firelab-*`).
- For `simulation_controller.py` / `scenario_store.py` threading issues: **report first, wait — do not fix inline.**
- Data ground truth: 24 scenarios, 481 frames × 49×101 cells per slice, TEMPERATURE + VELOCITY slices on
  disk, `.s3d` 3D smoke/soot data and `*_hrr.csv` HRR curves on disk **and still unused** — both become
  first-class citizens in this plan.

---

## 0. Design north star

> **The visitor stands between a real burning candle and the screen. The screen must feel like the
> chamber's living X-ray — not a chart.** Dark theatre-grade UI, one dominant visualization, physical
> controls that mirror the real room, and a fire that flickers, glows, and breathes.

Three experience pillars every phase serves:
1. **Cinematic** — the heatmap reads as *fire and smoke*, not as data. Emissive glow, transparency,
   drifting smoke, shimmer, flicker.
2. **Physical** — every on-screen control has a twin in the real chamber: candle cards, an animated door,
   vent flow arrows. Visitors change the virtual room and watch physics respond.
3. **Explanatory** — a right-hand inspector narrates in plain language what the fire is doing, live
   (reusing `auto_summary.py` sentence machinery + `summary_stats.py`).

**Performance covenant:** the demo runs at a locked **30 fps** playback with all effects on, on the demo
laptop, with 2× headroom (i.e., frame budget ≤ 16 ms measured). Every effect ships with an off-switch and
an automatic quality governor. We have ~57.7 fps of measured full-pipeline headroom today; effects spend
that budget deliberately.

---

## Phase 1 — Core Architecture & Navigation (the shell)

**Goal:** restructure the single-window app into a modern page-based application — left navigation rail,
`QStackedWidget` page host — *without rewriting any working view/controller code*. Existing widgets get
re-parented into pages, not rebuilt.

### Files to change
| File | Change |
|---|---|
| `src/main_window.py` (1853 lines) | The big one. Extract page assembly out of `MainWindow.__init__` into a new `src/pages/` package; `MainWindow` becomes shell (nav rail + stacked pages + status bar). Existing docks (browser, analytics) become page content instead of docks. |
| `src/pages/__init__.py`, `src/pages/base.py` | New. `Page` base: `title`, `icon`, `on_enter()`/`on_leave()` lifecycle (pause playback on leave, lazy-build on first enter — same lazy pattern that fixed the M3.1 startup regression). |
| `src/pages/home.py`, `live.py`, `compare.py`, `dataset.py`, `analysis.py`, `export_page.py` | New page classes. `live.py` initially hosts the existing 1×1 `ViewGrid` + `TimelineWidget` + control panel unchanged. |
| `src/nav.py` | New. `NavRail(QWidget)` — vertical icon+label buttons, active-state pill, QPainter icons in the vein of `schematic.py`'s `flame_icon`/`door_icon`. |
| `src/theme.py` | New tokens: `nav_bg`, `nav_active`, dark "theatre" palette variant (near-black `#0b0d12` background, ember-orange `#ff6b35` accent) as a third theme, `demo` mode. |
| `tests/test_pages.py` | New. Page lifecycle, lazy build, playback pauses on page switch. |

### Key features
- **Nav rail:** Home · Live Viewer · Compare · Dataset · Analysis · Export. Keyboard 1–6. Collapsible to
  icons-only (more canvas for the fire).
- **Theatre theme:** the demo default. Near-black surfaces so the fire's emissive palette is the only
  bright thing on screen (cinema rule: fire looks hot only against darkness).
- **State continuity:** `TimeController`, `ScenarioStore`, and the active scenario are app-level
  singletons owned by the shell and passed into pages — a page switch never reloads data.

### Risks
- `main_window.py` is the highest-coupling file in the repo; ~30 integration tests assume its current
  widget tree. Mitigation: extract mechanically (move, don't rewrite), keep object names/signals stable,
  run the suite after each extraction commit.
- Docks→pages changes QSettings persistence keys. Mitigation: version the settings group (`ui/v2/…`).

### Success criteria
- All existing tests pass (adapted only for widget-tree paths, not behavior).
- Page switch < 50 ms warm; playback state survives Live→Dataset→Live round-trip.
- App boots straight into Home in ≤ the current ~2 s.

**Effort:** 3–4 d.

---

## Phase 2 — Cinematic Visualization Engine ★ (the heart of the demo)

**Goal:** make the fire look *real*. This phase builds a compositing pipeline on top of the existing
`SliceView` blitting path, then layers effects in strictly increasing cost order. Each effect is a pass
in an ordered pipeline, individually toggleable, individually benchmarked.

### 2.0 Architecture: the `EffectsPipeline`

New file `src/cinema/pipeline.py`. Core idea: **stop letting matplotlib color the data.** Instead of
`imshow(temperature, cmap=…)`, we compute a final **RGBA uint8 frame in NumPy** each tick and hand it to a
*single* `imshow` artist (`set_data`) inside the existing `capture_background`/`blit_update` path — one
artist, one blit, unchanged render loop. All effects are array math on a small grid; that's why this is fast.

```
temperature (49×101 f32) ─┐
velocity    (49×101 f32) ─┼─► upsample 4–6× (bicubic zoom, ~300×600) ─► pass chain ─► RGBA ─► blit
HRR(t)      (from csv)  ──┘
Pass chain (each pass: (state, buffers) -> buffers, all preallocated):
  1 ToneMap  2 FireLUT  3 Bloom  4 Smoke  5 Shimmer  6 Particles  7 Vignette/Grade
```

Key components:
- `src/cinema/pipeline.py` — `EffectPass` protocol, `EffectsPipeline` (ordered passes, preallocated
  float32 buffers, per-pass `enabled` + `cost_ms` telemetry).
- `src/cinema/quality.py` — `QualityGovernor`: rolling frame-time EMA; if > 24 ms for 30 frames, drop the
  most expensive optional pass tier (particles → shimmer → bloom radius) and log it; restore when headroom
  returns. The demo can never stutter.
- `src/views.py` — `CinematicSliceView(SliceView)` implementing the same `PlotView` protocol, selected per
  cell (right-click menu already exists from M2.3). Research views remain untouched — **the science mode
  and the cinema mode coexist**, one keystroke apart (important for credibility questions at the demo).
- `tests/bench_effects.py` — per-pass and full-chain timing on real data; CI-style assertion: full chain
  ≤ 10 ms at 300×600 on the dev machine.

**Why upsample first:** effects like bloom and shimmer look right only at display-ish resolution; 49×101
is far too coarse. A one-time bicubic `scipy.ndimage.zoom` to ~300×600 costs ~1 ms and every later pass
works on that buffer. (6× on a 49×101 grid ≈ 180 k px — trivially NumPy-fast.)

### 2.1 Fire look, layer by layer (creative brainstorm → concrete plan)

**(a) Black-body "FireLUT" colormap with alpha — the single biggest visual jump. (Easy, ~0 ms)**
Real flames follow black-body radiation: transparent → deep red (~600 °C) → orange → yellow → white.
Build a custom LUT in `src/cinema/luts.py`:
- Color: piecewise ramp through measured black-body chromaticities (Kelvin→sRGB table, gamma-correct
  interpolation in linear light, not in sRGB — this is what makes it look *incandescent* rather than
  "heatmap orange").
- **Alpha ramp:** ambient temperature = fully transparent, revealing a dark chamber backdrop (room
  outline, floor gradient — drawn once into the static background that blitting already caches). The fire
  literally floats in the room instead of filling a rectangle. This alone transforms the perception.
- Keep `gist_heat`/scientific maps available; FireLUT is the cinema default, hazard-band isotherms still
  overlay on request.

**(b) Filmic tone mapping + auto-exposure. (Easy, <1 ms)**
- Map temperature → normalized intensity with a filmic S-curve (Reinhard or ACES-fit polynomial) instead
  of linear `Normalize`. Hot cores saturate to white gracefully; the mid-range gains contrast.
- **Auto-exposure:** vmax = EMA (τ ≈ 2 s) of the 99.5th percentile per frame. Like a camera's iris, the
  view adapts as the fire grows — early faint plumes are visible, later flashover doesn't clip. A "lock
  exposure" toggle keeps science mode honest.

**(c) Bloom / emissive glow. (Easy–Med, ~2–3 ms)**
Hot zones must *bleed light* like a lens:
- Threshold luminance above a knee → 3 Gaussian blurs at increasing σ (σ≈2, 6, 16 px via
  `scipy.ndimage.gaussian_filter`, or a 3-level downsample/blur/upsample "Kawase-style" pyramid for
  cheapness) → weighted additive composite in linear light.
- The bloom layer also spills over the room outline in the static backdrop — light *touching the walls*.
- Tuning: bloom strength modulated by HRR(t) from `*_hrr.csv` — the glow physically tracks the real
  heat-release curve. Data-driven cinematography.

**(d) Flicker & plume liveliness. (Easy, ~0 ms)**
Real fire is never steady between our 481 stored frames:
- **Sub-frame interpolation:** playback at 30 fps blends adjacent stored frames (`lerp(f[i], f[i+1], t)`)
  — smooth motion for free, since arrays are cached and tiny.
- **1/f flicker:** multiply exposure by `1 + a·pink_noise(t)`, amplitude `a` scaled by instantaneous HRR
  (precompute a pink-noise track once). Candle-like breathing, driven by the scenario's own energy curve.
- **Plume sway:** a very low-frequency horizontal sinusoid displacement (sub-pixel, via the shimmer warp
  below) makes the plume waver like convection actually does.

**(e) Heat shimmer / refraction. (Med, ~2–4 ms)**
The air above heat wobbles — the most visceral "this is hot" cue:
- Precompute 2–3 octaves of tileable value noise; per frame, sample it with a time offset and build a
  displacement field scaled by **local temperature above ambient** (so only hot air shimmers).
- Apply with `scipy.ndimage.map_coordinates(order=1)` on the composited RGB — a single warp of a 300×600
  buffer, ~2–3 ms. Subtlety knob default *low*: shimmer sells realism at 10 % strength and looks like a
  broken screen at 50 %.

**(f) Smoke layer — from data we already have. (Med–Hard, ~3–5 ms; highest wow-per-effort after FireLUT)**
Three tiers, in order of increasing fidelity; ship tier 1, then upgrade:
1. **Temperature-derived haze (ship first):** smoke density ∝ time-integrated "burnt" field — accumulate
   `max(T − T_smoke, 0)` with decay per frame into a persistent buffer, render as a soft gray-brown
   translucent layer with its own gentle upward advection (fixed buoyant drift + noise). Cheap, looks
   like a filling smoke layer under the ceiling — which is exactly what the real chamber does.
2. **Velocity-advected dye:** we already parse the VELOCITY slice (speed magnitude). Combine magnitude
   with a buoyancy-oriented direction prior (up + away from hot core, i.e. direction from ∇T) to build a
   pseudo-velocity field; advect the smoke buffer semi-Lagrangian (one `map_coordinates` call). Honest
   caveat for the executor: the stored slice is |v| only — **true u/w components would need M-SIM adding
   `U-VELOCITY`/`W-VELOCITY` slices to `fds/template.fds`** (cluster path already planned in ROADMAP.md;
   flag it in the M-SIM wishlist now, don't block on it).
3. **Real soot data (`.s3d`) — the sleeping giant:** every scenario ships Smokeview 3D soot-density files
   the app has never read. A `src/fds/s3d/` reader (RLE-compressed byte data — format documented in the
   Smokeview repo) extracting the same y-plane gives *actual simulated smoke*, not a proxy. Timebox a
   2-day spike exactly like M1.3s (parser spike → decision doc → then integrate). If the spike lands,
   FireLab renders **the real smoke the physics computed** — the single strongest scientific-credibility
   claim of the demo.
- Rendering for all tiers: second RGBA layer composited *under* the fire pass but *over* the backdrop
  (smoke occludes the room, fire glows through smoke — get this ordering right; it's what reads as depth).

**(g) Ember & flame particles. (Med, ~1–2 ms for ≤ 500 particles)**
`src/cinema/particles.py` — pure-NumPy structure-of-arrays particle pool (pos, vel, age, size, heat):
- Spawn ∝ local temperature above ignition threshold at the flame region; advect with the tier-2
  pseudo-velocity + buoyancy + jitter; fade & shrink with age.
- Render as **one** matplotlib `scatter` artist updated via `set_offsets`/`set_sizes`/`set_array`
  (additive-looking bright LUT) — a second blit-tracked artist; the M1.3 blitting code already supports
  capturing multiple animated artists (verify: `capture_background` must include the scatter — small
  `widgets.py` extension, listed below).
- Embers are punctuation, not content: sparse (30–100 alive), tiny, bright, short-lived. More reads as
  a screensaver.

**(h) Scene grading — the cinematic frame. (Easy, ~0 ms, static)**
Drawn once into the cached blit background:
- Dark chamber backdrop with the extent-true room outline (reuse `resolve_room_extent`), a subtle floor
  gradient "reflection" of the fire zone, door and vent drawn *in the scene* at their real positions
  (M2.6's deferred per-object `.smv` OBST/VENT parsing finally has its payoff — schedule the small
  mesh-index→physical-coords conversion here).
- Vignette multiplied into the final composite; colorbar redesigned as a slim vertical **hazard gauge**
  (ambient → untenable) with plain-language ticks ("room temperature", "sauna", "untenable", "flashover").

### 2.2 Performance budget (300×600 working buffer, per frame)

| Pass | Est. cost | Tier |
|---|---|---|
| Frame lerp + upsample | ~1.5 ms | core |
| ToneMap + FireLUT | ~0.5 ms | core |
| Bloom (3-level pyramid) | ~2.5 ms | A |
| Smoke advect + composite | ~3.5 ms | A |
| Shimmer warp | ~2.5 ms | B |
| Particles (500) | ~1.5 ms | B |
| Grade/vignette | ~0.3 ms | core |
| **Total** | **~12 ms** | 30 fps with margin |

Governor drops tier B first, then bloom radius. Benchmarks in `tests/bench_effects.py` are the DoD, not
these estimates — M2.4 taught us offscreen numbers overstate real-display performance ~4×, so the DoD
measurement is **on a real display**, like M2.4's re-measure.

### Files (Phase 2 summary)
New: `src/cinema/{pipeline,luts,quality,particles,smoke,noise}.py`, `src/fds/s3d/` (spike-gated),
`tests/test_cinema.py`, `tests/bench_effects.py`, `docs/spike-s3d.md`.
Changed: `src/views.py` (`CinematicSliceView`), `src/widgets.py` (multi-artist blit capture),
`src/config.py` (effect defaults), `src/theme.py` (hazard-gauge tokens), `src/main_window.py`/`pages/live.py` (view-mode switch).

### Risks
- **Blitting multiple artists:** the current capture path was built for one image artist. Verify with the
  scatter early (day 1 of particles); fallback is baking particles into the RGBA buffer as splats (still fast).
- **Taste risk:** effects can tip into video-game kitsch. Mitigation: every knob in a hidden dev panel
  (`Ctrl+Shift+E`), screenshot A/B review against reference photos of the *actual chamber's candles*, and
  the science view always one keystroke away.
- **`.s3d` format spike may fail** — that's why it's a timeboxed spike with tier-1/2 smoke already shipped.
- macOS + Fusion style is already validated; no new Qt styling risk.

### Success criteria
- Side-by-side test: a non-specialist shown the Live Viewer says "fire/smoke", not "chart" (actually run
  this hallway test; it's the phase's real DoD).
- ≥ 30 fps sustained on-real-display with tier A+B on, 481-frame full playback, quality governor never
  triggering on the demo machine.
- Science mode pixel-identical to pre-phase `SliceView` (regression-tested).

**Effort:** 8–10 d (FireLUT+tone+bloom+flicker ≈ 3 d; smoke tiers 1–2 ≈ 2 d; shimmer+particles ≈ 2 d;
grading + governor + benches ≈ 2 d; s3d spike +2 d gated).

---

## Phase 3 — Controls & Inspector (the physical mirror)

**Goal:** controls that look like the chamber, an inspector that explains like a guide.

### Key features
- **Candle cards** (`src/controls/candle_card.py`): big toggle cards with the QPainter flame icon
  *animated* (2–3 frame flicker via QTimer at 8 fps, painted, no GIFs) when that candle is lit in the
  scenario; extends `schematic.py`'s existing icon painters.
- **Animated door** (`src/controls/door_widget.py`): a top-view door arc that swings open/closed on
  toggle (`QPropertyAnimation` on a rotation property, 300 ms ease-out) — mirrors the physical door.
- **Vent flow indicators**: vent icons gain animated flow arrows (dash-offset animation) whose speed maps
  to the vent factor level (vod/voc), plain-language labels from M1.6 reused.
- **Right inspector** (`src/inspector.py`, on the Live page): live probe readout (M2.6's probe re-skinned
  large-type), a peak-temperature sparkline scrubbed in sync with `TimeController`, HRR gauge, and a
  **live narration line** — `auto_summary.py` templates extended with time-aware sentences ("The smoke
  layer is forming under the ceiling · The doorway is feeding fresh air to the flame").
- **Scenario transitions:** switching scenarios crossfades the fire (0.5 s alpha blend of old/new frames
  in the pipeline) instead of hard-swapping — cheap and hugely polished.

### Files
New: `src/controls/{candle_card,door_widget,vent_widget}.py`, `src/inspector.py`, `tests/test_controls.py`.
Changed: `src/pages/live.py`, `src/schematic.py` (share painters), `src/auto_summary.py` (time-aware
sentences — pure functions, easily tested), `src/main_window.py` (wiring: same toggle signals as today).

### Risks
- Control signals must keep driving the *same* `SimulationController` path (scenario index math in
  `manifest.py`) — re-skin, don't re-plumb. The threading-report-first rule applies if anything in the
  controller needs touching.

### Success criteria
- A visitor can set up any of the 24 scenarios without reading anything technical (hallway test #2).
- Narration line matches summary-stats facts (unit-tested templates, zero free generation).

**Effort:** 4–5 d.

---

## Phase 4 — Additional Pages

- **Home** (`pages/home.py`): full-bleed hero — a looping pre-rendered MP4 of the best scenario (made
  with our own M1.5 exporter + Phase 2 pipeline), title, "Start the fire →" CTA into Live, three stat
  tiles (24 experiments · 481 time steps · 2 physics fields). Attract mode returns here when idle (Phase 5).
- **Compare** (`pages/compare.py`): the existing `ViewGrid` 1×2/2×2 + `DifferenceView` re-hosted, with
  a "story preset" dropdown: *Door open vs closed* (velocity shows the effect — M2.3's verified finding),
  *One candle vs two*, *Ventilation strong vs weak*. Presets encode the honest findings already pinned in tests.
- **Dataset** (`pages/dataset.py`): `browser.py` table re-hosted with thumbnail strip per scenario
  (peak-frame PNG, generated once into cache).
- **Analysis** (`pages/analysis.py`): `analytics_panel.py` (PCA/clusters) + the M3.2 prediction 1×3 view,
  reframed for the public: "The AI guesses the next 8 seconds — here's where it's wrong" (the
  `DifferenceView` error panel becomes the star; honesty *is* the wow).
- **Export** (`pages/export_page.py`): M1.5 exporter UI + "demo postcard" one-click (current frame,
  cinematic grade, FireLab title card).

**Files:** the page files above; `src/browser.py`, `src/analytics_panel.py` (re-host, minimal edits);
thumbnail generation in `src/summary_stats.py`'s cache pattern.
**Risks:** low — composition of shipped parts. Watch the M3.1 lesson: thumbnails and presets must be
lazy/background, never at startup.
**Success criteria:** every page reachable in ≤ 2 clicks; no page blocks the GUI thread on first open.
**Effort:** 3–4 d.

---

## Phase 5 — Polish & Demo Features

- **Kiosk/attract mode:** after N min idle, drift back to Home's hero loop; any input returns to Live.
  F11 full-screen already exists; add cursor auto-hide.
- **Demo script mode:** number-key bookmarks (scenario + time + page) so the presenter jumps
  beat-to-beat; a `docs/demo-script.md` narrative to rehearse against.
- **Guided tour overlay:** 4-step first-run coach marks (Live page only), dismissible, QSettings-remembered.
- **Failure drills:** demo-data fallback path re-verified with the cinema pipeline; "effects off" master
  switch on `Esc` long-press; rehearse a mid-playback scenario-folder unplug.
- **Performance rehearsal on the actual demo laptop** (the M2.4 lesson institutionalized): full-day soak
  loop, thermal throttling check, governor logs reviewed.
- Final visual QA in both themes via the screenshot-diff discipline used in the GUI-modernization pass.

**Files:** `src/kiosk.py`, `src/tour.py`, `docs/demo-script.md`; small hooks in the shell.
**Risks:** time pressure — this phase is the buffer; cut tour before kiosk, kiosk before drills. Never cut drills.
**Success criteria:** 30-minute unattended soak with zero stutters/leaks (watch RSS); presenter completes
the scripted demo twice without touching a mouse.
**Effort:** 3–4 d + rehearsal.

---

## Top 5 first tasks for the Executor (start here, in order)

1. **`feat/firelab-cinema-core` — FireLUT + alpha transparency + filmic tone map + auto-exposure**
   (Phase 2 a+b). Build `src/cinema/{pipeline,luts}.py` with the single-RGBA-artist architecture and the
   black-body LUT over a dark extent-true backdrop. *This is 60 % of the visual transformation for ~2 days
   of work, and it establishes the pipeline every other effect plugs into.* DoD: side-by-side screenshot
   vs current `gist_heat` view; bench ≤ 4 ms/frame; science mode untouched.
2. **Bloom + HRR-driven flicker + sub-frame interpolation** (Phase 2 c+d). The fire starts to *breathe*:
   glow that tracks the real heat-release curve, 30 fps motion smoothness from frame lerp. First real use
   of `*_hrr.csv` in the render path. DoD: full chain ≤ 8 ms; on-display 30 fps sustained over all 481 frames.
3. **Smoke tier 1→2** (Phase 2 f): decaying accumulation haze, then velocity/∇T pseudo-advection, composited
   under the fire. Also file the M-SIM wishlist note (add `U-VELOCITY`/`W-VELOCITY` slices) so the cluster
   run captures it. DoD: smoke layer visibly pools under the ceiling and vents toward the open door.
4. **Navigation shell** (Phase 1): `pages/` + `NavRail` + theatre theme, existing widgets re-hosted. Done
   after the first three so the wow exists before the reorganization risk is taken — and so page design
   decisions are made around the real cinematic canvas. DoD: suite green, page switch < 50 ms.
5. **`.s3d` real-smoke spike** (timeboxed 2 d, Phase 2 f tier 3): reader spike + `docs/spike-s3d.md`
   decision doc, M1.3s-style. If it lands, the demo's smoke is *the actual simulated soot field* — the
   strongest possible answer to "is this real?" asked in front of the physical chamber. DoD: decision doc
   with go/no-go; if go, one scenario's soot slice rendered through the smoke compositor.

Shimmer, particles, controls, inspector, and the remaining pages follow in phase order once these five land.
