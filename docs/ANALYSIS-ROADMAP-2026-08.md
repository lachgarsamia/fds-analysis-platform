# Analysis Section Roadmap — August 2026 Pass

Follow-up to `docs/ANALYSIS-IMPROVEMENT-ROADMAP.md` (2026-07-27, Phases A–D),
which is fully implemented — cross-checked against the current code, every
item in it is done. This pass is a fresh audit after that work plus six more
rounds of consolidation (`git log`: Phases 1–6, an "Analysis final-polish
pass," and several "UX consolidation" commits), focused on what's still
worth doing now that the tab count has been cut hard and the panels
themselves are mostly solid.

**Method:** read `pages/analysis.py`'s current grouping + module docstring
(which already documents its own history), then every panel file in each
group for functional status for real (`fds/sim/`) data, current gating,
and known limitations; cross-checked against `docs/msim-preparation.md` for
what's blocked on new simulation output vs. what's an app-side gap.

## Current shape (for reference)

```
Overview & Interpretation   Dashboard · Hazard & Tenability · Narrative
Compare & Discover          Pairwise Comparison · PCA / Clustering
Probe & Measure             Spatial Probes (Devices/Zones/Velocity)
Factors & Sensitivity       Study (Factor influence/Correlation/Factor effects/Sensitivity)
Spatiotemporal Analysis     Field & Time Explorer (3 modes) · Space-time
Reference & Communication   Quantities · Graph · Ask
Experimental (collapsed)    Fire MRI · Attention · Why is it hot? · Forecasting
```

Navigation is 3–4 levels deep in three places (Study, Spatial Probes, Field
& Time Explorer each have their own inner mode-tabs on top of the outer
group → tab structure) — this is the main theme below, not the panels
themselves, which are individually in good shape.

## Do not re-propose (already tried, deliberately removed)

Response Curve (redundant with Factor influence), Tornado/What-if,
Calculator as its own tab (backend still called directly by
`main_window.py`), Sessions as its own tab (same), State space, Ensemble
spread, Parallel coordinates, "Quick probe" measurement tool, the Assistant
template-summary layer (replaced by Ask), Multi-plane cross-sections as a
top-level tab. All removed on purpose for not clearing "does this earn its
complexity/upkeep" against the roadmap's own bar — re-litigating any of
these needs a new reason, not just "it used to exist."

---

## A. Navigation & wayfinding (the actual gap)

1. **No breadcrumb anywhere.** A tab switch at any of the 3 levels already
   emits `AnalysisPage.tab_shown` — cheap to also update a one-line
   "Overview & Interpretation › Hazard & Tenability" label at the top of
   the page from the same signal, so a screenshot or a user re-orienting
   after Alt+Tab can tell where they are without hunting through nested
   tabs.
2. **No search/filter across ~25 leaf panels.** The 6-group structure is
   reasoned by research question, but that means finding a specific tool
   requires remembering *which question it answers*, not just its name. A
   lightweight "type to jump" filter (even just a `Ctrl+K`-style list of
   leaf-panel names that calls the same `AnalysisPage.show_tab()` already
   used for cross-navigation hand-offs) would pay for itself here more than
   almost anywhere else in the app.
3. **Investigation History exists but is invisible.** `history.py`'s
   back/forward is real, app-wide, and already bound to Alt+Left/Alt+Right
   — but it's a bare keyboard shortcut with zero UI surface. Given how deep
   Analysis navigation is, two small toolbar buttons (disabled/enabled from
   the same `InvestigationHistory` state the shortcuts already read) would
   make it discoverable exactly where it's most useful.
4. **Onboarding stops at Live.** `tour.py`'s guided tour is hard-wired to
   `pages/live.py` only — confirmed the only import site in `src/`. A
   first-time Analysis visit (6 groups, Experimental collapsed) gets zero
   orientation. Same low-cost fix as the existing tour: one dismiss-once
   coach-mark pointing at the group tabs + the Experimental toggle, reusing
   `should_show_tour`/`mark_tour_completed`'s exact QSettings pattern with
   its own key.
5. **"Experimental" is the one place using a different interaction pattern**
   (a `▶ Show experimental panels` push-button toggle) versus `QTabWidget`
   everywhere else on the page. This was a deliberate choice (keep
   lower-confidence tools from competing for attention) and works — flagged
   only as a minor consistency note, not something to change without a
   reason.

## B. Data-gated features: make the gating consistently visible

`docs/msim-preparation.md` confirms the real constraint: no CO output means
every hazard/tenability classification in the app is the **temperature-only
partial screen** today, and no U/W-VELOCITY means no real flow direction
anywhere. Both are external (cluster re-run) gates, not app work — but how
consistently the app *says so* varies:

6. **Partial-FED is stated in some places, silent in others.** Dashboard's
   caption and the Hazard & Tenability docstring both say "temperature-only
   partial screen (no CO/CO₂)" — good. Narrative's event chain and Study's
   panels use the same classification without repeating the caveat. Space-
   time is the only place that exposes Full FED as an explicit (gated)
   toggle. A user moving between panels has no consistent signal that
   they're always looking at the same partial screen. Worth a single shared
   small badge/tooltip (reusing `hazard_spaces`' own basis constant/text)
   wherever a hazard classification renders, rather than each panel
   deciding independently whether to mention it.
7. **"Why is it hot?" and Attention already disclaim well** ("association,
   not proven causation... needs real U/W-velocity, which this dataset does
   not have yet" / "NOT a physical field") — this is the right pattern; (6)
   is about extending the same honesty to hazard classification specifically.
8. **Forecasting's gate is environment-dependent, not data-dependent** — it
   degrades to a placeholder whenever `ml/train.py`/`ml/rollout.py` haven't
   been run locally, which is a different kind of "gated" than the CO/
   velocity cases (fixable by running a script, not by waiting on a cluster
   re-run). Its placeholder text ("No trained-model predictions available…")
   doesn't say that — pointing at `ml/README.md` would turn a dead end into
   a next step.

## C. Concrete, app-side fixes achievable without new simulation data

9. **Ask reads the raw store directly, not through `QuantityProvider`.**
   Every other panel that resolves a quantity goes through the
   gating-aware `.get()` (raises `GatedQuantityError` rather than
   fabricating a value). `query_panel.py`/`query_engine.py` don't — so a
   parsed question that happens to target a gated quantity (U-VELOCITY, CO,
   etc.) is the one path in the app that could return something misleading
   instead of an honest "not available" message. This is a real,
   independently-fixable correctness gap, not blocked on M-SIM.
10. **Reference & Communication is a grouping of convenience.** Quantities
    (a status table), Graph (the knowledge-graph browser), and Ask (Q&A)
    don't share a research question the way the other five groups do —
    it's "the tools that are about the app's own memory/reference, not an
    investigation." Each is individually solid; this is flagged as *worth
    a second look*, not a firm recommendation, since I don't have evidence
    the current split actually confuses anyone — just noting it's the one
    group whose name describes leftovers rather than a question.
11. **PCA/Clustering is the headline demo feature** ("groups scenarios by
    candle count without being told to," per `docs/demo-script.md`) and is
    functionally solid (lazy-loaded off the main thread, no playback-tick
    coupling) — but I haven't audited its own visual layout the way the
    Live Viewer just got this pass. Worth being the next thing looked at
    now that Live Viewer's iteration is wrapping up, given how much weight
    the demo script puts on it.

## D. Cross-cutting technical debt

12. **`main_window.py` is 4,543 lines** and every Analysis panel's
    cross-navigation, room-outline sync, ceiling-mask sync, and time-series
    data selection is wired through it via dozens of `_sync_cell_*`/
    `_on_cell_*` methods. Not urgent-breaking (the suite is green, 999
    passed / 1 skipped at last full run), but it's the single largest
    coupling point between Live Viewer internals and every Analysis
    feature — anything that touches one risks the other. Not a quick fix;
    flagged for awareness before any future large Analysis feature lands
    on top of it.

---

## Suggested phasing

**Quick wins (cheap, no data dependency):**
- A1 (breadcrumb from the existing `tab_shown` signal)
- A3 (surface Investigation History as two toolbar buttons)
- B8 (point Forecasting's placeholder at `ml/README.md`)
- C9 (route Ask through `QuantityProvider` for gating consistency)

**Next (a bit more design/eng effort):**
- A2 (searchable panel jump)
- A4 (one-time Analysis coach-mark, same mechanism as the Live tour)
- B6 (shared partial-FED badge wherever hazard classification renders)

**Worth evaluating, not committed:**
- C10 (whether Reference & Communication should restructure)
- C11 (a dedicated visual pass on PCA/Clustering)

**Awareness only, no immediate action:**
- D12 (`main_window.py` size/coupling)

**Blocked on external data (M-SIM cluster re-run), not app work:**
- Full FED/CO tenability becoming the default (not just Space-time's toggle)
- Real velocity streamlines/quiver, true causal tracing in "Why is it hot?"
