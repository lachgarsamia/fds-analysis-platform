# C10 / C11 / D12 — Investigation Findings (2026-08-06)

Companion to `docs/ANALYSIS-ROADMAP-2026-08.md`. Investigate-and-report only,
per instruction — nothing in this document has been implemented. Each
section ends with a recommendation; the decision is not made here.

---

## C10 — Reference & Communication grouping

**Question:** is there evidence the grouping (Quantities / Graph / Ask under
one tab) actually confuses, or is it a naming-of-convenience that works fine?

### What each panel actually does, behaviorally

| Panel | SelectionBus | Cross-nav in (from elsewhere) | Cross-nav out | Investigation History |
|---|---|---|---|---|
| **Quantities** | none | none found | none | never recorded |
| **Ask** | none — `results.insight_activated` wired only to its own `_show_answer`, never the bus | none found | none | never recorded |
| **Graph** | `set_bus()`, publishes on node click | none found | yes — jumps scenario/time/region | yes, recorded like any real investigation action |

Checked directly: `grep` for `_reveal(self.graph_panel)` / `_reveal(self.quantities_panel)`
/ `_reveal(self.query_panel)` / any `show_tab(...)` targeting these three —
**zero hits anywhere in `src/`.** Nothing in the app currently navigates a
user *into* any of these three from elsewhere. So there's no evidence of the
grouping "fighting" an attempted cross-nav pattern, because no such pattern
exists today in either direction.

### The actual inconsistency found

`pages/analysis.py`'s own module comment defines the group's charter:

> Reference & Communication: authoring/browsing/reporting tools that
> **aren't themselves an investigation** of the simulation

Graph doesn't fit that. It publishes to the `SelectionBus`, drives
navigation, and gets recorded in `InvestigationHistory` exactly like
Dashboard or Narrative — it behaves like a core investigation surface, not a
reference tool, despite sitting in the group whose charter explicitly
excludes that. Quantities and Ask, despite covering different subject
matter (a static status table vs. a closed-grammar Q&A), *do* both fit the
charter and cohere with each other: neither touches the bus, neither
publishes or reacts to a selection, both are self-contained lookups.

Also relevant: `context.py`'s `gather_context()` — a "what's related to
this point" mechanism that pulls from Graph alongside Devices/Zones/
Narrative/Cause/Sessions — has **zero callers anywhere in `src/`** (only
its own unit tests). Traced via `git log`: its UI (`context_panel.py`, a
"Context" tab) was deliberately deleted in `667c688` ("UX consolidation --
global scenario control + remove Context tab") as low-value, but the data
layer was explicitly kept "likely reused by a future Assistant restructure"
per that commit's own message. So Graph's deeper cross-panel integration is
real in the codebase but currently inert in the live app — not evidence for
or against the current grouping, just a dangling seam worth knowing about
before anyone assumes Graph's integration is more central than it is today.

### Recommendation

Not "restructure" and not a flat "leave as-is" — the evidence points at
something narrower: **Graph is the specific misfit**, not the grouping as a
whole. Quantities+Ask already cohere. This is a data point for a *possible*
small, targeted move (Graph joining a group that actually admits
investigation tools) rather than a full regrouping — but that's a design
call, not something the evidence forces. No cross-nav-fighting evidence
exists either way, so there's no urgency.

---

## C11 — PCA / Clustering visual pass

Rendered `AnalyticsPanelDock` (the tab's actual content widget) against the
real 24-scenario dataset, at four sizes, light and dark theme.

### Setup verified
- Real data loaded via the same lazy-load path a user triggers: `build_feature_index()`
  over all 24 real scenarios, `run_pca`, `run_clustering`. All 24 case
  indices present, 2 clusters, 83% candle-count alignment — matches
  `docs/demo-script.md`'s own claim about this feature.

### Defects found (named, at specific sizes)

1. **Title clips at moderate-to-small widths.** "Ensemble PCA — scenario
   clustering by fire behavior" (the full string, centered, `fontsize=11`)
   is cut off on the right edge at 1200×750 and 700×500 ("...fire behavic"
   / "...fire b"), and on **both** edges at 1000×650 ("mple PCA...fire b").
   Clean only at 1400×800. The title string is simply too long for the
   figure width at any DPI below that threshold, because font size is
   fixed in points while `subplots_adjust`'s margins are fixed fractions —
   neither adapts to the actual pixel width.

2. **Y-axis label clips on the left, specifically at 1000×650** (not at
   700×500 or 1400×800): "PC2 (20% variance explained)" renders as "C2 (20%
   variance explai" — the leading "P" is cut off and the trailing "ned)"
   runs past the plot border. Aspect-ratio-dependent, not simply
   "smaller = worse."

3. **X-axis label overlaps the bottom caption, specifically at 1000×650**
   (reproduced at 1200×750 in dark mode too): "PC1 (68% variance explained)"
   and the italic caption ("24 scenarios, 2 clusters — 83% match candle
   count.", placed via `fig.text(0.5, 0.01, ...)`) render on top of each
   other, both unreadable. Root cause: the caption is pinned 1% from the
   figure's bottom edge regardless of figure height, while the x-axis label
   sits at a fixed fraction (`bottom=0.13`) above it — at some aspect
   ratios these fixed offsets collide.

4. **Not a defect, initially suspected one:** `SliceView`-style panels hard-
   force `ax.set_facecolor(MplCanvas.PLOT_BG)` (always white) elsewhere in
   the app, which would read as a jarring white box in dark mode. Checked
   this panel specifically in dark theme (rendered, not assumed) — the
   axes background actually follows the theme correctly (dark), because
   `set_plot_theme()`'s `_style_ax()` re-applies the theme's own facecolor
   to every registered `MplCanvas`. No dark-mode contrast issue here.
   (Flagging that this was checked and cleared, not skipped.)

5. **Not a defect:** the two legends (Cluster upper-left, Candles
   upper-right) never overlap each other or the data points at any of the
   four sizes tested. Discrete two-legend design works as intended.

### Why this matters in practice, not just in a synthetic test

The Analysis page's real available width is roughly (screen width) minus
the nav rail (220–560px, user-resizable per this session's A3/A4 work) —
on a 1512px-wide laptop with the rail at its default ~340px, that's
~1170px, squarely in the range where defects #1–#3 reproduce. This isn't a
size nobody would hit.

### Recommendation

Worth fixing — these are real, reproducible, and land in a realistic
window-width range, on the single feature the demo script calls the
strongest "wow" moment for Analysis. Scope for a follow-up: font sizes and
margins need to respond to the actual canvas size (or the title needs
shortening/wrapping), not fixed points/fractions. Not scoped or fixed here
per instruction — this is the defect list to choose from.

---

## D12 — main_window.py coupling surface

**Not a refactor. Mapping only.**

### Scale

- `src/main_window.py`: **4,628 lines**, **195 methods** (grew +162 lines
  this session alone — every item in this session's work touched this file).
- **25 analysis panels** held as direct `self.<name>_panel` attributes,
  constructed and wired entirely inside this one file, plus 4
  `_build_*_panel` builder methods.

### The specific coupling surface named in the task

Three `_sync_cell_*` methods (`_sync_cell_extent`, `_sync_cell_ceiling_mask`,
`_sync_cell_timeseries_strip`, `main_window.py:2086/2148/2216`) each handle
one concern of "make this GridCell's view match its current
(scenario, quantity)": extent + room outline, the ceiling-obstruction mask,
and the time-series strip data. They are **not called from one place** —
they're invoked from **four different call sites**, inconsistently:

| Call site | What it calls | Cell scope |
|---|---|---|
| `_init_cell_view` (~2400) | extent + ceiling_mask passed as direct `init_plot()` kwargs (a *different* mechanism, not `_sync_cell_extent`/`_sync_cell_ceiling_mask`), plus `set_room_outline()` directly, plus `_sync_cell_timeseries_strip()` | one cell, on first view creation |
| `_sync_current_scenario` (~2996) | all three `_sync_cell_*`, guarded by `cell_type == "slice"` | **active cell only** |
| `_redraw_cell_now` (~3625) | all three `_sync_cell_*` | any cell (active or not) |
| `_on_cell_type_changed` (~3721) | only two of three (`_sync_cell_timeseries_strip`, `_sync_cell_ceiling_mask`) — relies on whichever type-specific render method it just called (`_init_cell_view`/`_render_difference_cell`/`_render_ensemble_cell`) to have already handled extent/room-outline via its own direct-kwarg path | any cell |

So there are **two different mechanisms** for the same three concerns
(direct `init_plot()` kwargs at creation time vs. the `_sync_cell_*` helper
trio afterward), spread across **four call sites**, with **inconsistent
completeness** (one site calls 2 of 3, relying on an implicit
already-handled assumption). Adding a fourth sync concern today means
finding and updating some subset of these four places by hand — nothing
enforces "these travel together."

### Direct evidence: both bugs this session trace to exactly this shape

1. **The ceiling-mask/stale-blit bug** (Live Viewer polish pass): a
   `set_ylim()` call added inside `views.py`'s `SliceView.set_room_outline()`
   needed a paired `capture_background()` call it didn't get on the first
   pass. `set_room_outline()` is invoked from `_sync_cell_extent`, one leg
   of this same three-method trio — the bug was a missing paired call
   *within* one already-fragile sync path, not caught until real-data
   testing surfaced a visibly stale, differently-zoomed background.
2. **The A3 history-button crash**: `_build_history_nav_bar()` (in
   `main_window.py`) builds a widget assuming `AnalysisPage` will always
   place it in a layout — but `AnalysisPage`'s own demo-mode branch
   doesn't. Nothing enforces that contract between the two files, so the
   widget got garbage-collected and the next `main_window.py` call into it
   crashed. Same root shape as #1: `main_window.py` orchestrates state
   across file boundaries via direct calls/shared assumptions, with no
   enforced contract, so a correct-looking change in one file breaks an
   assumption held somewhere else in `main_window.py`.

### A counter-example already in the codebase (this isn't uniformly bad)

`analysis_panel_base.py`'s `bind_to_bus()` / `AnalysisPanelBase` already
solves the *analogous* problem for a different, older concern: ~20 panels
all needing their `scenario_combo`/`frame_slider` kept in sync with the
`SelectionBus`. That was extracted into one shared, tested helper instead
of hand-wired per panel — described in its own docstring as "the go-forward
base for new analysis panels." The codebase has already done this kind of
extraction successfully once. The `_sync_cell_*` trio is a *newer*,
GridCell/SliceView-specific concern that hasn't received the same
treatment yet, and it's exactly the un-extracted part where both bugs
landed.

### Recommendation

A full main_window.py refactor is not justified by what's mapped here —
most of the file's size is 25 independently-simple panel wiring blocks,
not itself dangerous. But the **specific** `_sync_cell_*` surface is a
concentrated, already-twice-bitten risk, and there's a working precedent
(`bind_to_bus`) for how to fix exactly this shape of problem. A **contained,
incremental extraction** — one place that owns "what must stay in sync
when a cell's (scenario, quantity) changes," called from all four current
sites the same way — is worth doing *before* the next feature adds a fourth
concern to this trio, not as urgent standalone cleanup. Scope was not
sized here (no line-level plan was drafted) since this is a mapping report,
not a proposal.
