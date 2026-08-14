# C10 / C11 / D12 — Investigation Findings (2026-08-06)

Companion to `docs/ANALYSIS-ROADMAP-2026-08.md`. Investigate-and-report only,
per instruction — nothing in this document has been implemented. Each
section ends with a recommendation; the decision is not made here.

Also tracks later flagged findings from the same session, in the same
investigate-and-report spirit, even where they didn't originate from the
original C10/C11/D12 batch (see the Smoke Detector section below, added
2026-08-11).

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

**Status: done.** Executed later in this session as a contained
`cell_sync.py` extraction (`sync_cell`/`sync_extent`/`sync_ceiling_mask`/
`sync_timeseries_strip`), behavior-preserving by construction (every call
site kept its exact prior subset), verified against the exact two bug
surfaces named above. Two findings surfaced *by* that extraction remain
open, listed below rather than folded into the extraction itself.

---

## Open findings carried forward (undecided as of 2026-08-11)

Short-form list of everything from this document, plus later investigate-
and-report passes, that's still a decision rather than a closed item.

1. **C10 — Graph is misfiled** in Reference & Communication (it's bus-
   connected and history-recorded, contradicting that group's "not itself
   an investigation" charter; Quantities+Ask do fit). Small, optional,
   fold-in-when-nearby — not urgent. See the C10 section above.
2. **`_on_cell_type_changed`'s missing extent leg.** The D12 extraction's
   `sync_cell()` seam made this visible rather than fixing it: that one
   call site still calls only `sync_timeseries_strip`/`sync_ceiling_mask`
   directly, never `sync_extent`, exactly matching its pre-extraction
   behavior (`cell_sync.py`, `main_window.py`'s `_on_cell_type_changed`).
   Undecided: document as correct-by-design, or add the leg. One line
   either way.
3. **The difference/ensemble inline extent duplication.** `_render_difference_cell`/
   `_render_ensemble_cell` (`main_window.py`) each re-implement the same
   "extent may have changed, update if so" check inline rather than
   routing through `cell_sync.sync_extent` — a 5th/6th copy of that
   check, outside the four sites the D12 extraction covers. Small
   follow-up, deliberately not folded into that extraction.
4. **The sprinkler-physics finding.** A heat detector and a sprinkler at
   the same point can legitimately disagree (verified on case 17: heat
   detector trips at 0.5 s, sprinkler's RTI-modeled link never crosses
   68 °C because the fire is too short-lived for its thermal lag to catch
   up) — real behavior, not a bug, but worth a code comment on
   `devices.py`'s `compute_sprinkler` so the next person doesn't
   re-investigate "detector fired, sprinkler didn't, is this a bug?" from
   scratch. (The UI side of this is already done — `device_panel.py`'s
   "did not activate (link peaked at N °C, needs M °C)" message.)

---

## Smoke Detector — feasibility investigation (2026-08-11, investigate-only)

**Question:** is a defensible smoke-detector model achievable from this
app's real data, at the same rigor bar the sprinkler's RTI model clears?

**Verdict: not currently defensible** — the conversion formula is real
and citable, but this dataset's `SOOT DENSITY` field can't support a
trustworthy point-detector threshold. The current "smoke detection isn't
modeled" state is more honest than any threshold buildable today.

### What real smoke detectors sense vs. what exists here

Photoelectric (optical) detectors respond to light obscuration, which
*does* have a standard, citable relationship to soot mass concentration:
the mass extinction coefficient (`K = Kₘ · ρ_soot`, Beer-Lambert), Kₘ ≈
8700 m²/kg for flaming combustion — the same relation FDS itself uses
internally for its own `VISIBILITY` output (Jin's correlation, FDS
Technical Reference Guide). Ionization detectors respond to particle
number/size, not mass concentration — no clean formula exists for those,
so "smoke detector" here would only ever mean "photoelectric." The exact
UL 217/NFPA 72 trip-threshold percentage was not verified against the
actual standard text — flagged as needing that check before ever being
cited as authoritative, not stated as fact here.

### What's already in the codebase

Grepped for `extinction`/`obscuration`/`visibility`/`jin`/mass-extinction
patterns across `src/`: **nothing exists** beyond the one-line registry
interpretation text ("a proxy for smoke obscuration") and the empty,
gated `VISIBILITY` registry entry. `docs/msim-preparation.md` scopes
`VISIBILITY` as needing a *new* `&SLCF QUANTITY='VISIBILITY'` line (i.e.
computed by FDS itself, gated on M-SIM) — it does not consider deriving
it client-side from the `SOOT DENSITY` already on disk via the same
formula, which is mathematically possible without new simulation output.
Noted, not acted on — a real scope decision, not a mechanical fix.

### What the real data actually supports (re-verified directly, not assumed)

Checked `SOOT DENSITY` at the real room ceiling (z≈0.22 m, `schematic.ROOM_Z[1]`
— corrected after an initial wrong check at the domain's top edge, z=0.48,
which is empty buffer air above the solid ceiling slab, not room space) on
real scenarios (case 3, case 17):

- At a single plausible detector point (x=0.35 m, ceiling z=0.22 m, away
  from the candle): **exactly zero for the entire 120 s run, both
  scenarios.**
- Widened to a ceiling band across the room (x 0.27–0.8 m, still away
  from the immediate plume): only **16.1%** (case 3) / **5.1%** (case 17)
  of cell-time samples are ever nonzero at all, and where present it's
  sharply patchy between adjacent grid cells (0 to several thousand mg/m³
  one cell apart).

A heat detector/sprinkler reads TEMPERATURE — smooth and reliable
anywhere in the room. A soot-based smoke-detector threshold would be a
placement lottery: move the virtual device one grid cell and it could
flip from real signal to permanently dead for the whole run. That's the
disqualifier, independent of whether the conversion formula is legitimate
(it is).

### Recommendation

Do not build. If ever revisited, the smallest honest version would need:
Kₘ documented as a modeling choice (flaming vs. smoldering soot have
different cited values — same "domain expert must set this" flag
`msim-preparation.md` already puts on `CO_YIELD`), a UL/NFPA-verified
threshold, and a `reduced_model`-style disclosure caption every time it
renders. Not recommended even then — unlike the sprinkler's reduced-model
fallback (degrades to a documented worst-case assumption), this one's
failure mode is a device that silently never fires depending on exact
grid placement, which is a worse trap for a user than not modeling it at
all. No code changes made for this investigation.

### Closing note (2026-08-14)

A working prototype (`devices.py::compute_smoke_detector`, Beer-Lambert
obscuration, Kₘ=8700 m²/kg, a UL/NFPA-sourced 2.5 %/ft nominal photoelectric
threshold — the exact verification gap this doc originally flagged) was
built and tested against real case data. It **confirmed rather than
overturned** the verdict above: the resulting ~9.5 mg/m³ activation
threshold sits roughly three orders of magnitude below this dataset's real
nonzero soot values (bimodal — exactly 0, or already several thousand
mg/m³), so the model produces instant-trip-or-never-trip behavior at a
given point, not a graded response, with the same real grid-placement
sensitivity described above. Final decision: not shipped. "Smoke detection
isn't modeled" remains the honest state; no smoke-detector code is present
in the app.
