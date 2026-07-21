# V5 Architecture Stabilization — the Shared Selection Model

*Design note produced before V5-M1. Design only; no behaviour is changed by
this document. It records the V4 audit, specifies the SelectionModel /
SelectionBus, the Derived Quantities Framework, and how both integrate with the
existing Event Engine, Insight system, Session manager, Browser, and Quantity
Registry.*

## 1. Audit findings (V4)

### 1.1 Selection state is stored locally, ~17 times over
Every analysis panel is a standalone `QtWidgets.QWidget` (no shared base) that
re-implements the same scaffolding:

| Duplicated element | Panels affected |
|---|---|
| own `scenario_combo` (or `combo_a`/`combo_b`) | 16 |
| own `quantity_combo` + copy-pasted `kind == "slice2d"` filter | ~7 |
| own `frame_slider` + `idx / fps` time conversion (17 sites) | 7 |
| own per-`(scenario, quantity)` cache dict | 8 |
| `showEvent → ensure_loaded` lazy-load pattern | 14 |
| `phys_to_index` / `column_for_x` coordinate mapping (15 sites) | most |

The **live viewer** keeps the "real" selection separately again: the active grid
cell (`view_grid.active_cell()` → scenario + `quantity_key`), the playback time
(`time_controller.index`), and view state (`current_colormap`, `_link_clim`) —
236 references in `main_window.py`. Analysis panels do **not** share it.

### 1.2 There is exactly one cross-panel link today
`InsightList.insight_activated → MainWindow._on_insight_activated →
_on_seek_requested` (seek playback to an Insight's time). That is the seed of the
whole idea: an **`Insight` already carries `quantity`, `time_s`, `location`,
`region`, `value`** — it is a selection payload in all but name. The
SelectionModel should mirror those fields so the two unify.

### 1.3 The session already persists selection fragments
`active_index` (scenario), `time_index` (time), `time_window` (interval),
`zones` (region), `filters` — i.e. the session is *already* saving pieces of a
selection, ad hoc. A single model makes this coherent.

**Conclusion.** The work is not "add a bus on top of 17 local states." It is
"introduce one model, then delete the local states as panels adopt it." Doing
this *before* M2–M6 is what keeps them from each inventing a 17th copy.

## 2. The SelectionModel

One immutable value object — the single source of truth for "what the researcher
is looking at". Fields chosen to be a superset of `Insight` + the live-viewer
state + the session fragments:

```
Selection(
    scenario:   int | None          # case_index (primary)
    quantity:   str  = "TEMPERATURE"
    point:      (x, z) | None        # physical
    region:     (x0, x1, z0, z1) | None
    height:     float | None         # a chosen z (height workspace)
    time_s:     float | None         # instant
    interval:   (t0, t1) | None      # time-window selection
    phase:      str | None           # detected-phase name (events.py)
    comparison: (scenario_a, scenario_b) | None
)
```

Immutable + `with_(...)` copy-updates, so a change is always a new value and
diffing "what changed" is trivial. Pure/Qt-free (testable in isolation), living
in `selection.py`.

## 3. The SelectionBus

A thin `QObject` owning the current `Selection` and one signal:

```
class SelectionBus(QObject):
    changed = pyqtSignal(object, object)   # (new_selection, origin)
    def set(self, selection, origin=None)  # publish; no-op if unchanged
    def update(self, origin=None, **fields) # convenience partial update
    @property current -> Selection
```

- **Publish/subscribe.** MainWindow owns one bus. Panels take it in their
  constructor, `bus.changed.connect(self._on_selection)` to *react*, and call
  `bus.update(origin=self, point=...)` to *drive*.
- **No feedback loops.** Every publish carries an `origin`; a panel ignores
  changes it itself originated (`if origin is self: return`), and `set()` is a
  no-op when the new value equals the current one — so a click that lands on the
  same cell cannot ricochet. This is the single well-known failure mode of a
  shared bus, closed here by construction.
- **Coalescing.** `set()` compares by value; redundant republishes are dropped,
  so a drag that emits many identical selections costs one update.
- **Threading.** The bus is main-thread/UI only. It never touches
  `simulation_controller` or `scenario_store` (which stay off-limits); data
  still comes through the store exactly as today.

## 4. Panel refactor — `AnalysisPanelBase`

An optional base class that owns the shared scaffolding so panels stop
re-implementing it:

- builds the scenario/quantity selectors **bound to the bus** (change a combo →
  `bus.update`; bus changes → combos follow);
- the lazy `showEvent → ensure_loaded` and the per-`(scenario, quantity)` cache;
- `frame ↔ time` via one helper (`time_s = idx / fps`), killing the 17 copies;
- `phys ↔ index` via the existing `timeseries.phys_to_index` (one import, not 15
  re-derivations).

Panels override `render(selection)` only. **Migration is incremental**: the base
is additive; a panel adopts it one at a time, suite green after each, and the
live viewer is migrated last (it is the most entangled). No big-bang rewrite.

## 5. Derived Quantities Framework (infrastructure, from the start)

Per the request, this ships as infrastructure, not a milestone, so *every* V5
feature supports derived quantities for free.

**Today (M11):** `derived_quantities.py` has `DERIVED = {name: (source, fn)}`,
`derive()`, `source_quantity()`, and the registry supports `kind == "derived"` —
but derived fields do **not** flow through `scenario_store.get(scenario, key)`,
so tools cannot read them.

**Design:** a `QuantityProvider` that *wraps* the store (does **not** modify
`scenario_store`, which is off-limits):

```
class QuantityProvider:
    def get(self, scenario, key) -> np.ndarray:
        q = get_quantity(key.quantity)
        if q.kind == "derived":
            src = SliceKey(source_quantity(key.quantity), key.direction, key.offset)
            return derive(key.quantity, self._store.get(scenario, src))
        return self._store.get(scenario, key)          # real quantities unchanged
    def get_extent(self, scenario, key): ...            # delegates
```

Panels (via `AnalysisPanelBase`) read through the provider instead of the raw
store. Then a derived quantity is just a registry entry + a function, and it
appears in every combo, plot, notebook, dashboard, comparison, and report
automatically. The registry entry defines a **user-definable** derived quantity
(`dT/dt`, thermal dose, temperature gradient, heat accumulation, hot-layer
thickness, a combined hazard index) whose function reduces existing fields; each
carries the same honesty metadata as M11 (unit, interpretation, and a `basis`
so downstream Insights stay traceable).

Multi-frame derived quantities (e.g. `dT/dt`) need the time axis, so the provider
signature also allows a whole-series `derive_series(name, source_series, fps)`;
single-frame and series functions register the same way.

## 6. Integration with existing subsystems

- **Event Engine (`events.py`).** Detected phases already have `(name, t0, t1)`.
  A phase is a selection: selecting one sets `interval` + `phase`, and the
  Fire-Story timeline both drives and reflects the bus. No change to detection.
- **Insight system (`insight.py`).** `Insight` and `Selection` share four
  fields. Add two pure adapters — `Insight.to_selection()` and
  `Selection.from_insight()` — so the existing `insight_activated` link becomes
  "publish this Insight's selection to the bus", and *every* panel reacts, not
  just playback. This subsumes today's single cross-panel link into the general
  mechanism with no loss.
- **Session manager (`session.py`).** Add one `selection` block to the schema
  (the fields in §2). It composes with the existing `active_index` / `time_index`
  / `time_window` / `zones` rather than replacing them (back-compat: older
  sessions simply have no `selection`). Saving/restoring a session restores the
  exact viewpoint — the M6 promise, now literal.
- **Browser (`browser.py`).** Scenario selection becomes two-way: picking a row
  publishes `scenario`; a bus scenario change highlights the row. The existing
  filter state is unaffected.
- **Quantity Registry (`registry.py`).** `quantity` is a selection field; the
  registry stays the source of truth for units/colormap/thresholds/kind, and the
  Derived Quantities Framework (§5) plugs in through `kind == "derived"` with no
  new registry concept.

## 7. Risks and how the design closes them

| Risk | Mitigation |
|---|---|
| Feedback loops between panels | `origin` guard + value-equality no-op in `set()` |
| Big-bang refactor breaks V4 | `AnalysisPanelBase` is additive; migrate one panel at a time, suite green each step; live viewer last |
| Touching the off-limits data/threading layer | `QuantityProvider` wraps the store; `scenario_store`/`simulation_controller` are never edited |
| Derived quantities diverging from honesty rules | derived entries carry the same registry metadata + `basis`; reuse M11's gating helpers |
| Selection model drift as V5 grows | one immutable value object with explicit fields; new needs extend it, not per-panel state |

## 8. Recommended sequence for V5-M1

1. `selection.py` — the immutable `Selection` + `SelectionBus`, fully unit-tested
   (equality, `with_`, no-op-on-equal, origin passthrough). *No UI yet.*
2. Insight/Session adapters (`to_selection` / `from_insight`; session `selection`
   block) — pure, tested.
3. `QuantityProvider` + one generalized derived quantity end-to-end (e.g.
   `dT/dt`) — tested through a panel.
4. `AnalysisPanelBase`; migrate **two** representative panels (one field panel:
   Height; one series panel: Linked Inspection) to prove the pattern.
5. Wire the bus into MainWindow, replacing the single `insight_activated` link;
   Browser two-way.
6. Migrate remaining panels incrementally in later steps; the live viewer last.

Steps 1–5 are the M1 deliverable (foundation + proof on two panels); the rest is
mechanical follow-through that M2–M6 can ride on.
