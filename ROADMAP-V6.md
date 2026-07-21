# ROADMAP V6 — Gated Capabilities (data-blocked, interfaces prepared)

> V6 is the work that **cannot be built honestly on the current output**. Each
> capability is blocked on data the study does not yet have — the M-SIM cluster
> re-run (`docs/msim-preparation.md`) or an experimental dataset. This document
> records, per capability, the data prerequisite and the **interface hook already
> prepared in the code**, so each drops in without re-deriving anything. Nothing
> here is implemented in V5 (per the V5 Phase 7 rule).

## Principle: prepare the seam, not the feature

For every gated capability the seam already exists: a registry entry, a provider
method, or a panel comment marking exactly where the data wires in. When the data
lands, the change is additive at that seam — no rearchitecting.

---

## V6-1 — True 3D velocity: streamlines / quiver / volume

**Blocked on:** U-VELOCITY and W-VELOCITY components (the current `VELOCITY` is
speed magnitude only). M-SIM `&SLCF QUANTITY='U-VELOCITY'` / `'W-VELOCITY'` at
the read plane (`docs/msim-preparation.md` §3).

**Prepared hooks:**
- `registry.py` — `U-VELOCITY`, `W-VELOCITY` registered (units/colormap/
  interpretation), `gated=True`.
- `quantity_provider.py` — `get_vector(scenario, direction, offset)` stub raises
  `GatedQuantityError`; wire it to the two component reads when present. The
  provider already resolves raw + derived fields, so a vector field is one more
  resolution behind the same call.
- Volume: the `.s3d` backbone (`src/fds/s3d/`) already decodes volumetric SOOT;
  a velocity/temperature volume reuses that path. New surface: a streamline /
  quiver / volume panel that calls `get_vector`.

## V6-2 — Multi-plane linked cross-sections (XY / XZ / YZ)

**Blocked on:** slices on additional planes. Today only the y-normal read plane
is output; the store keys slices by `(quantity, direction, offset)` already, so
only the extra `&SLCF` output is missing, not a data path.

**Prepared hooks:**
- `spacetime_panel.py` — comment marks where plane selectors and per-plane
  `SliceKey(direction, offset)` reads attach; the panel already reshapes one
  plane into x–time / z–time, so XY/XZ/YZ is the same reduction on more planes.
- `slice_key.py` — `SliceKey` already carries direction + offset; no change
  needed to address new planes.

## V6-3 — Full FED / CO / smoke toxicity

**Blocked on:** a CO output. M-SIM needs a `CO_YIELD` in `&REAC` (a combustion-
modeling choice for a domain expert) plus a CO `&SLCF` — flagged as a modeling
task, not a mechanical edit (`docs/msim-preparation.md` §3).

**Prepared hooks:**
- `registry.py` — `CARBON MONOXIDE VOLUME FRACTION` registered with hazard
  bands, `gated=True`.
- `tenability.py` — comment marks where `fed_gas_dose(co_field, fps)` sums with
  the convected-heat dose into a full FED; the partial-screen disclaimer retires
  at that point.
- `hazard_spaces.py` — the class bands generalize to a full-FED axis once the
  gas dose exists (today: temperature-only, stated in `BASIS`).

## V6-4 — Validation toolkit (simulation vs experiment)

**Blocked on:** an experimental dataset (thermocouples / sensors). None is
bundled with the study, so there is nothing to validate against.

**Prepared hooks:**
- `validation.py` — the full interface is stubbed (`load_experimental_series`,
  `rmse`, `arrival_time_error`, `validation_table`), each raising
  `ValidationGate` until data exists. The V6 validation panel calls these and
  overlays sensor curves on the existing time-series machinery.
- Reuse: `timeseries` (curves/overlays), `report_builder` (a validation table
  in a report), `figure_export` (publication figure).

---

## Sequencing

Lead with **V6-1 (U/W-velocity)** — the single feature most blocked by current
output and the one that unlocks honest streamlines and the cinema Tier-2
advection. Then **V6-3 (CO → full FED)** once the modeling review sets
`CO_YIELD`. **V6-2 (multi-plane)** is cheap the moment the planes are output.
**V6-4 (validation)** waits on external experimental data and is independent of
M-SIM.

## Non-negotiable principles (carried from V2–V5)

- No feature ships until its data exists; gated interfaces raise a clear,
  honest error (`GatedQuantityError` / `ValidationGate`) rather than fabricating.
- Every conclusion stays a template over computed values with a `basis`.
- The suite stays green and the app runnable; each gated capability drops in
  additively at its prepared seam.
