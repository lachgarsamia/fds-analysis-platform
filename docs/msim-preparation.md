# M-SIM cluster re-run — preparation & gate (V2 roadmap M3.5)

**Status: NO-GO for execution in this environment.** This document prepares the M-SIM re-run so it can be executed the moment its gate opens, without re-deriving anything. It deliberately makes **no edit to `fds/template.fds`** and **no code change** — both would violate the gate below.

## 1. What M-SIM is

A one-time re-run of the 24-scenario candle study on the FZJ cluster (`fds/start_job.batch`, FDS 6.7.1) after adding richer slice/species output to the FDS input decks, purely to unblock app features that the *current* output cannot support. Not a new research axis — the factorial design is unchanged.

## 2. The gate (why this is NO-GO now)

The V2 roadmap (§6 M3.5) states the gate condition is **"met once M0.1 closes"**, and the V1 roadmap (§0) states **"editing `fds/template.fds` does not begin until parser validation is confirmed solid — not merely until cluster access exists."** Two independent blockers, both currently unsatisfied:

| Blocker | State | Evidence |
|---|---|---|
| **M0.1 edge-column adjudication** (Phase 0) | **OPEN** — Phase 0 was skipped | `docs/spike-parser-validation.md` §3: an isolated outer-edge-column discrepancy vs `fdsreader` was found, characterized, and filed, but not adjudicated (needs FDS binary-format docs or a Smokeview check). The `.sf`/`.s3d` parsers are otherwise well-validated (interior <4 °C, per-frame max exact; `.s3d` matches `fdsreader` exactly after the M2.2 Fortran-order fix). |
| **Cluster access** | **UNAVAILABLE** here | No SLURM/FDS in this environment; `start_job.batch` targets FZJ `slfire`. |

**The gate rationale is scientific, not bureaucratic:** M-SIM produces *new* simulation output; the parser must be trusted on it before that output is trusted, and the one open parser question (edge column) touches exactly the domain-boundary cells a re-run would newly populate. Closing M0.1 first is what makes the new output trustworthy.

**Ungated exception (per V1 §0):** `sim.0` — re-running the *existing, unedited* templates on the cluster as a baseline — does not depend on M0.1 and may run whenever cluster access is secured. It just reproduces the current output; it is a cluster smoke-test, not the feature-unblocking run.

## 3. Ready-to-apply template changes (apply only once the gate opens)

Both `fds/template.fds` and `fds/template_hvac.fds` currently emit, at the app's read plane and its cell-centered twin:

```
&SLCF PBY = 0.000, QUANTITY = 'TEMPERATURE' /
&SLCF PBY = 0.000, QUANTITY = 'VELOCITY' /          ! speed magnitude only
```

Add, at the same `PBY = 0.000` plane (and, matching the existing convention, a `CELL_CENTERED` twin at `PBY = -0.005` if desired):

```
&SLCF PBY = 0.000, QUANTITY = 'U-VELOCITY' /        ! in-plane x-component
&SLCF PBY = 0.000, QUANTITY = 'W-VELOCITY' /        ! in-plane z-component
```

For a y-normal (PBY) slice, **U (x) and W (z) are the two in-plane components** — exactly the true vector field F14 needs; the existing `VELOCITY` is |v| magnitude and cannot give direction. This is the lowest-risk, highest-value edit and is a pure output addition (no physics change).

**CO for real FED (higher-risk, needs modeling review — do not apply blind):** M3.2 tenability is temperature-only *partial* FED because there is no CO output. Producing CO requires more than a slice line: the `&REAC` (currently `SOOT_YIELD=0.01, FUEL='METHANE'`) needs a `CO_YIELD` (a combustion-modeling choice a fire-science domain expert must set), after which a `&SLCF PBY=0.000, QUANTITY='CARBON MONOXIDE VOLUME FRACTION'` (or the `.s3d` `SMOKF3D` equivalent) becomes meaningful. Flagged as a modeling task, not a mechanical edit.

**Optional 2D soot at the read plane:** a `&SLCF PBY=0.000, QUANTITY='SOOT DENSITY'` would give a cheap 2D soot read on the app's plane, avoiding the volumetric `.s3d` decode M2.1/M2.2 handle today. Nice-to-have, not required.

## 4. What each addition unblocks in the app (the wishlist, consolidated)

- **U/W-VELOCITY → F14 real streamlines / quiver** (roadmap Phase 3–4, gated). Today the science views cannot draw honest streamlines because only |v| is stored; the cinema mode's ∇T pseudo-advection is explicitly *not* science-grade. This is the single feature most blocked by current output.
- **CO → real FED tenability** (M3.2 becomes full FED instead of temperature-only partial). The M3.2 disclaimer ("no CO/CO₂ → not a full FED") is written to be retired exactly when this lands.
- **U/W-VELOCITY also feeds the cinema smoke Tier-2 advection** (`ROADMAP-FIRELAB.md` Phase 2.1f flagged this same wishlist item).
- **U/V/W-VELOCITY → neutral-plane diagnostic for DYNAMIC PRESSURE** (Live-polish follow-up, height_panel.py's vertical profile). DYNAMIC PRESSURE = 0.5·ρ·|v|² is derived from the scalar speed and can never go negative, so its vertical profile cannot show the classic vent-flow neutral plane (the height where signed pressure crosses zero and flow direction reverses) -- it can only show where flow forcing concentrates, labeled "flow-forcing profile" rather than claiming a neutral-plane finder it isn't. Revisit once a future run exposes signed U/V/W and dynamic pressure can be computed with a real sign.

## 5. How to execute (once the gate opens)

1. Close **M0.1** (adjudicate + fix the edge-column discrepancy; re-run the `fdsreader` cross-validation). This is the gate.
2. Apply §3's SLCF additions to `fds/template.fds` (+ `template_hvac.fds`), on a branch, per the standing "branch before first edit" rule.
3. Regenerate the 24 decks: `cd fds && python3 generate_sim.py` (unchanged; it only substitutes placeholders and copies the templates).
4. Submit on the cluster: `sbatch start_job.batch` in each scenario folder (or a sweep wrapper), FDS 6.7.1 as pinned.
5. **Re-validate the new output** against `fdsreader` before the app trusts it (the gate's whole point) — especially the new `U-VELOCITY`/`W-VELOCITY` slices, which the app's parser has never read.
6. Then build **F14** (streamlines/quiver from the real components) and, if CO landed, upgrade **M3.2** to full FED.

## 6. Decision

**NO-GO to execute or to edit `fds/template.fds` now** — the gate (M0.1 closed) is not met and no cluster is available. Everything needed to execute the instant both conditions hold is specified above; nothing further can be done in this environment without violating the gate or fabricating output.

## 7. Registered-but-gated quantities (V4-M11)

The quantity registry (`src/registry.py`) now carries the target quantities
as first-class entries (units, colormap, hazard bands, plain-language
interpretation) so every analysis tool works on them the moment data
arrives. They are marked `gated=True` with `gate_reason=MSIM_GATE` and are
**excluded from the data-driven quantity discovery**, so they never enter
the tool combos and cannot break any feature; they appear only in the
read-only **Quantities** reference panel. Wire real data by having FDS emit
the corresponding `&SLCF` (see §3) — no registry change is then needed.

| Registered quantity | Unblocks | Needs (per §3) |
|---|---|---|
| `U-VELOCITY`, `W-VELOCITY` | true vector field / streamlines / quiver | `&SLCF QUANTITY='U-VELOCITY'`, `'W-VELOCITY'` |
| `CARBON MONOXIDE VOLUME FRACTION` | full FED tenability (retires the temperature-only disclaimer) | `CO_YIELD` in `&REAC` + CO `&SLCF` (modeling review) |
| `PRESSURE` | vent-driving / doorway flow | `&SLCF QUANTITY='PRESSURE'` |
| `VISIBILITY` | egress tenability | `&SLCF QUANTITY='VISIBILITY'` |
| `HEAT FLUX` | burn/ignition thresholds | surface/gas `&SLCF` heat-flux quantity |
| `SOOT MASS FRACTION` | 2D smoke read without the `.s3d` decode | `&SLCF QUANTITY='SOOT MASS FRACTION'` |

**Ungated now (V4-M11):** two *derived* quantities computed from existing
fields ship immediately — `TEMPERATURE RISE` (T − ambient) and
`DYNAMIC PRESSURE` (½ρ|v|²) — implemented in `src/derived_quantities.py`
and previewable in the Quantities panel. These need no re-run.

## 8. Prepared code hooks (V5 Phase 7 — V6 preparation)

The V5 Phase 7 pass added the *interface seams* so each gated capability drops in
without re-deriving anything (full plan: `ROADMAP-V6.md`). No gated feature is
implemented; each seam raises a clear, honest error until its data exists.

| Gated capability | Data prerequisite | Prepared seam in code |
|---|---|---|
| 3D streamlines / quiver / volume | U/W-VELOCITY (§3) | `registry` U/W-VELOCITY (gated); `quantity_provider.get_vector()` stub (`GatedQuantityError`) |
| Multi-plane XY/XZ/YZ cross-sections | slices on more planes | `spacetime_panel` plane-selector comment; `SliceKey(direction, offset)` already addresses planes |
| Full FED / CO / smoke toxicity | CO output (`CO_YIELD` in `&REAC`, §3) | `registry` CO (gated); `tenability` `fed_gas_dose` comment; `hazard_spaces.BASIS` states the partial screen |
| Validation (sim vs experiment) | experimental sensor data | `validation.py` stub interface (`ValidationGate`) |

`QuantityProvider.get()` now raises `GatedQuantityError(gate_reason)` for any
gated quantity, so a mistaken request fails loudly instead of silently reading
absent data. When M-SIM lands, the gated flags flip and these seams become plain
reads — no rearchitecting.
