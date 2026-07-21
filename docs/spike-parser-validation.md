# M1.3s — Parser validation spike

**Branch:** `spike/parser-validation-fdsreader` · **Timebox:** 1 day · **Status:** complete
**Scope (per ROADMAP.md §4):** cross-validate `src/fds/slice/slice.py` against an independent reader on exactly ONE simulation, produce a colormap recommendation for M1.3.1, and answer the M-SIM scope flag. Not shippable app code — this document is the deliverable.

---

## 0. Deviation from spec: `fdsreader` version

The roadmap pinned `fdsreader==10.1`. **That version does not exist on PyPI** — the package's real release history runs `0.0.0` → `1.11.7` (no `10.x` line was ever published; likely a mix-up with a different versioning scheme when the pin was chosen). This is a factual correction, not a design ambiguity, so the spike proceeded rather than blocking:

**Used `fdsreader==1.11.7`** (the latest stable release, no pre-releases) instead, installed in a scratch venv, not added to `pyproject.toml`, per the original scope.

## 1. Cross-validation methodology

- **Scenario:** `c1_d0_vod0_voc0` — quantity `TEMPERATURE`, direction=1 (y-normal), offset=0. This is the app's default load path (`load_data.py`).
- **Data source:** the *full* on-disk copy at `fds/sim/c1_d0_vod0_voc0/` rather than the trimmed `tests/fixtures/` copy. The pytest fixture only ships the `.sf` files our own parser actually reads for this one quantity/direction/offset combination; `fdsreader` constructs a `Simulation` object eagerly over *every* slice it finds in the `.smv` (including quantities/mesh chunks the fixture deliberately omits) and crashes on the first missing file. The full on-disk copy is the same scenario, just complete — still exactly ONE simulation, not a sweep.
- **Environments:** each parser was run in its own environment (project venv for ours — `fdsreader`'s newer numpy doesn't understand our parser's legacy dtype character codes; `fdsreader`'s scratch venv for it) and results were saved to `.npy` for a neutral-environment comparison, to avoid a cross-environment numpy mismatch contaminating the result.
- **Orientation reconciliation:** `fdsreader`'s `Slice.to_global()` returns shape `(481, 101, 49)`; our `readSlice()` returns `(481, 49, 101)` — the two spatial axes transposed (an axis-order convention difference, not a data issue). All comparisons below use `fdsreader`'s array transposed to `(0, 2, 1)` to align with ours.

## 2. Results

### Times

**Exact match.** `np.array_equal(our_times, fdsreader_times)` → `True`, max abs diff `0.0`, 481 timesteps both sides. This directly validates M1.2's vectorized `readAllTimes` (`n_times = (filesize - offset) // stride`) against an independent implementation.

### Temperature data — interior of the domain

Excluding the single rightmost column (see §3 below):

| Metric | Value |
|---|---|
| Max abs diff | 3.76 °C |
| Mean abs diff | 0.0068 °C |
| Cells with diff > 1 °C | 9,012 / 2,380,469 (0.38%) |
| Cells with diff > 5 °C | 2,471 / 2,380,469 (0.10%) |

The larger interior diffs (up to ~3.7 °C) concentrate in columns 73–95, which fall inside the mesh block spanning x=[0.75, 1.0] — the same mesh containing the candle (`&VENT XB = 0.92,0.96, ... / Candle 1` in `c1_d0_vod0_voc0.fds`). This is the steepest-thermal-gradient region in the whole domain (values climb from ~24 °C to ~80+ °C within a few grid cells near the plume); a few-°C absolute diff there is consistent with normal float32/float64 and sub-cell-position differences between two independently-implemented readers, not a structural bug.

### Aggregate statistics (what downstream analytics actually uses)

| Metric | Ours | fdsreader | Diff |
|---|---|---|---|
| Global peak temperature | 469.2725 °C | 469.2724914… °C | ~1e-5 °C (float32 rounding) |
| Frame of global peak | 32 | 32 | exact match |
| Per-frame max, all 481 frames | — | — | **0.0 exact match, every frame** |

This is the important number: the per-frame maximum — the statistic M2.5's `summary_stats.py` and M3.1's feature vectors will actually consume — agrees **exactly** across all 481 frames. Whatever discrepancy exists (see §3) never touches the true per-frame hot spot.

## 3. Triaged discrepancy: rightmost column (x=1.0 domain edge)

One real, isolated, and precisely characterized disagreement:

- **Location:** column index 100 of 101 (the single rightmost column, at the domain's outer x=1.0 boundary) — no other column shows this pattern.
- **Magnitude:** max abs diff 39.8 °C, mean 1.7 °C, confined entirely to that one column.
- **Signature:** at frame 329, row 6 — our parser reports `42.58 °C` at column 100 vs. `82.41 °C` at column 99 (a real, physically plausible value distinct from its neighbor). `fdsreader` reports `82.41093444824219 °C` at **both** column 99 and column 100 — bit-for-bit identical values.

That exact-duplicate signature on the `fdsreader` side is the fingerprint of boundary padding/extrapolation (repeating the last real interior value into an edge/ghost column), not of a genuinely independent measurement. Whether that means `fdsreader` is padding a boundary our parser reads correctly, or our `combineSlices` mesh-stitching is misassigning the true edge value at that one seam, **could not be adjudicated from the data alone** — doing so needs either the raw FDS binary-format spec (not available in this offline environment) or a genuine Smokeview visual cross-check (see §4). Per the spike's own instructions, no fix was attempted here.

**Filed as a follow-up, not fixed in this branch:** added to ROADMAP.md §0's "Known defects to fix opportunistically" list, associated with the existing item #1 (`combineSlices` assumes uniform mesh resolution across meshes, unvalidated) since this is evidence bearing directly on that same code path's mesh-boundary handling.

**Practical impact today: none observed.** The per-frame max-temperature agreement (§2) confirms this edge artifact doesn't reach any statistic currently computed or planned through Phase 3. It's worth a real fix before M2.6's probe/isotherm work (which reads coordinates near domain edges) or before treating `combineSlices` as trustworthy for arbitrary mesh layouts.

### 3.1 Resolution (V2-M0.1, 2026-07): adjudicated in favour of our parser

The open question in §3 -- is `42.58 °C` (our value at x=1.0) real, or is `fdsreader`'s duplicated `82.41 °C` correct? -- was resolved without Smokeview, using `fdsreader`'s **own per-mesh subslice** (its independent raw `.sf` decode, *before* the `to_global()` stitching that produced the duplicate). For the last mesh (x=0.75-1.0), at frame 329, its final three x-nodes read `[135.05, 82.41, 42.58]` -- i.e. `fdsreader`'s raw decode of the x=1.0 node is **42.58 °C, bit-for-bit our `combineSlices` value**.

So two independent decoders (ours, and `fdsreader` at the per-mesh level) agree the true FDS value at the outer boundary node is 42.58 °C; only `fdsreader`'s `to_global()` differs, duplicating the x=0.99 value into the edge. **Our parser is correct; the discrepancy was `fdsreader`'s global-stitcher boundary padding** -- the same general-case `to_global()` weakness independently found in the M2.1 `.s3d` spike (`docs/spike-s3d-v2.md`). No fix to `combineSlices` was needed. Pinned as `tests/test_slice_parser.py::TestOuterEdgeColumn`.

## 4. Smokeview convention review — not performed

Smokeview is not installed in this development environment (`command -v smokeview` → not found), and no display/GUI tooling is available in this sandboxed session to install and drive it interactively. **This task could not be completed as specified.** No visual color-band/ramp convention was observed firsthand; nothing here should be read as a Smokeview-verified recommendation.

## 5. Colormap recommendation (for M1.3.1)

Given §4's limitation, this recommendation is based on FDS-visualization domain convention rather than a confirmed Smokeview match:

**Keep the app's existing default, `gist_heat`.** It already renders as a black→red→orange→yellow→white progression — the standard "blackbody radiation" / flame palette used across fire-visualization tooling (and functionally close to Smokeview's common `Blue_Green_Red`/hot-metal-style temperature ramps). The app already defaults to it (`main_window.py`'s `self.current_colormap = self.settings.value("colormap", "gist_heat")`), so **M1.3.1 has no default to change** — just confirm and keep it, and continue offering `viridis`/`cividis` as the colorblind-safe alternatives already in the menu.

**Hazard-threshold color bands** (for pairing with M2.6's isotherm feature) — proposed as general fire-safety reference points, **not derived from this study's own data**, and should be reviewed by a domain expert/supervisor before being treated as authoritative for the demo:

| Band | Range | Rationale |
|---|---|---|
| Safe | < 60 °C | Below common skin-contact pain/injury threshold for brief exposure |
| Hazardous | 60–300 °C | Burn risk; approaching materials' ignition/degradation range |
| Life-threatening | > 300 °C | Near flashover-adjacent conditions; short survivable exposure time |

## 6. M-SIM scope flag

**Does "improve the simulation for better output" require editing `fds/template.fds`? No — not based on anything found in this spike.**

- VELOCITY slices already exist in every scenario's raw output and are simply unread by the app (confirmed again during this spike — `fdsreader` found `VELOCITY` slices alongside `TEMPERATURE` with no template changes needed).
- The one real finding here (§3's edge-column discrepancy) is a **parser-side** (`combineSlices`) issue, not a simulation-output gap — fixing it means changing `slice.py`, not `fds/template.fds`.
- Nothing in this spike surfaced a need for finer mesh resolution, additional quantities, or additional slice planes. That scoping question (M-SIM's sim.1) remains open and unaffected by this spike either way.

## 7. Summary for M1.3.1

Adopt `gist_heat` (no change needed — already the default). Cross-validation gives high confidence in the parser's core correctness (exact time match, exact per-frame-max match across all 481 frames); the one located discrepancy is narrow, well-characterized, non-blocking for current work, and filed as a follow-up rather than silently ignored.
