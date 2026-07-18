# M2.1 `.s3d` Science Backbone spike (V2 roadmap)

**Timebox:** 1–2 days (spent: well under one day) · **Status:** complete · **Scope:** determine whether a y=0-plane-only extraction (not full-domain) from `.s3d` volumetric data is viable as V2-M2.1's data path, with value-level cross-validation against the trusted `.sf` TEMPERATURE ground truth. Investigation only — no `src/` changes. Builds on the prior FireLab-roadmap Tier-3 spike (`docs/spike-s3d.md`), re-verifying its findings and going further (real extraction + real cross-check, not estimates).

Method: `fdsreader==1.11.7` (ad hoc, not added to `pyproject.toml`) against the real `fds/sim/c1_d0_vod0_voc0/` scenario, same interpreter/numpy the app's own test suite runs under (`.venv`'s numpy 2.5.1 is incompatible with `slice.py`'s dtype strings — noted for anyone reusing this script).

## Findings

1. **Only 12 of 24 submeshes are needed for the y=0 plane, not 24.** The domain's `&MULT` splits Y into exactly 2 rows (`J_UPPER=1`); y=0 is the boundary between them, so either row's face suffices. Confirmed via `mesh.extent.y_start == 0.0`.

2. **Submesh grids are node-centered (26×16×17), not cell-centered (25×15×16) — one shared boundary node per adjacent pair.** Naively concatenating 12 tiles gave a (51×104) plane; after trimming the shared node at each seam (the same convention `slice.py`'s `combineSliceGeometry()` already uses for `.sf`), the stitched plane is **exactly (49×101) — an exact match to the `.sf` grid shape.** Strong corroboration that `.s3d` and `.sf` share one coordinate system, and that `combineSliceGeometry()`'s stitching logic is directly reusable rather than needing a new algorithm.

3. **TEMPERATURE (`SMOKG3D` channel 03) is dead data in this dataset — root-caused, not just re-observed.** The prior spike flagged "upper bound always 0" as an open question; this pass confirms it at the raw file level: every line of `c1_d0_vod0_voc0_0001_03.s3d.sz`'s 4th column (`upper_bound`) is `0.0000000E+00`, across all 481 frames. This is baked into the FDS-generated output, not a decode or API bug — decoded values are correctly zero everywhere. **Consequence: the roadmap's planned value-level cross-check ("extracted `.s3d` TEMPERATURE vs. `.sf` TEMPERATURE") cannot be completed with this data.** Geometry/shape cross-validated cleanly (finding 2); physical values did not, for a data reason outside this app's control. Worth an M-SIM wishlist entry (does the input deck need an explicit bound/output setting for this channel?) — not chased further within this timebox.

4. **SOOT DENSITY (the actual target quantity) looks physically sensible.** Nonzero-cell fraction over the 12-submesh y=0 plane grows across sampled frames: 0 → 0.00022 → 0.00057 → 0.00052 → 0.00136 (frames 0/120/240/360/480) — small (a candle fire, coarse mesh) but a real, mostly-increasing smoke signal, consistent with the prior spike's per-submesh finding.

5. **Performance is quantity-dependent and one number is anomalous.** Full RLE decode, 12 submeshes × 481 frames: **SOOT DENSITY 0.43 s** (≈36 ms/submesh, close to the prior spike's ~64 ms/submesh estimate); **TEMPERATURE 13.8 s** (≈1.15 s/submesh, ~32× slower) despite decoding to all-zero data. Not chased further within the timebox — flagged as an open question for whoever integrates this, since a 32× swing is large enough to break an interactive-load budget if it turns out to be common rather than specific to this dead channel.

6. **Decode cost cannot be reduced by wanting only the y=0 face.** RLE decompression is per-full-submesh-frame; there is no partial-cube decode. The "only 12 of 24" saving (finding 1) is the only real lever — once a submesh is decoded, its cost is fixed regardless of which face is kept.

7. **Memory matches the prior spike's estimate.** Raw uint8, 12 submeshes, one quantity, all 481 frames: 40.8 MB (measured, SOOT). Doubling for all 24 submeshes (full-domain, one quantity) gives ≈82 MB — matches the Tier-3 spike's independent estimate almost exactly.

## Go / No-Go

**Conditional GO — narrower than the prior spike's, with one scope cut.**

- **GO** for SOOT DENSITY (and structurally-identical HRRPUV/CO₂ DENSITY, untested here — verify before integration, don't assume) via: 12-submesh y=0 extraction, node-boundary-trimmed stitching reusing `combineSliceGeometry()`'s pattern, uint8-levels + per-frame `.sz` bounds memory model (per-scenario, not eager across the ensemble).
- **NO-GO** for TEMPERATURE via `.s3d` in this dataset — genuinely zero data at the source. Drop the `.sf`-vs-`.s3d` TEMPERATURE cross-check from M2.1's scope; SOOT DENSITY has no `.sf` equivalent to cross-check against, so its correctness bar is physical plausibility (finding 4) plus the geometry cross-check (finding 2), not a direct ground-truth diff.
- **Before integration:** re-measure decode cost for SOOT/HRRPUV/CO₂ specifically (don't assume finding 5's SOOT number generalizes) and confirm on a second scenario that finding 3 (dead TEMPERATURE channel) isn't scenario-specific.
