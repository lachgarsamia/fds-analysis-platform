# FireLab Phase 2.1f Tier 3 — `.s3d` real soot-data spike

**Branch:** `feat/firelab-s3d-spike` · **Timebox:** 2 days (spent: well under one day of investigation) · **Status:** complete
**Scope (per `ROADMAP-FIRELAB.md` Phase 2, section f, Tier 3):** determine whether the on-disk `.s3d` Smokeview 3D smoke/soot files can feasibly become the smoke compositor's real data source, in place of (or alongside) the Tier 1/2 heuristic haze already shipped. Not shippable app code — this document plus the throwaway investigation scripts below are the deliverable; no `src/` changes.

---

## 0. What exists on disk

Every scenario folder (e.g. `fds/sim/c2_d0_vod2_voc0/`) ships, per mesh block, four `.s3d` binary files plus a matching `.s3d.sz` text sidecar each, declared in the `.smv` file as:

```
SMOKF3D     <mesh_index>
 <case>_XXXX_01.s3d      SOOT DENSITY        rho_C0.9H0.1   kg/m3
SMOKF3D     <mesh_index>
 <case>_XXXX_02.s3d      HRRPUV              hrrpuv         kW/m3
SMOKG3D     <mesh_index>
 <case>_XXXX_03.s3d      TEMPERATURE         temp           C
SMOKG3D     <mesh_index>
 <case>_XXXX_04.s3d      CARBON DIOXIDE DENSITY  rho_CO2    kg/m3
```

The domain in this scenario is decomposed into **24 mesh blocks** (parallel-FDS mesh splitting), so a full-domain reconstruction of one quantity needs 24 `.s3d` files, not one. This was not previously visible to the app — `slice.py` only ever reads 2D `.sf` slice files, which apparently come from a single logical plane regardless of the 3D mesh split.

## 1. Methodology

1. **Manual binary inspection** (no library): read one `.s3d` file as a sequence of Fortran unformatted records (4-byte little-endian length markers before/after each record — verified zero markers mismatched across the whole file). Found: header record = `ONE=1` + 6 ints (mesh index bounds, e.g. `(0,25, 0,15, 0,16)` — matching the same mesh block bounds already visible in the `.smv`'s `SLCF` declarations); then, per frame: a 4-byte `TIME` float, an 8-byte `(NPTS, NCHARS)` int pair (`NPTS` = total grid points in this block, confirmed exactly equal to `(25+1)×(15+1)×(16+1) = 7072`; `NCHARS` = compressed byte count of the next record), then the `NCHARS`-byte compressed payload itself.
2. **Cross-validation against a real reader, not guesswork**: installed `fdsreader` (latest PyPI release, `1.11.7` — same `10.1`-doesn't-exist correction M1.3s already found; not added to `pyproject.toml`, ad hoc for this spike only) and loaded a real scenario's `Simulation`. It correctly enumerates all four quantities per mesh block and exposes a `Smoke3D`/`SubSmoke3D` API with a documented RLE decoder (`fdsreader/smoke3d/smoke3d.py`) that matches the byte pattern found by hand: a `mark=255` escape byte followed by `(value, repeat_count)`, else a literal byte — i.e. classic run-length encoding, not a proprietary/undocumented scheme.
3. **Read `fdsreader`'s actual source** (small, pure-Python + numpy package) rather than trusting its convenience methods blindly — this surfaced the one real gotcha below.

## 2. Findings

### 2.1 The format is genuinely decodable, with high confidence — not a guess
The RLE scheme is simple and was independently confirmed two ways (byte-level manual inspection *and* a real, versioned, community-maintained library's own decoder implementation agreeing with it). This removes the spike's biggest a priori risk (silently decoding garbage that happens to "look like" data).

### 2.2 Gotcha: `fdsreader`'s convenience API returns raw quantization levels, not physical units
`SubSmoke3D.data` / `Smoke3D.to_global()` decode the RLE bytes into raw **0–255 integer levels** and stop there — they do **not** rescale against the per-frame upper bound. That bound lives in a separate `<file>.s3d.sz` sidecar (`time, npts, nchars, upper_bound` per line), parsed by `fdsreader` into `SubSmoke3D.upper_bounds` but never applied to `.data`. Calling `to_global()` naively (as a first-pass sanity check did) yields values like `{0.0, 1.0, 2.0}` for SOOT DENSITY — technically correct raw levels, but meaningless as physical density until rescaled: `physical ≈ (level / 255) × upper_bounds[frame]`, applied **per sub-mesh, before stitching** (each mesh block has its own upper bound per frame).

After applying that rescale to one sub-mesh's SOOT DENSITY series: the upper bound itself climbs plausibly over the burn (`0.5 → 0.64 → 0.79 → 0.93 → 1.08` at t≈0/30/60/90/120s), and the fraction of nonzero cells in that block grows from `0.0` to `0.84` by the end of the run — a coherent, physically sensible "smoke is filling this region" signal, not noise.

*Open, non-blocking question:* the same rescale applied to the TEMPERATURE (`SMOKG3D`) channel for this scenario returned an upper bound of exactly `0.0` for every frame, in every one of the 24 sub-meshes — i.e., this specific scenario's `.sz` sidecar records zero temperature-channel bound throughout. Not chased further within the timebox (TEMPERATURE isn't Tier 3's actual target quantity — SOOT DENSITY is, and that one behaved sensibly); flagged here rather than silently ignored.

### 2.3 Real, quantified costs — this is heavier than the 2D data the app uses today
- **Decode speed, single sub-mesh:** ~64 ms for one sub-mesh's *entire* 481-frame series (comparable order of magnitude to M1.2's existing cold-load slice costs). Cheap.
- **`to_global()`'s general-purpose grid-stitching across all 24 sub-meshes: ~1.7–5 s** for one quantity, one scenario — far too slow for an interactive scenario switch. This cost is `to_global()`'s own coordinate-matching machinery, not the RLE decode itself; a purpose-built stitcher (we already know the mesh grid is regular, unlike `fdsreader`'s general case) would very likely be much faster, but that's unverified follow-up work, not something this spike built.
- **Size:** one quantity, one scenario, all 24 sub-meshes, all 481 frames, as float32 ≈ **327 MB**. The *entire* existing 2D dataset (all 24 scenarios, both quantities) is ≈230 MB per `ROADMAP.md`'s own ground truth. Eagerly caching even one `.s3d` quantity for every scenario the way M1.2 caches slices (≈24 × 327 MB ≈ 7.8 GB) is not viable; only the active scenario's data should ever be decoded/cached, matching the existing LRU pattern's spirit rather than its literal "keep several in memory" sizing.

## 3. Go / no-go decision

**Conditional GO — but not a drop-in for the pipeline as it exists today.**

The format is decodable with real confidence (not the spike's original biggest risk), and SOOT DENSITY specifically produces a physically coherent signal once correctly rescaled. That is enough to justify building this properly. It is **not**, however, a quick addition to `cinema/smoke.py`'s existing tier structure without a scoped follow-up milestone, because:

1. A correct reader needs to do what `fdsreader` doesn't: rescale per sub-mesh against its own `.s3d.sz` upper bound *before* stitching sub-meshes together — get the order wrong and it silently looks plausible while being physically meaningless (exactly the class of bug this spike exists to catch before it ships).
2. The multi-second full-domain stitch cost is incompatible with live/interactive playback without a dedicated cache (an `.npy`-backed, per-scenario, lazy-loaded cache in the direct spirit of M1.2's disk cache — not built here, out of this spike's timebox).
3. It's a new runtime dependency (`fdsreader`, pure-Python + numpy, no heavy native deps) if adopted as the reader rather than hand-rolling the ~20-line RLE decoder ourselves — a small, cheap decision either way, but a real one to make deliberately rather than by accident.

### Recommended follow-up (separate, scoped milestone — not this spike)
- A small `src/fds/s3d/` reader: either wrap `fdsreader`'s `SubSmoke3D` + apply the missing rescale, or port the ~20-line RLE decoder directly (avoids the new dependency; the algorithm is now well-understood and documented above).
- A per-scenario, lazy, disk-cached (`.npy`, mtime-invalidated — same pattern as `ScenarioStore`) loader for exactly the active scenario's SOOT DENSITY, not an eager all-scenario cache.
- Bench the real stitching cost of a purpose-built (regular-grid-aware) combiner before assuming it's fast enough; `to_global()`'s 1.7–5 s is not.
- Estimated effort: **1.5–2.5 days** — comparable to M1.3s + M1.3's own combined scope, not a multi-week rebuild.

Until that follow-up lands, Tier 1/2 (temperature-derived haze + velocity-advected dye, already shipped) remain the smoke compositor's only data source — a reasonable, already-working fallback, not a blocked/broken state.
