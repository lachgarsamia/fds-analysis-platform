# FireScope: the physical model and design rationale

Written for the person defending this app's design choices, not for a
developer reading the code. Every claim below is grounded in the actual
source and the actual FDS input decks in `fds/sim/`, cited in parentheses so
you can go verify or go deeper. Where I'm inferring rather than reading a
direct statement, I say so explicitly — treat unmarked claims as directly
confirmed in code/data, and marked ones as my own reading of intent.

---

## 1. Quantities — what's real, what's derived, what's a proxy

**TEMPERATURE** is real: a direct read of an FDS `.sf` slice file, no
computation (`registry.py`'s `TEMPERATURE` entry, `kind="slice2d"`). The
display range is fixed at 20–170 °C (`AMBIENT_C=20` to `AMBIENT_C+150`), not
auto-scaled per scenario. That's a deliberate trade, not an oversight: the
comment on the registry entry says the real 24-scenario dataset was checked
directly and every scenario's peak lands 382–469 °C — well above 170 — so
this range *deliberately* saturates the flame plume itself in exchange for
spreading the room's ambient-to-hazardous gradient (the part an occupant
actually experiences) across the visible ramp instead of crushing it into a
sliver below the peak (`registry.py`, comment above the `TEMPERATURE` entry).

**VELOCITY** is real but partial: it's the FDS `VELOCITY` slice, which is
**speed magnitude |v| only — direction is not stored.** The registry entry
says this plainly: "direction is not stored (the in-plane U/W components are
gated)" (`registry.py`, `VELOCITY` entry). This isn't a UI limitation, it's
a data limitation: the FDS input deck's `&SLCF` lines only ever request
`QUANTITY = 'TEMPERATURE'` and `QUANTITY = 'VELOCITY'` at the read plane
(confirmed directly — I grepped every `.fds` deck in `fds/sim/`, none
requests `U-VELOCITY`/`W-VELOCITY`). Real streamlines would need those two
signed components; adding them is a known, scoped, *not-yet-executed* change
(`docs/msim-preparation.md` §3) gated behind re-validating the slice parser
first (§2) — the project's own stated rule is that new simulation output
isn't trusted until the parser is proven on it, so the gate is deliberate,
not neglect.

**SOOT DENSITY** is real but structurally different from the other two: it's
decoded from the *volumetric* `.s3d` output, not a declared 2D slice (no
`SOOT DENSITY` `&SLCF` line exists in any deck — confirmed by grep). The app
extracts a 2D plane from that 3D volume at a specific physical position
(`SliceKey.plane_pos`, `slice_key.py`), which is why it's plumbed through a
separate code path (`load_data.py`'s dispatch) rather than the normal slice
read. Its display range is fixed at 0–20 000 mg/m³, and the registry comment
states this was "tuned to the real 24-scenario dataset's measured
max-over-run (~19 289 mg/m³ ... verified directly, not assumed)"
(`registry.py`, `SOOT DENSITY` entry) — same reasoning as TEMPERATURE: one
fixed scale so the same color means the same concentration in every
scenario, not a per-run auto-scale that would make color meaningless across
scenarios. The registry's own one-line description calls it "a proxy for
smoke obscuration" (`registry.py`, `interpretation` field) — it's a real
measured field, but *soot mass concentration* is being used as a stand-in
for the visibility/obscuration an occupant would experience, not a direct
readout of visibility itself (that would be the still-gated `VISIBILITY`
quantity).

**TEMPERATURE RISE (ΔT)** is derived, computed on the fly, never stored:
`T − 20 °C` (`derived_quantities.py:23–25`, `temperature_rise()`). Physical
meaning: it isolates the fire's own contribution from the room's starting
temperature — useful for comparing runs that might not share the exact same
ambient baseline.

**DYNAMIC PRESSURE** is derived: `½ρ|v|²` with a fixed air density
`ρ = 1.2 kg/m³` (`derived_quantities.py:20,28–31`). Two things worth being
able to defend here:
- Because it's built from `VELOCITY`'s *unsigned* magnitude, it can never go
  negative. A true dynamic-pressure field driven by signed velocity
  components would show a "neutral plane" — the height where the sign flips
  and flow reverses direction through a vent. This one can't; it can only
  show where flow forcing concentrates. `docs/msim-preparation.md` §4 states
  this directly and recommends the panel be read as a "flow-forcing
  profile," not a neutral-plane finder, until real signed velocity exists.
- Its display default (1 Pa ceiling, not the dataset's ~9.4 Pa max) exists
  because the quantity is genuinely bimodal across the 24 scenarios: natural-
  ventilation runs peak at 0.57–0.69 Pa, HVAC-forced runs peak at 6–9.4 Pa,
  "with nothing in between" (`registry.py`, comment on the `DYNAMIC PRESSURE`
  entry, which states this was verified directly against all 24 scenarios).
  A ceiling picked from the dataset max would crush the 16/24 natural-
  ventilation scenarios — the common case — into under 7% of the color ramp.
  The trade is the same one made for TEMPERATURE: sacrifice legibility on the
  rare extreme (HVAC scenarios, which saturate to the top color) to keep the
  common case readable; the slider is the documented escape hatch for
  inspecting an HVAC scenario in detail.

**Gated (registered, not yet available):** `CARBON MONOXIDE VOLUME
FRACTION`, `U-VELOCITY`, `W-VELOCITY`, `V-VELOCITY`, `PRESSURE`,
`VISIBILITY`, `HEAT FLUX`, `SOOT MASS FRACTION` — every one carries
`gated=True` and a `gate_reason` pointing at `docs/msim-preparation.md`
(`registry.py`, `MSIM_GATE` constant + each entry). Mechanically,
`QuantityProvider.get()` checks `q.gated` and raises `GatedQuantityError`
*immediately*, before ever touching the data store (`quantity_provider.py`,
`get()`, the `if q.gated: raise GatedQuantityError(q.gate_reason)` check) —
every caller in the app (tenability, devices, the Ask grammar) is written to
catch that specific exception and fall back to an honest partial model,
never to silently return empty or fabricated data. Worth being precise about
*why* CO specifically is harder than the velocity components: adding
U/W-VELOCITY is a pure output addition (add two `&SLCF` lines); adding CO
requires the `&REAC` line to gain a `CO_YIELD` parameter, which
`docs/msim-preparation.md` §3 calls "a combustion-modeling choice a fire-
science domain expert must set" — not a mechanical edit anyone can make.

---

## 2. Slices — what planes exist, and why

The room is a small physical scale model, not a full-size compartment. Its
geometry comes straight from the FDS deck's `&MESH`/`&MULT` lines (I read
`fds/sim/c1_d0_vod0_voc0/c1_d0_vod0_voc0.fds` directly, and this geometry is
identical across every scenario's own deck):

```
&MULT ID='m1', DX=0.25,DY=0.15,DZ=0.16, I_UPPER=3,J_UPPER=1,K_UPPER=2/
&MESH IJK=25,15,16, XB=0.0,0.25,-0.15,0.0,0.0,0.16, MULT_ID='m1' /
```

One 25×15×16-cell mesh block, tiled 4× along x, 2× along y, 3× along z. That
gives a total simulated domain of **x ∈ [0, 1.0] m, y ∈ [−0.15, 0.15] m,
z ∈ [0, 0.48] m** — a box roughly a meter long, 30 cm deep, half a meter
tall. This matches `get_extent()`'s actual return value for the app's read
plane, `[0.0, 1.0, 0.0, 0.48]`, which I confirmed directly against the
running provider.

**The app reads exactly one plane: y-normal, at y = 0** (the domain's own
mid-depth line), giving the x–z side-view cross-section every screen in this
app shows. This is the *only* plane any deck requests data on — every deck's
`&SLCF` block is just:

```
&SLCF PBY =  0.000, QUANTITY = 'TEMPERATURE' /
&SLCF PBY = -0.005, QUANTITY = 'TEMPERATURE', CELL_CENTERED = .TRUE. /
&SLCF PBY =  0.000, QUANTITY = 'VELOCITY' /
&SLCF PBY = -0.005, QUANTITY = 'VELOCITY', CELL_CENTERED = .TRUE. /
```

There's a second, cell-centered copy at `PBY=−0.005` for each quantity, but
the app's default `SliceKey` (`direction=1` → y-axis, per
`slice_key.py`'s `DIRECTION_TO_AXIS`) reads the face-centered one at
`PBY=0.000` — confirmed by the array shape: `TEMPERATURE`'s frames come back
as `(49, 101)` per timestep, matching **grid nodes** (100 x-cells + 1,
48 z-cells + 1), not cell centers. **There is no x-normal or z-normal slice
anywhere in the data** — the app can't offer a front or top view today not
because of a UI choice, but because the simulation was never asked to output
one.

Frame rate: the deck's `&DUMP DT_SLCF=0.25` line writes a slice every
0.25 s, i.e. **4 frames per second** — and that's exactly what the running
app reports as `timesteps_per_second` (confirmed directly). `&TIME
T_END=120` matches the 0–120 s timeline every panel shows.

SOOT DENSITY doesn't ride this same mechanism — no deck has a `SOOT DENSITY`
`&SLCF` line, so it's read from the volumetric `.s3d` dump instead, at a
specific physical plane position rather than a declared slice
(`slice_key.py`'s `plane_pos` field exists specifically for this). A code
comment in `main_window.py` (near the SOOT overlay logic) states this is
confirmed to work at "the y=0 plane (offset 0), the one plane SOOT DENSITY
is confirmed for" — *I haven't independently re-verified that specific
claim beyond reading the comment; treat it as the codebase's own documented
confidence, not something I re-derived myself.*

---

## 3. Detection/device models — the ones worth getting exactly right

Three device types exist: thermocouple, heat detector, sprinkler
(`devices.py`, `KINDS = ("thermocouple", "heat_detector", "sprinkler")`).
**There is no smoke detector.** I grepped the entire source tree for
`smoke_detector`, "smoke detector," and any optical-density threshold logic
— nothing exists. SOOT DENSITY is shown as a field (see §1) but nothing in
the app thresholds it into a detection or alarm event. If asked "what about
smoke alarms," the honest answer is: not modeled, at all, today.

**Thermocouple** is the simplest: a point sample of `TEMPERATURE`,
bilinearly interpolated at the placed (x, z) every frame
(`measure.py:probe_value`, using `scipy.ndimage.map_coordinates` with
`order=1`). It reports the full history, its max, peak heating rate
(`np.gradient(temp) * fps`), and threshold-crossing times at the standard
60/100/300 °C bands, plus a heat-only FED dose at that single point
(`devices.py:compute_thermocouple`). It has no activation state — it's an
instrument, not a detector.

**Heat detector** activates the instant local `TEMPERATURE` crosses a fixed
threshold (default 74 °C) — a plain comparator, no lag, no thermal mass
modeled (`devices.py:compute_heat_detector`, `_crossing_time`). Optionally
it can also trigger on rate-of-rise, whichever condition is met first. This
is a deliberately simple, instantaneous model.

**Sprinkler is the one that needs care, because it is *not* instantaneous.**
It implements the standard RTI (Response Time Index) thermal-lag ordinary
differential equation, Euler-integrated at the simulation's own frame rate:

```
dT_link/dt = (√|u| / RTI) · (T_gas − T_link)
```

(`devices.py:compute_sprinkler`, and the exact same formula string is
surfaced live in the UI as the device's "basis" text.) Defaults: RTI = 100
(m·s)^0.5, activation threshold 68 °C (`devices.py`, `_DEFAULT_PARAMETERS`).
Plain-language version: a real sprinkler doesn't sense the room's gas
temperature directly — it senses the temperature of its own small thermal
mass (the fusible link or glass bulb), which only heats up as fast as heat
transfers *into* it from the surrounding air. That transfer rate depends on
both the RTI (a property of the physical sprinkler head — lower RTI means a
faster-responding head) and the local air velocity (faster-moving air
transfers heat into the link faster, hence the `√u` term). This is standard
fire-protection engineering, not something FireScope invented — I'm stating
this as domain framing on top of the code, and it matches the formula
exactly as implemented.

The practical consequence — and this is the "hard question" version — is
that **a sprinkler can genuinely fail to trip on a fire a heat detector
catches instantly, without either device being wrong.** I verified this
directly against a real scenario in this dataset (case 17, `2 candles ·
Narrow door · Vent 1 HVAC · Vent 2 closed`, near the candle corner): gas
temperature spiked to ~248 °C, tripping the heat detector at 0.5 s. The
sprinkler's link temperature, chasing that same gas temperature with a lag
set by RTI=100 and a local velocity around 0.7–1.5 m/s, only ever reached
about 60 °C before the short-lived candle flame died down and the room
cooled again — never crossing its 68 °C threshold. The fire was too brief
for the slow-responding thermal mass to catch up. This is exactly why real
"quick response" sprinklers (RTI ≈ 20–50) exist as a distinct product class
from "standard response" ones (RTI ≈ 80–350, per general fire-protection
convention) — *that RTI-class terminology is my own domain framing, not a
label present in the code,* but the mechanism (lower RTI ⇒ faster catch-up)
is exactly what the formula above implements. When a device's status reads
"did not activate," this session added the peak link temperature and the
threshold to that message specifically so this isn't opaque
(`device_panel.py:_headline`).

One more honesty point on the sprinkler: it falls back to a "reduced model"
(`u` fixed at 1.0 m/s, explicitly labeled as such in its basis text) only
when `VELOCITY` isn't registered for that plane at all — it never invents a
measured velocity (`devices.py:compute_sprinkler`, `_has_velocity` /
`reduced_model`).

---

## 4. Hazard/tenability classification

The default classification (`hazard_spaces.py`) sorts every cell, every
frame, into one of four classes from a temperature threshold plus a
cumulative-exposure rule:

```
Safe        T < 60 °C
Warning     60 ≤ T < 100 °C
Critical    100 ≤ T < 300 °C
Untenable   T ≥ 300 °C,  OR  cumulative time above 60 °C ≥ 30 s (a heat-dose proxy)
```

(`hazard_spaces.py:classify_series`, exact logic; the 30-second exposure
limit is `DEFAULT_EXPOSURE_LIMIT_S`.)

**Where the 60/100/300 numbers themselves come from is worth being precise
about, because it's not "derived from this experiment."** A project spike
document states plainly that these bands were "proposed as general
fire-safety reference points, **not derived from this study's own data**,
and should be reviewed by a domain expert/supervisor before being treated as
authoritative for the demo" (`docs/spike-parser-validation.md` §5). Each
band has a stated physical rationale in that same document: Safe is below
the common skin-contact pain/injury threshold for brief exposure; the
60–300 °C band is burn risk approaching materials' ignition/degradation
range; above 300 °C is described as "near flashover-adjacent conditions,
short survivable exposure time." These are standard fire-safety literature
values applied to this specific small-scale candle experiment — a
reasonable choice, but one you should be ready to say out loud is a
literature convention, not something calibrated against this dataset.

**"Temperature-only, partial screen"** — the caption every hazard-rendering
panel shows (`hazard_spaces.basis_caption()`) — means exactly what it says:
because CO is gated (see §1), the Untenable escalation rule above uses a
convected-heat-only exposure proxy, not a real toxic-gas dose. The moment CO
becomes available, the same code path switches to full FED automatically
(`classify_series`'s `co_field` branch) and the caption changes to name that
basis instead (`FULL_FED_BASIS`).

**Full FED**, when CO *is* available, is the standard ISO 13571 / SFPE
Handbook (Purser) dose model, in its commonly-cited simplified form (no
respiratory-minute-volume or activity-level adjustment — `tenability.py`'s
own module docstring states this explicitly):

```
FED_CO (toxic gas)    : d(FED)/dt = [CO_ppm]^1.036 / 35000   per minute
FED_heat (convected)  : d(FED)/dt = 1 / t_I(T),  t_I(T) = exp(5.1849 − 0.0273·T) minutes
FED_full              : FED_CO + FED_heat
```

(`tenability.py:fed_gas_dose`, `fed_heat_dose`, `full_fed`.) FED ≥ 1.0
conventionally marks incapacitation (`FED_INCAPACITATION = 1.0`). The
module docstring is explicit that this is never presented as "a certified
life-safety calculation" — it's the standard textbook equations at the same
level of simplification the app already uses elsewhere.

One separate, related indicator: a **flashover indicator** flags any frame
whose *peak* temperature exceeds 500 °C. The code comment is explicit this
is "an indicator only, not a flashover/combustion model"
(`hazard_spaces.py`, module docstring and `FLASHOVER_INDICATOR_C`) — it's a
threshold on peak temperature, not a physics-based flashover onset
criterion (real flashover involves radiative feedback loops this app
doesn't model at all).

A related but separate honest simplification: **smoke-layer height**
(shown elsewhere, e.g. in scenario comparisons) uses "a documented
simplification of the rigorous two-zone (Cooper) method" — it integrates
the *domain-mean* vertical excess-temperature profile from the ceiling down
to the half-integral point, rather than computing a per-column height and
averaging afterward, "traded for O(n_times) cost on data already cached"
(`layer_height.py`, module docstring, near-verbatim quote).

---

## 5. The scenario space — what the 24 runs actually vary

24 scenarios = a full factorial of four two-or-three-level factors:
candles (1 or 2), door (narrow or wide), Vent 1/"vod" (open / closed /
HVAC), Vent 2/"voc" (open / closed) — 2×2×3×2 = 24
(`manifest.py`, `_FACTORS = ('candles', 'door', 'vod', 'voc')`,
`FACTOR_LABELS`). I cross-checked every factor's physical meaning directly
against the `.fds` decks rather than trusting the folder-name convention
alone:

**Candles** are literal FDS burner vents: `&VENT ... SURF_ID='BURNER'`, each
a roughly 4 cm × 4 cm floor patch near the room's far wall, with a fixed
`HRRPUA = 48 kW/m²` and `FUEL='METHANE'` (`&REAC` line). One candle's
nominal heat-release rate works out to `48 kW/m² × 0.0016 m² ≈ 0.077 kW`
(~77 W) — genuinely candle-scale, not a room fire; *that specific
multiplication is my own arithmetic on the deck's stated numbers, not a
value the code computes and stores anywhere.* The 2-candle decks add a
second, identical burner immediately adjacent (confirmed by diffing
`c1_d0_vod0_voc0.fds` against `c2_d0_vod0_voc0.fds`). Real candle flames
aren't modeled as solid-wax pyrolysis; they're a fixed-HRR gas (methane)
burner — a standard, deliberate simplification for a small controlled flame,
not solid-fuel burning.

**Door** ("Narrow door" = 0.05, "Wide door" = 0.15, per the deck's own
`&HEAD TITLE` text) is a `&HOLE` cut through the compartment's dividing
wall. Comparing the narrow- and wide-door decks directly: the only
difference is the hole's **z-extent (height)** — narrow is `z = −0.01 to
0.05` (~6 cm tall), wide is `z = −0.01 to 0.15` (~16 cm tall); the
horizontal (x/y) footprint of the opening is identical in both. *Worth
flagging plainly: the deck's own label calls this "door width," but what
actually changes in the physics is the doorway's vertical opening height,
not its side-to-side width — that's my own reading of the geometry, not a
literal statement anywhere in the code, and it's exactly the kind of detail
that trips people up defending a model.*

**Vent 1 ("vod")** and **Vent 2 ("voc")** are two independent ceiling
openings. The deck's own inline comments name them: `vod` = "vertical
opening door" (a ceiling hole near the doorway side, x ≈ 0.32–0.40); `voc` =
"vertical opening candle" (a ceiling hole directly above the candle burners,
x ≈ 0.86–0.94). Both, when "open," are plain rectangular holes in the
ceiling slab — a passive opening with no forcing, buoyancy-driven flow only.
When "closed," the hole is simply **absent**: the ceiling stays solid there
(confirmed by diffing decks with different `voc`/`vod` values — the closed
variants have no `&HOLE` line at that location at all).

**Vent 1 = HVAC is a genuinely different mechanism, not just "more open."**
Only `vod` (Vent 1) has this third state; `voc` never does
(`manifest.py`'s `_VOC_STATES` only has open/closed). In the HVAC decks, the
plain hole is replaced by two boundary vents at the ceiling's two faces
(z=0.22 "IN," z=0.24 "OUT," both `SURF_ID='HVAC'`), wired through an actual
`&HVAC` duct (`LENGTH=0.2 m, AREA=0.04 m²`) to a fan with a **fixed
volumetric flow rate**, `VOLUME_FLOW = 0.016 m³/s`. Physically: a small
duct-and-fan assembly that draws air in at the ceiling's underside and
forces it out the topside at a constant rate, independent of the room's own
buoyancy — real mechanical/forced ventilation, not just a bigger passive
opening. This is exactly why DYNAMIC PRESSURE reads roughly ten times higher
in HVAC scenarios than in any open/closed-vent scenario (see §1's registry
comment, verified against the real dataset).

**Room geometry vs. simulated domain:** the visually-drawn "room" interior
(`schematic.py`'s `ROOM_X = (0.27, 1.0)`, `ROOM_Z = (0.0, 0.22)`) is smaller
than the full simulated box (x: 0–1.0, z: 0–0.48) for two physical reasons,
both directly visible in the deck: the dividing wall sits at x = 0.26–0.28
(so x < 0.27 is a corridor/antechamber outside the modeled "room," not room
space), and the ceiling slab itself is only 2 cm thick at z = 0.22–0.24 —
everything above that is empty buffer air the FDS mesh needs numerically but
the app deliberately doesn't draw as occupiable room space.

---

## 6. The honest limitations — stated before the audience asks

A consolidated list, each one directly stated in the code or docs, that you
should be able to say out loud before being asked:

- **VELOCITY is a speed magnitude only.** No direction is stored, so there
  are no real streamlines or vector-field visualizations today — that's
  gated on adding `U-VELOCITY`/`W-VELOCITY` output, which is itself gated on
  re-validating the slice parser first (§1, §2).
- **DYNAMIC PRESSURE can't show a neutral plane.** Built from an
  always-non-negative speed magnitude, it can only show where flow forcing
  concentrates, not where flow reverses direction — the project's own docs
  relabel it a "flow-forcing profile" for exactly this reason (§1).
- **Hazard/tenability is a temperature-only partial screen** until CO
  exists, and CO's absence isn't a simple oversight — it needs a
  domain-expert combustion-modeling decision (`CO_YIELD`), not a mechanical
  fix (§1, §4).
- **The 60/100/300 °C hazard bands are literature reference values,
  explicitly not calibrated against this dataset**, and the project's own
  docs say they need domain-expert review before being called authoritative
  (§4).
- **Smoke-layer height is a simplified domain-mean method**, documented as
  a simplification of the rigorous two-zone (Cooper) approach (§4).
- **The flashover indicator is a peak-temperature threshold, not a
  flashover model** — no radiative feedback physics is simulated for it
  (§4).
- **The Physics Attention Map is explicitly "a heuristic saliency map, not
  a physical field"** — its own module docstring states it "does not
  measure any single simulated quantity," it's a weighted combination of
  rates of change (|dT/dt|, |dV/dt|, |∇T|, |dHRR/dt|)
  (`attention.py`, near-verbatim quote).
- **Advanced Comparison's "Physics — associated drivers" are explicitly
  association, not causation** — the exact phrase "association with the
  difference, not a proven cause" is generated into the UI text itself, not
  just documentation (`advanced_compare.py`, three separate citations).
- **No smoke detector exists.** SOOT DENSITY is shown as a field, but
  nothing in the app thresholds it into a detection/alarm event (§3).
- **The Narrative panel deliberately carries no hazard badge**, and that's
  a considered choice, not an omission: it reports raw threshold-crossing
  facts (via `events.py`'s `detect_events`, reading the same 60/100/300 °C
  levels directly) but never calls `hazard_spaces.classify_series` or
  `worst_class` — it never issues a *classification*, so badging it with the
  partial-FED disclaimer would have implied a claim the panel doesn't
  actually make.
- **Nothing is ever fabricated for a gated quantity.** `QuantityProvider`
  raises `GatedQuantityError` before touching the data store the instant a
  gated quantity is requested, and every consumer in the app is written to
  catch that and fall back to its documented partial model — never to
  silently substitute estimated or empty data (§1).

---

*Grounding note: every code citation above was read directly from this
repository during this session (`registry.py`, `derived_quantities.py`,
`quantity_provider.py`, `slice_key.py`, `devices.py`, `measure.py`,
`hazard_spaces.py`, `tenability.py`, `layer_height.py`, `attention.py`,
`advanced_compare.py`, `events.py`, `manifest.py`, `schematic.py`, and the
`.fds` decks under `fds/sim/`), not recalled from general fire-simulation
convention. The handful of explicitly-marked inferences (RTI "quick/standard
response" class terminology, the door-height reading, the candle wattage
arithmetic) are flagged in place above and nowhere else in this document —
everything else is a direct citation.*
