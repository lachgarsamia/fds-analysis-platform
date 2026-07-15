# FireLab Digital Twin — Demo Script

A narrative to rehearse against, alongside the physical chamber (candles, door, ventilation) the app
sits next to. Written for a presenter who wants to run the whole demo without touching a mouse.

## Setup

- Theatre theme active (View → Theme → Theatre) — the demo default.
- App boots on **Home**. Idle for 3+ minutes drifts back here automatically (kiosk/attract mode);
  any key or mouse movement returns to **Live Viewer**.
- Keyboard-only controls used throughout:
  - `1`–`6` — jump to a page (Home / Live Viewer / Compare / Dataset / Analysis / Export), same order
    as the nav rail.
  - `Space` — play/pause. `←`/`→` — step one frame. `Shift+←`/`Shift+→` — step one second.
  - `Ctrl+Shift+<1-9>` — record a bookmark (current page + scenario + time) into that slot.
  - `Shift+<1-9>` — jump to a recorded bookmark. Record your beats during rehearsal, then the live
    run is just `Shift+1`, `Shift+2`, ... in order.
  - `F11` — toggle full screen.
  - Hold `Esc` for ~0.6s — emergency "effects off" switch (falls back to the plain scientific heatmap
    if the cinematic pipeline ever looks wrong live; hold again to restore).

## Beats

1. **Home (`1`).** "This is FireLab — a digital twin of the physical chamber next to us." Point out
   the three stat tiles (24 experiments, 481 time steps, 2 physics fields measured per run).

2. **Start the fire.** Click (or `2`) into **Live Viewer**. First-run visitors see a 4-step guided
   tour automatically; presenters who've dismissed it once won't see it again (remembered via
   QSettings — if rehearsing on a fresh machine, expect it once).

3. **Light the candle(s).** Use the candle cards to match whatever's actually lit in the physical
   chamber. Press `Space` to start playback. Narrate what the Inspector (right panel) says as the
   fire develops — the narration line is generated from the same numbers driving the heatmap, not
   free text, so it's always consistent with what's on screen.

4. **Open the door / vents.** Toggle the door and vent controls live and point at the smoke layer
   responding (Tier 1/2 haze + shimmer). This is the single best "wow" moment — pause here.

5. **Compare (`3`).** Pick "Door open vs. closed." This jumps into a 1×2 grid showing air speed
   side by side with the difference view — the honest, verified finding (M2.3) that the door's
   effect shows up in airflow, not raw temperature.

6. **Dataset (`4`).** Sort by peak temperature, filter by candle count. "Every one of the 24
   experiments is browsable here."

7. **Analysis (`5`).** The PCA/clustering scatter groups scenarios by candle count without being told
   to. If a trained model exists, open a prediction view from the Dataset page's "View model
   prediction" button: "the AI guesses the next few seconds — here's where it's wrong." Honesty is
   the point, not a perfect model.

8. **Export (`6`).** One-click demo postcard (PNG, current frame + title card) as a takeaway.

## Failure drills (rehearse before the real thing)

- **Cinematic pipeline looks wrong live:** hold `Esc` ~0.6s to fall back to the plain scientific
  heatmap immediately; hold again to restore once fixed. Verified: this is the same View →
  "Cinematic fire view" toggle every cell already listens to, not new per-cell state.
- **Demo-data fallback:** if `fds/sim/` isn't found (e.g. a scenario folder got unplugged/moved), the
  app falls back to generated demo data rather than crashing — re-verify this path still renders
  correctly through the cinematic pipeline before the demo, not just the plain heatmap, since Phase 2
  added a lot of pipeline surface since this fallback was last exercised end to end.
- **Mid-playback scenario-folder unplug:** rehearse actually disconnecting/renaming the dataset
  folder while a scenario is mid-playback, to see what the presenter sees and how to recover
  (restart the app onto demo data) without a live crash.

## Performance rehearsal (do this on the actual demo laptop, not a dev machine)

- 30+ minute unattended soak with the cinematic pipeline on: watch process RSS for leaks, watch for
  stutters. (The M2.4 lesson: offscreen/dev-machine FPS numbers overstate real-display performance —
  re-measure on the real hardware, don't trust a laptop-vs-demo-rig assumption.)
- Confirm no thermal throttling kicks in over that soak on the specific machine being used.
- Full visual QA pass in both the dark and theatre themes side by side (screenshot-diff discipline
  from the GUI-modernization pass) — check every page, not just Live.

## Success criteria

- A presenter completes the scripted demo twice, keyboard-only, without touching a mouse.
- 30-minute unattended soak: zero stutters, zero visible memory growth.
