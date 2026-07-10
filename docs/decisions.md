# Architecture decisions log

Short, dated entries recording gate/decision outcomes referenced from ROADMAP.md, so the reasoning behind a "don't migrate" or "keep as-is" call is preserved alongside the numbers that produced it, not just the conclusion.

---

## M2.4 — pyqtgraph migration gate (2026-07-10)

**Gate (per ROADMAP.md §4 M2.4):** timeboxed, 2 days max. Adopt pyqtgraph only if matplotlib-blit can't hold ≥15 fps on a 2×2 synced-playback grid. Otherwise the `PlotView` interface stays (M2.2) and migration defers to Phase 4, where volume rendering (`.s3d`) will want the GL stack anyway.

**Decision: do not migrate.** matplotlib-blit clears the bar with a wide margin on both measurements taken.

### Numbers

| Rendering path | fps | ms/tick | Notes |
|---|---|---|---|
| Offscreen (`QT_QPA_PLATFORM=offscreen`) | ~247 fps | ~4.0 ms | `tests/bench_grid_fps.py`, no real compositing/paint pipeline |
| **Real display** (native `cocoa` Qt backend, actual on-screen window, `app.processEvents()` pumped per tick) | **~57.7 fps** | ~17.3 ms | Same 2×2 grid, 4 distinct real scenarios (`c1_d0_vod0_voc0`, `c1_d1_vod0_voc0`, `c2_d0_vod0_voc0`, `c2_d1_vod0_voc0`), same `_on_time_changed` code path, ad hoc script (not yet a committed benchmark) |

Both measurements: 2×2 grid, 4 distinct scenarios, TEMPERATURE quantity, 120 timeline ticks, this dev machine (Apple Silicon MacBook Pro, macOS 24.5.0).

**Both readings clear the ≥15 fps DoD threshold by 3.8× (real display) to 16× (offscreen)** — not a marginal pass either way.

### Caveat on the offscreen number (carried over from M1.3.3's own benchmark note)

M1.3.3's blitting benchmark already established that this app's offscreen numbers don't reliably predict on-screen numbers — measured ~1.3× speedup offscreen against a ≥5× prediction, because offscreen skips real compositing/paint overhead that a live display incurs. **The same caveat applies here, and this entry now has a concrete illustration of it: the real-display number (57.7 fps) is actually the *lower* of the two, not higher** — offscreen rendering skipped enough real compositing cost that it overstated performance by ~4×. Anyone citing "247 fps" alone (as the initial M2.2 report did) is citing the less representative number; 57.7 fps is the one that reflects an actual user's screen.

### On the real-display measurement's own limits

This session's environment turned out to have a genuine, actively-composited display attached (`QApplication.platformName()` returns `cocoa`, not `offscreen`; `Quartz.CGGetActiveDisplayList` reports one active display) — not something to assume in general for an agent-driven dev session, so this was verified before relying on it rather than assumed. The 57.7 fps figure is real, but still a single spot-check: one machine, one run, a benchmark harness that pumps `app.processEvents()` explicitly rather than a live `QTimer` loop (adds its own small overhead, if anything making 57.7 fps a slight underestimate rather than overestimate). **Flagged for a follow-up spot-check** on a different real machine/monitor configuration if one becomes available before the demo (2026-09-11) — not because there's a specific reason to doubt 57.7 fps, but because a single sample this far from the threshold is enough to close the *gate*, not enough to call the number itself fully characterized.

### Conclusion

Matplotlib-blit stays as the sole `PlotView` backend through Phase 2 and Phase 3. `PyQtGraphSliceView` is not implemented. Revisit only if Phase 4's volume-rendering work (`.s3d`, GL stack) makes a backend switch worth it on its own merits — not because of a 2×2-grid FPS concern, which this gate closes.
