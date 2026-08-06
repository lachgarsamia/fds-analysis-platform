"""The `_sync_cell_*` seam (D12 extraction): what must stay in sync on a
GridCell's view whenever its (scenario, quantity) changes, after the view
already exists -- extent + room outline, the ceiling-obstruction mask, and
the time-series strip.

Pure structural move out of main_window.py (~2400 lines' worth of unrelated
orchestration away from this specific concern); every function body here is
unchanged from the MainWindow methods it replaces. The `_<x>_for` value
computation (`_extent_for`, `_room_outline_for`, `_ceiling_mask_for`,
`_timeseries_strip_data`) stays on MainWindow -- it owns caches and store
routing that are out of scope for this move -- so every function here takes
`main_window` and calls back into it for those values, the same duck-typed
"take the owning object" convention `analysis_panel_base.bind_to_bus`
already established for panel/bus wiring.

`sync_cell()` is the enforced-complete entry point: it always runs all three
legs, no per-leg opt-out. A future 4th sync concern is added once, here, and
every call site using `sync_cell()` picks it up automatically. The one
call site that is genuinely incomplete today (_on_cell_type_changed, which
never re-syncs extent -- see main_window.py's own comment there) calls
`sync_ceiling_mask`/`sync_timeseries_strip` directly instead of `sync_cell`,
so its incompleteness is visible as "doesn't call the seam" rather than
hidden behind a keyword flag.

`_init_cell_view`'s extent/ceiling_mask are deliberately NOT routed through
here: at that point the view has no artists yet, so those values are
construction-time `init_plot()` kwargs, not post-init setter pushes -- a
different, legitimate mechanism, not a third leg of this seam. Forcing that
path through `sync_extent`'s setter call after the fact would mean an extra
set_room_outline() -> set_ylim() call on a freshly-initialized view, exactly
the ordering this session's ceiling-mask/stale-blit bug came from -- so it
stays exactly as it was.
"""

from __future__ import annotations


def sync_extent(main_window, cell) -> None:
    """Update a cell's plotted extent if its current (case, quantity)
    lives on a different physical plane than what the view was last
    drawn with (M2.2 -- SOOT's doorway slice differs from the standard
    side view). A no-op for every same-plane change (the extents
    compare equal), so scenario switches and TEMPERATURE/VELOCITY/
    SOOT-side toggles cost nothing here."""
    new_extent = main_window._extent_for(cell.case_index, cell.quantity_key)
    if new_extent != getattr(cell.view, "_extent", None):
        cell.view.set_extent(new_extent)
    cell.view.set_room_outline(main_window._room_outline_for(cell))


def sync_ceiling_mask(main_window, cell) -> None:
    """Push _ceiling_mask_for's result to `cell`'s view -- call whenever a
    cell's scenario/quantity changes (same call sites as sync_extent/
    sync_timeseries_strip)."""
    cell.view.set_ceiling_mask(main_window._ceiling_mask_for(cell))


def sync_timeseries_strip(main_window, cell) -> None:
    """Show/refresh `cell`'s time-series strip for its current quantity,
    or hide it -- call whenever a cell's scenario/quantity changes (same
    call sites as sync_extent)."""
    result = main_window._timeseries_strip_data(cell)
    if result is None:
        cell.timeseries_strip.setVisible(False)
        return
    series, colors, y_range, title, y_label, caption, band_labels = result
    cell.timeseries_strip.set_series(series, colors, y_range, title=title,
                                     y_label=y_label, caption=caption, band_labels=band_labels)
    cell.timeseries_strip.set_index(main_window.time_controller.index)
    cell.timeseries_strip.setVisible(True)


def sync_cell(main_window, cell) -> None:
    """The complete set, unconditionally -- extent, ceiling mask, and the
    time-series strip. No per-leg opt-out: a call site that needs a
    subset calls the individual sync_* functions above directly instead,
    so its incompleteness is visible at the call site rather than hidden
    behind a flag here."""
    sync_extent(main_window, cell)
    sync_ceiling_mask(main_window, cell)
    sync_timeseries_strip(main_window, cell)
