"""Session files (V2 roadmap M2.4, feature F10): save/restore the full
grid workspace -- layout, per-cell scenario/quantity/type selections,
playback time, and the shared link/colormap/isotherm toggles -- as a
JSON file, so a comparison setup can be reopened exactly as left, or
handed to a colleague.

Serialization is pure (operates on plain values, not Qt objects);
MainWindow.collect_session_state()/apply_session_state() are the only
methods that touch live app state, kept there (not here) since they need
the real GridCell/ViewGrid/TimeController instances.
"""

from __future__ import annotations

import json

SESSION_VERSION = 2          # v2 adds the Evidence Notebook (V4-M2)
_SUPPORTED_VERSIONS = (1, 2)  # v1 sessions still load (notebook simply absent)


def cell_to_dict(cell) -> dict:
    """One GridCell's restorable state. quantity_key is stored as its
    `.quantity` string only -- direction/offset aren't yet user-selectable
    (single fixed plane, see slice_key.py), so the string round-trips via
    quantity lookup on load rather than reconstructing a SliceKey here."""
    d = {"cell_type": cell.cell_type, "quantity": cell.quantity_key.quantity}
    if cell.cell_type == "slice":
        d["case_index"] = int(cell.case_index)
    elif cell.cell_type == "difference":
        d["case_index_a"] = int(cell.case_index_a)
        d["case_index_b"] = int(cell.case_index_b)
    elif cell.cell_type == "ensemble":
        d["ensemble_case_indices"] = [int(c) for c in cell.ensemble_case_indices]
        d["ensemble_stat"] = cell.ensemble_stat
    return d


def build_session_dict(layout_name: str, cells: list, active_index: int, time_index: int,
                        link_clim: bool, colormap: str, isotherms_enabled: bool,
                        notebook: list | None = None, zones: list | None = None,
                        name: str = "", intent: str = "", metadata: dict | None = None,
                        time_window: dict | None = None, filters: dict | None = None,
                        measurements: list | None = None,
                        selection: dict | None = None,
                        calculated_fields: list | None = None,
                        devices: list | None = None) -> dict:
    return {
        "version": SESSION_VERSION,
        "layout": layout_name,
        "cells": [cell_to_dict(c) for c in cells],
        "active_index": active_index,
        "time_index": time_index,
        "link_clim": link_clim,
        "colormap": colormap,
        "isotherms_enabled": isotherms_enabled,
        "notebook": notebook or [],  # V4-M2 Evidence Notebook (serialized entries)
        "zones": zones or [],        # V4-M4 named zones (physical rectangles)
        # V4-M6 named-analysis-session fields (optional; older readers ignore).
        "name": name,
        "intent": intent,
        "metadata": metadata or {},
        "time_window": time_window or {},   # V4-M5 interval selection
        "filters": filters or {},           # experiment-browser filter state
        "measurements": measurements or [], # V4-M7 on-canvas measurements
        "selection": selection or {},       # V5-M1 shared selection (scenario/time/point/...)
        "calculated_fields": calculated_fields or [],  # V6-M1 Field Calculator definitions
        "devices": devices or [],  # V6-M2 Virtual Device Network placements + cached results
    }


def write_session(path: str, session: dict) -> None:
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


def read_session(path: str) -> dict:
    """Raises ValueError on a missing/unreadable file, wrong version, or
    malformed shape -- callers show this to the user rather than crashing
    (same convention as scenario_store.py's corrupted-cache fallback)."""
    try:
        with open(path) as f:
            session = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"could not read session file: {e}") from e
    if not isinstance(session, dict) or session.get("version") not in _SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported session file (expected version {SESSION_VERSION})")
    if "layout" not in session or "cells" not in session:
        raise ValueError("session file is missing required fields")
    return session
