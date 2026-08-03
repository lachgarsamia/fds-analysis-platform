"""The Context Panel's data layer (V6-M4): "what's related to what the
researcher is looking at right now."

`gather_context(app, selection)` is the one place that answers this,
reusing the exact pattern graph_panel.py's own `_gather()` established --
duck-type through `getattr(app, ...)` at already-cached live attributes
(never recomputing anything, never touching the store), and defensive
local imports around filesystem reads. No new data path: every value
returned here already lives somewhere in the app (a device's `.results`,
a probe's `.results`, a notebook entry, a zone, a measurement, a graph
node, a saved session) -- this module only filters and collects.

Runs on `SelectionBus.changed` (interaction rate), never per playback
tick, and is Qt-free/duck-typed so it's unit-testable with a plain fake
object -- no `QApplication` needed (see tests/test_context.py).
"""

from __future__ import annotations

from selection import Selection


def _live(app, attr: str, sub: str):
    """The same closure graph_panel._gather() uses: `app.<attr>.<sub>`, or
    None if either link is absent (no manifest, panel not built, demo mode)."""
    panel = getattr(app, attr, None) if app is not None else None
    return getattr(panel, sub, None) if panel is not None else None


def _matches_scenario(obj_scenario, scenario) -> bool:
    return scenario is None or obj_scenario == scenario


def _near(point_a, point_b, tol: float = 0.5) -> bool:
    if point_a is None or point_b is None:
        return False
    return abs(point_a[0] - point_b[0]) <= tol and abs(point_a[1] - point_b[1]) <= tol


def _related_devices(app, selection: Selection) -> list:
    devices = _live(app, "device_panel", "_devices") or []
    return [d for d in devices if _matches_scenario(d.scenario, selection.scenario)]


def _related_probes(app, selection: Selection) -> list:
    probes = _live(app, "velocity_panel", "_probes") or []
    return [p for p in probes if _matches_scenario(p.scenario, selection.scenario)]


def _related_notebook(app, selection: Selection) -> list:
    """Notebook entries near the selected point. An Insight carries no
    scenario, so without a point there is nothing honest to filter on --
    every entry is returned rather than fabricating a scenario match."""
    notebook = _live(app, "evidence_dock", "notebook")
    entries = list(notebook.entries) if notebook is not None else []
    if selection.point is None:
        return entries
    return [e for e in entries if _near(e.insight.location, selection.point)]


def _related_zones(app, selection: Selection) -> list:
    zones = _live(app, "zone_panel", "_zones") or []
    if selection.point is None:
        return list(zones)
    x, z = selection.point
    return [z_ for z_ in zones if z_.x0 <= x <= z_.x1 and z_.z0 <= z <= z_.z1]


def _related_measurements(app, selection: Selection) -> list:
    measurements = _live(app, "measurement_panel", "_measurements") or []
    if selection.point is None:
        return list(measurements)
    return [m for m in measurements if any(_near(p, selection.point) for p in m.points)]


def _related_graph_nodes(app, selection: Selection) -> list:
    graph = _live(app, "graph_panel", "_graph")
    if graph is None:
        return []
    nodes = []
    for node in graph.nodes.values():
        if selection.scenario is not None and node.scenario == selection.scenario:
            nodes.append(node)
        elif selection.point is not None and _near(node.point, selection.point):
            nodes.append(node)
    return nodes


def _nearest_narrative_event(app, selection: Selection):
    """The narrative event closest in time to the selection, for this
    scenario. Reuses narrative_panel's own cached, per-scenario event
    detection (`_events`) -- the same store-read-once-then-cache cost
    graph_panel._events_for already accepts at this same bus-changed rate;
    a scenario already visited in the Narrative tab is a pure cache hit."""
    panel = getattr(app, "narrative_panel", None)
    if panel is None or selection.scenario is None:
        return None
    try:
        events = panel._events(selection.scenario)
    except Exception:
        return None
    timed = [e for e in events if e.primary_time() is not None]
    if not timed:
        return None
    t = selection.time_s if selection.time_s is not None else 0.0
    return min(timed, key=lambda e: abs(e.primary_time() - t))


def _related_cause_chain(app, selection: Selection) -> list:
    """The Cause Explorer's last-computed chain, only when the researcher
    has already traced a hot spot near the selected point in that panel --
    already-cached on the panel (no new store read; context.py never
    touches the store), consistent with "if available" rather than a
    fresh trace on every selection change."""
    panel = getattr(app, "cause_panel", None)
    if panel is None or selection.point is None:
        return []
    if not _near(getattr(panel, "_last_point", None), selection.point):
        return []
    return list(getattr(panel, "_last_insights", None) or [])


def _point_story(app, selection: Selection) -> str:
    """A short combined paragraph for the selected point (Analysis-
    improvement roadmap Phase C): nearest Narrative event + a local
    measurement/zone reading + the Cause Explorer's chain, if available.
    Every clause reuses an existing engine's own already-computed result --
    nothing here fabricates a new number. Empty string if nothing applies
    (no point selected, or nothing related found yet)."""
    if selection.point is None:
        return ""
    sentences = []
    event = _nearest_narrative_event(app, selection)
    if event is not None:
        sentences.append(f"Nearest narrative event: {event.statement}")
    reading = None
    for m in _related_measurements(app, selection):
        if m.readout:
            reading = f"{m.label or m.kind}: {m.readout}"
            break
    if reading is None:
        zones = _related_zones(app, selection)
        if zones:
            reading = f"inside zone \"{zones[0].name}\""
    if reading is not None:
        sentences.append(f"Local reading: {reading}")
    chain = _related_cause_chain(app, selection)
    if chain:
        sentences.append(f"Cause trace (association, not causation): {chain[-1].statement}")
    if not sentences:
        return ""
    return " ".join(sentences)


def _related_sessions(app, selection: Selection) -> list:
    """Saved sessions whose grid references the selected scenario --
    determined by scanning each session file's cells (bounded by the
    typically-small number of saved sessions; never on a playback tick).
    Defensive: a missing/malformed session directory must never break
    Context Panel rendering."""
    if selection.scenario is None:
        return []
    try:
        import session_store
        directory = getattr(app, "_sessions_dir", None) or session_store.default_sessions_dir()
        matches = []
        for info in session_store.list_sessions(directory):
            try:
                session = session_store.load_session(info.path)
            except ValueError:
                continue
            for cell in session.get("cells", []) or []:
                refs = (cell.get("case_index"), cell.get("case_index_a"), cell.get("case_index_b"),
                       *(cell.get("ensemble_case_indices", []) or []))
                if selection.scenario in refs:
                    matches.append(info)
                    break
        return matches
    except Exception:
        return []


def gather_context(app, selection: Selection) -> dict:
    """Every related artifact for `selection`, grouped by kind. Every value
    is always a list (never None), so callers never need defensive `.get`
    scaffolding."""
    return {
        "devices": _related_devices(app, selection),
        "probes": _related_probes(app, selection),
        "notebook": _related_notebook(app, selection),
        "zones": _related_zones(app, selection),
        "measurements": _related_measurements(app, selection),
        "graph_nodes": _related_graph_nodes(app, selection),
        "sessions": _related_sessions(app, selection),
        "point_story": _point_story(app, selection),
    }
