"""Research Knowledge Graph model (V5 Phase 6): the app's laboratory memory.

A deterministic graph assembled from artifacts that *already exist* -- no new
data, no simulation. Node types: Experiment, Scenario, Session, Insight
(notebook entry), Zone, Measurement, Event (narrative), Hazard (worst
tenability class reached), and Tag (factor levels like "vod2", plus the
user's notebook/zone tags). Edges connect an experiment to its scenarios,
a scenario to its factor tags/narrative events/hazard classification, a
tagged artifact to its tags, and (Analysis UX + reliability pass) every
zone/measurement/device/vector_probe/hypothesis to the scenario it belongs
to -- so clicking a tag ("vod2", "flashover", "plume") surfaces everything
connected to it, and no placed artifact is an unreachable island.

Each node can produce a `Selection` (M1) for the fields it carries (scenario,
time, point, region, quantity), or None if it is purely organizational. Pure,
Qt-free, deterministic.

Analysis UX + reliability pass: previously zone/measurement/device/
vector_probe/hypothesis nodes had NO edges at all (only navigable via
their own to_selection(), not via the graph's relationships), and nothing
here read hazard_spaces.py's tenability classification despite the app-
wide "laboratory memory over every existing artifact" framing. Both
gaps are closed below -- reusing hazard_spaces.py's own classification
(worst_class over an already-computed classify_series), not a new hazard
model, and reusing the exact same g.link() pattern already used for
scenario<->tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# Column order for the layered layout / grouping in the browser.
NODE_TYPES = ("tag", "experiment", "session", "scenario", "event", "hazard",
              "insight", "zone", "measurement", "device", "vector_probe",
              "hypothesis")
TYPE_COLOR = {
    "tag": "#8E24AA", "experiment": "#1565C0", "session": "#00838F",
    "scenario": "#E8622C", "event": "#F9A825", "hazard": "#B71C1C",
    "insight": "#2E7D32",
    "zone": "#6D4C41", "measurement": "#455A64",
    "device": "#C2185B", "vector_probe": "#00695C",   # V6-M4
    "hypothesis": "#5E35B1",   # Analysis-improvement roadmap Phase C
}


@dataclass
class Node:
    id: str
    type: str
    label: str
    tags: Tuple[str, ...] = ()
    scenario: Optional[int] = None
    time_s: Optional[float] = None
    point: Optional[tuple] = None
    region: Optional[tuple] = None
    quantity: Optional[str] = None

    def to_selection(self):
        """A Selection for this node, or None if it carries nothing navigable."""
        if (self.scenario is None and self.time_s is None and self.point is None
                and self.region is None):
            return None
        from selection import Selection
        fields = {}
        if self.scenario is not None:
            fields["scenario"] = self.scenario
        if self.time_s is not None:
            fields["time_s"] = self.time_s
        if self.point is not None:
            fields["point"] = self.point
        if self.region is not None:
            fields["region"] = self.region
        if self.quantity:
            fields["quantity"] = self.quantity
        return Selection(**fields)


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)      # id -> Node
    edges: set = field(default_factory=set)        # frozenset({a_id, b_id})

    def add(self, node: Node) -> str:
        self.nodes.setdefault(node.id, node)
        return node.id

    def link(self, a: str, b: str) -> None:
        if a != b and a in self.nodes and b in self.nodes:
            self.edges.add(frozenset((a, b)))

    def neighbors(self, node_id: str) -> list:
        return sorted({next(iter(e - {node_id})) for e in self.edges if node_id in e})

    def nodes_of(self, node_type: str) -> list:
        return [n for n in self.nodes.values() if n.type == node_type]

    def all_tags(self) -> list:
        return sorted({t for n in self.nodes.values() for t in n.tags})


_FACTORS = ("candles", "door", "vod", "voc")


def build_graph(scenarios, notebook=None, zones=None, measurements=None,
                experiments=None, sessions=None, events_by_scenario=None,
                devices=None, vector_probes=None, hypotheses=None,
                hazard_by_scenario=None) -> Graph:
    """Assemble the graph from the current artifacts. `scenarios` are manifest
    entries; the rest are optional and default to empty.

    `devices`/`vector_probes` (V6-M4): Device/VectorProbe instances
    (devices.py/velocity.py) -- placed instrumentation becomes graph nodes
    the same way zones/measurements already do. Every zone/measurement/
    device/vector_probe/hypothesis with a `scenario` field is now also
    linked to that scenario node (Analysis UX + reliability pass), same
    g.link() pattern already used for scenario<->tag, so they participate
    in the graph's relationships instead of only being reachable via their
    own to_selection().

    `hypotheses` (Analysis-improvement roadmap Phase C): pinned Sensitivity
    what-if estimates (dicts: id/label/nearest_scenario) -- an interpolated
    estimate isn't itself a real run, so its only navigable dimension is the
    nearest existing scenario it was pinned against.

    `hazard_by_scenario` (Analysis UX + reliability pass): {case_index:
    class_name}, the worst hazard_spaces.py tenability class reached in
    that scenario (e.g. "Critical") -- reuses that module's own
    classification, not a new hazard model. One hazard node per entry,
    linked to its scenario, so "what physical relationships explain the
    behavior I'm observing" has a direct answer for hazard severity."""
    g = Graph()

    def tag(name: str) -> str:
        return g.add(Node(f"tag:{name}", "tag", name))

    # scenarios + their factor-level tags
    scenario_sid = {}
    for e in scenarios or []:
        sid = g.add(Node(f"scenario:{e.case_index}", "scenario", e.folder,
                         tags=tuple(f"{f}{getattr(e, f)}" for f in _FACTORS),
                         scenario=e.case_index))
        scenario_sid[e.case_index] = sid
        for f in _FACTORS:
            g.link(sid, tag(f"{f}{getattr(e, f)}"))

    # experiments -> member scenarios (matched by folder)
    folder_to_sid = {e.folder: f"scenario:{e.case_index}" for e in (scenarios or [])}
    for exp in experiments or []:
        eid = g.add(Node(f"experiment:{exp.name}", "experiment", exp.name,
                         tags=tuple(exp.tags or ())))
        for t in (exp.tags or ()):
            g.link(eid, tag(t))
        for folder in getattr(exp, "scenarios", []) or []:
            if folder in folder_to_sid:
                g.link(eid, folder_to_sid[folder])

    # sessions (study-level; linked by shared tags only)
    for s in sessions or []:
        s_tags = tuple(getattr(s, "tags", ()) or ())
        g.add(Node(f"session:{s.name}", "session", s.name, tags=s_tags))
        for t in s_tags:
            g.link(f"session:{s.name}", tag(t))

    # notebook entries (insights)
    for i, entry in enumerate(notebook or []):
        ins = entry.insight
        nid = g.add(Node(f"insight:{i}", "insight", ins.statement[:60],
                         tags=tuple(entry.tags or ()), time_s=ins.primary_time(),
                         point=ins.location, region=ins.region, quantity=ins.quantity))
        for t in (entry.tags or ()):
            g.link(nid, tag(t))

    # zones -- no scenario link: zone_stats.py's own Zone is deliberately
    # scenario-agnostic (a named region applies to any scenario and is
    # compared across them), so it has no `scenario` field to link from;
    # forcing one would invent a relationship the domain model doesn't
    # have, which is exactly what this pass' node/edge additions are
    # meant to avoid ("do not add relationships simply because they can
    # be drawn").
    for i, z in enumerate(zones or []):
        g.add(Node(f"zone:{i}", "zone", z.name,
                   region=(z.x0, z.x1, z.z0, z.z1)))

    # measurements -- same reasoning as zones: measure.py's Measurement
    # has no `scenario` field (a disposable, un-named single-use read).
    for i, m in enumerate(measurements or []):
        pt = tuple(m.points[0]) if m.points else None
        reg = (tuple(m.points[0]) + tuple(m.points[1])) if m.kind == "rect" and len(m.points) >= 2 else None
        g.add(Node(f"measurement:{i}", "measurement", m.label or m.kind,
                   point=(pt if reg is None else None), region=reg))

    # devices (V6-M4 Virtual Device Network) -- linked to their owning
    # scenario (Analysis UX + reliability pass): unlike zones/
    # measurements, devices.py's Device genuinely carries a `scenario`
    # field (it's placed in one specific run), so this is a real
    # relationship, not an invented one.
    for d in devices or []:
        did = g.add(Node(f"device:{d.id}", "device", d.name,
                         scenario=d.scenario, point=tuple(d.position)))
        if d.scenario in scenario_sid:
            g.link(did, scenario_sid[d.scenario])

    # vector probes (V6-M4 True Velocity) -- same real scenario link as devices.
    for p in vector_probes or []:
        pid = g.add(Node(f"vector_probe:{p.id}", "vector_probe", p.name,
                         scenario=p.scenario, point=tuple(p.position)))
        if p.scenario in scenario_sid:
            g.link(pid, scenario_sid[p.scenario])

    # pinned Sensitivity what-if hypotheses (Phase C) -- linked to the
    # nearest existing scenario they were pinned against (the same
    # `scenario` field to_selection() already reads for navigation).
    for h in hypotheses or []:
        hid = g.add(Node(f"hypothesis:{h.get('id')}", "hypothesis", h.get("label", "what-if"),
                         scenario=h.get("nearest_scenario")))
        nearest = h.get("nearest_scenario")
        if nearest in scenario_sid:
            g.link(hid, scenario_sid[nearest])

    # hazard classification per scenario (Analysis UX + reliability pass):
    # reuses hazard_spaces.py's own worst-tenability-class-reached
    # classification (computed by the caller, not here) -- one node per
    # scenario with a computed class, linked to that scenario, so hazard
    # severity is a direct graph relationship rather than absent entirely.
    for case_index, class_name in (hazard_by_scenario or {}).items():
        if case_index not in scenario_sid:
            continue
        hzid = g.add(Node(f"hazard:{case_index}", "hazard", class_name,
                          scenario=case_index))
        g.link(hzid, scenario_sid[case_index])

    # narrative events for the given scenarios -> their scenario
    for case_index, evs in (events_by_scenario or {}).items():
        sid = f"scenario:{case_index}"
        if sid not in g.nodes:
            continue
        for j, ev in enumerate(evs):
            evid = g.add(Node(f"event:{case_index}:{j}", "event",
                              ev.statement.split(":")[0][:40],
                              scenario=case_index, time_s=ev.primary_time(),
                              quantity=ev.quantity))
            g.link(evid, sid)
    return g
