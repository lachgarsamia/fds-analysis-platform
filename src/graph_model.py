"""Research Knowledge Graph model (V5 Phase 6): the app's laboratory memory.

A deterministic graph assembled from artifacts that *already exist* -- no new
data, no simulation. Node types: Experiment, Scenario, Session, Insight
(notebook entry), Zone, Measurement, Event (narrative), and Tag (factor levels
like "vod2", plus the user's notebook/zone tags). Edges connect an experiment
to its scenarios, a scenario to its factor tags and its narrative events, and a
tagged artifact to its tags -- so clicking a tag ("vod2", "flashover", "plume")
surfaces everything connected to it.

Each node can produce a `Selection` (M1) for the fields it carries (scenario,
time, point, region, quantity), or None if it is purely organizational. Pure,
Qt-free, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# Column order for the layered layout / grouping in the browser.
NODE_TYPES = ("tag", "experiment", "session", "scenario", "event",
              "insight", "zone", "measurement", "device", "vector_probe")
TYPE_COLOR = {
    "tag": "#8E24AA", "experiment": "#1565C0", "session": "#00838F",
    "scenario": "#E8622C", "event": "#F9A825", "insight": "#2E7D32",
    "zone": "#6D4C41", "measurement": "#455A64",
    "device": "#C2185B", "vector_probe": "#00695C",   # V6-M4
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
                devices=None, vector_probes=None) -> Graph:
    """Assemble the graph from the current artifacts. `scenarios` are manifest
    entries; the rest are optional and default to empty.

    `devices`/`vector_probes` (V6-M4): Device/VectorProbe instances
    (devices.py/velocity.py) -- placed instrumentation becomes graph nodes
    the same way zones/measurements already do, linked to nothing else
    (their `scenario` alone makes them navigable via `to_selection()`)."""
    g = Graph()

    def tag(name: str) -> str:
        return g.add(Node(f"tag:{name}", "tag", name))

    # scenarios + their factor-level tags
    for e in scenarios or []:
        sid = g.add(Node(f"scenario:{e.case_index}", "scenario", e.folder,
                         tags=tuple(f"{f}{getattr(e, f)}" for f in _FACTORS),
                         scenario=e.case_index))
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

    # zones
    for i, z in enumerate(zones or []):
        g.add(Node(f"zone:{i}", "zone", z.name,
                   region=(z.x0, z.x1, z.z0, z.z1)))

    # measurements
    for i, m in enumerate(measurements or []):
        pt = tuple(m.points[0]) if m.points else None
        reg = (tuple(m.points[0]) + tuple(m.points[1])) if m.kind == "rect" and len(m.points) >= 2 else None
        g.add(Node(f"measurement:{i}", "measurement", m.label or m.kind,
                   point=(pt if reg is None else None), region=reg))

    # devices (V6-M4 Virtual Device Network)
    for d in devices or []:
        g.add(Node(f"device:{d.id}", "device", d.name,
                   scenario=d.scenario, point=tuple(d.position)))

    # vector probes (V6-M4 True Velocity)
    for p in vector_probes or []:
        g.add(Node(f"vector_probe:{p.id}", "vector_probe", p.name,
                   scenario=p.scenario, point=tuple(p.position)))

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
