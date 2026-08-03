"""Research Knowledge Graph panel (V5 Phase 6), an Analysis-page tab.

A navigable view of the study's laboratory memory: a layered node-link diagram
plus a filterable list browser over the graph_model. Clicking a node highlights
its connections and publishes its Selection to the SelectionBus (M1), jumping
the workspace to the relevant scenario / time / region. Filter by node type or
by tag (e.g. "vod2", a notebook tag) to isolate a slice of the memory.

Built deterministically from existing artifacts (scenarios, experiments,
sessions, notebook, zones, measurements, placed devices/vector probes (V6-M4),
the selected scenario's narrative events and hazard classification (Analysis
UX + reliability pass -- hazard_spaces.py's own worst-tenability-class-
reached, not a new model)). "Refresh" re-gathers them. Reuses graph_model,
events, hazard_spaces, and the M1 bus.

Analysis UX + reliability pass also added a "Focus on selected" toggle:
hides everything more than one hop from the currently-selected node
(reusing the same neighbor computation click-highlighting already used),
so the graph stays legible as it gains real edges from the additions
above instead of becoming a hairball.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from slice_key import SliceKey
from descriptors import compute_descriptors
from events import detect_events
import hazard_spaces as hz
import graph_model as gm


class GraphPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, fps: int, app=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._app = app                 # main_window, for live artifacts
        self._bus = None
        self._graph = gm.Graph()
        self._pos = {}
        self._selected = None
        self._event_cache = {}
        self._hazard_cache = {}
        self._quantity_cache = {}
        self._current_scenario = None
        self._focus_selected = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Knowledge graph")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem("All types", None)
        for t in gm.NODE_TYPES:
            self.type_combo.addItem(t.capitalize(), t)
        header.addWidget(self.type_combo)
        header.addWidget(QtWidgets.QLabel("Tag:"))
        self.tag_combo = QtWidgets.QComboBox()
        header.addWidget(self.tag_combo)
        self.focus_checkbox = QtWidgets.QCheckBox("Focus on selected")
        self.focus_checkbox.setToolTip(
            "Hide everything more than one connection away from the selected node, "
            "so the graph stays readable as it grows")
        self.focus_checkbox.toggled.connect(self._on_focus_toggled)
        header.addWidget(self.focus_checkbox)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._rebuild)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(
            "The study's memory. Click a node to highlight its links and jump the "
            "workspace to it; click a tag to surface everything connected to it.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        # Compact type legend (Analysis final-polish pass): node color was
        # previously only decodable via the x-axis column headers and the
        # tree browser's own grouping -- a small always-visible swatch row
        # makes "what does this color mean" immediate.
        self.legend = QtWidgets.QLabel(" &middot; ".join(
            f'<span style="color:{gm.TYPE_COLOR[t]}">&#9679;</span> {t.capitalize()}'
            for t in gm.NODE_TYPES))
        self.legend.setTextFormat(QtCore.Qt.RichText)
        self.legend.setWordWrap(True)
        self.legend.setProperty("role", "caption")
        layout.addWidget(self.legend)

        body = QtWidgets.QSplitter()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setAccessibleName("Graph browser")
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_tree)
        body.addWidget(self.tree)
        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Knowledge graph view")
        body.addWidget(self.canvas)
        body.setStretchFactor(0, 2); body.setStretchFactor(1, 5)
        layout.addWidget(body, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setProperty("role", "caption")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.type_combo.currentIndexChanged.connect(self._render)
        self.tag_combo.currentIndexChanged.connect(self._render)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._loaded = False

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)

    def _on_selection(self, sel, origin) -> None:
        if origin is self or sel.scenario == self._current_scenario:
            return
        self._current_scenario = sel.scenario
        if self._loaded:
            self._rebuild()

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            self._rebuild()
            # Analysis final-polish pass: start focused on the current
            # scenario (if any) instead of dumping every node/edge on
            # first open -- avoids the "hairball by default" problem while
            # the "Focus on selected" checkbox stays available to widen
            # the view back out.
            sid = (f"scenario:{self._current_scenario}"
                   if self._current_scenario is not None else None)
            if sid is not None and sid in self._graph.nodes:
                self.focus_checkbox.setChecked(True)
                self._select_node(sid)

    def _events_for(self, case_index):
        if case_index is None:
            return []
        if case_index not in self._event_cache:
            try:
                data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
                extent = self._store.get_extent(case_index, SliceKey("TEMPERATURE"))
                self._event_cache[case_index] = detect_events(
                    compute_descriptors(data, extent, self._fps))
            except Exception:
                self._event_cache[case_index] = []
        return self._event_cache[case_index]

    def _quantities_for(self, case_index):
        """A small, fixed set of summary_stats.py's already-computed
        per-scenario metrics for `case_index` -- deliberately just 3 (peak
        temperature, peak HRR, minimum smoke-layer height), not every
        ScenarioSummary field, so the graph answers "what did this
        scenario measure?" without growing into another dense, hairball
        view. Cached and scoped to the same "currently selected scenario
        only" bound as _hazard_class_for/_events_for."""
        if case_index is None:
            return []
        if case_index not in self._quantity_cache:
            readings = []
            summaries = getattr(self._app, "_scenario_summaries", None) or []
            summary = next((s for s in summaries if s.case_index == case_index), None)
            if summary is not None:
                if summary.max_temp_c is not None:
                    readings.append(("Peak temperature", f"{summary.max_temp_c:.0f} °C"))
                if summary.peak_hrr_kw is not None:
                    readings.append(("Peak HRR", f"{summary.peak_hrr_kw:.0f} kW"))
                if summary.layer_min_height_m is not None:
                    readings.append(("Min. smoke-layer height", f"{summary.layer_min_height_m:.2f} m"))
            self._quantity_cache[case_index] = readings
        return self._quantity_cache[case_index]

    def _hazard_class_for(self, case_index):
        """Worst hazard_spaces.py tenability class reached in `case_index`
        (a class *name*, e.g. "Critical") -- reuses that module's own
        classification (temperature thresholds + cumulative exposure), not
        a new hazard model. Cached like _events_for, and scoped to the
        same "currently selected scenario only" bound for the same reason
        (keeps the graph's node/edge growth bounded rather than
        classifying all 24 scenarios' full arrays on every rebuild)."""
        if case_index is None:
            return None
        if case_index not in self._hazard_cache:
            try:
                data = np.asarray(self._store.get(case_index, SliceKey("TEMPERATURE")))
                thresholds = hz.band_thresholds("TEMPERATURE")
                classes = hz.classify_series(data, thresholds, self._fps)
                worst = int(hz.worst_class(classes).max())
                self._hazard_cache[case_index] = hz.CLASS_NAMES[worst]
            except Exception:
                self._hazard_cache[case_index] = None
        return self._hazard_cache[case_index]

    # ------------------------------------------------------------- build
    def _gather(self):
        app = self._app
        def live(attr, sub):
            panel = getattr(app, attr, None) if app is not None else None
            return getattr(panel, sub, None) if panel is not None else None
        notebook = getattr(getattr(app, "evidence_dock", None), "notebook", None)
        entries = list(notebook.entries) if notebook is not None else []
        zones = live("zone_panel", "_zones") or []
        # V6-M4: placed devices/vector probes become graph nodes too, the
        # same way zones/measurements already do.
        devices = live("device_panel", "_devices") or []
        vector_probes = live("velocity_panel", "_probes") or []
        # Phase C: pinned Sensitivity what-if estimates become hypothesis nodes.
        hypotheses = live("sensitivity_panel", "_hypotheses") or []
        experiments = []
        sessions = []
        try:
            import experiment as ex
            experiments = [ex.load_experiment(i.path)
                           for i in ex.list_experiments(ex.default_experiments_dir())]
        except Exception:
            experiments = []
        try:
            import session_store
            sess_dir = getattr(app, "_sessions_dir", None) or session_store.default_sessions_dir()
            sessions = session_store.list_sessions(sess_dir)
        except Exception:
            sessions = []
        ev = {}
        hazard = {}
        quantities = {}
        if self._current_scenario is not None:
            ev[self._current_scenario] = self._events_for(self._current_scenario)
            cls = self._hazard_class_for(self._current_scenario)
            if cls is not None:
                hazard[self._current_scenario] = cls
            q = self._quantities_for(self._current_scenario)
            if q:
                quantities[self._current_scenario] = q
        return (entries, zones, experiments, sessions, ev, devices,
               vector_probes, hypotheses, hazard, quantities)

    def _rebuild(self) -> None:
        (entries, zones, experiments, sessions, ev, devices,
         vector_probes, hypotheses, hazard, quantities) = self._gather()
        self._graph = gm.build_graph(self._manifest, notebook=entries, zones=zones,
                                     experiments=experiments,
                                     sessions=sessions, events_by_scenario=ev,
                                     devices=devices, vector_probes=vector_probes,
                                     hypotheses=hypotheses, hazard_by_scenario=hazard,
                                     quantities_by_scenario=quantities)
        # tag filter options
        self.tag_combo.blockSignals(True)
        self.tag_combo.clear()
        self.tag_combo.addItem("All tags", None)
        for t in self._graph.all_tags():
            self.tag_combo.addItem(t, t)
        self.tag_combo.blockSignals(False)
        self._layout_positions()
        self._populate_tree()
        self._render()

    def _layout_positions(self) -> None:
        self._pos = {}
        for col, t in enumerate(gm.NODE_TYPES):
            ns = self._graph.nodes_of(t)
            ys = np.linspace(0.05, 0.95, len(ns)) if len(ns) > 1 else [0.5]
            for node, y in zip(ns, ys):
                self._pos[node.id] = (col, float(y))

    # ------------------------------------------------------------- filters
    def _visible_ids(self):
        t = self.type_combo.currentData()
        tag = self.tag_combo.currentData()
        ids = []
        for node in self._graph.nodes.values():
            if t is not None and node.type != t:
                continue
            if tag is not None and tag not in node.tags and f"tag:{tag}" != node.id:
                continue
            ids.append(node.id)
        visible = set(ids)
        # Focus on selected (Analysis UX + reliability pass): restrict to
        # the selected node's one-hop neighborhood -- as the graph gains
        # real edges (hazard/device/vector_probe/hypothesis links above),
        # this is what keeps it from becoming a hairball, reusing the same
        # neighbor computation click-highlighting already relies on.
        if self._focus_selected and self._selected is not None and self._selected in self._graph.nodes:
            neighborhood = {self._selected, *self._graph.neighbors(self._selected)}
            visible &= neighborhood
        return visible

    def _on_focus_toggled(self, checked: bool) -> None:
        self._focus_selected = checked
        self._render()

    def _populate_tree(self) -> None:
        self.tree.clear()
        visible = self._visible_ids()
        for t in gm.NODE_TYPES:
            ns = [n for n in self._graph.nodes_of(t) if n.id in visible]
            if not ns:
                continue
            top = QtWidgets.QTreeWidgetItem([f"{t.capitalize()} ({len(ns)})"])
            for node in ns:
                item = QtWidgets.QTreeWidgetItem([node.label])
                item.setData(0, QtCore.Qt.UserRole, node.id)
                top.addChild(item)
            self.tree.addTopLevelItem(top)
        self.tree.expandToDepth(0)

    # ------------------------------------------------------------- render
    def _render(self) -> None:
        self._populate_tree()
        visible = self._visible_ids()
        highlight = set()
        if self._selected is not None and self._selected in self._graph.nodes:
            highlight = {self._selected, *self._graph.neighbors(self._selected)}
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        for edge in self._graph.edges:
            a, b = tuple(edge)
            if a not in visible or b not in visible:
                continue
            xa, ya = self._pos[a]; xb, yb = self._pos[b]
            hot = bool(highlight) and a in highlight and b in highlight
            ax.plot([xa, xb], [ya, yb], "-",
                    color=("#00E5FF" if hot else "#ddd"),
                    linewidth=(1.4 if hot else 0.4),
                    zorder=(2 if hot else 1), alpha=(0.9 if hot else 0.5))
        for node in self._graph.nodes.values():
            if node.id not in visible:
                continue
            x, y = self._pos[node.id]
            dim = bool(highlight) and node.id not in highlight
            ax.plot(x, y, "o", color=gm.TYPE_COLOR[node.type],
                    markersize=(9 if node.id == self._selected else 6),
                    alpha=(0.25 if dim else 1.0), zorder=3)
            if node.type != "scenario" or node.id == self._selected:
                ax.annotate(node.label[:16], (x, y), fontsize=5,
                            alpha=(0.25 if dim else 0.9),
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xticks(range(len(gm.NODE_TYPES)))
        ax.set_xticklabels([t.capitalize() for t in gm.NODE_TYPES], fontsize=6, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        fig.subplots_adjust(top=0.98, bottom=0.12, left=0.02, right=0.98)
        self.canvas.draw_idle()

    # ------------------------------------------------------------- interact
    def _select_node(self, node_id) -> None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return
        self._selected = node_id
        neigh = self._graph.neighbors(node_id)
        self.status.setText(f"<b>{node.type}: {node.label}</b> — {len(neigh)} connection(s)")
        sel = node.to_selection()
        if sel is not None and self._bus is not None:
            self._bus.set(sel, origin=self)
        self._render()

    def _on_tree(self, item, _col) -> None:
        node_id = item.data(0, QtCore.Qt.UserRole)
        if node_id:
            self._select_node(node_id)

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or not self._pos:
            return
        visible = self._visible_ids()
        best, best_d = None, 0.06
        for nid, (x, y) in self._pos.items():
            if nid not in visible:
                continue
            d = ((x - event.xdata) / max(1, len(gm.NODE_TYPES))) ** 2 + (y - event.ydata) ** 2
            if d < best_d:
                best_d, best = d, nid
        if best is not None:
            self._select_node(best)
