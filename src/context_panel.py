"""Context Panel (V6-M4): the unified "what's related to what I'm looking
at right now" hub -- the centerpiece of connecting Devices, Vector Fields,
Knowledge Graph, Notebook, Zones, Measurements, and saved Sessions into one
investigation workflow, plus the Investigation History.

Purely a *consumer* of the existing shared-navigation substrate: it reuses
`InsightList` (the app's one clickable-statement widget, already used by
narrative/semantic-diff/advanced-compare/evidence-notebook) for every
Insight-shaped group -- devices, vector probes, notebook entries, zones,
measurements (the last two as synthetic Insights; zones/measurements have
no scenario of their own, matching how the Knowledge Graph already treats
them) -- so those rows fire the exact same `insight_activated` signal every
other feature in the app already emits. MainWindow needs zero new dispatch
code for them (see its existing `_on_insight_activated`).

Graph nodes and saved sessions are handled directly instead: some graph
nodes carry a `scenario` an Insight cannot represent (Insight has no
scenario field, by design -- see insight.py), and a session isn't
Insight-shaped at all. The Investigation History strip is the same case --
a `Selection` (not an Insight) must survive the round trip so `scenario` is
never lost on replay.

Performance (V6-M4 objective 8): playback publishes `time_s` on the
SelectionBus every tick (main_window.py's `_on_time_changed`), so this
panel must not re-gather context on every tick. It re-renders only when
scenario/point/region/quantity actually change -- the same guard
graph_panel.py already uses (`sel.scenario == self._current_scenario`) --
never on a time-only update.

Reuses context.gather_context (the data layer), InsightList, and the
SelectionBus. No new data path: everything here already exists somewhere
in the app.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from insight import Insight, InsightList
from context import gather_context


class ContextPanel(QtWidgets.QWidget):
    insight_activated = QtCore.pyqtSignal(object)       # devices/probes/notebook/zones/measurements
    session_reveal_requested = QtCore.pyqtSignal(str)   # a saved session's path
    hover_changed = QtCore.pyqtSignal(object)           # (scenario, point) or None -- linked hover

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app
        self._bus = None
        self._selection = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Context")
        title.setProperty("role", "section-title")
        layout.addWidget(title)

        self.summary = QtWidgets.QLabel("Select anything to see what's related.")
        self.summary.setAccessibleName("Context summary")
        self.summary.setToolTip("The current selection: scenario, quantity, time, and point/region.")
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "value")
        layout.addWidget(self.summary)

        # Synthesized point-story (Analysis-improvement roadmap Phase C):
        # nearest narrative event + local measurement/zone reading + cause
        # chain, combined into one paragraph -- pure reuse via
        # context.gather_context, nothing computed here.
        self.point_story = QtWidgets.QLabel("")
        self.point_story.setAccessibleName("Point story")
        self.point_story.setToolTip(
            "A synthesized summary of what's known about the selected point: the "
            "nearest narrative event, a local measurement/zone reading, and the "
            "Cause Explorer's chain (if already traced nearby).")
        self.point_story.setWordWrap(True)
        self.point_story.setProperty("role", "caption")
        layout.addWidget(self.point_story)

        history_row = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("← Back")
        self.back_button.setAccessibleName("History back")
        self.back_button.setToolTip("Return to the previous selection in this investigation")
        self.back_button.clicked.connect(self._on_back)
        self.forward_button = QtWidgets.QPushButton("Forward →")
        self.forward_button.setAccessibleName("History forward")
        self.forward_button.setToolTip("Advance to the next selection in this investigation")
        self.forward_button.clicked.connect(self._on_forward)
        history_row.addWidget(self.back_button)
        history_row.addWidget(self.forward_button)
        history_row.addStretch(1)
        layout.addLayout(history_row)

        self.tabs = QtWidgets.QTabWidget()
        self.devices_list = self._make_insight_list("Related devices")
        self.probes_list = self._make_insight_list("Related vector probes")
        self.notebook_list = self._make_insight_list("Related notebook entries")
        self.zones_list = self._make_insight_list("Related zones")
        self.measurements_list = self._make_insight_list("Related measurements")

        self.graph_list = QtWidgets.QListWidget()
        self.graph_list.setAccessibleName("Related graph nodes")
        self.graph_list.setToolTip("Click a node to jump to it (Reveal in Knowledge Graph). "
                                   "Hovering highlights its location in the Live Viewer.")
        self.graph_list.itemClicked.connect(self._on_graph_item)
        # Linked hover (V6-M4 objective 4): highlight a node's location on
        # the Live Viewer on hover, without changing the selection.
        self.graph_list.setMouseTracking(True)
        self.graph_list.itemEntered.connect(self._on_graph_hover)

        self.sessions_list = QtWidgets.QListWidget()
        self.sessions_list.setAccessibleName("Related sessions")
        self.sessions_list.setToolTip("Saved sessions whose grid references this scenario.")
        self.sessions_list.itemClicked.connect(self._on_session_item)

        self.history_list = QtWidgets.QListWidget()
        self.history_list.setAccessibleName("Investigation history")
        self.history_list.setToolTip("Every selection recorded this session -- click to revisit.")
        self.history_list.itemClicked.connect(self._on_history_item)

        for widget, label, tip in (
                (self.devices_list, "Devices", "Virtual devices placed on this scenario"),
                (self.probes_list, "Vector probes", "Vector probes placed on this scenario"),
                (self.notebook_list, "Notebook", "Evidence Notebook entries near this point"),
                (self.zones_list, "Zones", "Named zones containing this point"),
                (self.measurements_list, "Measurements", "Measurements near this point"),
                (self.graph_list, "Graph", "Knowledge Graph nodes for this scenario/point"),
                (self.sessions_list, "Sessions", "Saved sessions referencing this scenario"),
                (self.history_list, "History", "This investigation's recorded selections")):
            i = self.tabs.addTab(widget, label)
            self.tabs.setTabToolTip(i, tip)
        layout.addWidget(self.tabs, 1)

    def _make_insight_list(self, name: str) -> InsightList:
        lst = InsightList(self)
        lst.setAccessibleName(name)
        lst.setToolTip("Click a row to reveal it across the app (same shared navigation "
                       "every insight in the app uses).")
        lst.insight_activated.connect(self.insight_activated.emit)
        return lst

    # ------------------------------------------------------------- bus (M1)
    def set_bus(self, bus) -> None:
        self._bus = bus
        bus.changed.connect(self._on_selection)
        self._on_selection(bus.current, None)

    @staticmethod
    def _relevant_fields(sel):
        return (sel.scenario, sel.point, sel.region, sel.quantity)

    def _on_selection(self, selection, origin) -> None:
        prev = self._selection
        significant = prev is None or self._relevant_fields(selection) != self._relevant_fields(prev)
        self._selection = selection
        self.summary.setText(self._summary_text(selection))
        if significant:
            self._render()
        else:
            self._refresh_history()   # cheap: reflects a new history entry without a full re-gather

    # ------------------------------------------------------------- render
    def _render(self) -> None:
        if self._selection is None:
            return
        ctx = gather_context(self._app, self._selection)
        self.point_story.setText(ctx["point_story"])

        device_insights = [ins for ins in (d.summary_insight() for d in ctx["devices"]) if ins is not None]
        self.devices_list.set_insights(device_insights)
        probe_insights = [ins for ins in (p.summary_insight() for p in ctx["probes"]) if ins is not None]
        self.probes_list.set_insights(probe_insights)
        self.notebook_list.set_insights([e.insight for e in ctx["notebook"]])
        self.zones_list.set_insights([self._zone_insight(z) for z in ctx["zones"]])
        self.measurements_list.set_insights([self._measurement_insight(m) for m in ctx["measurements"]])

        self.graph_list.clear()
        for node in ctx["graph_nodes"]:
            item = QtWidgets.QListWidgetItem(f"{node.type}: {node.label}")
            item.setData(QtCore.Qt.UserRole, node)
            self.graph_list.addItem(item)

        self.sessions_list.clear()
        for info in ctx["sessions"]:
            item = QtWidgets.QListWidgetItem(info.name)
            item.setData(QtCore.Qt.UserRole, info.path)
            item.setToolTip(info.preview())
            self.sessions_list.addItem(item)

        self._refresh_history()

    def _refresh_history(self) -> None:
        history = getattr(self._app, "history", None)
        self.back_button.setEnabled(bool(history) and history.can_back())
        self.forward_button.setEnabled(bool(history) and history.can_forward())
        self.history_list.clear()
        for entry in (history.entries[-50:] if history is not None else []):
            item = QtWidgets.QListWidgetItem(entry.label or self._summary_text(entry.selection))
            item.setData(QtCore.Qt.UserRole, entry.selection)
            self.history_list.addItem(item)

    @staticmethod
    def _summary_text(sel) -> str:
        parts = []
        if sel.scenario is not None:
            parts.append(f"scenario {sel.scenario}")
        parts.append(sel.quantity)
        if sel.time_s is not None:
            parts.append(f"t = {sel.time_s:.1f} s")
        if sel.point is not None:
            parts.append(f"({sel.point[0]:.2f}, {sel.point[1]:.2f}) m")
        return " · ".join(parts)

    @staticmethod
    def _zone_insight(zone) -> Insight:
        return Insight(statement=f"Zone: {zone.name}",
                       region=(zone.x0, zone.x1, zone.z0, zone.z1))

    @staticmethod
    def _measurement_insight(m) -> Insight:
        point = tuple(m.points[0]) if m.points else None
        return Insight(statement=f"{m.label or m.kind}: {m.readout}", location=point)

    # ----------------------------------------------------------- interaction
    def _on_graph_item(self, item) -> None:
        node = item.data(QtCore.Qt.UserRole)
        sel = node.to_selection() if node is not None else None
        if sel is not None and self._bus is not None:
            self._bus.set(sel, origin=self)

    def _on_graph_hover(self, item) -> None:
        node = item.data(QtCore.Qt.UserRole)
        if node is not None and node.scenario is not None and node.point is not None:
            self.hover_changed.emit((node.scenario, node.point))

    def _on_session_item(self, item) -> None:
        path = item.data(QtCore.Qt.UserRole)
        if path:
            self.session_reveal_requested.emit(path)

    def _on_history_item(self, item) -> None:
        sel = item.data(QtCore.Qt.UserRole)
        history = getattr(self._app, "history", None)
        if sel is not None and self._bus is not None:
            self._bus.set(sel, origin=history)

    def _on_back(self) -> None:
        history = getattr(self._app, "history", None)
        if history is None or self._bus is None:
            return
        sel = history.back()
        if sel is not None:
            self._bus.set(sel, origin=history)

    def _on_forward(self) -> None:
        history = getattr(self._app, "history", None)
        if history is None or self._bus is None:
            return
        sel = history.forward()
        if sel is not None:
            self._bus.set(sel, origin=history)
