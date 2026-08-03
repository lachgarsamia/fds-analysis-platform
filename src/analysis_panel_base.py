"""AnalysisPanelBase + the bus-binder (V5-M1 / Phase 0) — Layer 3 → Layer 2.

`AnalysisPanelBase` is the go-forward base for new analysis panels: it holds the
`SelectionBus` and `QuantityProvider`, does the lazy `showEvent → ensure_loaded`,
and offers `publish(**fields)` / `react(ctx)` so a panel drives and follows the
shared selection without ever referencing another panel.

`bind_to_bus` is the *additive* migration path for the existing V4 panels: it
wires a panel's own `scenario_combo` and `frame_slider` to the bus **without
changing the panel's class or __init__** — changing a combo publishes the
scenario, a bus change syncs the combo, and a re-entrancy guard stops the echo
from re-publishing. This realizes "every panel depends only on Layer 2" for the
shared fields (scenario, time) while preserving all V4 behaviour; deeper
per-panel point/region sync rides along as panels are individually touched
(M2–M6), exactly as docs/architecture-selection-model.md scopes.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from manifest import scenario_label


def populate_scenario_combo(combo: QtWidgets.QComboBox, entries: list) -> None:
    """Fill `combo` with one item per manifest/summary entry: display text
    is a human-readable factor-level summary (manifest.scenario_label)
    instead of the raw disk folder name, with the folder itself kept as
    each item's tooltip (still there for provenance/debugging, just not
    the visible label every scenario picker across the app used to show).
    Shared so ~20 panels don't each carry their own copy of this loop."""
    for entry in entries:
        combo.addItem(scenario_label(entry), entry.case_index)
        combo.setItemData(combo.count() - 1, entry.folder, QtCore.Qt.ToolTipRole)


def bind_to_bus(panel, bus, fps: int) -> None:
    """Synchronize `panel`'s scenario_combo / frame_slider with `bus`. Safe to
    call on any panel; fields the panel lacks are skipped."""
    fps = max(1, fps)
    combo = getattr(panel, "scenario_combo", None)
    slider = getattr(panel, "frame_slider", None)
    qcombo = getattr(panel, "quantity_combo", None)
    qopts = getattr(panel, "_quantity_options", None)
    quantity_synced = qcombo is not None and qopts
    if combo is None and slider is None and not quantity_synced:
        return
    guard = {"syncing": False}

    if quantity_synced:
        def _on_qcombo(_i, c=qcombo, opts=qopts):
            if guard["syncing"]:
                return
            i = c.currentIndex()
            if 0 <= i < len(opts):
                bus.update(origin=panel, quantity=opts[i][1].quantity)
        qcombo.currentIndexChanged.connect(_on_qcombo)

    if combo is not None:
        def _on_combo(_i, c=combo):
            if guard["syncing"]:
                return
            data = c.currentData()
            if data is not None:
                bus.update(origin=panel, scenario=data)
        combo.currentIndexChanged.connect(_on_combo)

    if slider is not None:
        def _on_slider(_v, s=slider):
            if guard["syncing"]:
                return
            bus.update(origin=panel, time_s=s.value() / fps)
        slider.valueChanged.connect(_on_slider)

    def _on_selection(sel, origin):
        if origin is panel:
            return
        # RC polish (playback): a hidden analysis tab must not re-render on
        # every playback tick. Only the visible panel follows time live;
        # scenario/quantity (rarer, from explicit picks) still sync so a
        # hidden panel is correct the moment it is shown.
        time_live = panel.isVisible()
        guard["syncing"] = True
        try:
            if combo is not None and sel.scenario is not None:
                idx = combo.findData(sel.scenario)
                if idx >= 0 and idx != combo.currentIndex():
                    combo.setCurrentIndex(idx)
            if time_live and slider is not None and sel.time_s is not None:
                fi = min(max(int(round(sel.time_s * fps)), 0), slider.maximum())
                if fi != slider.value():
                    slider.setValue(fi)
            if quantity_synced and sel.quantity:
                for i, (_label, key) in enumerate(qopts):
                    if key.quantity == sel.quantity and i != qcombo.currentIndex():
                        qcombo.setCurrentIndex(i)
                        break
        finally:
            guard["syncing"] = False
    bus.changed.connect(_on_selection)


class AnalysisPanelBase(QtWidgets.QWidget):
    """Optional base for new panels. Existing panels use bind_to_bus instead."""

    def __init__(self, bus=None, provider=None, fps: int = 1, parent=None):
        super().__init__(parent)
        self._bus = bus
        self._provider = provider
        self._fps = max(1, fps)
        self._loaded = False
        if bus is not None:
            bus.changed.connect(self._on_selection)

    # --- selection plumbing ---
    def publish(self, **fields) -> None:
        if self._bus is not None:
            self._bus.update(origin=self, **fields)

    def _on_selection(self, selection, origin) -> None:
        if origin is self or self._bus is None:
            return
        self.react(self._bus.context(self._provider))

    def react(self, ctx) -> None:
        """Override to follow the shared selection."""

    # --- lazy load ---
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self.load()

    def load(self) -> None:
        """Override for one-time population."""
