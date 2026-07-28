"""Time-Window & Interval Analysis panel (V4-M5), an Analysis-page tab.

Time as a selectable dimension. Pick a scenario and quantity; the timeline
shows the field's spatial-mean and spatial-max over time with the Fire
Story's detected phase boundaries marked. Click twice to select any
interval (or pick a detected phase) and read its statistics -- mean, peak,
integral, trend; or switch to before/after mode, click one instant, and
compare the two halves. Findings are navigable, savable Insights.

Static/lazy; the two per-frame series and the phase list are computed once
per (scenario, quantity) and cached. Reuses descriptors + events (phase
boundaries) and the registry.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas
from registry import get_quantity, AMBIENT_C
from insight import InsightList, Insight
from descriptors import compute_descriptors
from events import detect_events
import time_window as tw


def _short(statement: str) -> str:
    """A compact phase label from an event statement."""
    return statement.split(":")[0].split("(")[0].strip().rstrip(".")


class TimeWindowPanel(QtWidgets.QWidget):
    def __init__(self, store, manifest: list, quantity_options: list, fps: int, parent=None):
        super().__init__(parent)
        self._store = store
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._quantity_options = [(label, key) for label, key in quantity_options
                                  if get_quantity(key.quantity).kind == "slice2d"]
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}       # (case, quantity) -> dict(series...)
        self._series = None
        self._mode = "window"  # "window" | "split"
        self._clicks = []      # pending window clicks (times)
        self._t0 = None
        self._t1 = None
        self._split = None
        self._bus = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Interval analysis")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Interval scenario")
        header.addWidget(self.scenario_combo)
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.setAccessibleName("Interval quantity")
        for label, _key in self._quantity_options:
            self.quantity_combo.addItem(label)
        self.quantity_combo.setEnabled(len(self._quantity_options) > 1)
        header.addWidget(self.quantity_combo)
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setAccessibleName("Interval mode")
        self.mode_combo.addItems(["Window (click two times)", "Before / after (click one)"])
        controls.addWidget(self.mode_combo)
        controls.addWidget(QtWidgets.QLabel("Phase:"))
        self.phase_combo = QtWidgets.QComboBox()
        self.phase_combo.setAccessibleName("Detected phase")
        self.phase_combo.setMinimumWidth(160)
        controls.addWidget(self.phase_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.caption = QtWidgets.QLabel(
            "Click the timeline to select an interval; dotted lines mark the "
            "detected phases. Statistics update for the selection.")
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Interval timeline")
        layout.addWidget(self.canvas, 1)

        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setProperty("role", "caption")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.stats_label)

        self.insights = InsightList()
        self.insights.setMaximumHeight(100)
        layout.addWidget(self.insights)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.quantity_combo.currentIndexChanged.connect(self._reload)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.phase_combo.activated.connect(self._on_phase_selected)
        self.insights.insight_activated.connect(self._on_insight)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    @property
    def _key(self):
        idx = max(0, self.quantity_combo.currentIndex())
        return self._quantity_options[idx][1] if self._quantity_options else None

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest or not self._quantity_options:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        for entry in self._manifest:
            self.scenario_combo.addItem(entry.folder, entry.case_index)
        self.scenario_combo.blockSignals(False)
        self._reload()

    # ------------------------------------------------------------- data load
    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        key = self._key
        if case_index is None or key is None:
            return
        cache_key = (case_index, key.quantity)
        if cache_key not in self._cache:
            data = np.asarray(self._store.get(case_index, key))
            extent = self._store.get_extent(case_index, key)
            times = np.arange(data.shape[0]) / self._fps
            desc = compute_descriptors(data, extent, self._fps)
            events = detect_events(desc, quantity=key.quantity)
            phases = tw.phase_windows(
                [(e.primary_time(), _short(e.statement)) for e in events],
                float(times[-1]) if len(times) else 0.0)
            self._cache[cache_key] = {
                "times": times,
                "mean": data.mean(axis=(1, 2)),
                "max": data.max(axis=(1, 2)),
                "phases": phases,
                "unit": get_quantity(key.quantity).unit,
                "label": get_quantity(key.quantity).label,
            }
        self._series = self._cache[cache_key]
        # default window: the whole run
        t = self._series["times"]
        self._t0, self._t1 = (float(t[0]), float(t[-1])) if len(t) else (0.0, 0.0)
        self._split = float(t[len(t) // 2]) if len(t) else 0.0
        self._clicks = []
        self._refresh_phase_combo()
        self._render()
        self._compute()

    def _refresh_phase_combo(self) -> None:
        self.phase_combo.blockSignals(True)
        self.phase_combo.clear()
        self.phase_combo.addItem("(whole run)")
        for name, _a, _b in self._series["phases"]:
            self.phase_combo.addItem(name)
        self.phase_combo.blockSignals(False)

    # --------------------------------------------------------------- events
    def _on_mode_changed(self, idx: int) -> None:
        self._mode = "window" if idx == 0 else "split"
        self._clicks = []
        self._render()
        self._compute()

    def _on_phase_selected(self, idx: int) -> None:
        if idx <= 0 or self._series is None:
            t = self._series["times"]
            self._t0, self._t1 = float(t[0]), float(t[-1])
        else:
            _name, a, b = self._series["phases"][idx - 1]
            self._t0, self._t1 = a, b
        self._mode = "window"
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.blockSignals(False)
        self._render()
        self._compute()
        self._publish_selection()

    def _on_click(self, event) -> None:
        if self._series is None or event.inaxes is None or event.xdata is None:
            return
        t = float(event.xdata)
        if self._mode == "split":
            self._split = t
        else:
            self._clicks.append(t)
            if len(self._clicks) == 2:
                self._t0, self._t1 = sorted(self._clicks)
                self._clicks = []
            else:
                self._render()  # show the first click, wait for the second
                return
        self._render()
        self._compute()
        self._publish_selection()

    def _on_insight(self, insight) -> None:
        pass  # interval stats are spans; nothing to seek here

    # ------------------------------------------------------------- bus (Phase 2)
    def set_bus(self, bus) -> None:
        """Publishes the current window as Selection.interval -- previously
        local-only despite Selection.interval existing for exactly this.
        One-way (publish only), same scope as TimeSeriesPanel's probe.
        Split mode is deliberately not published: unlike interval/point/
        region, Selection.time_s also drives the shared playback frame
        everywhere, so publishing it from a stats-only split click would
        silently relocate every other panel's current frame -- a louder
        side effect than this phase's "shared context, not surprise
        actions" scope."""
        self._bus = bus

    def _publish_selection(self) -> None:
        if self._bus is None or self._mode != "window":
            return
        if self._t0 is not None and self._t1 is not None:
            self._bus.update(origin=self, interval=(self._t0, self._t1))

    # -------------------------------------------------- session state (V4-M6)
    def get_state(self) -> dict:
        """The current interval selection, for a named session."""
        return {"scenario": self.scenario_combo.currentData(),
                "quantity": self.quantity_combo.currentIndex(),
                "mode": self._mode, "t0": self._t0, "t1": self._t1,
                "split": self._split}

    def set_state(self, state: dict) -> None:
        if not state:
            return
        self.ensure_loaded()
        case = state.get("scenario")
        if case is not None:
            i = self.scenario_combo.findData(case)
            if i >= 0:
                self.scenario_combo.setCurrentIndex(i)  # triggers _reload
        qi = state.get("quantity")
        if qi is not None and 0 <= qi < self.quantity_combo.count():
            self.quantity_combo.setCurrentIndex(qi)
        self._mode = state.get("mode", "window")
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0 if self._mode == "window" else 1)
        self.mode_combo.blockSignals(False)
        if state.get("t0") is not None:
            self._t0 = state["t0"]
        if state.get("t1") is not None:
            self._t1 = state["t1"]
        if state.get("split") is not None:
            self._split = state["split"]
        self._clicks = []
        self._render()
        self._compute()

    # --------------------------------------------------------------- render
    def _render(self) -> None:
        if self._series is None:
            return
        s = self._series
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(s["times"], s["mean"], color="#E8622C", label=f"{s['label']} mean")
        ax.plot(s["times"], s["max"], color="#B91C1C", linewidth=0.8, label=f"{s['label']} max")
        for _name, a, _b in s["phases"]:
            ax.axvline(a, color="#888", linewidth=0.6, linestyle=":")
        if self._mode == "split" and self._split is not None:
            ax.axvline(self._split, color="#00E5FF", linewidth=1.4)
        else:
            if self._t0 is not None and self._t1 is not None:
                ax.axvspan(self._t0, self._t1, color="#00E5FF", alpha=0.15)
            for c in self._clicks:
                ax.axvline(c, color="#00E5FF", linewidth=1.0, linestyle="--")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel(f"{s['label']} ({s['unit']})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        fig.subplots_adjust(top=0.96, bottom=0.16, left=0.12, right=0.97)
        self.canvas.draw_idle()

    def _fmt(self, st: dict, unit: str) -> str:
        trend = "rising" if st["slope"] > 1e-6 else "falling" if st["slope"] < -1e-6 else "flat"
        return (f"mean {st['mean']:.1f} {unit} &nbsp; peak {st['peak']:.1f} {unit} &nbsp; "
                f"integral {st['integral']:.0f} {unit}·s &nbsp; trend {trend} "
                f"({st['slope']:+.2f} {unit}/s, net {st['delta']:+.1f} {unit})")

    def _compute(self) -> None:
        if self._series is None:
            return
        s = self._series
        unit = s["unit"]
        if self._mode == "split" and self._split is not None:
            before, after = tw.before_after_split(s["mean"], s["max"], s["times"], self._split)
            self.stats_label.setText(
                f"<b>Before {self._split:.1f} s</b> ({before['t0']:.1f}–{before['t1']:.1f} s): "
                f"{self._fmt(before, unit)}<br>"
                f"<b>After {self._split:.1f} s</b> ({after['t0']:.1f}–{after['t1']:.1f} s): "
                f"{self._fmt(after, unit)}")
            self._build_split_insights(before, after, unit)
        else:
            st = tw.interval_stats(s["mean"], s["max"], s["times"], self._t0, self._t1)
            self.stats_label.setText(
                f"<b>{st['t0']:.1f}–{st['t1']:.1f} s</b> ({st['n_frames']} frames): "
                f"{self._fmt(st, unit)}")
            self._build_window_insights(st, unit)

    def _build_window_insights(self, st: dict, unit: str) -> None:
        trend = "rose" if st["delta"] > 0 else "fell" if st["delta"] < 0 else "held"
        self.insights.set_insights([
            Insight(f"Over {st['t0']:.1f}–{st['t1']:.1f} s, {self._series['label'].lower()} "
                    f"{trend} by {abs(st['delta']):.1f} {unit} "
                    f"(mean {st['mean']:.1f}, peak {st['peak']:.1f} {unit}).",
                    category="query", quantity=self._key.quantity, time_s=(st["t0"], st["t1"]),
                    value=st["mean"], unit=unit,
                    basis="mean/peak over the window; net change of the spatial-mean series")])

    def _build_split_insights(self, before: dict, after: dict, unit: str) -> None:
        d = after["mean"] - before["mean"]
        direction = "higher" if d > 0 else "lower" if d < 0 else "unchanged"
        self.insights.set_insights([
            Insight(f"After {self._split:.1f} s the mean is {abs(d):.1f} {unit} {direction} "
                    f"({before['mean']:.1f} → {after['mean']:.1f} {unit}).",
                    category="query", quantity=self._key.quantity, time_s=self._split,
                    value=after["mean"], unit=unit,
                    basis="before/after split of the spatial-mean series at the chosen instant")])
