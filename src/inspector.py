"""Live Inspector (FireLab roadmap Phase 3; static/dynamic split added in
the scientific-visualization completion pass): a right-hand panel on the
Live page showing a large-type cursor-probe readout, a peak-temperature
sparkline scrubbed in sync with TimeController, an HRR gauge, and a
deterministic live-narration line (auto_summary.narrate_frame).

Split into two groups so the panel reads as stable during playback
instead of "continuously refreshing": static metadata (scenario,
quantity, grid size, slice location, duration, frame count) only changes
when MainWindow's `key != self._inspector_series_key` guard fires (a
scenario/quantity switch), via set_static_info(); dynamic info (frame,
time, min/max, probe) changes every tick via set_time()/set_probe(), but
only as text mutations on pre-built QLabels -- nothing here is ever
recreated mid-playback.

Pure presentation, same split as schematic.py/views.py: state arrives via
setters MainWindow calls, nothing here fetches data itself.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from auto_summary import narrate_frame
from insight import InsightList

AMBIENT_DEFAULT_C = 20.0


class _Sparkline(QtWidgets.QWidget):
    """A minimal line-plot + moving marker -- custom-painted rather than a
    second matplotlib canvas, since this is a small, low-data-rate widget
    (one line, one marker) where a full MplCanvas would be overkill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._series: list = []
        self._index = 0
        self._color = "#4C8DFF"

    def set_series(self, series: list) -> None:
        self._series = list(series)
        self._index = 0
        self.update()

    def set_index(self, index: int) -> None:
        self._index = index
        self.update()

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QtGui.QPainter) -> None:
        rect = self.rect().adjusted(2, 2, -2, -2)
        if not self._series or rect.width() <= 0 or rect.height() <= 0:
            return
        lo, hi = min(self._series), max(self._series)
        span = max(hi - lo, 1e-6)
        n = len(self._series)

        def point(i: int, value: float) -> QtCore.QPointF:
            x = rect.left() + (i / max(n - 1, 1)) * rect.width()
            y = rect.bottom() - ((value - lo) / span) * rect.height()
            return QtCore.QPointF(x, y)

        path = QtGui.QPainterPath()
        path.moveTo(point(0, self._series[0]))
        for i, v in enumerate(self._series[1:], start=1):
            path.lineTo(point(i, v))

        # Subtle filled area under the curve for readability.
        area = QtGui.QPainterPath(path)
        area.lineTo(point(n - 1, self._series[-1]).x(), rect.bottom())
        area.lineTo(rect.left(), rect.bottom())
        area.closeSubpath()
        fill = QtGui.QColor(self._color)
        fill.setAlpha(38)
        painter.fillPath(area, fill)

        painter.setPen(QtGui.QPen(QtGui.QColor(self._color), 1.5))
        painter.drawPath(path)

        # Peak marker (hollow) -- shows where the hottest moment sits
        # relative to the current playback time.
        peak_i = max(range(n), key=lambda i: self._series[i])
        peak_pt = point(peak_i, self._series[peak_i])
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor("#E8622C"), 1.4))
        painter.drawEllipse(peak_pt, 3.2, 3.2)

        # Moving playback cursor: a vertical line at the current time plus a
        # filled bullet on the curve (user feedback -- follows playback).
        idx = min(self._index, n - 1)
        marker = point(idx, self._series[idx])
        painter.setPen(QtGui.QPen(QtGui.QColor(self._color), 1.0, QtCore.Qt.DashLine))
        painter.drawLine(QtCore.QPointF(marker.x(), rect.top()),
                         QtCore.QPointF(marker.x(), rect.bottom()))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(self._color))
        painter.drawEllipse(marker, 3.5, 3.5)


class InspectorPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("inspectorPanel")
        self.setMinimumWidth(200)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Inspector")
        title.setProperty("role", "title")
        layout.addWidget(title)

        # --- Static metadata: set once per scenario/quantity change (see
        # MainWindow._update_inspector's key != self._inspector_series_key
        # guard), never touched per playback tick. One pre-built QLabel per
        # field, text-only updates -- never recreated. ---------------------
        static_caption = QtWidgets.QLabel("Scenario")
        static_caption.setProperty("role", "caption")
        layout.addWidget(static_caption)

        self._static_labels: dict = {}
        static_grid = QtWidgets.QFormLayout()
        static_grid.setContentsMargins(0, 0, 0, 0)
        static_grid.setSpacing(4)
        static_grid.setLabelAlignment(QtCore.Qt.AlignLeft)
        for field in ("Scenario", "Quantity", "Grid size", "Slice", "Duration", "Frames"):
            key_label = QtWidgets.QLabel(f"{field}:")
            key_label.setProperty("role", "caption")
            value_label = QtWidgets.QLabel("—")
            value_label.setProperty("role", "value")
            value_label.setWordWrap(True)
            self._static_labels[field] = value_label
            static_grid.addRow(key_label, value_label)
        layout.addLayout(static_grid)

        layout.addWidget(_divider())

        # --- Dynamic: updated every playback tick, text-only. ---------------
        dynamic_caption = QtWidgets.QLabel("Live")
        dynamic_caption.setProperty("role", "caption")
        layout.addWidget(dynamic_caption)

        self.frame_label = QtWidgets.QLabel("—")
        self.frame_label.setProperty("role", "value")
        layout.addWidget(self.frame_label)

        self.range_label = QtWidgets.QLabel("—")
        self.range_label.setProperty("role", "value")
        layout.addWidget(self.range_label)

        # Difference cells only (A-B) -- hidden/blank for a plain slice
        # cell, never recreated, just shown/hidden + text-updated.
        self.diff_stats_caption = QtWidgets.QLabel("Difference statistics (A − B)")
        self.diff_stats_caption.setProperty("role", "caption")
        self.diff_stats_caption.setVisible(False)
        layout.addWidget(self.diff_stats_caption)

        self.diff_stats_label = QtWidgets.QLabel("")
        self.diff_stats_label.setProperty("role", "value")
        self.diff_stats_label.setWordWrap(True)
        self.diff_stats_label.setVisible(False)
        layout.addWidget(self.diff_stats_label)

        # V2 roadmap M1.5: the stats above are one frame's snapshot; this
        # button opens a curve of RMS/max|delta| across the whole
        # timeline, showing *when* two scenarios diverge, not just by how
        # much right now.
        self.diff_plot_button = QtWidgets.QPushButton("Plot difference over time…")
        self.diff_plot_button.setVisible(False)
        layout.addWidget(self.diff_plot_button)

        self.probe_label = QtWidgets.QLabel("Hover the plot to inspect a point.")
        self.probe_label.setProperty("role", "value")
        self.probe_label.setWordWrap(True)
        layout.addWidget(self.probe_label)

        self._sparkline_caption = QtWidgets.QLabel("Peak temperature over time")
        self._sparkline_caption.setProperty("role", "caption")
        layout.addWidget(self._sparkline_caption)
        self.sparkline = _Sparkline()
        layout.addWidget(self.sparkline)

        self._hrr_caption = QtWidgets.QLabel("Heat release rate")
        self._hrr_caption.setProperty("role", "caption")
        layout.addWidget(self._hrr_caption)
        self.hrr_gauge = QtWidgets.QProgressBar()
        self.hrr_gauge.setRange(0, 100)
        self.hrr_gauge.setTextVisible(True)
        self.hrr_gauge.setFormat("%p% of peak")
        layout.addWidget(self.hrr_gauge)
        # RC polish: an explicit HRR state so the gauge is never a silent grey
        # bar -- it either reads a value or says why it can't.
        self.hrr_state_label = QtWidgets.QLabel("")
        self.hrr_state_label.setWordWrap(True)
        self.hrr_state_label.setProperty("role", "caption")
        layout.addWidget(self.hrr_state_label)

        self.narration_label = QtWidgets.QLabel("")
        self.narration_label.setWordWrap(True)
        self.narration_label.setProperty("role", "value")
        layout.addWidget(self.narration_label)

        # Fire story (V3-M2): the scenario's detected events as a clickable
        # list, plus a live "current phase" line during playback.
        self._story_caption = QtWidgets.QLabel("Fire story")
        self._story_caption.setProperty("role", "caption")
        layout.addWidget(self._story_caption)
        self.phase_label = QtWidgets.QLabel("")
        self.phase_label.setWordWrap(True)
        self.phase_label.setProperty("role", "value")
        layout.addWidget(self.phase_label)
        self.story_list = InsightList()
        self.story_list.setMaximumHeight(150)
        layout.addWidget(self.story_list)

        # V6 polish: full detail by default (unchanged single-cell Live
        # Viewer behavior) -- InspectorStack switches a section to compact
        # only when 2+ scenarios are being compared (see its ensure_count),
        # since that's the case that needs each section's essentials
        # (Scenario/Quantity/Live readout) to fit without heavy scrolling.
        # The sparkline/HRR/narration/Fire story keep computing and
        # updating underneath either way (set_compact only hides widgets,
        # never stops feeding them), so switching back to full shows
        # current data immediately, not stale placeholders.
        self._compact = False

        layout.addStretch(1)

        self._events: list = []
        self._events_fps = 1

        self._series: list = []
        self._ambient_c = AMBIENT_DEFAULT_C
        self._door_wide_open = True
        self._n_frames = 0
        self._fps = 1.0

    def set_palette(self, palette) -> None:
        self.sparkline.set_color(palette.accent)

    def set_compact(self, compact: bool) -> None:
        """Show/hide the HRR/narration/Fire story block, keeping
        Scenario/Quantity/Grid size/Slice/Duration/Frames + the Live
        frame/min-max/probe readout + the peak-temperature sparkline (kept
        visible even when compact -- comparing scenarios still wants that
        curve, just not the HRR gauge/narration/Fire story). Hiding (not
        removing) these widgets means they keep updating from underneath
        -- toggling back to non-compact shows current data immediately, no
        stale placeholder."""
        self._compact = compact
        expanded = not compact
        self._sparkline_caption.setVisible(True)
        self.sparkline.setVisible(True)
        for w in (self._hrr_caption, self.hrr_gauge, self.hrr_state_label,
                 self.narration_label, self._story_caption, self.phase_label, self.story_list):
            w.setVisible(expanded)

    def set_static_info(self, scenario: str, quantity: str, grid_size: str,
                        slice_location: str, n_frames: int, fps: float) -> None:
        """Called only on scenario/quantity change (see module docstring)
        -- never per tick. Duration is derived from n_frames/fps rather
        than taking a precomputed string, so this stays the single source
        of truth set_time()'s frame counter also uses."""
        self._n_frames = n_frames
        self._fps = max(fps, 1e-6)
        duration_s = (n_frames - 1) / self._fps if n_frames > 0 else 0.0
        self._static_labels["Scenario"].setText(scenario)
        self._static_labels["Quantity"].setText(quantity)
        self._static_labels["Grid size"].setText(grid_size)
        self._static_labels["Slice"].setText(slice_location)
        self._static_labels["Duration"].setText(f"{duration_s:.1f} s")
        self._static_labels["Frames"].setText(str(n_frames))

    def set_difference_stats(self, min_v: float, max_v: float, mean_v: float, rms_v: float,
                              unit: str = "") -> None:
        """Called every tick a difference cell is active -- min_v/max_v/
        mean_v/rms_v are computed by the caller from the same A-B array
        already fetched for display, never re-read here."""
        self.diff_stats_caption.setVisible(True)
        self.diff_stats_label.setVisible(True)
        self.diff_plot_button.setVisible(True)
        self.diff_stats_label.setText(
            f"min = {min_v:.1f}{unit}   max = {max_v:.1f}{unit}\n"
            f"mean = {mean_v:.1f}{unit}   RMS = {rms_v:.1f}{unit}"
        )

    def clear_difference_stats(self) -> None:
        self.diff_stats_caption.setVisible(False)
        self.diff_stats_label.setVisible(False)
        self.diff_plot_button.setVisible(False)
        self.diff_stats_label.setText("")

    def set_probe(self, x, z, value, unit: str = "") -> None:
        if x is None:
            self.probe_label.setText("Hover the plot to inspect a point.")
            return
        value_text = f"{value:.1f}{unit}" if value is not None else "—"
        self.probe_label.setText(f"x = {x:.3f} m\nz = {z:.3f} m\nvalue = {value_text}")

    def set_story(self, events: list, fps: int) -> None:
        """Populate the Fire story list with a scenario's detected events
        (V3-M2), computed from TEMPERATURE regardless of which quantity the
        cell currently displays. `events` are insight.Insight objects from
        events.py; None or [] clears the list (e.g. a difference/ensemble
        cell, which has no single scenario to narrate, or a scenario where
        no events were detected)."""
        self._events = list(events or [])
        self._events_fps = max(1, fps)
        self.story_list.set_insights(self._events)
        # RC polish: an explicit empty-state instead of a blank grey list.
        self.phase_label.setText(
            "" if self._events else "No fire events detected for this scenario.")

    def set_story_index(self, index: int) -> None:
        """Update the live "current phase" line to the most recent event at
        or before the current frame."""
        if not self._events:
            self.phase_label.setText("")
            return
        current = None
        for ev in self._events:
            fi = ev.frame_index(self._events_fps)
            if fi is not None and fi <= index:
                current = ev
        self.phase_label.setText(f"Now: {current.statement}" if current is not None
                                 else "Now: before ignition")

    def set_scenario(self, peak_temp_by_frame: list, ambient_c: float, door_wide_open: bool) -> None:
        """Called on scenario/quantity change -- resets the sparkline
        series and the narration's static context (ambient/door)."""
        self._series = list(peak_temp_by_frame)
        self._ambient_c = ambient_c
        self._door_wide_open = door_wide_open
        self.sparkline.set_series(self._series)

    def set_time(self, index: int, hrr_fraction: float = None,
                 frame_min: float = None, frame_max: float = None, unit: str = "") -> None:
        """Called every playback tick: scrubs the sparkline marker, updates
        the HRR gauge (0-100%, None leaves it at its last value -- no HRR
        data available for this scenario), regenerates the narration line
        from already-computed numbers, and refreshes the frame/time/min-
        max readout -- frame_min/frame_max are the caller's already-
        computed current-frame extremes (this widget never touches the
        data array itself)."""
        self.sparkline.set_index(index)
        if hrr_fraction is not None:
            self.hrr_gauge.setValue(int(round(max(0.0, min(1.0, hrr_fraction)) * 100)))
            self.hrr_gauge.setFormat("%p% of peak")
            self.hrr_state_label.setText("")
        else:
            self.hrr_gauge.setValue(0)
            self.hrr_gauge.setFormat("no data")
            self.hrr_state_label.setText(
                "No heat-release-rate output for this scenario (no _hrr.csv).")

        t_now = index / self._fps if self._fps else 0.0
        self.frame_label.setText(f"Frame {index + 1} / {max(self._n_frames, 1)}   ·   t = {t_now:.1f} s")
        if frame_min is not None and frame_max is not None:
            self.range_label.setText(f"min = {frame_min:.1f}{unit}   max = {frame_max:.1f}{unit}")
        else:
            self.range_label.setText("—")

        if not self._series:
            self.narration_label.setText("")
            return
        idx = min(index, len(self._series) - 1)
        current_temp_c = self._series[idx]
        peak_temp_c = max(self._series)
        self.narration_label.setText(
            narrate_frame(current_temp_c, peak_temp_c, self._ambient_c, self._door_wide_open)
        )
        self.set_story_index(index)

    def clear(self) -> None:
        """Neutral state for a cell type/quantity the inspector doesn't
        narrate (e.g. a difference/ensemble cell, or Air speed) -- keeps
        the probe readout (still meaningful) but blanks the rest rather
        than showing stale or misleading numbers."""
        self._series = []
        self.sparkline.set_series([])
        self.hrr_gauge.setValue(0)
        self.narration_label.setText("")


def _divider() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setObjectName("divider")
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFixedHeight(1)
    return line


class InspectorStack(QtWidgets.QWidget):
    """One InspectorPanel per visible grid cell, stacked vertically in a
    scroll area (V6 polish: comparing 2+ cells needs each cell's own full
    detail, not just whichever one is "active" -- the single-InspectorPanel
    design couldn't show more than one scenario's stats at a time). Grows
    lazily and hides extra sections rather than destroying them, same
    convention as GridLayout's _grow_to/_place_cells, so switching layouts
    back and forth never loses a section's state. Section 0 is never
    hidden/destroyed, so callers that only ever cared about "the" inspector
    (single-cell layouts) can keep holding a direct reference to it."""

    section_added = QtCore.pyqtSignal(int, object)  # index, new InspectorPanel

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        scroll.setWidget(content)

        self._sections: list = []   # [(label QLabel, InspectorPanel), ...]

    def _add_section(self) -> None:
        label = QtWidgets.QLabel()
        label.setProperty("role", "section-title")
        panel = InspectorPanel()
        insert_at = self._layout.count() - 1   # before the trailing stretch
        self._layout.insertWidget(insert_at, label)
        self._layout.insertWidget(insert_at + 1, panel)
        self._sections.append((label, panel))
        self.section_added.emit(len(self._sections) - 1, panel)

    def ensure_count(self, n: int) -> None:
        """Grow to at least `n` sections if needed, then show exactly the
        first `n` (hiding, never destroying, any extra). Per-section "Cell
        N" labels only show once there's more than one. A single visible
        section (the Live Viewer's normal 1x1 case) always gets full detail
        -- comparing 2+ scenarios switches every visible section to
        InspectorPanel.set_compact so they fit without heavy scrolling."""
        n = max(1, n)
        while len(self._sections) < n:
            self._add_section()
        for i, (label, panel) in enumerate(self._sections):
            visible = i < n
            panel.setVisible(visible)
            label.setVisible(visible and n > 1)
            if visible:
                label.setText(f"Cell {i + 1}")
                panel.set_compact(n > 1)

    def section(self, i: int) -> InspectorPanel:
        return self._sections[i][1]

    def count(self) -> int:
        return len(self._sections)

    def set_palette(self, palette) -> None:
        for _label, panel in self._sections:
            panel.set_palette(palette)
