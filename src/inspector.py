"""Live Inspector (FireLab roadmap Phase 3): a right-hand panel on the
Live page showing a large-type cursor-probe readout, a peak-temperature
sparkline scrubbed in sync with TimeController, an HRR gauge, and a
deterministic live-narration line (auto_summary.narrate_frame).

Pure presentation, same split as schematic.py/views.py: state arrives via
setters MainWindow calls, nothing here fetches data itself.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from auto_summary import narrate_frame

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
        painter.setPen(QtGui.QPen(QtGui.QColor(self._color), 1.5))
        painter.drawPath(path)

        idx = min(self._index, n - 1)
        marker = point(idx, self._series[idx])
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(self._color))
        painter.drawEllipse(marker, 3, 3)


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

        self.probe_label = QtWidgets.QLabel("Hover the plot to inspect a point.")
        self.probe_label.setProperty("role", "value")
        self.probe_label.setWordWrap(True)
        layout.addWidget(self.probe_label)

        layout.addWidget(QtWidgets.QLabel("Peak temperature over time"))
        self.sparkline = _Sparkline()
        layout.addWidget(self.sparkline)

        layout.addWidget(QtWidgets.QLabel("Heat release rate"))
        self.hrr_gauge = QtWidgets.QProgressBar()
        self.hrr_gauge.setRange(0, 100)
        self.hrr_gauge.setTextVisible(False)
        layout.addWidget(self.hrr_gauge)

        self.narration_label = QtWidgets.QLabel("")
        self.narration_label.setWordWrap(True)
        self.narration_label.setProperty("role", "value")
        layout.addWidget(self.narration_label)

        layout.addStretch(1)

        self._series: list = []
        self._ambient_c = AMBIENT_DEFAULT_C
        self._door_wide_open = True

    def set_palette(self, palette) -> None:
        self.sparkline.set_color(palette.accent)

    def set_probe(self, x, z, value, unit: str = "") -> None:
        if x is None:
            self.probe_label.setText("Hover the plot to inspect a point.")
            return
        value_text = f"{value:.1f}{unit}" if value is not None else "—"
        self.probe_label.setText(f"x = {x:.3f} m\nz = {z:.3f} m\nvalue = {value_text}")

    def set_scenario(self, peak_temp_by_frame: list, ambient_c: float, door_wide_open: bool) -> None:
        """Called on scenario/quantity change -- resets the sparkline
        series and the narration's static context (ambient/door)."""
        self._series = list(peak_temp_by_frame)
        self._ambient_c = ambient_c
        self._door_wide_open = door_wide_open
        self.sparkline.set_series(self._series)

    def set_time(self, index: int, hrr_fraction: float = None) -> None:
        """Called every playback tick: scrubs the sparkline marker, updates
        the HRR gauge (0-100%, None leaves it at its last value -- no HRR
        data available for this scenario), and regenerates the narration
        line from already-computed numbers."""
        self.sparkline.set_index(index)
        if hrr_fraction is not None:
            self.hrr_gauge.setValue(int(round(max(0.0, min(1.0, hrr_fraction)) * 100)))

        if not self._series:
            self.narration_label.setText("")
            return
        idx = min(index, len(self._series) - 1)
        current_temp_c = self._series[idx]
        peak_temp_c = max(self._series)
        self.narration_label.setText(
            narrate_frame(current_temp_c, peak_temp_c, self._ambient_c, self._door_wide_open)
        )

    def clear(self) -> None:
        """Neutral state for a cell type/quantity the inspector doesn't
        narrate (e.g. a difference/ensemble cell, or Air speed) -- keeps
        the probe readout (still meaningful) but blanks the rest rather
        than showing stale or misleading numbers."""
        self._series = []
        self.sparkline.set_series([])
        self.hrr_gauge.setValue(0)
        self.narration_label.setText("")
