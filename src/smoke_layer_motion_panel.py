"""Smoke-Layer Motion panel (Analysis roadmap follow-up): a position-
velocity *teaching analogy* grounded entirely in real simulation data.
The UI itself never calls the bottom panel "velocity" -- this dataset
already has a real, distinct VELOCITY quantity (registry.py), and reusing
that word for an interface-descent speed would collide with it. The
bottom panel is labeled "descent rate" throughout.

The only FireScope quantity that's literally kinematic is smoke-layer
height (layer_height.smoke_layer_height_series): a real height in meters
that descends over time. This panel shows that height (the "position")
locked to its own time-derivative, the descent rate (the "velocity" of
the teaching analogy, never the UI label) -- np.gradient(height) * fps,
the same rate-from-position idiom devices.py already uses for a
thermocouple's heating rate. One scrub cursor (the frame_slider, the same
control every other Analysis panel uses) drives both panels together.

Reuses layer_height.py's existing computation verbatim -- this panel adds
no new physics, only a new view over an existing one. TEMPERATURE is
never gated (see tenability.fed_heat_dose's own docstring), so the
GatedQuantityError path below is defensive, not one expected to fire on
this dataset today: it exists so a future gated scenario fails with an
honest empty state instead of a guessed curve, per this widget's own
non-negotiable.

Sign-convention note (cleanup pass): summary_stats.py separately reports
`smoke_descent_rate_m_s`, a positive-only *scalar* -- the single worst
frame-to-frame drop across a run, clamped >=0, used as one of several
severity-style response fields in study_analytics.py's factor-effects
comparisons (peak temp, HRR, hazard duration, etc. are all positive-
magnitude there too). This panel's `rate` is deliberately signed and
full-series instead (negative = descending, positive = rising) -- it
carries the rise/fall information the scalar's `max(0.0, ...)` clamp
discards, which is the whole point of a position/velocity view. The two
are related (summary_stats.py's scalar is, in spirit, `-min(rate)` over
a run, though computed via np.diff rather than np.gradient -- see that
module's own comment), not contradictory: one is a full curve for
exploration, the other a single ranking statistic. Not unified into one
sign convention on purpose -- study_analytics.py's callers assume a
positive magnitude (see its own pinned test), so flipping the scalar's
sign would ripple into an unrelated comparison feature for no benefit;
documenting the relationship at both sites instead.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from widgets import MplCanvas, plot_fg_color
from registry import AMBIENT_C
from slice_key import SliceKey
from quantity_provider import GatedQuantityError
from layer_height import smoke_layer_height_series
from analysis_panel_base import populate_scenario_combo

_CAPTION = (
    "Smoke-layer height (top) and its descent rate (bottom), locked to one "
    "time cursor. Height is the domain-mean excess-temperature half-integral "
    "from the ceiling down -- a documented simplification of the rigorous "
    "two-zone (Cooper) method, not a directly measured interface "
    "(layer_height.py). Descent rate is the real time-derivative of that "
    "same height curve, nothing else -- negative means the layer is "
    "descending."
)

_NO_DATA_CAPTION = "Smoke-layer height is not available for this scenario."


def _presmoke_frame_count(height: np.ndarray, ceiling_z: float, tol: float = 1e-9) -> int:
    """Count of *leading* frames layer_height.py itself reports as "no
    distinguishable smoke yet" -- height held exactly at the ceiling
    default (z1) because the excess-temperature integral never crossed
    that module's own _MIN_INTEGRAL floor (see smoke_layer_height_series'
    docstring). Only the leading run counts: a later return to the
    ceiling value means smoke genuinely cleared -- a real event, not a
    pre-smoke artifact -- and stays fully plotted."""
    n = 0
    for h in height:
        if abs(float(h) - ceiling_z) <= tol:
            n += 1
        else:
            break
    return n


class SmokeLayerMotionPanel(QtWidgets.QWidget):
    def __init__(self, provider, manifest: list, fps: int, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._manifest = sorted(manifest, key=lambda e: e.case_index)
        self._fps = max(1, fps)
        self._loaded = False
        self._cache = {}       # case_index -> {"height":..., "rate":...} or None
        self._series = None    # the current scenario's cache entry

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Smoke-layer motion")
        title.setProperty("role", "section-title")
        header.addWidget(title)
        header.addStretch(1)
        self.scenario_combo = QtWidgets.QComboBox()
        self.scenario_combo.setAccessibleName("Smoke-layer motion scenario")
        header.addWidget(self.scenario_combo)
        layout.addLayout(header)

        self.caption = QtWidgets.QLabel(_CAPTION)
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.caption)

        self.canvas = MplCanvas(self)
        self.canvas.setAccessibleName("Smoke-layer height and descent rate")
        layout.addWidget(self.canvas, 1)

        frame_row = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("t = 0.0 s")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setAccessibleName("Smoke-layer motion frame")
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        layout.addLayout(frame_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("role", "value")
        layout.addWidget(self.status_label)

        self.scenario_combo.currentIndexChanged.connect(self._reload)
        self.frame_slider.valueChanged.connect(self._on_frame)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded or not self._manifest:
            return
        self._loaded = True
        self.scenario_combo.blockSignals(True)
        populate_scenario_combo(self.scenario_combo, self._manifest)
        self.scenario_combo.blockSignals(False)
        self._reload()

    def _compute(self, case_index: int):
        """height/rate for `case_index`, or None if honestly unavailable --
        never a guessed curve."""
        try:
            data = np.asarray(self._provider.get(case_index, SliceKey("TEMPERATURE")))
            extent = self._provider.get_extent(case_index, SliceKey("TEMPERATURE"))
        except GatedQuantityError:
            return None
        if extent is None or data.shape[0] < 2:
            return None
        height = smoke_layer_height_series(data, extent, AMBIENT_C)
        # Same rate-from-position idiom devices.py's compute_thermocouple
        # already uses for heating rate (np.gradient(temp) * fps) -- the
        # real time-derivative of a real measured series, not a separate
        # model. Left exactly as-is, unmasked, unmodified -- the display
        # logic in _render() decides what to plot, this is still the real
        # signed derivative of the real height series.
        rate = np.gradient(height) * self._fps
        presmoke_n = _presmoke_frame_count(height, float(extent[3]))
        return {"height": height, "rate": rate, "presmoke_n": presmoke_n}

    def _reload(self) -> None:
        if not self._loaded:
            return
        case_index = self.scenario_combo.currentData()
        if case_index is None:
            return
        if case_index not in self._cache:
            self._cache[case_index] = self._compute(case_index)
        self._series = self._cache[case_index]
        n = len(self._series["height"]) if self._series is not None else 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(n - 1, 0))
        if self.frame_slider.value() >= n:
            self.frame_slider.setValue(int(n * 0.6))
        self.frame_slider.blockSignals(False)
        self._render()

    def _on_frame(self, _v) -> None:
        self.frame_label.setText(f"t = {self.frame_slider.value() / self._fps:.1f} s")
        self._render()

    def _render(self) -> None:
        fig = self.canvas.fig
        fig.clear()
        if self._series is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, _NO_DATA_CAPTION, ha="center", va="center",
                    fontsize=9, wrap=True, color=plot_fg_color())
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            self.status_label.setText("")
            self.caption.setText(_NO_DATA_CAPTION)
            return
        self.caption.setText(_CAPTION)

        height = self._series["height"]
        rate = self._series["rate"]
        presmoke_n = self._series["presmoke_n"]
        n = len(height)
        idx = min(self.frame_slider.value(), n - 1)
        times = np.arange(n) / self._fps
        t_cursor = idx / self._fps

        # np.gradient's boundary handling turns the pre-smoke -> real-smoke
        # transition into an artificially huge spike -- not a real descent
        # event: a one-sided difference at the leading edge (index
        # presmoke_n-1, referencing only the ceiling default before it),
        # then a centered difference one frame later (index presmoke_n)
        # that still partly references that same ceiling value. Both
        # indices are masked from the *displayed* rate line and shaded on
        # both panels; `rate` itself (used everywhere else, incl. the
        # cursor scatter math below) stays the complete, unmodified
        # np.gradient(height)*fps array -- nothing about the underlying
        # derivative changes, only what's drawn as trustworthy signal.
        mask_end = min(presmoke_n, n - 1) if presmoke_n >= 1 else -1
        rate_display = rate.copy()
        if mask_end >= 0:
            rate_display[: mask_end + 1] = np.nan

        # Two locked, time-aligned panels sharing one x-axis (sharex) --
        # position on top, descent rate on bottom, same cursor drawn on both.
        ax_h = fig.add_subplot(211)
        ax_h.plot(times, height, color="#2563EB", linewidth=1.4)
        ax_h.axvline(t_cursor, color="#00E5FF", linewidth=1.2)
        ax_h.scatter([t_cursor], [height[idx]], color="#2563EB", zorder=5, s=26)
        ax_h.set_ylabel("layer height (m)", fontsize=8)
        ax_h.set_title("Smoke-layer height (position)", fontsize=9, fontweight="bold")
        ax_h.tick_params(labelsize=7, labelbottom=False)

        ax_r = fig.add_subplot(212, sharex=ax_h)
        ax_r.plot(times, rate_display, color="#E8622C", linewidth=1.4)
        ax_r.axhline(0.0, color="#888888", linewidth=0.6)
        ax_r.axvline(t_cursor, color="#00E5FF", linewidth=1.2)
        if mask_end < 0 or idx > mask_end:
            ax_r.scatter([t_cursor], [rate[idx]], color="#E8622C", zorder=5, s=26)
        ax_r.set_xlabel("time (s)", fontsize=8)
        ax_r.set_ylabel("descent rate (m/s)", fontsize=8)
        ax_r.set_title("Smoke-layer descent rate — negative = descending", fontsize=9)
        ax_r.tick_params(labelsize=7)

        if mask_end >= 0:
            t_mask_end = mask_end / self._fps
            for ax in (ax_h, ax_r):
                ax.axvspan(0.0, t_mask_end, color="#94A3B8", alpha=0.25, zorder=0)
            ax_r.text(t_mask_end / 2, 0.5, "no smoke\ndetected yet",
                      transform=ax_r.get_xaxis_transform(), ha="center", va="center",
                      fontsize=7, color="#64748B")

        fig.subplots_adjust(top=0.92, bottom=0.12, left=0.15, right=0.97, hspace=0.35)
        self.canvas.draw_idle()

        if mask_end >= 0 and idx <= mask_end:
            self.status_label.setText(
                f"t = {t_cursor:.1f} s  ·  height {height[idx]:.3f} m  ·  "
                f"no smoke detected yet (rate not meaningful here)")
        else:
            direction = "descending" if rate[idx] < -1e-6 else ("rising" if rate[idx] > 1e-6 else "steady")
            self.status_label.setText(
                f"t = {t_cursor:.1f} s  ·  height {height[idx]:.3f} m  ·  "
                f"rate {rate[idx]:+.4f} m/s ({direction})")
