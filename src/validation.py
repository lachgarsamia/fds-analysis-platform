"""Validation toolkit (V6, GATED — interface only, not implemented).

Compare a simulation to *experimental* measurements: overlay thermocouple /
sensor curves, compute RMSE and arrival-time errors, and threshold-crossing
differences for a publication validation table.

This is prepared but **not implemented**: the study ships no experimental
dataset, so there is nothing to validate against and no honest way to compute
these numbers. The signatures below name the interface the V6 panel will call;
each raises `ValidationGate` until an experimental dataset is provided. See
ROADMAP-V6.md and docs/msim-preparation.md.
"""

from __future__ import annotations

GATE = ("Validation needs an experimental dataset (thermocouples / sensors) to "
        "compare against; none is bundled with this study. Prepared for V6.")


class ValidationGate(NotImplementedError):
    """Raised by every validation entry point until experimental data exists."""


def load_experimental_series(path: str):
    """(times, values, sensor_metadata) from an experimental measurement file.
    GATED — no experimental data is available."""
    raise ValidationGate(GATE)


def rmse(sim_series, exp_series):
    """Root-mean-square error between an aligned sim and experimental series.
    GATED — requires experimental data."""
    raise ValidationGate(GATE)


def arrival_time_error(sim_series, exp_series, threshold, fps):
    """Difference in the time each series first crosses `threshold`.
    GATED — requires experimental data."""
    raise ValidationGate(GATE)


def validation_table(sim, experiment):
    """Assemble the publication validation table (RMSE, arrival-time and
    threshold errors per sensor). GATED — requires experimental data."""
    raise ValidationGate(GATE)
