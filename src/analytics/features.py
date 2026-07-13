"""Per-scenario feature vectors for ensemble analytics (M3.1.1).

Pure NumPy over arrays ScenarioStore has already loaded/cached -- no new
I/O beyond what the app was already doing, and no randomness, so the same
scenario always produces the same feature vector (the DoD's "deterministic
-> snapshot tests" requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slice_key import DEFAULT_SLICE_KEY

# Fixed length every scenario's time-series curves are resampled to, so
# scenarios with slightly different frame counts (FDS's adaptive
# timestepping can produce a handful more/fewer than the nominal 481
# frames -- see M2.3's finding on this) still produce equal-length,
# directly comparable feature vectors for PCA/clustering.
CURVE_POINTS = 24

# "Hot" for the hot-area-fraction curve: the lowest M1.3s.4 hazard-band
# threshold (docs/spike-parser-validation.md §4) -- consistent with
# M2.6's isotherm default levels (config.ISOTHERM_LEVELS), same
# fire-safety reference points, not a new number invented for this module.
HOT_THRESHOLD_C = 60.0

# Sentinel for "never crossed this threshold" when building a numeric PCA
# vector (see as_vector()): the full simulated duration, since a scenario
# that never gets hot enough is meaningfully "slower/safer" than one that
# does, not a missing value to be dropped or zero-filled (zero would
# wrongly look like "crossed immediately").
NEVER_CROSSED_SECONDS = 999.0


def _downsample(curve: np.ndarray, n_points: int = CURVE_POINTS) -> list:
    """Resample a 1-D curve of any length to exactly n_points, evenly
    spaced across its own timeline -- np.interp handles both down- and
    up-sampling and any source length (including n_points itself)."""
    n = len(curve)
    if n == 0:
        return [0.0] * n_points
    if n == 1:
        return [float(curve[0])] * n_points
    src_x = np.linspace(0.0, 1.0, n)
    dst_x = np.linspace(0.0, 1.0, n_points)
    return [float(v) for v in np.interp(dst_x, src_x, curve)]


def _first_crossing_seconds(curve: np.ndarray, fps: int, threshold: float) -> float | None:
    hits = np.flatnonzero(curve > threshold)
    if hits.size == 0:
        return None
    return float(hits[0] / fps)


@dataclass(frozen=True)
class ScenarioFeatures:
    """One scenario's feature vector, plus the factor labels needed to
    color/marker a PCA scatter without a second lookup."""
    case_index: int
    folder: str
    candles: int
    door: int
    vod: int
    voc: int
    max_temp_curve: list = field(default_factory=list)          # CURVE_POINTS-length, °C
    hot_area_fraction_curve: list = field(default_factory=list)  # CURVE_POINTS-length, 0..1
    spatial_mean_curve: list = field(default_factory=list)       # CURVE_POINTS-length, °C
    time_to_100c_s: float | None = None
    time_to_300c_s: float | None = None
    time_to_600c_s: float | None = None

    def as_vector(self) -> np.ndarray:
        """Flat numeric vector for PCA/clustering. Time-to-threshold Nones
        (never crossed) become NEVER_CROSSED_SECONDS -- see that
        constant's docstring for why that's not a missing-data problem."""
        times = [
            NEVER_CROSSED_SECONDS if t is None else t
            for t in (self.time_to_100c_s, self.time_to_300c_s, self.time_to_600c_s)
        ]
        return np.array(
            self.max_temp_curve + self.hot_area_fraction_curve + self.spatial_mean_curve + times,
            dtype=float,
        )


def compute_scenario_features(entry, store, fps: int,
                               quantity_key=DEFAULT_SLICE_KEY,
                               hot_threshold_c: float = HOT_THRESHOLD_C) -> ScenarioFeatures:
    """entry: manifest.ScenarioEntry-shaped (duck-typed, same convention
    as views.py's EnsemblePickerDialog -- no manifest.py import needed
    here either). store: anything with ScenarioStore's .get() interface."""
    data = np.asarray(store.get(entry.case_index, quantity_key))
    max_by_frame = data.max(axis=(1, 2))
    hot_fraction_by_frame = (data > hot_threshold_c).mean(axis=(1, 2))
    mean_by_frame = data.mean(axis=(1, 2))

    return ScenarioFeatures(
        case_index=entry.case_index,
        folder=entry.folder,
        candles=entry.candles,
        door=entry.door,
        vod=entry.vod,
        voc=entry.voc,
        max_temp_curve=_downsample(max_by_frame),
        hot_area_fraction_curve=_downsample(hot_fraction_by_frame),
        spatial_mean_curve=_downsample(mean_by_frame),
        time_to_100c_s=_first_crossing_seconds(max_by_frame, fps, 100.0),
        time_to_300c_s=_first_crossing_seconds(max_by_frame, fps, 300.0),
        time_to_600c_s=_first_crossing_seconds(max_by_frame, fps, 600.0),
    )


def build_feature_matrix(features_list: list) -> tuple:
    """Stack ScenarioFeatures.as_vector() for every scenario into one
    (n_scenarios, n_features) matrix, case_index-ordered -- the shape
    sklearn's PCA/KMeans expect. Returns (matrix, case_indices) rather
    than a bare matrix so callers can map rows back to scenarios without
    assuming the input list's order."""
    ordered = sorted(features_list, key=lambda f: f.case_index)
    matrix = np.stack([f.as_vector() for f in ordered]) if ordered else np.zeros((0, 0))
    case_indices = [f.case_index for f in ordered]
    return matrix, case_indices


def build_feature_index(entries: list, store, fps: int,
                         quantity_key=DEFAULT_SLICE_KEY) -> list:
    """compute_scenario_features() for every manifest entry, case_index-
    ordered. No disk cache of its own (unlike summary_stats.py's
    build_summary_index) -- computing all 24 scenarios' features is cheap
    (numpy reductions over already-disk-cached arrays, no new I/O), so a
    second cache layer on top would be one more thing to invalidate for
    no measured benefit."""
    return [
        compute_scenario_features(entry, store, fps, quantity_key)
        for entry in sorted(entries, key=lambda e: e.case_index)
    ]
