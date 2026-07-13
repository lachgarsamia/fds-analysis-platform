"""Deterministic per-scenario auto-summaries (M3.1.3).

Every number in a generated summary comes from summary_stats.ScenarioSummary
(already computed for the experiment browser, M2.5) or a direct array lookup
via ScenarioStore -- nothing here is inferred or generated text, only a
fixed sentence template filled in with computed values, per the DoD's
explicit "all numbers computed, none generated."
"""

from __future__ import annotations

import numpy as np

from slice_key import DEFAULT_SLICE_KEY

# Landmark x-zones (meters) a peak-temperature pixel is classified against
# for the "(near ...)" spatial descriptor. Derived from the real .fds
# geometry (candle burners at x=[0.84,0.96], door at x=[0.25,0.29] --
# confirmed via M2.3's investigation, re-verified here against all 24
# scenarios' actual peak locations before shipping this module: every one
# falls in the candle zone, none in the door zone or elsewhere, so this
# isn't a hypothetical classifier, it's been checked against what the
# real dataset actually does).
LANDMARK_ZONES = (
    ("the candle", 0.75, 1.00),
    ("the door", 0.20, 0.35),
)

VOD_STATE_LABELS = {0: "vent-1-open", 1: "vent-1-closed", 2: "vent-1-HVAC"}


def _spatial_descriptor(store, case_index: int, quantity_key, peak_frame_index: int) -> str:
    """" (near the candle)" or similar, or "" if extent/data aren't
    available (demo mode) or the peak doesn't fall in any known zone."""
    try:
        extent = store.get_extent(case_index, quantity_key)
        data = np.asarray(store.get(case_index, quantity_key))
    except Exception:  # noqa: BLE001 - a missing descriptor must not break the summary
        return ""
    if extent is None or peak_frame_index >= data.shape[0]:
        return ""
    frame = data[peak_frame_index]
    row, col = np.unravel_index(np.argmax(frame), frame.shape)
    x0, x1, _z0, _z1 = extent
    n_x = frame.shape[1]
    if n_x <= 1 or x1 == x0:
        return ""
    x_phys = x0 + col / (n_x - 1) * (x1 - x0)
    for label, lo, hi in LANDMARK_ZONES:
        if lo <= x_phys <= hi:
            return f" (near {label})"
    return ""


def _vent_comparison_sentence(entry, own_summary, all_summaries: list) -> str:
    """"<other vent-1 state> variants peaked X°C <lower/higher> on
    average" -- compares this scenario's own peak against the mean peak
    of same-candle-count scenarios in the *other* vent-1 (VOD) state
    group (open vs. not-open), whichever this scenario isn't already in.
    Returns "" if there's no real comparison group (e.g. only one vod
    level present, or this is the only scenario at this candle count)."""
    same_candle = [s for s in all_summaries if s.candles == entry.candles]
    if entry.vod == 0:
        comparison_group = [s for s in same_candle if s.vod != 0]
        comparison_label = "closed/HVAC vent-1"
    else:
        comparison_group = [s for s in same_candle if s.vod == 0]
        comparison_label = VOD_STATE_LABELS[0]
    if not comparison_group:
        return ""

    own_mean = float(np.mean([s.max_temp_c for s in same_candle if s.vod == entry.vod]))
    comparison_mean = float(np.mean([s.max_temp_c for s in comparison_group]))
    diff = own_mean - comparison_mean
    # Capitalize only the leading letter -- str.capitalize() would also
    # lowercase the rest, turning the "HVAC" acronym into "hvac".
    label = comparison_label[0].upper() + comparison_label[1:]
    if abs(diff) < 0.5:
        return f" {label} variants peaked about the same on average."
    direction = "lower" if diff > 0 else "higher"
    return f" {label} variants peaked {abs(diff):.0f}°C {direction} on average."


def generate_summary(entry, summary, all_summaries: list, store,
                      fps: int, quantity_key=DEFAULT_SLICE_KEY) -> str:
    """entry: manifest.ScenarioEntry-shaped. summary: this scenario's
    summary_stats.ScenarioSummary. all_summaries: every scenario's
    ScenarioSummary (for the cross-scenario comparison sentence)."""
    peak_frame = int(np.argmax(summary.max_temp_by_frame_c)) if summary.max_temp_by_frame_c else 0
    peak_time_s = peak_frame / fps
    descriptor = _spatial_descriptor(store, entry.case_index, quantity_key, peak_frame)
    sentence1 = f"Peak {summary.max_temp_c:.0f}°C at t={peak_time_s:.0f}s{descriptor}."

    if summary.time_to_300c_s is not None:
        sentence2 = f" Exceeded 300°C at t={summary.time_to_300c_s:.0f}s."
    else:
        sentence2 = " Never exceeded 300°C."

    comparison_sentence = _vent_comparison_sentence(entry, summary, all_summaries)
    return sentence1 + sentence2 + comparison_sentence


def generate_all_summaries(entries: list, summaries: list, store, fps: int,
                            quantity_key=DEFAULT_SLICE_KEY) -> dict:
    """case_index -> summary text, for every scenario. Building all 24 at
    once (rather than one at a time) means the vent-comparison sentence's
    group means are computed consistently across the whole set."""
    by_case_entry = {e.case_index: e for e in entries}
    result = {}
    for summary in summaries:
        entry = by_case_entry.get(summary.case_index)
        if entry is None:
            continue
        result[summary.case_index] = generate_summary(entry, summary, summaries, store, fps, quantity_key)
    return result


def export_markdown(entries: list, summaries: list, store, fps: int, path: str,
                     quantity_key=DEFAULT_SLICE_KEY) -> None:
    """Writes one Markdown file with every scenario's auto-summary --
    the same generate_summary() output shown in the app, so the export
    can never drift from what a user already saw on screen."""
    texts = generate_all_summaries(entries, summaries, store, fps, quantity_key)
    by_case_entry = {e.case_index: e for e in entries}
    lines = ["# FDS Visualizer -- Scenario Auto-Summaries", ""]
    for case_index in sorted(texts):
        entry = by_case_entry[case_index]
        lines.append(f"## {entry.folder}")
        lines.append("")
        lines.append(texts[case_index])
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
