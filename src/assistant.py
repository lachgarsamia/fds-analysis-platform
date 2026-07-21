"""Safe Assistant engine (V4-M12), strictly bounded.

This is NOT a generative model. It is a deterministic template engine that
*organizes computed evidence* -- the Evidence Notebook, the session, saved
measurements, and interval statistics -- into summaries, finding lists,
report outlines, comparisons, and figure captions. Every output is filled
purely from numbers that were already computed with a traceable basis.

Hard rule (roadmap 3.12), enforced here in code:
  physics conclusions come only from computed evidence with a `basis`.
The assistant never asserts a physical cause. A free-text request is
routed only to one of SAFE_ACTIONS; anything phrased as "why / what caused"
is refused (interpret_request -> "refuse"), so there is no path by which a
causal or invented claim can be produced.

Pure, Qt-free, deterministic.
"""

from __future__ import annotations

DISCLAIMER = ("Generated deterministically from computed values and saved "
              "evidence; it organizes results and infers no physical causes.")

REFUSAL = ("I can only organize computed evidence and generate deterministic "
           "summaries, comparisons, and captions from values that were already "
           "measured. I cannot infer why something happened or assert a physical "
           "cause -- those would need computed evidence with a stated basis. Try "
           "\"summarize the session\", \"list key findings\", or \"figure caption\".")

# The only things the assistant can do. Each maps a free-text intent to a
# deterministic action; there is deliberately no open-ended path.
SAFE_ACTIONS = (
    "summarize_session",
    "list_key_findings",
    "report_outline",
    "compare_intervals",
    "figure_caption",
)

_CAUSAL_WORDS = ("why", "cause", "caused", "because", "reason for", "explain why",
                 "what makes", "responsible for")

_INTENT_KEYWORDS = {
    "summarize_session": ("summarize", "summary", "overview", "recap"),
    "list_key_findings": ("finding", "findings", "key result", "results"),
    "report_outline": ("outline", "organize", "sections", "structure the report"),
    "compare_intervals": ("compare interval", "compare the interval", "before and after",
                          "compare intervals", "interval comparison"),
    "figure_caption": ("caption", "figure caption", "describe the view"),
}


def interpret_request(text: str) -> str:
    """Map a free-text request to a SAFE_ACTIONS name, or 'refuse'. A causal
    phrasing is refused *unless* it also clearly names a safe action, so the
    assistant can never be steered into asserting a cause."""
    t = (text or "").strip().lower()
    if not t:
        return "refuse"
    matched = next((action for action, kws in _INTENT_KEYWORDS.items()
                    if any(k in t for k in kws)), None)
    if any(w in t for w in _CAUSAL_WORDS) and matched is None:
        return "refuse"
    return matched or "refuse"


def _with_disclaimer(body: str) -> str:
    return f"{body}\n\n— {DISCLAIMER}"


def _fmt_time(t):
    return f"t = {float(t):.1f} s" if isinstance(t, (int, float)) else None


def summarize_session(session: dict) -> str:
    """A traceable overview of a saved/collected session: its identity and
    the counts + a few statements from its computed evidence."""
    notebook = session.get("notebook", []) or []
    zones = session.get("zones", []) or []
    measurements = session.get("measurements", []) or []
    tw = session.get("time_window", {}) or {}
    lines = [f"Session: {session.get('name') or '(unnamed)'}"]
    if session.get("intent"):
        lines.append(f"Intent: {session['intent']}")
    lines.append(f"Contents: {len(notebook)} finding(s), {len(zones)} zone(s), "
                 f"{len(measurements)} measurement(s).")
    if tw.get("t0") is not None and tw.get("t1") is not None:
        lines.append(f"Time window: {float(tw['t0']):.1f}–{float(tw['t1']):.1f} s.")
    if notebook:
        lines.append("Top findings:")
        for entry in notebook[:3]:
            ins = entry.get("insight", {}) or {}
            when = _fmt_time(ins.get("time_s"))
            prefix = f"  · {when}: " if when else "  · "
            lines.append(prefix + str(ins.get("statement", "")))
    return _with_disclaimer("\n".join(lines))


def list_key_findings(notebook: list) -> str:
    """The Evidence Notebook's findings as a numbered, report-ready list --
    verbatim computed statements, with their time, note, and basis."""
    if not notebook:
        return _with_disclaimer("No findings saved yet. Right-click any "
                                "measurement and choose \"Save to Evidence Notebook\".")
    lines = ["Key findings:"]
    for i, entry in enumerate(notebook, 1):
        ins = entry.get("insight", {}) or {}
        when = _fmt_time(ins.get("time_s"))
        line = f"{i}. " + (f"[{when}] " if when else "") + str(ins.get("statement", ""))
        if entry.get("note"):
            line += f" (note: {entry['note']})"
        lines.append(line)
        if ins.get("basis"):
            lines.append(f"     basis: {ins['basis']}")
    return _with_disclaimer("\n".join(lines))


def report_outline(notebook: list) -> str:
    """Organize the findings into report sections, grouped by tag (untagged
    findings fall under 'General'). Ordering only -- no new claims."""
    if not notebook:
        return _with_disclaimer("No findings to organize yet.")
    sections: dict = {}
    for entry in notebook:
        tags = entry.get("tags") or ["General"]
        statement = str((entry.get("insight", {}) or {}).get("statement", ""))
        for tag in tags:
            sections.setdefault(tag, []).append(statement)
    lines = ["Suggested report outline:"]
    for tag in sorted(sections):
        lines.append(f"\n## {tag}")
        for s in sections[tag]:
            lines.append(f"  - {s}")
    return _with_disclaimer("\n".join(lines))


def compare_intervals(stats_a: dict, stats_b: dict, label_a: str, label_b: str,
                      unit: str = "") -> str:
    """Describe the difference between two interval-statistics dicts
    (time_window.interval_stats output). States the numbers and their
    delta; attributes no cause."""
    u = f" {unit}" if unit else ""
    d_mean = stats_b.get("mean", 0.0) - stats_a.get("mean", 0.0)
    direction = "higher" if d_mean > 0 else "lower" if d_mean < 0 else "equal"
    lines = [
        f"Interval comparison ({label_a} vs {label_b}):",
        f"  {label_a} ({stats_a.get('t0', 0):.1f}–{stats_a.get('t1', 0):.1f} s): "
        f"mean {stats_a.get('mean', 0):.1f}{u}, peak {stats_a.get('peak', 0):.1f}{u}.",
        f"  {label_b} ({stats_b.get('t0', 0):.1f}–{stats_b.get('t1', 0):.1f} s): "
        f"mean {stats_b.get('mean', 0):.1f}{u}, peak {stats_b.get('peak', 0):.1f}{u}.",
        f"  {label_b}'s mean is {abs(d_mean):.1f}{u} {direction} than {label_a}'s.",
    ]
    return _with_disclaimer("\n".join(lines))


def figure_caption(scenario: str, quantity_label: str, unit: str, time_s: float,
                   peak_value: float) -> str:
    """A publication-ready caption for the current view, built from its
    computed peak. Descriptive only."""
    body = (f"{quantity_label} field for scenario {scenario} at t = {time_s:.1f} s "
            f"(peak {peak_value:.1f} {unit}).")
    return _with_disclaimer(body)
