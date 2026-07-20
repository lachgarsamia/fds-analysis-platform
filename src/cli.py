"""Headless CLI / Python API (V2 roadmap M3.4, feature F12).

Batch stats / figure export / report generation / session rendering
against a study directory, with no GUI -- the layered data pipeline
(data_provider, summary_stats, figure_export, report_builder,
auto_summary) is all importable and runnable without a QApplication, so
this is lab-automation glue over the same code the app uses, not a
parallel implementation.

Subcommands:
  stats           print/write the summary-stats index (JSON or CSV)
  export          render one scenario's field to a publication figure
  report          build a per-scenario or A-vs-B HTML report
  session-render  render a saved session's cells to images

Run as `python -m cli ...` (with src on the path) or, after
`pip install -e .`, as the `fdsvis-cli` console command.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

# Defensive: matplotlib/PyQt import paths must never require a display in
# a headless run. This never creates a QApplication.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from data_provider import load_study, DataLoadError
from slice_key import SliceKey, DEFAULT_SLICE_KEY, SOOT_QUANTITY
from config import QUANTITY_DISPLAY, ISOTHERM_LEVELS
from summary_stats import build_summary_index
from auto_summary import generate_summary


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def resolve_case_index(manifest: list, scenario: str) -> int | None:
    """A scenario argument is either a folder name or an integer
    case_index."""
    for e in manifest:
        if e.folder == scenario:
            return e.case_index
    try:
        idx = int(scenario)
    except ValueError:
        return None
    return idx if any(e.case_index == idx for e in manifest) else None


def resolve_quantity_key(quantity: str) -> SliceKey:
    """Quantity name -> SliceKey. SOOT resolves to the y=0 side plane
    (the CLI doesn't expose per-plane selection); everything else is a
    plain `.sf` slice."""
    if quantity == SOOT_QUANTITY:
        return SliceKey(SOOT_QUANTITY, 1, 0, 0.0)
    return SliceKey(quantity)


def _study_summaries(sim_data, study_root):
    cache_path = os.path.join(study_root, ".cache", "summaries.json")
    return build_summary_index(sim_data.manifest, sim_data.store,
                               sim_data.timesteps_per_second, cache_path)


def _peak_frame(summary) -> int:
    return int(np.argmax(summary.max_temp_by_frame_c)) if summary.max_temp_by_frame_c else 0


# ------------------------------------------------------------------ stats
def cmd_stats(args) -> int:
    sim_data = load_study(args.study)
    if not sim_data.manifest:
        return _fail("study has no scenarios")
    summaries = _study_summaries(sim_data, args.study)
    from dataclasses import asdict
    rows = [asdict(s) for s in summaries]

    if args.format == "json":
        text = json.dumps(rows, indent=2)
    else:
        buf = io.StringIO()
        fields = [k for k in rows[0].keys() if k != "max_temp_by_frame_c"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        text = buf.getvalue()

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {len(summaries)} scenario summaries to {args.output}")
    else:
        print(text)
    return 0


# ----------------------------------------------------------------- export
def cmd_export(args) -> int:
    from figure_export import export_publication_figure, provenance_line
    sim_data = load_study(args.study)
    case_index = resolve_case_index(sim_data.manifest, args.scenario)
    if case_index is None:
        return _fail(f"no scenario matching {args.scenario!r}")
    entry = next(e for e in sim_data.manifest if e.case_index == case_index)
    key = resolve_quantity_key(args.quantity)
    display = QUANTITY_DISPLAY[key.quantity]
    data = np.asarray(sim_data.store.get(case_index, key))

    if args.frame is not None:
        frame_idx = min(max(args.frame, 0), data.shape[0] - 1)
    else:
        summary = next(s for s in _study_summaries(sim_data, args.study) if s.case_index == case_index)
        frame_idx = min(_peak_frame(summary), data.shape[0] - 1)

    extent = sim_data.store.get_extent(case_index, key)
    provenance = provenance_line(entry.path, entry.folder, frame_idx / sim_data.timesteps_per_second)
    export_publication_figure(
        np.asarray(data[frame_idx]), args.output, cmap=display['cmap'], vmin=display['vmin'],
        vmax=display['slider_default'], extent=extent,
        colorbar_label=f"{display['label']} ({display['unit']})", title=entry.folder,
        isotherm_levels=ISOTHERM_LEVELS.get(key.quantity), provenance=provenance)
    print(f"wrote figure for {entry.folder} (frame {frame_idx}) to {args.output}")
    return 0


# ----------------------------------------------------------------- report
def cmd_report(args) -> int:
    from figure_export import figure_png_bytes, provenance_line
    from report_builder import build_scenario_report, build_comparison_report, write_report
    sim_data = load_study(args.study)
    summaries = _study_summaries(sim_data, args.study)
    fps = sim_data.timesteps_per_second
    key = DEFAULT_SLICE_KEY
    display = QUANTITY_DISPLAY[key.quantity]

    ca = resolve_case_index(sim_data.manifest, args.scenario)
    if ca is None:
        return _fail(f"no scenario matching {args.scenario!r}")
    entry_a = next(e for e in sim_data.manifest if e.case_index == ca)
    summary_a = next(s for s in summaries if s.case_index == ca)

    if args.vs is None:
        data = np.asarray(sim_data.store.get(ca, key))
        peak = min(_peak_frame(summary_a), data.shape[0] - 1)
        png = figure_png_bytes(
            np.asarray(data[peak]), cmap=display['cmap'], vmin=display['vmin'],
            vmax=display['slider_default'], extent=sim_data.store.get_extent(ca, key),
            colorbar_label=f"{display['label']} ({display['unit']})", title=entry_a.folder,
            isotherm_levels=ISOTHERM_LEVELS.get(key.quantity))
        prov = provenance_line(entry_a.path, entry_a.folder, peak / fps)
        text = generate_summary(entry_a, summary_a, summaries, sim_data.store, fps, key)
        html = build_scenario_report(entry_a, summary_a, text, png, prov)
    else:
        cb = resolve_case_index(sim_data.manifest, args.vs)
        if cb is None:
            return _fail(f"no scenario matching {args.vs!r}")
        entry_b = next(e for e in sim_data.manifest if e.case_index == cb)
        summary_b = next(s for s in summaries if s.case_index == cb)
        data_a = np.asarray(sim_data.store.get(ca, key))
        data_b = np.asarray(sim_data.store.get(cb, key))
        peak = min(_peak_frame(summary_a), data_a.shape[0] - 1, data_b.shape[0] - 1)
        diff = np.asarray(data_a[peak]) - np.asarray(data_b[peak])
        vmax = float(np.max(np.abs(diff))) or 1.0
        png = figure_png_bytes(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               extent=sim_data.store.get_extent(ca, key),
                               colorbar_label=f"Δ{display['label']} ({display['unit']})",
                               title=f"{entry_a.folder} − {entry_b.folder}")
        prov_a = provenance_line(entry_a.path, entry_a.folder, peak / fps)
        prov_b = provenance_line(entry_b.path, entry_b.folder, peak / fps)
        text_a = generate_summary(entry_a, summary_a, summaries, sim_data.store, fps, key)
        text_b = generate_summary(entry_b, summary_b, summaries, sim_data.store, fps, key)
        html = build_comparison_report(entry_a, entry_b, summary_a, summary_b,
                                       text_a, text_b, png, prov_a, prov_b)

    write_report(args.output, html)
    print(f"wrote report to {args.output}")
    return 0


# --------------------------------------------------------- session-render
def _cell_frame(store, cell: dict, key: SliceKey, index: int):
    """The (frame, cmap, symmetric) tuple to render for one session cell."""
    ctype = cell.get("cell_type", "slice")
    if ctype == "difference":
        a = np.asarray(store.get(cell["case_index_a"], key))
        b = np.asarray(store.get(cell["case_index_b"], key))
        i = min(index, a.shape[0] - 1, b.shape[0] - 1)
        return a[i] - b[i], "RdBu_r", True
    if ctype == "ensemble":
        cases = cell.get("ensemble_case_indices", [])
        if not cases:
            return None, None, False
        arrays = [np.asarray(store.get(c, key)) for c in cases]
        i = min(index, min(a.shape[0] for a in arrays) - 1)
        stat = cell.get("ensemble_stat", "mean")
        stacked = np.stack([a[i] for a in arrays], axis=0)
        return getattr(np, stat)(stacked, axis=0), None, False
    a = np.asarray(store.get(cell["case_index"], key))
    return a[min(index, a.shape[0] - 1)], None, False


def cmd_session_render(args) -> int:
    from figure_export import export_publication_figure
    from session import read_session
    sim_data = load_study(args.study)
    try:
        session = read_session(args.session)
    except ValueError as e:
        return _fail(str(e))
    os.makedirs(args.output_dir, exist_ok=True)
    index = session.get("time_index", 0)

    written = 0
    for i, cell in enumerate(session.get("cells", [])):
        key = resolve_quantity_key(cell.get("quantity", DEFAULT_SLICE_KEY.quantity))
        display = QUANTITY_DISPLAY[key.quantity]
        frame, cmap_override, symmetric = _cell_frame(sim_data.store, cell, key, index)
        if frame is None:
            continue
        case_ref = cell.get("case_index", cell.get("case_index_a", 0))
        extent = sim_data.store.get_extent(case_ref, key)
        if symmetric:
            vmax = float(np.max(np.abs(frame))) or 1.0
            vmin, vmax, cmap = -vmax, vmax, cmap_override
        else:
            vmin, vmax, cmap = display['vmin'], display['slider_default'], cmap_override or display['cmap']
        out = os.path.join(args.output_dir, f"cell_{i}_{cell.get('cell_type', 'slice')}.png")
        export_publication_figure(
            np.asarray(frame), out, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
            colorbar_label=f"{display['label']} ({display['unit']})")
        written += 1
    print(f"rendered {written} cell(s) to {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdsvis-cli", description="Headless FDS study tools (V2 M3.4).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="print/write the summary-stats index")
    p.add_argument("study", help="FDS study directory")
    p.add_argument("--format", choices=("json", "csv"), default="csv")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="render a scenario field to a publication figure")
    p.add_argument("study")
    p.add_argument("--scenario", required=True, help="folder name or case index")
    p.add_argument("--quantity", default=DEFAULT_SLICE_KEY.quantity)
    p.add_argument("--frame", type=int, default=None, help="frame index (default: peak-temperature frame)")
    p.add_argument("-o", "--output", required=True, help="figure path (.png/.svg/.pdf)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("report", help="build an HTML report")
    p.add_argument("study")
    p.add_argument("--scenario", required=True)
    p.add_argument("--vs", default=None, help="second scenario for an A-vs-B comparison")
    p.add_argument("-o", "--output", required=True, help="report path (.html)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("session-render", help="render a saved session's cells to images")
    p.add_argument("session", help="session .json file")
    p.add_argument("study", help="FDS study directory the session refers to")
    p.add_argument("-o", "--output-dir", default="session_render")
    p.set_defaults(func=cmd_session_render)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataLoadError as e:
        return _fail(f"{e.message} {e.technical_detail}".strip())


if __name__ == "__main__":
    sys.exit(main())
