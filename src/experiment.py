"""Experiment model + repository (V4-M9).

An `Experiment` organizes a batch of related scenarios into one named,
tagged study: a description, a list of scenarios (by stable folder name),
a designated baseline, and shared parameters for a parametric sweep. It is
the organizational layer above individual runs -- create an experiment,
group scenarios, check their availability, and hand a baseline-vs-other
pair straight into the Advanced Comparison workflow (V4-M8).

Honesty note: this application organizes *pre-computed* cluster runs (the
M-SIM data gate, docs/msim-preparation.md). There is no in-app FDS
solver, so "run/prepare" here means checking and loading a scenario's
existing data, not launching a simulation -- experiment_status reports
each scenario as `ready` (data present) or `missing`, never a fabricated
run state.

Pure model + a small JSON repository (reusing session_store's slug/time
helpers). Qt-free.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List

from session_store import slugify, now_iso, make_metadata


@dataclass
class Experiment:
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)   # stable folder names
    baseline: str = ""                                   # folder name, or ""
    shared_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": "experiment",
            "name": self.name, "description": self.description,
            "tags": list(self.tags), "scenarios": list(self.scenarios),
            "baseline": self.baseline, "shared_params": dict(self.shared_params),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        return cls(
            name=str(d.get("name", "")), description=str(d.get("description", "")),
            tags=[str(t) for t in d.get("tags", [])],
            scenarios=[str(s) for s in d.get("scenarios", [])],
            baseline=str(d.get("baseline", "")),
            shared_params=dict(d.get("shared_params", {}) or {}),
            metadata=dict(d.get("metadata", {}) or {}))


def experiment_status(exp: Experiment, available_folders) -> dict:
    """Per-scenario availability + completion for the status dashboard.
    `available_folders` is the set of folder names the store can load."""
    avail = set(available_folders or [])
    statuses = {s: ("ready" if s in avail else "missing") for s in exp.scenarios}
    ready = sum(1 for v in statuses.values() if v == "ready")
    total = len(exp.scenarios)
    return {
        "statuses": statuses,
        "ready": ready,
        "missing": total - ready,
        "total": total,
        "completion": (ready / total) if total else 0.0,
    }


# ------------------------------------------------------------- repository
@dataclass
class ExperimentInfo:
    path: str
    name: str
    description: str
    tags: List[str]
    n_scenarios: int
    baseline: str
    modified: str

    def preview(self) -> str:
        parts = [self.description or "(no description)",
                 f"{self.n_scenarios} scenario(s)"]
        if self.baseline:
            parts.append(f"baseline {self.baseline}")
        if self.tags:
            parts.append("tags: " + ", ".join(self.tags))
        return "  ·  ".join(parts)


def default_experiments_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".fdsvis", "experiments")


def _read(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("kind") != "experiment":
        raise ValueError("not an experiment file")
    return data


def save_experiment(directory: str, exp: Experiment, path: str = None) -> str:
    os.makedirs(directory, exist_ok=True)
    meta = dict(exp.metadata or {})
    if not meta:
        meta = make_metadata()
    meta["modified"] = now_iso()
    meta.setdefault("created", meta["modified"])
    exp.metadata = meta
    if path is None:
        base = os.path.join(directory, slugify(exp.name or "experiment"))
        path, i = base + ".json", 2
        while os.path.exists(path):
            path = f"{base}-{i}.json"
            i += 1
    with open(path, "w") as f:
        json.dump(exp.to_dict(), f, indent=2)
    return path


def load_experiment(path: str) -> Experiment:
    return Experiment.from_dict(_read(path))


def delete_experiment(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def list_experiments(directory: str) -> list:
    infos = []
    for path in glob.glob(os.path.join(directory, "*.json")):
        try:
            d = _read(path)
        except (ValueError, OSError):
            continue
        infos.append(ExperimentInfo(
            path=path, name=d.get("name", ""), description=d.get("description", ""),
            tags=[str(t) for t in d.get("tags", [])],
            n_scenarios=len(d.get("scenarios", []) or []),
            baseline=d.get("baseline", ""),
            modified=(d.get("metadata", {}) or {}).get("modified", "")))
    infos.sort(key=lambda i: (i.modified or "", i.name), reverse=True)
    return infos
