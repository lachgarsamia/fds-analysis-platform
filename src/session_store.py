"""Named analysis-session repository (V4-M6).

V2/V4 sessions are a JSON snapshot of an investigation (session.py builds
the state dict: grid + view config + Evidence Notebook + zones + time
window + browser filters). This module turns single files into a *named,
browsable library*: many sessions per data run, each with a name, an
intent, and metadata (created / modified / author / data fingerprint), so
an investigation can be saved, listed, previewed, reloaded exactly, and
exported to a report.

Pure logic (slugify, fingerprint, conflict detection) is separated from
the small filesystem layer so it is deterministic and unit-testable. No
new physics -- serialization and state management only.
"""

from __future__ import annotations

import getpass
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from session import read_session, write_session

DRAFT_SLUG = "__draft__"


# --------------------------------------------------------------- pure helpers
def now_iso() -> str:
    """Current UTC time as a sortable ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name: str) -> str:
    """Filesystem-safe slug for a session name; never empty."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "session"


def data_fingerprint(manifest: list) -> str:
    """A short, stable id for the underlying data run: the sorted scenario
    folder names hashed. Lets a session record which run it belongs to and
    a load warn on a mismatch (version pinning)."""
    folders = sorted(getattr(e, "folder", str(e)) for e in (manifest or []))
    digest = hashlib.sha1("\n".join(folders).encode("utf-8")).hexdigest()
    return f"{len(folders)}:{digest[:12]}"


def make_metadata(author: str = None, data_version: str = "", created: str = None,
                  modified: str = None) -> dict:
    ts = now_iso()
    return {
        "author": author if author is not None else _current_user(),
        "data_version": data_version,
        "created": created or ts,
        "modified": modified or ts,
        "app": "fdsvisualizer",
    }


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def detect_conflict(baseline_modified: str, disk_modified: str) -> bool:
    """True when the on-disk session was modified after the baseline we
    started from -- i.e. another save happened underneath us. ISO strings
    compare lexically. Missing values never conflict."""
    if not baseline_modified or not disk_modified:
        return False
    return disk_modified > baseline_modified


@dataclass
class SessionInfo:
    path: str
    name: str
    intent: str
    created: str
    modified: str
    data_version: str
    n_notebook: int
    n_zones: int
    is_draft: bool

    def preview(self) -> str:
        parts = [self.intent or "(no description)",
                 f"{self.n_notebook} note(s), {self.n_zones} zone(s)"]
        if self.modified:
            parts.append(f"modified {self.modified}")
        return "  ·  ".join(parts)


def session_info_from_dict(path: str, session: dict) -> SessionInfo:
    meta = session.get("metadata", {}) or {}
    name = session.get("name", "") or os.path.splitext(os.path.basename(path))[0]
    return SessionInfo(
        path=path, name=name, intent=session.get("intent", ""),
        created=meta.get("created", ""), modified=meta.get("modified", ""),
        data_version=meta.get("data_version", ""),
        n_notebook=len(session.get("notebook", []) or []),
        n_zones=len(session.get("zones", []) or []),
        is_draft=os.path.basename(path).startswith(DRAFT_SLUG))


# ----------------------------------------------------------- filesystem layer
def default_sessions_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".fdsvis", "sessions")


def _unique_path(directory: str, slug: str) -> str:
    base = os.path.join(directory, slug)
    path = base + ".json"
    i = 2
    while os.path.exists(path):
        path = f"{base}-{i}.json"
        i += 1
    return path


def save_session(directory: str, session: dict, path: str = None) -> str:
    """Write a session to `directory` (or overwrite `path`). Stamps
    metadata.modified. Returns the file path."""
    os.makedirs(directory, exist_ok=True)
    meta = dict(session.get("metadata", {}) or {})
    meta["modified"] = now_iso()
    meta.setdefault("created", meta["modified"])
    session = dict(session, metadata=meta)
    if path is None:
        path = _unique_path(directory, slugify(session.get("name", "session")))
    write_session(path, session)
    return path


def save_draft(directory: str, session: dict) -> str:
    """Overwrite the single autosave draft file."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, DRAFT_SLUG + ".json")
    session = dict(session, name=session.get("name") or "(autosaved draft)")
    return save_session(directory, session, path=path)


def load_session(path: str) -> dict:
    """Read a session file (reuses session.read_session's validation)."""
    return read_session(path)


def delete_session(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def list_sessions(directory: str) -> list:
    """Every readable session in `directory` as SessionInfo, newest first;
    unreadable files are skipped, not fatal."""
    infos = []
    for path in glob.glob(os.path.join(directory, "*.json")):
        try:
            infos.append(session_info_from_dict(path, read_session(path)))
        except ValueError:
            continue
    infos.sort(key=lambda i: (i.modified or "", i.name), reverse=True)
    return infos
