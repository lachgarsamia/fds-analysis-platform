"""Launches the Junior Fire Scientist kids app from FireScope's own
"Back" nav-rail button (see nav.py's module comment on that button), or
activates it if it's already running rather than spawning a duplicate.

Mirror of Junior Fire Scientist's own public/firescope_launcher.py --
that module launches FireScope from the kids app's Grown-ups button (and
activates it instead of relaunching if it's already running); this one
does the reverse, so the two repositories stay independent in both
directions with no duplicate-window buildup either way.

How "already running" is known without any real IPC: when the kids app
launches FireScope, it passes its own pid via the JUNIOR_FIRE_SCIENTIST_PID
environment variable (see that repo's firescope_launcher.py). This
module reads it back, checks the pid is still alive, and if so activates
its window (macOS only, via System Events/osascript) instead of starting
a second kids-app process. If FireScope wasn't launched that way at all
(no env var -- e.g. started standalone during development) or the
activation fails, Back falls back to a fresh launch, same as before.

Locating the kids app: JUNIOR_FIRE_SCIENTIST_APP_PATH (if set) points
directly at its checkout's root; otherwise this assumes the conventional
local layout, a sibling directory next to this repo's own root (both
under the same parent, e.g. ~/Desktop/fds_visualizer kids/ next to
~/Desktop/FireScope). Interpreter: JUNIOR_FIRE_SCIENTIST_PYTHON (if
set), otherwise the same anaconda interpreter FireScope's own dev
workflow already runs on (both repos' .venv lack working Qt/offscreen
support -- see FireScope's own established restart convention)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

_DEFAULT_SIBLING_NAME = "fds_visualizer kids"
_DEFAULT_PYTHON = "/opt/anaconda3/bin/python"


def find_kids_app_root() -> Optional[Path]:
    """Junior Fire Scientist repo root, or None if it can't be located.
    Checked in order: JUNIOR_FIRE_SCIENTIST_APP_PATH env var, then the
    conventional sibling-directory layout. Verified by the presence of
    src/main.py, not just the directory, so a stale/half-moved checkout
    is reported as missing rather than launched broken."""
    override = os.environ.get("JUNIOR_FIRE_SCIENTIST_APP_PATH")
    if override:
        candidate = Path(override).expanduser()
        return candidate if (candidate / "src" / "main.py").is_file() else None

    this_repo_root = Path(__file__).resolve().parent.parent
    candidate = this_repo_root.parent / _DEFAULT_SIBLING_NAME
    return candidate if (candidate / "src" / "main.py").is_file() else None


def pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists. Signal 0 probes
    existence/permission without actually sending a signal -- the
    standard portable way to check a pid without owning it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    except OSError:
        return False
    return True


def activate_pid(pid: int) -> bool:
    """Bring pid's window(s) to the front. macOS only (System Events via
    osascript -- the standard no-extra-dependency way to activate
    another process's window by pid, no PyObjC/extra package needed);
    returns False elsewhere or on any failure, so the caller can fall
    back to a fresh launch instead of silently doing nothing."""
    if sys.platform != "darwin":
        return False
    script = (
        f'tell application "System Events" to set frontmost of '
        f'(first process whose unix id is {pid}) to true'
    )
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def find_running_kids_app_pid() -> Optional[int]:
    """The kids app's pid, if this FireScope process was launched by it
    (JUNIOR_FIRE_SCIENTIST_PID, set by that repo's firescope_launcher.py)
    and that process is still alive. None if FireScope wasn't launched
    that way, the env var is malformed, or the process has since exited
    -- callers should launch a fresh one in every None case."""
    raw = os.environ.get("JUNIOR_FIRE_SCIENTIST_PID")
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid_alive(pid) else None


def launch_kids_app() -> Tuple[Optional[subprocess.Popen], str]:
    """Best-effort launch of the kids app, landing on its Welcome page
    (--welcome), as its own detached process (start_new_session=True: it
    must keep running as its own process group after this one closes,
    the same as launching any standalone sibling app). Returns
    (process, message) -- process is the launched Popen on success (None
    on failure), message is empty on success or a short, honest
    explanation on failure for the caller to show the user, never
    swallowed."""
    root = find_kids_app_root()
    if root is None:
        return None, (
            "Junior Fire Scientist isn't installed where this launcher "
            "expects it. Set the JUNIOR_FIRE_SCIENTIST_APP_PATH "
            "environment variable to its checkout, or place it as a "
            f"sibling folder named '{_DEFAULT_SIBLING_NAME}'."
        )

    python = os.environ.get("JUNIOR_FIRE_SCIENTIST_PYTHON", _DEFAULT_PYTHON)
    if not Path(python).is_file():
        python = sys.executable

    src_dir = root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    try:
        process = subprocess.Popen(
            [python, "main.py", "--welcome"],
            cwd=str(src_dir),
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return None, f"Couldn't start Junior Fire Scientist: {exc}"
    return process, ""
