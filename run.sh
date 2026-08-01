#!/bin/bash
# Resilient launcher for the FDS SLCF Visualizer.
#
# Why this exists: on this machine, PyQt5's Cocoa platform-plugin init
# repeatedly failed to start ("Could not find the Qt platform plugin
# cocoa"). Root-caused: the project (and its old .venv) lived under
# ~/Desktop, which has iCloud "Desktop & Documents" sync enabled. Qt's own
# QDir::entryList() -- used to enumerate the Qt5/plugins/platforms
# directory at startup -- silently returned zero entries there (confirmed
# directly: Python's os.listdir() saw all 4 platform .dylib files fine at
# the exact same path, QDir saw none; the identical QDir call worked
# instantly once tested against a copy outside ~/Desktop). That's why it
# looked "intermittent, byte-identical env, one launch works one doesn't"
# -- it tracked iCloud's local materialization state for that folder, not
# a broken install. The venv now lives at ~/.venvs/fds_visualizer,
# outside any iCloud-synced folder, which fixes this at the root.
#
# A second, genuinely separate failure mode was also seen once: the
# installed PyQt5/PyQt5-Qt5/PyQt5-sip trio drifting out of a mutually
# consistent state (pip resolving a newer PyQt5-Qt5 than the bindings
# were built against). Plain relaunching never fixes that one -- only
# reinstalling the matched, pinned trio does -- so that recovery path
# stays below as a fallback even though the Desktop/iCloud issue was the
# actual cause of most crashes seen this session.
#
# Both failure modes abort inside Qt's C++ layer via qFatal() before
# Python ever gets a chance to catch anything, so neither can be fixed by
# retrying *within* one process -- only by relaunching or reinstalling
# then relaunching the process from the outside.

set -u
cd "$(dirname "$0")"

VENV="$HOME/.venvs/fds_visualizer"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
MAX_ATTEMPTS=4
STARTUP_WINDOW_S=6   # a crash within this many seconds of launch is treated as a startup-only failure, not a real runtime issue
PYQT_VERSION="5.15.11"
PYQT_SIP_VERSION="12.18.0"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found -- expected a venv at $VENV (deliberately" >&2
    echo "outside ~/Desktop/iCloud sync, see comment above). Create it with:" >&2
    echo "  python3 -m venv $VENV && $PIP install -e $(dirname "$0")" >&2
    exit 1
fi

try_launch_once() {
    start_ts=$(date +%s)
    PYTHONPATH=src "$PYTHON" src/main.py "$@"
    code=$?
    elapsed=$(( $(date +%s) - start_ts ))
    return_code=$code
    return_elapsed=$elapsed
}

reinstall_pyqt() {
    echo "[run.sh] Every plain retry failed at startup -- that's the signature of a" >&2
    echo "[run.sh] mismatched PyQt5/PyQt5-Qt5/PyQt5-sip install, not the transient race." >&2
    echo "[run.sh] Reinstalling the matched, pinned trio once..." >&2
    "$PIP" uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip >&2
    "$PIP" install --no-cache-dir \
        "PyQt5==${PYQT_VERSION}" "PyQt5-Qt5==${PYQT_VERSION}" "PyQt5-sip==${PYQT_SIP_VERSION}" >&2
}

run_attempts() {
    local n="$1"
    shift
    local attempt=1
    while [ "$attempt" -le "$n" ]; do
        if [ "$attempt" -gt 1 ]; then
            echo "[run.sh] Startup race hit (attempt $((attempt - 1))/$n) -- relaunching..."
        fi
        try_launch_once "$@"
        if [ "$return_code" -eq 0 ]; then
            return 0
        fi
        if [ "$return_elapsed" -ge "$STARTUP_WINDOW_S" ]; then
            # Ran for a while before exiting -- either the user closed the
            # window normally or a real (non-startup) error occurred.
            # Retrying (or reinstalling) won't help and would be wrong.
            return "$return_code"
        fi
        attempt=$((attempt + 1))
    done
    return "$return_code"
}

run_attempts "$MAX_ATTEMPTS" "$@"
code=$?
if [ "$code" -eq 0 ] || [ "$return_elapsed" -ge "$STARTUP_WINDOW_S" ]; then
    exit "$code"
fi

reinstall_pyqt
run_attempts "$MAX_ATTEMPTS" "$@"
code=$?
if [ "$code" -eq 0 ] || [ "$return_elapsed" -ge "$STARTUP_WINDOW_S" ]; then
    exit "$code"
fi

echo "[run.sh] Still failing at startup after retries AND a clean PyQt5 reinstall --" >&2
echo "[run.sh] this is something new. Check the output above." >&2
exit 1
