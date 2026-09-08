#!/bin/bash
set -u
cd "$(dirname "$0")"

VENV="$HOME/.venvs/fds_visualizer"
PYTHON="$VENV/bin/python"
MAX_ATTEMPTS=4
STARTUP_WINDOW_S=6
PYQT_VERSION="5.15.11"
PYQT_SIP_VERSION="12.18.0"

for f in pyproject.toml src/main.py; do
    if [ ! -f "$f" ]; then
        echo "error: $f not found -- run this from the repository root." >&2
        exit 1
    fi
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found. Install Python 3 and retry." >&2
    exit 1
fi

# venv kept outside the repo: under an iCloud-synced folder Qt can't
# enumerate its Cocoa plugin dir and the app aborts at startup.
ARCH_PREFIX=""
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && command -v arch >/dev/null 2>&1; then
    ARCH_PREFIX="arch -arm64"
fi

if [ ! -x "$PYTHON" ]; then
    echo "Creating virtual environment at $VENV..."
    $ARCH_PREFIX python3 -m venv "$VENV" || exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "error: venv creation did not produce $PYTHON" >&2
    exit 1
fi

if ! "$PYTHON" -m pip show fdsvis >/dev/null 2>&1; then
    echo "Installing project dependencies..."
    $ARCH_PREFIX "$PYTHON" -m pip install -e . || exit 1
fi

try_launch_once() {
    start_ts=$(date +%s)
    PYTHONPATH=src $ARCH_PREFIX "$PYTHON" src/main.py "$@"
    code=$?
    elapsed=$(( $(date +%s) - start_ts ))
    return_code=$code
    return_elapsed=$elapsed
}

reinstall_pyqt() {
    echo "[run.sh] repeated startup failures -- reinstalling the pinned PyQt5 trio..." >&2
    $ARCH_PREFIX "$PYTHON" -m pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip >&2
    $ARCH_PREFIX "$PYTHON" -m pip install --no-cache-dir --force-reinstall \
        "PyQt5==${PYQT_VERSION}" "PyQt5-Qt5==${PYQT_VERSION}" "PyQt5-sip==${PYQT_SIP_VERSION}" >&2
}

run_attempts() {
    local n="$1"
    shift
    local attempt=1
    while [ "$attempt" -le "$n" ]; do
        if [ "$attempt" -gt 1 ]; then
            echo "[run.sh] startup failed (attempt $((attempt - 1))/$n) -- relaunching..."
        fi
        try_launch_once "$@"
        if [ "$return_code" -eq 0 ]; then
            return 0
        fi
        if [ "$return_elapsed" -ge "$STARTUP_WINDOW_S" ]; then
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

echo "[run.sh] still failing at startup after retries and a clean PyQt5 reinstall." >&2
exit 1
