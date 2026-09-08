# FireScope

Interactive desktop visualizer for FDS (Fire Dynamics Simulator) slice-file
ensembles. Load a family of simulation runs, scrub through time, compare
scenarios side by side, probe values, and export figures and reports.

Built with PyQt5, matplotlib, NumPy, SciPy and scikit-learn. All application
code lives under `src/`.

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.9 or newer

That is the only thing you need to install yourself. `run.sh` takes care of the
virtualenv and every Python dependency.

## Quick start

```
git clone https://github.com/lachgarsamia/fds-analysis-platform.git
cd fds-analysis-platform
chmod +x run.sh
./run.sh
```

The first run spends a minute building the environment. Every run after that
launches straight away.

`chmod +x` is needed because a fresh Git clone doesn't always keep the
executable bit on `run.sh`.

## What `run.sh` does

- Creates a virtualenv at `~/.venvs/fds_visualizer` if it isn't there yet
- Installs the project into it with `pip install -e .`
- Launches `src/main.py`
- Retries the transient PyQt/Cocoa startup failures seen on macOS, and
  reinstalls the pinned PyQt5 packages once if every attempt fails immediately

The environment lives in `~/.venvs`, not inside the repo, on purpose: under an
iCloud-synced folder (`~/Desktop`, `~/Documents`) Qt cannot read its own plugin
directory and the app won't start.

To rebuild from scratch, delete the venv and run again:

```
rm -rf ~/.venvs/fds_visualizer
./run.sh
```

## Data

FireScope reads FDS slice output from `fds/sim/` (or `fds/sim_stage1_prep/`).
That data is large and is not stored in Git. When it is absent the app opens in
demo mode on a synthetic dataset, marked "demo" in the title bar, so the
interface is still fully usable.

To work with real data, put your FDS run folders under `fds/sim/` before
launching.

## Command line

A headless CLI is installed next to the app:

```
~/.venvs/fds_visualizer/bin/fdsvis-cli --help
```

Subcommands: `stats` (summary-stats index), `export` (render a field to a
publication figure), `report` (build an HTML report), `session-render` (render a
saved session's cells to images).

## Tests

```
~/.venvs/fds_visualizer/bin/pip install -e ".[dev]"
~/.venvs/fds_visualizer/bin/pytest
```

## Layout

```
src/     application code: PyQt5 UI, FDS slice/s3d readers, analysis
tests/   pytest suite and fixtures
fds/     simulation templates and generators (sim output is gitignored)
docs/    design notes and roadmaps
ml/      experimental forecasting models (separate environment)
```
