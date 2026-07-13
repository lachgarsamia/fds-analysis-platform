# ml/ — forecasting model (M3.2)

Training pipeline for a Fourier Neural Operator (FNO) that forecasts the
next TEMPERATURE frame from the previous 8, plus the baselines it has to
beat. Lives entirely outside `src/` and has its own dependencies (torch,
neuraloperator, scikit-image) and its own test suite — the app itself
never imports anything here, so `torch` is not required to run the
visualizer.

## Setup

```bash
pip install torch neuraloperator scikit-image
```

Everything else (`numpy`, `scikit-learn`) is already a dependency of the
app. No `ml/requirements.txt` is tracked separately; the three packages
above are the only ones `ml/` needs beyond `pyproject.toml`'s own set.

Requires the real dataset at `fds/sim/` (see the repo root README) — every
script here assumes all 24 real scenarios are present and will raise if
they're not.

## Pipeline, in order

```bash
python3 ml/baselines.py   # persistence + linear-extrapolation baselines -> ml/baseline_results.json
python3 ml/train.py       # trains the FNO -> ml/checkpoints/fno_<hash>.pt
python3 ml/rollout.py     # evaluates the latest checkpoint, writes ml/model_results.json +
                           # ml/comparison_report.json, exports predictions/<case>.npy
```

Each step reads only what the previous one wrote (or, for `train.py`,
only the raw dataset) — there's no hidden shared state.

## Reproducibility

- **Scenario split is deterministic**: `dataset.SPLIT_SEED = 42` fixes a
  20/4 train/test split by scenario (never by time — an entire scenario's
  time series is wholly train or wholly test), then a further 3-scenario
  carve-out from the 20 for validation/early-stopping. `ml/train.py` also
  seeds `torch.manual_seed()` from this same value, so weight
  initialization and DataLoader shuffle order are reproducible too, not
  just which scenarios land in which bucket (CPU runs are fully
  deterministic this way; MPS may still have minor kernel-level
  non-determinism in a few fused ops regardless). Every script here
  calls `dataset.scenario_split()`/`train_val_split()` with the same
  default seed, so re-running any step reproduces the identical split.
- **Checkpoints are traceable**: `ml/train.py` saves
  `ml/checkpoints/fno_<config_hash>.pt`, where `<config_hash>` is a
  12-character hash of the exact training config (`dataset.config_hash()`)
  — architecture, learning rate, batch size, split seed. The checkpoint
  file also embeds the config dict itself, plus the normalization stats
  and the exact train/val/test case indices used, so a checkpoint is
  self-describing: nothing about how it was produced has to be
  remembered separately.
- **Normalization stats** are computed once, on the fit-split scenarios
  only (never val or test), and saved inside the checkpoint — `rollout.py`
  reuses them from the checkpoint rather than recomputing, so evaluation
  is guaranteed to use the exact stats the model was trained with.

To retrain with different hyperparameters:

```bash
python3 ml/train.py --max-epochs 60 --patience 10 --lr 1e-3 --batch-size 16
```

(`ml/train.py --help` for the full list.) A different config produces a
different `config_hash`, hence a different checkpoint filename — old
checkpoints are never overwritten.

## Model

FNO2d (`neuralop.models.FNO`), the k=8 input frames treated as channels
(`in_channels=8`, `out_channels=1`) rather than a genuine 3rd axis — the
standard way to feed a sliding window into a 2D FNO. `n_modes=(16, 16)`,
`hidden_channels=32`, `n_layers=4` by default (`ml/model.py`).
`positional_embedding` is disabled: neuralop's default "grid" embedding
builds a plain CPU tensor that doesn't follow `model.to(device)`, which
crashes on Apple Silicon's MPS backend (`RuntimeError: Passed CPU tensor
to MPS op`). Disabling it is a one-line workaround, not the >1-day
integration friction that would have triggered the milestone's pre-agreed
ConvLSTM fallback — the FNO's spectral convolutions still see the full 2D
grid each layer, just without an explicit coordinate hint concatenated
onto the input.

Training is **single-step**: 8 real frames in, the 9th real frame out,
MSE loss in normalized space. Multi-step forecasts (autoregressive
rollout, feeding a prediction back in as the newest "real" frame) only
happen at evaluation time, in `ml/rollout.py` / `ml/metrics.py`'s
`evaluate_rollout()` — never during training. See `dataset.py`'s module
docstring for why that reading of the spec was chosen.

Device: MPS if available (`torch.backends.mps.is_available()`), else CPU
(`ml/model.py:get_device()`).

## Evaluation

`ml/metrics.py`'s `evaluate_rollout()` is the single harness used for
*both* the baselines and the trained model, so their RMSE/SSIM-vs-lead-
time curves are directly comparable — same rollout starts (every 20
frames within each test scenario), same 4 held-out test scenarios, same
metrics, same 8-step horizon. The only thing that differs between a
baseline and the model is which one-step `forecast_step_fn` gets plugged
in (`ml/baselines.py`'s `persistence_step`/`linear_extrapolation_step`
vs. `ml/model.py`'s `model_forecast_step()`).

`ml/rollout.py` additionally writes `ml/comparison_report.json`: an
explicit, saved (not just printed) fno-vs-persistence comparison per lead
time, plus a top-level `beats_persistence_at_lead_4_or_more` boolean —
the milestone's own non-negotiable ("the FNO must beat persistence or the
report says so honestly") answered as a durable artifact.

## In-app evaluation view

`ml/rollout.py`'s `export_full_scenario_predictions()` writes
`predictions/<case_index>.npy` for each of the 4 test scenarios (plus
`predictions/manifest.json`) — a full `(n_frames, H, W)` array: the first
8 frames are copied straight from ground truth (the model's own required
seed, never itself a prediction), the rest are the model's autoregressive
rollout for the scenario's *entire* real length, denormalized to physical
units.

The app's `src/prediction_store.py` reads this directory and, if present,
enables a "View model prediction" button in the experiment browser: select
a test-set scenario, click the button, and a 1x3 grid opens showing ground
truth, prediction, and their difference side by side (reusing the app's
existing grid/view rendering, not a separate display path). Absent
entirely if `predictions/` doesn't exist — e.g. nobody has run this
pipeline yet.

## Tests

```bash
python3 -m pytest ml/tests/ -q
```

Run separately from the app's own `pytest` (which only collects `tests/`,
per `pyproject.toml`) — this is what keeps `torch` out of the app's
required dependency set even though `ml/`'s own suite exercises it freely.
Tests that need the real dataset or a trained checkpoint are guarded with
`pytest.mark.skipif` and skip cleanly if either is absent.
