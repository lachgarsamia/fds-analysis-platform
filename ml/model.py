"""FNO2d wrapper for M3.2's forecasting model (3.2.3).

Treats the k=8 input frames as channels (in_channels=WINDOW_IN) rather
than a genuine 3rd spatial/temporal axis -- the standard way to feed a
sliding window into neuralop's 2D FNO for next-frame prediction.
out_channels=1: a single predicted frame, consistent with dataset.py's
single-step-training design (see its module docstring).

positional_embedding=None: neuralop's default "grid" embedding builds its
coordinate grid as a plain CPU tensor that doesn't follow model.to(device)
(confirmed: `RuntimeError: Passed CPU tensor to MPS op` on this Apple
Silicon Mac's MPS backend). Disabling it is a one-line workaround, not
the >1-day integration friction that would trigger the ConvLSTM fallback
the milestone pre-agreed to cap at -- the FNO's spectral convolutions
still see the full 2D grid each layer, just without an explicit
coordinate hint concatenated onto the input.
"""

from __future__ import annotations

import os
import sys

import torch
from neuralop.models import FNO

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from dataset import WINDOW_IN  # noqa: E402

DEFAULT_N_MODES = (16, 16)
DEFAULT_HIDDEN_CHANNELS = 32
DEFAULT_N_LAYERS = 4


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_fno(n_modes=DEFAULT_N_MODES, hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
              n_layers: int = DEFAULT_N_LAYERS, in_channels: int = WINDOW_IN) -> FNO:
    return FNO(
        n_modes=tuple(n_modes),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        positional_embedding=None,
    )


def model_forecast_step(model: FNO, device: torch.device):
    """Returns a forecast_step_fn(window: (WINDOW_IN, H, W) normalized
    ndarray) -> (H, W) normalized ndarray, matching ml/metrics.py's
    evaluate_rollout() calling convention exactly -- the same harness
    used for the persistence/linear-extrapolation baselines, so the
    model's RMSE/SSIM-vs-lead-time curve is directly comparable (3.2.4)."""
    model.eval()

    def step(window):
        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0).float().to(device)
            y = model(x)
            return y.squeeze(0).squeeze(0).cpu().numpy()

    return step
