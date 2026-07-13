"""Tests for ml/model.py. Fast/synthetic -- no real dataset or training
needed, just shape/type contracts for build_fno() and
model_forecast_step()."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import WINDOW_IN  # noqa: E402
from model import build_fno, get_device, model_forecast_step  # noqa: E402


class TestBuildFno:
    def test_output_shape(self):
        model = build_fno(n_modes=(4, 4), hidden_channels=4, n_layers=2)
        x = torch.randn(2, WINDOW_IN, 12, 16)
        y = model(x)
        assert y.shape == (2, 1, 12, 16)

    def test_default_in_channels_matches_window_in(self):
        model = build_fno(n_modes=(4, 4), hidden_channels=4, n_layers=2)
        assert model.in_channels == WINDOW_IN


class TestGetDevice:
    def test_returns_a_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("mps", "cpu")


class TestModelForecastStep:
    def test_output_shape_and_dtype(self):
        model = build_fno(n_modes=(4, 4), hidden_channels=4, n_layers=2)
        device = torch.device("cpu")
        step_fn = model_forecast_step(model, device)
        window = np.random.default_rng(0).normal(size=(WINDOW_IN, 12, 16)).astype(np.float32)
        result = step_fn(window)
        assert result.shape == (12, 16)
        assert result.dtype == np.float32

    def test_does_not_require_grad_tracking(self):
        model = build_fno(n_modes=(4, 4), hidden_channels=4, n_layers=2)
        device = torch.device("cpu")
        step_fn = model_forecast_step(model, device)
        window = np.zeros((WINDOW_IN, 8, 8), dtype=np.float32)
        result = step_fn(window)
        assert np.isfinite(result).all()
