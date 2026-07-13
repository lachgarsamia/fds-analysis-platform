"""Training loop for M3.2's FNO forecasting model (3.2.3).

Single-step supervision (8 frames in, 9th frame out -- see dataset.py's
module docstring for why), MSE loss in normalized space. Scenario-level
fit/val split from dataset.py; the 4 TEST scenarios are never touched
here, only in 3.2.4's rollout evaluation.

Early stopping on validation loss, best checkpoint + config hash saved
for reproducibility (a checkpoint file's name traces back to exactly the
hyperparameters that produced it, per dataset.py's config_hash()).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from dataset import (  # noqa: E402
    build_dataset, compute_normalization, config_hash, load_entries,
    scenario_split, train_val_split,
)
from model import DEFAULT_HIDDEN_CHANNELS, DEFAULT_N_LAYERS, DEFAULT_N_MODES, build_fno, get_device  # noqa: E402

CHECKPOINT_DIR = os.path.join(_ML_DIR, "checkpoints")

DEFAULT_CONFIG = {
    "n_modes": list(DEFAULT_N_MODES),
    "hidden_channels": DEFAULT_HIDDEN_CHANNELS,
    "n_layers": DEFAULT_N_LAYERS,
    "lr": 1e-3,
    "batch_size": 16,
    "max_epochs": 100,
    "patience": 10,
    "split_seed": 42,
}


def train(config: dict = None, verbose: bool = True) -> dict:
    config = {**DEFAULT_CONFIG, **(config or {})}
    device = get_device()

    entries = load_entries()
    train_cases, test_cases = scenario_split(entries, seed=config["split_seed"])
    fit_cases, val_cases = train_val_split(train_cases, seed=config["split_seed"])
    stats = compute_normalization(entries, fit_cases)

    fit_inputs, fit_targets = build_dataset(entries, fit_cases, stats)
    val_inputs, val_targets = build_dataset(entries, val_cases, stats)

    fit_loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_inputs), torch.from_numpy(fit_targets)),
        batch_size=config["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_targets)),
        batch_size=config["batch_size"], shuffle=False,
    )

    model = build_fno(
        n_modes=config["n_modes"],
        hidden_channels=config["hidden_channels"],
        n_layers=config["n_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(config["max_epochs"]):
        model.train()
        train_losses = []
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                pred = model(x)
                val_losses.append(loss_fn(pred, y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} ({elapsed:.0f}s)")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            # state_dict() includes a non-tensor "_metadata" entry in some
            # torch versions -- filter to real tensors only.
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                if torch.is_tensor(v)
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    chash = config_hash(config)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"fno_{chash}.pt")
    torch.save({
        "model_state_dict": best_state,
        "config": config,
        "config_hash": chash,
        "stats": stats,
        "train_cases": fit_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "best_val_loss": best_val_loss,
        "history": history,
    }, checkpoint_path)
    if verbose:
        print(f"saved checkpoint to {checkpoint_path}")

    return {
        "checkpoint_path": checkpoint_path,
        "config_hash": chash,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train M3.2's FNO forecasting model.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    args = parser.parse_args()
    train({
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
    })


if __name__ == "__main__":
    main()
