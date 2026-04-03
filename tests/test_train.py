"""Tests for training loop — uses synthetic data and tmp_path, never real files."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.lstm import LSTMForecaster
from src.training.train import _train_one_epoch, _validate, train


def _mini_config(tmp_path: Path) -> dict[str, Any]:
    """Return a minimal config for fast training tests."""
    return {
        "model": {
            "input_size": 3,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "output_size": 1,
        },
        "training": {
            "lookback": 5,
            "horizon": 2,
            "batch_size": 4,
            "epochs": 3,
            "learning_rate": 1e-3,
            "max_norm": 1.0,
            "patience": 2,
            "seed": 42,
        },
        "data": {"target_col": "target"},
        "mlflow": {"experiment_name": "test-run"},
    }


def _write_synthetic_csvs(tmp_path: Path, n_rows: int = 50) -> None:
    """Write synthetic train/val/test CSVs to tmp_path/data/processed/."""
    rng = np.random.default_rng(0)
    cols = ["feat_a", "feat_b", "target"]
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for split in ("train", "val", "test"):
        df = pd.DataFrame(rng.standard_normal((n_rows, 3)), columns=cols)
        df.to_csv(processed / f"{split}.csv")


def test_train_one_epoch_returns_float(tmp_path: Path) -> None:
    """_train_one_epoch returns a finite float loss."""
    cfg = _mini_config(tmp_path)
    model = LSTMForecaster(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    rng = np.random.default_rng(1)
    data = rng.standard_normal((50, 3)).astype(np.float32)
    from src.data.torch_dataset import SlidingWindowDataset

    ds = SlidingWindowDataset(data, lookback=5, horizon=2, target_idx=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    loss = _train_one_epoch(model, loader, optimizer, criterion, 1.0, device)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_validate_returns_float(tmp_path: Path) -> None:
    """_validate returns a finite float loss."""
    cfg = _mini_config(tmp_path)
    model = LSTMForecaster(cfg)
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    rng = np.random.default_rng(2)
    data = rng.standard_normal((50, 3)).astype(np.float32)
    from src.data.torch_dataset import SlidingWindowDataset

    ds = SlidingWindowDataset(data, lookback=5, horizon=2, target_idx=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    loss = _validate(model, loader, criterion, device)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_validate_no_grad(tmp_path: Path) -> None:
    """Validation must not accumulate gradients."""
    cfg = _mini_config(tmp_path)
    model = LSTMForecaster(cfg)
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    rng = np.random.default_rng(3)
    data = rng.standard_normal((50, 3)).astype(np.float32)
    from src.data.torch_dataset import SlidingWindowDataset

    ds = SlidingWindowDataset(data, lookback=5, horizon=2, target_idx=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    _validate(model, loader, criterion, device)
    for p in model.parameters():
        assert p.grad is None, "Validation should not set gradients"


def test_train_saves_checkpoint(tmp_path: Path, monkeypatch: Any) -> None:
    """Full train() saves a checkpoint .pt file."""
    cfg = _mini_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    _write_synthetic_csvs(tmp_path)

    # Monkeypatch CWD so train() reads from tmp_path
    monkeypatch.chdir(tmp_path)

    # Patch build_dataloaders to use tmp_path data
    from src.data.torch_dataset import SlidingWindowDataset

    def mock_build(config: dict[str, Any]) -> tuple:  # type: ignore[type-arg]
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 3)).astype(np.float32)
        ds = SlidingWindowDataset(data, lookback=5, horizon=2, target_idx=2)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        return loader, loader, loader, 2

    monkeypatch.setattr("src.training.train.build_dataloaders", mock_build)

    train(str(config_path))

    ckpt_path = tmp_path / "models" / "lstm_checkpoint.pt"
    assert ckpt_path.exists(), "Checkpoint must be saved"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert "model_state" in ckpt
    assert "config" in ckpt
    assert "epoch" in ckpt
    assert "val_loss" in ckpt


def test_gradient_clipping_respected(tmp_path: Path) -> None:
    """Gradients are clipped to max_norm after backward."""
    cfg = _mini_config(tmp_path)
    cfg["training"]["max_norm"] = 0.01  # very small to force clipping
    model = LSTMForecaster(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    rng = np.random.default_rng(4)
    data = (rng.standard_normal((50, 3)) * 100).astype(np.float32)
    from src.data.torch_dataset import SlidingWindowDataset

    ds = SlidingWindowDataset(data, lookback=5, horizon=2, target_idx=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    _train_one_epoch(model, loader, optimizer, criterion, 0.01, device)

    # After training with tiny max_norm, grad norms should be small
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm**0.5
    assert total_norm < 1.0, f"Gradient norm {total_norm} too large after clipping"
