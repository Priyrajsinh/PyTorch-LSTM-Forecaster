"""Tests for LSTMForecaster architecture and checkpoint round-trip."""

from pathlib import Path

import torch
import yaml

from src.models.lstm import LSTMForecaster


def _load_config() -> dict:
    """Load config from config/config.yaml."""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def test_lstm_output_shape(tmp_path: Path) -> None:
    """Forward pass produces [batch, horizon] output."""
    cfg = _load_config()
    model = LSTMForecaster(cfg)
    batch = cfg["training"]["batch_size"]
    x = torch.randn(batch, cfg["training"]["lookback"], cfg["model"]["input_size"])
    out, (h, c) = model(x)
    assert out.shape == (batch, cfg["training"]["horizon"])


def test_lstm_save_load_checkpoint(tmp_path: Path) -> None:
    """Save and reload checkpoint preserves parameter count."""
    cfg = _load_config()
    model = LSTMForecaster(cfg)
    save_path = tmp_path / "test.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, save_path)
    ckpt = torch.load(str(save_path), map_location="cpu")
    model2 = LSTMForecaster(ckpt["config"])
    model2.load_state_dict(ckpt["model_state"])
    assert model2.count_parameters() == model.count_parameters()


def test_lstm_hidden_state_shape(tmp_path: Path) -> None:
    """Hidden and cell states have shape [num_layers, batch, hidden_size]."""
    cfg = _load_config()
    model = LSTMForecaster(cfg)
    batch = 4
    x = torch.randn(batch, cfg["training"]["lookback"], cfg["model"]["input_size"])
    _, (h, c) = model(x)
    n_layers = cfg["model"]["num_layers"]
    hidden = cfg["model"]["hidden_size"]
    assert h.shape == (n_layers, batch, hidden)
    assert c.shape == (n_layers, batch, hidden)
