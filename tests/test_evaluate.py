"""Tests for evaluation module — synthetic data only, never real model files."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluate import (
    _inverse_scale,
    evaluate_multi_horizon,
    plot_forecast,
)
from src.exceptions import ModelLoadError
from src.models.lstm import LSTMForecaster


def _mini_config() -> dict[str, Any]:
    """Minimal config for fast evaluation tests."""
    return {
        "model": {
            "input_size": 3,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "training": {
            "lookback": 5,
            "horizon": 4,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 1e-3,
            "max_norm": 1.0,
            "patience": 2,
            "seed": 42,
        },
        "data": {"target_col": "feat_c"},
        "evaluation": {"horizons": [2, 4]},
        "mlflow": {"experiment_name": "test-run"},
    }


def _write_processed(tmp_path: Path, n_rows: int = 60) -> None:
    """Write synthetic scaled CSVs to tmp_path/data/processed/.

    All 3 columns (including the target) are written in standardised form,
    matching the production preprocessing that scales all features.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    cols = ["feat_a", "feat_b", "feat_c"]
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for split in ("train", "val", "test"):
        # Standard-normal data mimics already-scaled CSVs
        df = pd.DataFrame(rng.standard_normal((n_rows, 3)), columns=cols)
        df.to_csv(processed / f"{split}.csv")


def _make_scaler(n_features: int = 3) -> StandardScaler:
    """Return a fitted StandardScaler covering ALL n_features columns (incl. target)."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((200, n_features)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def _make_checkpoint(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Save a synthetic checkpoint and scaler, return checkpoint path."""
    import joblib

    model = LSTMForecaster(config)
    ckpt_path = tmp_path / "models" / "lstm_checkpoint.pt"
    scaler_path = tmp_path / "models" / "scaler.joblib"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": config, "epoch": 0}, ckpt_path
    )
    # Scaler covers all 3 features including target (feat_c at index 2)
    joblib.dump(_make_scaler(n_features=3), scaler_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_inverse_scale_roundtrip() -> None:
    """_inverse_scale(scaled) recovers original values within tolerance."""
    scaler = _make_scaler(n_features=3)
    rng = np.random.default_rng(5)
    original = rng.standard_normal(50).astype(np.float32) * 5 + 10

    # Manually scale the target column (idx=2)
    dummy = np.zeros((50, 3), dtype=np.float32)
    dummy[:, 2] = original
    scaled = scaler.transform(dummy)[:, 2]

    recovered = _inverse_scale(scaled, scaler, n_features=3, target_idx=2)
    np.testing.assert_allclose(recovered, original, atol=1e-4)


def test_inverse_scale_output_shape() -> None:
    """_inverse_scale output has the same length as input."""
    scaler = _make_scaler(n_features=3)
    arr = np.zeros(20, dtype=np.float32)
    out = _inverse_scale(arr, scaler, n_features=3, target_idx=0)
    assert out.shape == (20,)


def test_load_model_and_scaler_raises_on_missing_file(tmp_path: Path) -> None:
    """load_model_and_scaler raises ModelLoadError if checkpoint is absent."""
    from src.evaluation import evaluate as ev_mod

    orig = ev_mod.CHECKPOINT_PATH
    try:
        ev_mod.CHECKPOINT_PATH = tmp_path / "nonexistent.pt"  # type: ignore[assignment]
        with pytest.raises(ModelLoadError):
            from src.evaluation.evaluate import load_model_and_scaler

            load_model_and_scaler(_mini_config())
    finally:
        ev_mod.CHECKPOINT_PATH = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Integration-style tests (monkeypatch CWD + synthetic files)
# ---------------------------------------------------------------------------


def test_evaluate_multi_horizon_returns_correct_keys(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """evaluate_multi_horizon returns one key per configured evaluation horizon."""
    cfg = _mini_config()
    _write_processed(tmp_path)
    _make_checkpoint(tmp_path, cfg)
    monkeypatch.chdir(tmp_path)

    results = evaluate_multi_horizon(cfg)

    assert set(results.keys()) == {"h2", "h4"}
    for h_key, metrics in results.items():
        assert set(metrics.keys()) == {"mse", "mae", "rmse"}
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0


def test_evaluate_multi_horizon_metrics_are_finite(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """All returned metric values must be finite floats."""
    cfg = _mini_config()
    _write_processed(tmp_path)
    _make_checkpoint(tmp_path, cfg)
    monkeypatch.chdir(tmp_path)

    results = evaluate_multi_horizon(cfg)

    for metrics in results.values():
        for val in metrics.values():
            assert np.isfinite(val), f"Non-finite metric: {val}"


def test_evaluate_rmse_equals_sqrt_mse(tmp_path: Path, monkeypatch: Any) -> None:
    """RMSE must equal sqrt(MSE) for every horizon (mathematical identity)."""
    cfg = _mini_config()
    _write_processed(tmp_path)
    _make_checkpoint(tmp_path, cfg)
    monkeypatch.chdir(tmp_path)

    results = evaluate_multi_horizon(cfg)

    for h_key, metrics in results.items():
        expected_rmse = metrics["mse"] ** 0.5
        assert (
            abs(metrics["rmse"] - expected_rmse) < 1e-5
        ), f"{h_key}: rmse={metrics['rmse']} != sqrt(mse)={expected_rmse}"


def test_plot_forecast_creates_png(tmp_path: Path, monkeypatch: Any) -> None:
    """plot_forecast saves a non-empty PNG to reports/figures/forecast_vs_actual.png."""
    cfg = _mini_config()
    _write_processed(tmp_path)
    _make_checkpoint(tmp_path, cfg)
    monkeypatch.chdir(tmp_path)

    plot_forecast(cfg)

    png_path = tmp_path / "reports" / "figures" / "forecast_vs_actual.png"
    assert png_path.exists(), "PNG not created by plot_forecast"
    assert png_path.stat().st_size > 0, "PNG is empty"
