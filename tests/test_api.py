"""Tests for FastAPI endpoints.

Model and scaler are injected via monkeypatch so no real checkpoint files are
needed.  torch.load and joblib.load are patched before the lifespan runs,
which keeps tests fully hermetic and avoids any .gitignored artefacts.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

import src.api.app as api_module
from src.models.lstm import LSTMForecaster


def _load_config() -> dict[str, Any]:
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return _load_config()


@pytest.fixture()
def client(cfg: dict[str, Any]) -> TestClient:
    """TestClient with lifespan driven by patched torch.load / joblib.load."""
    model = LSTMForecaster(cfg)
    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": cfg,
        "epoch": 5,
    }

    mock_scaler = MagicMock()
    mock_scaler.n_features_in_ = cfg["model"]["input_size"]
    # Pass-through: return the same array so shapes stay correct
    mock_scaler.transform.side_effect = lambda x: np.array(x, dtype=np.float32)
    mock_scaler.inverse_transform.side_effect = lambda x: np.array(x, dtype=np.float32)

    with (
        patch("torch.load", return_value=ckpt),
        patch("joblib.load", return_value=mock_scaler),
    ):
        with TestClient(api_module.app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_200(client: TestClient) -> None:
    """Health endpoint returns 200 with required keys."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert "uptime_seconds" in body
    assert "memory_mb" in body
    assert "checkpoint_epoch" in body


def test_health_checkpoint_epoch(client: TestClient) -> None:
    """checkpoint_epoch reflects the value stored in the checkpoint."""
    r = client.get("/api/v1/health")
    assert r.json()["checkpoint_epoch"] == 5


# ── /forecast ─────────────────────────────────────────────────────────────────


def test_forecast_valid_shape(client: TestClient, cfg: dict[str, Any]) -> None:
    """Valid input returns forecast of length == horizon with trace_id."""
    lookback = cfg["training"]["lookback"]
    n_features = cfg["model"]["input_size"]
    target_col = cfg["data"]["target_col"]
    # Build feature_names that include the target column at the last position
    feature_names = [f"feat_{i}" for i in range(n_features - 1)] + [target_col]
    payload = {
        "values": [[0.1] * n_features for _ in range(lookback)],
        "feature_names": feature_names,
    }
    r = client.post("/api/v1/forecast", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == cfg["training"]["horizon"]
    assert body["horizon_hours"] == cfg["training"]["horizon"]
    assert "trace_id" in body


def test_forecast_wrong_row_count(client: TestClient, cfg: dict[str, Any]) -> None:
    """forecast returns 422 when too few rows are provided."""
    n_features = cfg["model"]["input_size"]
    target_col = cfg["data"]["target_col"]
    feature_names = [f"feat_{i}" for i in range(n_features - 1)] + [target_col]
    payload = {
        "values": [[0.0] * n_features],  # only 1 row, not lookback
        "feature_names": feature_names,
    }
    r = client.post("/api/v1/forecast", json=payload)
    assert r.status_code == 422
    assert "Expected" in r.json()["detail"]


def test_forecast_missing_target_col(client: TestClient, cfg: dict[str, Any]) -> None:
    """forecast returns 422 when feature_names omits the target column."""
    lookback = cfg["training"]["lookback"]
    n_features = cfg["model"]["input_size"]
    payload = {
        "values": [[0.0] * n_features for _ in range(lookback)],
        "feature_names": [f"feat_{i}" for i in range(n_features)],  # no T (degC)
    }
    r = client.post("/api/v1/forecast", json=payload)
    assert r.status_code == 422


# ── /model_info ───────────────────────────────────────────────────────────────


def test_model_info_schema(client: TestClient, cfg: dict[str, Any]) -> None:
    """model_info returns correct architecture metadata."""
    r = client.get("/api/v1/model_info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "LSTMForecaster"
    assert body["hidden_size"] == cfg["model"]["hidden_size"]
    assert body["num_layers"] == cfg["model"]["num_layers"]
    assert body["lookback"] == cfg["training"]["lookback"]
    assert body["horizon"] == cfg["training"]["horizon"]


def test_model_info_includes_results_when_present(
    client: TestClient, tmp_path: Any
) -> None:
    """model_info includes 'results' key when reports/metrics.json exists."""
    import json

    metrics = {"h6": {"mse": 1.0, "mae": 0.5, "rmse": 1.0}}
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(metrics))

    original_path = api_module.METRICS_PATH
    api_module.METRICS_PATH = metrics_file  # type: ignore[assignment]
    try:
        r = client.get("/api/v1/model_info")
        assert "results" in r.json()
    finally:
        api_module.METRICS_PATH = original_path


# ── /metrics (Prometheus) ─────────────────────────────────────────────────────


def test_metrics_endpoint_reachable(client: TestClient) -> None:
    """Prometheus /metrics endpoint returns 200 with text/plain content."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
