"""Multi-horizon evaluation for LSTMForecaster."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.exceptions import ModelLoadError
from src.logger import get_logger
from src.models.lstm import LSTMForecaster

logger = get_logger(__name__)

CHECKPOINT_PATH = Path("models/lstm_checkpoint.pt")
SCALER_PATH = Path("models/scaler.joblib")
METRICS_PATH = Path("reports/metrics.json")
PLOT_PATH = Path("reports/figures/forecast_vs_actual.png")


def load_model_and_scaler(
    config: dict[str, Any],
) -> tuple[LSTMForecaster, Any, dict[str, Any]]:
    """Load best checkpoint and scaler from models/.

    Args:
        config: Full config dict (unused here; model config read from checkpoint).

    Returns:
        Tuple of (model in eval mode, fitted scaler, raw checkpoint dict).
    """
    try:
        ckpt: dict[str, Any] = torch.load(  # nosec B614
            str(CHECKPOINT_PATH),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model = LSTMForecaster(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        scaler = joblib.load(SCALER_PATH)
    except (FileNotFoundError, KeyError) as exc:
        raise ModelLoadError(f"Cannot load checkpoint or scaler: {exc}") from exc
    logger.info("Checkpoint and scaler loaded", extra={"epoch": ckpt.get("epoch")})
    return model, scaler, ckpt


def _inverse_scale(
    arr_1d: np.ndarray,
    scaler: Any,
    n_features: int,
    target_idx: int,
) -> np.ndarray:
    """Inverse-scale a 1D array of target values back to °C.

    Args:
        arr_1d: 1-D array of scaled predictions or actuals.
        scaler: Fitted StandardScaler with n_features_in_ columns.
        n_features: Total number of features the scaler was fitted on.
        target_idx: Column index of the target variable.

    Returns:
        1-D array in original °C units.
    """
    dummy = np.zeros((len(arr_1d), n_features), dtype=np.float32)
    dummy[:, target_idx] = arr_1d
    result: np.ndarray = scaler.inverse_transform(dummy)[:, target_idx]
    return result


def evaluate_multi_horizon(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Evaluate MSE/MAE/RMSE at each horizon in config.evaluation.horizons.

    Args:
        config: Full config dict.

    Returns:
        Nested dict keyed by 'h{horizon}' with 'mse', 'mae', 'rmse' in °C units.
    """
    model, scaler, _ckpt = load_model_and_scaler(config)
    test_df = pd.read_csv("data/processed/test.csv", index_col=0)
    target_col: str = config["data"]["target_col"]
    feature_cols = list(test_df.columns)
    target_idx: int = feature_cols.index(target_col)
    test_arr = test_df.values.astype(np.float32)
    n_features: int = int(scaler.n_features_in_)
    lookback: int = config["training"]["lookback"]
    results: dict[str, dict[str, float]] = {}

    from torch.utils.data import DataLoader

    from src.data.torch_dataset import SlidingWindowDataset

    for horizon in config["evaluation"]["horizons"]:
        ds = SlidingWindowDataset(test_arr, lookback, horizon, target_idx)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds_list: list[np.ndarray] = []
        actuals_list: list[np.ndarray] = []

        with torch.no_grad():
            for X_b, y_b in loader:
                lstm_out, _ = model.lstm(X_b)
                last_h = model.dropout(lstm_out[:, -1, :])
                out = torch.stack(
                    [model.fc(last_h) for _ in range(horizon)], dim=1
                ).squeeze(-1)
                preds_list.append(out.detach().cpu().numpy())
                actuals_list.append(y_b.detach().cpu().numpy())

        preds = np.concatenate(preds_list)  # [N, horizon]
        actuals = np.concatenate(actuals_list)  # [N, horizon]

        # CRITICAL: inverse-transform to °C before metrics — scaled errors meaningless
        preds_c = _inverse_scale(preds.flatten(), scaler, n_features, target_idx)
        actuals_c = _inverse_scale(actuals.flatten(), scaler, n_features, target_idx)

        mse = float(mean_squared_error(actuals_c, preds_c))
        mae = float(mean_absolute_error(actuals_c, preds_c))
        rmse = float(np.sqrt(mse))
        results[f"h{horizon}"] = {"mse": mse, "mae": mae, "rmse": rmse}
        logger.info(
            f"horizon={horizon}h  MSE={mse:.4f}°C²  MAE={mae:.4f}°C  RMSE={rmse:.4f}°C",
            extra={"horizon": horizon, "mse": mse, "mae": mae, "rmse": rmse},
        )

    return results


def plot_forecast(config: dict[str, Any]) -> None:
    """Plot actual vs predicted temperature for first 200 test windows.

    Uses the training horizon (config.training.horizon) and saves PNG to
    reports/figures/forecast_vs_actual.png.

    Args:
        config: Full config dict.
    """
    model, _scaler, _ = load_model_and_scaler(config)
    test_df = pd.read_csv("data/processed/test.csv", index_col=0)
    feature_cols = list(test_df.columns)
    target_idx: int = feature_cols.index(config["data"]["target_col"])
    test_arr = test_df.values.astype(np.float32)
    lookback: int = config["training"]["lookback"]
    horizon: int = config["training"]["horizon"]

    from torch.utils.data import DataLoader

    from src.data.torch_dataset import SlidingWindowDataset

    ds = SlidingWindowDataset(test_arr, lookback, horizon, target_idx)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_preds: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []

    with torch.no_grad():
        for X_b, y_b in loader:
            out, _ = model(X_b)
            all_preds.append(out.detach().cpu().numpy())
            all_actual.append(y_b.detach().cpu().numpy())

    preds = np.concatenate(all_preds)[:200, 0]  # first forecast step
    actuals = np.concatenate(all_actual)[:200, 0]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actuals, label="Actual T (degC)", color="steelblue", linewidth=1.2)
    ax.plot(
        preds,
        label="Predicted T (degC)",
        color="coral",
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_title("LSTM Forecast vs Actual — First 200 Test Points (24h horizon)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Temperature (degC, scaled)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(PLOT_PATH), dpi=150)
    plt.close()
    logger.info("Forecast plot saved", extra={"path": str(PLOT_PATH)})


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    results = evaluate_multi_horizon(cfg)
    plot_forecast(cfg)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation"):
        for h_key, metrics in results.items():
            mlflow.log_metrics({f"{h_key}_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(PLOT_PATH))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(METRICS_PATH), "w") as fout:
        json.dump(results, fout, indent=2)
    logger.info("Evaluation complete", extra={"path": str(METRICS_PATH)})
