"""Training loop for LSTMForecaster with MLflow tracking."""

import argparse
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloaders
from src.logger import get_logger
from src.models.lstm import LSTMForecaster
from utils.seed import set_seed

logger = get_logger(__name__)

CHECKPOINT_DIR = Path("models")
CHECKPOINT_PATH = CHECKPOINT_DIR / "lstm_checkpoint.pt"


def _train_one_epoch(
    model: LSTMForecaster,
    loader: DataLoader[tuple[Any, Any]],
    optimizer: torch.optim.Adam,
    criterion: nn.MSELoss,
    max_norm: float,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    losses: list[float] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output, _ = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _validate(
    model: LSTMForecaster,
    loader: DataLoader[tuple[Any, Any]],
    criterion: nn.MSELoss,
    device: torch.device,
) -> float:
    """Run validation and return mean loss."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output, _ = model(batch_x)
            losses.append(criterion(output, batch_y).item())
    return float(np.mean(losses))


def train(config_path: str = "config/config.yaml") -> None:
    """Train LSTMForecaster with early stopping and MLflow tracking."""
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device", extra={"device": str(device)})

    # --- Label sanity check ---
    train_df = pd.read_csv("data/processed/train.csv", index_col=0)
    target_col: str = config["data"]["target_col"]
    assert target_col in train_df.columns, f"Target {target_col} not in columns"
    assert train_df[target_col].isna().sum() == 0, "NaN in target column"
    logger.info(
        "Target check passed",
        extra={
            "target": target_col,
            "min": float(train_df[target_col].min()),
            "max": float(train_df[target_col].max()),
        },
    )

    # --- DataLoaders ---
    train_loader, val_loader, _test_loader, target_idx = build_dataloaders(config)

    # --- Model ---
    model = LSTMForecaster(config).to(device)
    logger.info("Model created", extra={"params": model.count_parameters()})

    # --- Optimizer + Loss + Scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- MLflow ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(
            {
                "hidden_size": config["model"]["hidden_size"],
                "num_layers": config["model"]["num_layers"],
                "lookback": config["training"]["lookback"],
                "horizon": config["training"]["horizon"],
                "lr": config["training"]["learning_rate"],
                "batch_size": config["training"]["batch_size"],
                "max_norm": config["training"]["max_norm"],
                "dropout": config["model"]["dropout"],
            }
        )

        best_val_loss = float("inf")
        patience_count = 0
        max_norm: float = config["training"]["max_norm"]
        patience: int = config["training"]["patience"]
        epochs: int = config["training"]["epochs"]

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, criterion, max_norm, device
            )
            val_loss = _validate(model, val_loader, criterion, device)

            logger.info(
                "Epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                },
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

            scheduler.step(val_loss)
            current_lr: float = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", current_lr, step=epoch)

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    CHECKPOINT_PATH,
                )
                logger.info(
                    "Checkpoint saved",
                    extra={"val_loss": round(val_loss, 4)},
                )
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info("Early stopping", extra={"epoch": epoch})
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        logger.info(
            "Training complete",
            extra={"best_val_loss": round(best_val_loss, 4)},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)
