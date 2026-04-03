"""Build train/val/test DataLoaders from processed CSV splits."""

from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.torch_dataset import SlidingWindowDataset
from src.logger import get_logger

logger = get_logger(__name__)


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[
    DataLoader[tuple[Any, Any]],
    DataLoader[tuple[Any, Any]],
    DataLoader[tuple[Any, Any]],
    int,
]:
    """Build train/val/test DataLoaders from processed splits.

    Args:
        config: Parsed config.yaml dict.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, target_idx).
    """
    train_df = pd.read_csv("data/processed/train.csv", index_col=0)
    val_df = pd.read_csv("data/processed/val.csv", index_col=0)
    test_df = pd.read_csv("data/processed/test.csv", index_col=0)

    target_col: str = config["data"]["target_col"]
    feature_cols = list(train_df.columns)
    target_idx = feature_cols.index(target_col)

    lookback: int = config["training"]["lookback"]
    horizon: int = config["training"]["horizon"]
    batch: int = config["training"]["batch_size"]

    train_arr = train_df.values.astype(np.float32)
    val_arr = val_df.values.astype(np.float32)
    test_arr = test_df.values.astype(np.float32)

    train_ds = SlidingWindowDataset(train_arr, lookback, horizon, target_idx)
    val_ds = SlidingWindowDataset(val_arr, lookback, horizon, target_idx)
    test_ds = SlidingWindowDataset(test_arr, lookback, horizon, target_idx)

    # shuffle=False always for time series — order must be preserved
    train_loader: DataLoader[tuple[Any, Any]] = DataLoader(
        train_ds, batch_size=batch, shuffle=False, drop_last=True
    )
    val_loader: DataLoader[tuple[Any, Any]] = DataLoader(
        val_ds, batch_size=batch, shuffle=False
    )
    test_loader: DataLoader[tuple[Any, Any]] = DataLoader(
        test_ds, batch_size=batch, shuffle=False
    )

    logger.info(
        "DataLoaders built",
        extra={
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        },
    )
    return train_loader, val_loader, test_loader, target_idx
