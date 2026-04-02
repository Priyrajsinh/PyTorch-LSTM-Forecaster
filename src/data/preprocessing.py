"""Chronological train/val/test split and StandardScaler fitting."""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)


def chronological_split_and_scale(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Chronological 70/15/15 split. Scaler fit on train ONLY.

    Args:
        df: Validated, hourly Jena Climate DataFrame.
        config: Parsed config.yaml dict.

    Returns:
        (train_df, val_df, test_df, scaler) — all splits scaled, scaler saved to disk.
    """
    n = len(df)
    train_end = int(n * config["data"]["train_split"])
    val_end = int(n * (config["data"]["train_split"] + config["data"]["val_split"]))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Chronological split",
        extra={"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    )
    logger.info(
        "Train range",
        extra={"start": str(train_df.index[0]), "end": str(train_df.index[-1])},
    )
    logger.info(
        "Val range", extra={"start": str(val_df.index[0]), "end": str(val_df.index[-1])}
    )
    logger.info(
        "Test range",
        extra={"start": str(test_df.index[0]), "end": str(test_df.index[-1])},
    )

    feature_cols = [c for c in df.columns if c != config["data"]["target_col"]]

    # Fit ONLY on train — never on val or test to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    logger.info("Scaler fitted on train and saved to models/scaler.joblib")

    # Transform all splits with the same fitted scaler
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv("data/processed/train.csv")
    val_df.to_csv("data/processed/val.csv")
    test_df.to_csv("data/processed/test.csv")
    logger.info("Processed CSVs written to data/processed/")

    return train_df, val_df, test_df, scaler
