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
    """Chronological 70/15/15 split. Scaler fit on train ONLY, applied to all splits.

    All 15 features (including the target T (degC)) are standardised so the LSTM
    receives a consistent input scale.  The scaler is saved to models/scaler.joblib
    and used during evaluation to inverse-transform predictions back to °C.

    Args:
        df: Validated, hourly Jena Climate DataFrame (15 columns).
        config: Parsed config.yaml dict.

    Returns:
        (train_df, val_df, test_df, scaler) — all splits scaled, scaler saved to disk.
    """
    n = len(df)
    train_end = int(n * config["data"]["train_split"])
    val_end = int(n * (config["data"]["train_split"] + config["data"]["val_split"]))

    train_raw = df.iloc[:train_end].copy()
    val_raw = df.iloc[train_end:val_end].copy()
    test_raw = df.iloc[val_end:].copy()

    logger.info(
        "Chronological split",
        extra={"train": len(train_raw), "val": len(val_raw), "test": len(test_raw)},
    )
    logger.info(
        "Train range",
        extra={"start": str(train_raw.index[0]), "end": str(train_raw.index[-1])},
    )
    logger.info(
        "Val range",
        extra={"start": str(val_raw.index[0]), "end": str(val_raw.index[-1])},
    )
    logger.info(
        "Test range",
        extra={"start": str(test_raw.index[0]), "end": str(test_raw.index[-1])},
    )

    cols = list(df.columns)

    # Fit ONLY on train — never on val or test to prevent data leakage.
    # All 15 features (including target) are scaled for consistent LSTM inputs.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw.values)
    val_scaled = scaler.transform(val_raw.values)
    test_scaled = scaler.transform(test_raw.values)

    train_df = pd.DataFrame(train_scaled, index=train_raw.index, columns=cols)
    val_df = pd.DataFrame(val_scaled, index=val_raw.index, columns=cols)
    test_df = pd.DataFrame(test_scaled, index=test_raw.index, columns=cols)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    logger.info(
        "Scaler (15 features) fitted on train and saved to models/scaler.joblib"
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv("data/processed/train.csv")
    val_df.to_csv("data/processed/val.csv")
    test_df.to_csv("data/processed/test.csv")
    logger.info("Processed CSVs written to data/processed/")

    return train_df, val_df, test_df, scaler
