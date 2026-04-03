"""Jena Climate dataset loader — download, resample, validate."""

import glob
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.exceptions import DataLoadError, DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)


def load_jena(config: Dict[str, Any]) -> pd.DataFrame:
    """Download Jena Climate, resample to hourly, encode wind direction, validate.

    Args:
        config: Parsed config.yaml dict.

    Returns:
        Hourly DataFrame with 15 features (wd (deg) replaced by wd_sin/wd_cos).

    Raises:
        DataLoadError: If download or CSV parsing fails.
        DataValidationError: If the pandera schema check fails.
    """
    raw_dir = Path("data/raw")
    raw_path = raw_dir / "jena_climate.csv"
    zip_path = raw_dir / "jena_climate.zip"

    if not raw_path.exists():
        try:
            url: str = config["data"]["url"]
            if not url.startswith("https://"):
                raise DataLoadError(f"Only https:// URLs are permitted, got: {url}")
            logger.info("Downloading Jena Climate dataset...")
            urllib.request.urlretrieve(url, str(zip_path))  # nosec B310
            with zipfile.ZipFile(str(zip_path)) as z:
                z.extractall(str(raw_dir))
            csv_files = glob.glob(str(raw_dir / "*.csv"))
            if csv_files and csv_files[0] != str(raw_path):
                shutil.move(csv_files[0], str(raw_path))
        except Exception as exc:
            raise DataLoadError(f"Failed to download Jena Climate: {exc}") from exc

    try:
        df = pd.read_csv(
            str(raw_path),
            index_col="Date Time",
            parse_dates=True,
            dayfirst=True,
        )
        df.index = pd.DatetimeIndex(df.index)
    except Exception as exc:
        raise DataLoadError(f"Failed to parse Jena Climate CSV: {exc}") from exc

    # Resample to hourly (mean) and drop rows with NaN
    df = df.resample(config["data"]["resample_freq"]).mean()
    df = df.dropna()

    # Circular encoding — 359° and 1° are 2° apart, linear distance would be wrong
    df["wd_sin"] = np.sin(df["wd (deg)"] * np.pi / 180)
    df["wd_cos"] = np.cos(df["wd (deg)"] * np.pi / 180)
    df = df.drop(columns=["wd (deg)"])
    logger.info("Wind direction encoded as sin/cos. input_size is now 15.")

    # SHA-256 checksum of the processed DataFrame
    checksum = hashlib.sha256(df.to_csv().encode()).hexdigest()
    checksums_path = raw_dir / "checksums.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(str(checksums_path), "w") as fh:
        json.dump({"jena_climate": checksum}, fh)
    logger.info(
        "Jena Climate loaded",
        extra={"rows": len(df), "checksum_prefix": checksum[:12]},
    )

    # Validate schema BEFORE any split or transformation
    try:
        from src.data.validation import CLIMATE_SCHEMA

        CLIMATE_SCHEMA.validate(df)
    except Exception as exc:
        raise DataValidationError(f"Schema validation failed: {exc}") from exc

    return df
