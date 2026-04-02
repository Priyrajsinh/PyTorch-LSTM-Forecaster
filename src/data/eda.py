"""EDA — exploratory plots saved to reports/figures/."""

from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.logger import get_logger

matplotlib.use("Agg")  # non-interactive backend — no display needed

logger = get_logger(__name__)

FIGURES_DIR = Path("reports/figures")


def run_eda(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Generate and save all EDA figures, log summary statistics.

    Args:
        train_df: Scaled training split.
        val_df: Scaled validation split.
        test_df: Scaled test split.
        config: Parsed config.yaml dict.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    target_col = config["data"]["target_col"]

    full_df = pd.concat([train_df, val_df, test_df])

    # Log per-split stats for target column
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        col = split[target_col]
        logger.info(
            f"{name} {target_col} stats",
            extra={
                "min": round(float(col.min()), 3),
                "max": round(float(col.max()), 3),
                "mean": round(float(col.mean()), 3),
                "std": round(float(col.std()), 3),
            },
        )

    # 1. Temperature time-series
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(full_df.index, full_df[target_col], linewidth=0.4)
    ax.set_title(f"{target_col} — full series (hourly)")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_timeseries.png", dpi=120)
    plt.close(fig)
    logger.info("Saved temperature_timeseries.png")

    # 2. Feature distributions (histograms)
    feature_cols = [c for c in full_df.columns if c != target_col]
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    flat_axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        flat_axes[i].hist(full_df[col].dropna(), bins=50, edgecolor="none")
        flat_axes[i].set_title(col, fontsize=8)
    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)
    fig.suptitle("Feature distributions (scaled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=120)
    plt.close(fig)
    logger.info("Saved feature_distributions.png")

    # 3. Correlation heatmap
    corr = full_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")

    # 4. Train/val/test split timeline
    fig, ax = plt.subplots(figsize=(14, 2))
    ax.barh(
        0, len(train_df), left=0, color="#4C72B0", label=f"Train ({len(train_df):,})"
    )
    ax.barh(
        0,
        len(val_df),
        left=len(train_df),
        color="#DD8452",
        label=f"Val ({len(val_df):,})",
    )
    ax.barh(
        0,
        len(test_df),
        left=len(train_df) + len(val_df),
        color="#55A868",
        label=f"Test ({len(test_df):,})",
    )
    ax.set_xlabel("Hourly timesteps")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title("Chronological train / val / test split")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "train_val_test_split.png", dpi=120)
    plt.close(fig)
    logger.info("Saved train_val_test_split.png")
