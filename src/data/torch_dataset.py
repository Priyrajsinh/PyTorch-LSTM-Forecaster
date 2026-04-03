"""SlidingWindowDataset — fixed-lookback windows for time-series forecasting."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """PyTorch Dataset using a sliding window over time series data.

    Args:
        data: Scaled array of shape [T, n_features].
        lookback: Input window size in hours.
        horizon: Forecast steps ahead.
        target_idx: Column index of target variable T (degC).
    """

    def __init__(
        self,
        data: np.ndarray,
        lookback: int,
        horizon: int,
        target_idx: int,
    ) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.target_idx = target_idx

    def __len__(self) -> int:
        # CRITICAL formula — never use len(data) - lookback:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_window, target_sequence).

        Args:
            idx: Window start index.

        Returns:
            Tuple of (X of shape [lookback, n_features], y of shape [horizon]).
        """
        X = self.data[idx : idx + self.lookback]  # [lookback, n_features]
        y = self.data[
            idx + self.lookback : idx + self.lookback + self.horizon,
            self.target_idx,
        ]  # [horizon]
        return X, y
