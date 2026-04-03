"""Tests for SlidingWindowDataset — all use synthetic data, never real files."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import SlidingWindowDataset


def test_sliding_window_len() -> None:
    """len must equal T - lookback - horizon + 1."""
    data = np.random.randn(1000, 14).astype(np.float32)
    ds = SlidingWindowDataset(data, lookback=72, horizon=24, target_idx=1)
    expected_len = 1000 - 72 - 24 + 1
    assert len(ds) == expected_len, f"Expected {expected_len}, got {len(ds)}"


def test_sliding_window_shapes() -> None:
    """X must be (lookback, features) and y must be (horizon,)."""
    data = np.random.randn(500, 14).astype(np.float32)
    ds = SlidingWindowDataset(data, lookback=72, horizon=24, target_idx=1)
    X, y = ds[0]
    assert X.shape == (72, 14), f"X shape {X.shape}"
    assert y.shape == (24,), f"y shape {y.shape}"


def test_dataloader_no_shuffle() -> None:
    """DataLoader with shuffle=False must return identical first batches on two passes."""
    data = np.random.randn(300, 14).astype(np.float32)
    ds = SlidingWindowDataset(data, lookback=24, horizon=12, target_idx=1)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    first_batch_1, _ = next(iter(loader))
    first_batch_2, _ = next(iter(loader))
    assert torch.allclose(first_batch_1, first_batch_2), "Same seed must give same order"


def test_y_is_target_col_only() -> None:
    """y must contain values from the target column, not all features."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 5)).astype(np.float32)
    target_idx = 3
    ds = SlidingWindowDataset(data, lookback=10, horizon=5, target_idx=target_idx)
    _, y = ds[0]
    # The target column values starting at offset lookback
    expected = torch.tensor(data[10:15, target_idx])
    assert torch.allclose(y, expected), "y must match target column slice"


def test_last_window_is_valid() -> None:
    """The last index (len-1) must not raise an IndexError."""
    data = np.random.randn(100, 14).astype(np.float32)
    ds = SlidingWindowDataset(data, lookback=10, horizon=5, target_idx=0)
    last_idx = len(ds) - 1
    X, y = ds[last_idx]
    assert X.shape == (10, 14)
    assert y.shape == (5,)


def test_consecutive_windows_overlap_by_lookback_minus_one() -> None:
    """Window i+1 shares lookback-1 timesteps with window i."""
    data = np.arange(200 * 3, dtype=np.float32).reshape(200, 3)
    ds = SlidingWindowDataset(data, lookback=10, horizon=5, target_idx=0)
    X0, _ = ds[0]
    X1, _ = ds[1]
    # X1[0] should equal X0[1]
    assert torch.allclose(X1[0], X0[1]), "Consecutive windows must share lookback-1 steps"
