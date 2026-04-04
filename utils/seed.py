"""Reproducibility utilities for B3 LSTM Forecaster."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across random, numpy, and torch.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
