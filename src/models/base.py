"""Abstract base class for all B3 forecasting models."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseForecaster(ABC, nn.Module):
    """Abstract base for all PyTorch forecasting modules.

    Subclasses must implement forward() and from_checkpoint().
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, hidden: Any) -> tuple[torch.Tensor, Any]:
        """Run a forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, features).
            hidden: Hidden state tuple for recurrent models.

        Returns:
            Tuple of (output tensor, updated hidden state).
        """

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path: Path) -> "BaseForecaster":
        """Load a model from a saved checkpoint.

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            Instantiated model with loaded weights.
        """
