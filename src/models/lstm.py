"""LSTMForecaster — multi-step time series forecaster."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.logger import get_logger

logger = get_logger(__name__)


class LSTMForecaster(nn.Module):
    """LSTM-based multi-step time series forecaster."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise LSTM layers and projection head from config.

        Args:
            config: Full config dict — all sizes from config['model'].
        """
        super().__init__()
        self.config = config
        m = config["model"]

        self.lstm = nn.LSTM(
            input_size=m["input_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=m["dropout"])
        self.fc = nn.Linear(m["hidden_size"], m["output_size"])
        self.horizon: int = config["training"]["horizon"]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run forward pass through LSTM + linear projection.

        Args:
            x: Input tensor of shape [batch, lookback, input_size].

        Returns:
            Tuple of (predictions [batch, horizon], (h_n, c_n)).
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]
        last_hidden = self.dropout(last_hidden)
        out = torch.stack(
            [self.fc(last_hidden) for _ in range(self.horizon)], dim=1
        )  # [batch, horizon, output_size]
        out = out.squeeze(-1)  # [batch, horizon]
        return out, (h_n, c_n)

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, path: Path) -> "LSTMForecaster":
        """Load a model from a saved checkpoint.

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            Instantiated model with loaded weights.
        """
        checkpoint: dict[str, Any] = torch.load(  # nosec B614
            str(path), map_location="cpu", weights_only=False
        )
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        logger.info("Model loaded from checkpoint", extra={"path": str(path)})
        return model
