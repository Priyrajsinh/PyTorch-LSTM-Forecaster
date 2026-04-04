"""LSTMForecaster — direct multi-step (MIMO) time series forecaster."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.logger import get_logger

logger = get_logger(__name__)


class LSTMForecaster(nn.Module):
    """LSTM encoder with a single direct multi-step projection head (MIMO).

    The LSTM encodes the lookback window into a hidden state; one linear layer
    then projects that hidden state to all `horizon` future steps simultaneously.
    Each output neuron specialises in its own forecast horizon, so predictions
    are not identical across steps (unlike a repeated single-step approach).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise LSTM layers and MIMO projection head from config.

        Args:
            config: Full config dict — all sizes read from config['model'] and
                config['training'].
        """
        super().__init__()
        self.config = config
        m = config["model"]
        self.horizon: int = config["training"]["horizon"]

        self.lstm = nn.LSTM(
            input_size=m["input_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=m["dropout"])
        # MIMO head: one projection for all horizon steps at once.
        # Each of the `horizon` output neurons specialises in its own step.
        self.fc = nn.Linear(m["hidden_size"], self.horizon)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run forward pass: LSTM encoder → dropout → MIMO linear projection.

        Args:
            x: Input tensor of shape [batch, lookback, input_size].

        Returns:
            Tuple of (predictions [batch, horizon], (h_n, c_n)).
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = self.dropout(lstm_out[:, -1, :])  # [batch, hidden_size]
        out = self.fc(last_hidden)  # [batch, horizon] — all steps in one shot
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
            Instantiated model with loaded weights, set to eval mode.
        """
        checkpoint: dict[str, Any] = torch.load(  # nosec B614
            str(path), map_location="cpu", weights_only=False
        )
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        logger.info("Model loaded from checkpoint", extra={"path": str(path)})
        return model
