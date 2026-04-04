"""Self-contained Hugging Face Space app for LSTM Temperature Forecaster.

LSTMForecaster is inlined here so the Space has zero dependency on the main
repo's src/ package — everything needed is in this single file.

Before deploying, copy the artefacts into hf_space/models/:
    cp models/lstm_checkpoint.pt hf_space/models/lstm_checkpoint.pt
    cp models/scaler.joblib      hf_space/models/scaler.joblib

The uploaded CSV must have exactly 72 rows and the same feature columns as
the training data (including "T (degC)") in raw unscaled units.
"""

from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── LSTMForecaster (inlined — no src/ dependency) ────────────────────────────
class LSTMForecaster(nn.Module):
    """LSTM encoder with a single direct multi-step projection head (MIMO)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
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
        self.fc = nn.Linear(m["hidden_size"], self.horizon)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = self.dropout(lstm_out[:, -1, :])
        out = self.fc(last_hidden)
        return out, (h_n, c_n)


# ── Load artefacts once at module import ──────────────────────────────────────
_MODELS_DIR = Path("models")

_ckpt: dict[str, Any] = torch.load(  # nosec B614
    str(_MODELS_DIR / "lstm_checkpoint.pt"), map_location="cpu", weights_only=False
)
_hf_model = LSTMForecaster(_ckpt["config"])
_hf_model.load_state_dict(_ckpt["model_state"])
_hf_model.eval()

_hf_scaler = joblib.load(_MODELS_DIR / "scaler.joblib")
_n_features: int = int(_hf_scaler.n_features_in_)
_target_col: str = _ckpt["config"]["data"]["target_col"]
_lookback: int = _ckpt["config"]["training"]["lookback"]
_horizon: int = _ckpt["config"]["training"]["horizon"]


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_from_csv(csv_file: str) -> str:
    """Load CSV, scale input, run LSTM inference, inverse-transform to °C.

    Args:
        csv_file: File path provided by gr.File.  Must contain exactly
                  `lookback` rows and the same feature columns as training,
                  including the target column (e.g. "T (degC)").

    Returns:
        Multi-line string with one "Hour +N: X.XX°C" entry per forecast step,
        or an error message if the input is invalid.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading CSV: {exc}"

    if len(df) != _lookback:
        return f"Expected {_lookback} rows, got {len(df)}."

    if _target_col not in df.columns:
        return f"CSV must contain a column named '{_target_col}'."

    target_idx: int = list(df.columns).index(_target_col)
    arr = df.values.astype(np.float32)
    arr_scaled = _hf_scaler.transform(arr)

    x = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out, _ = _hf_model(x)

    preds_scaled: np.ndarray = out.squeeze(0).numpy()

    dummy = np.zeros((len(preds_scaled), _n_features), dtype=np.float32)
    dummy[:, target_idx] = preds_scaled
    forecast_c: np.ndarray = _hf_scaler.inverse_transform(dummy)[:, target_idx]

    return "\n".join(
        f"Hour +{i + 1:2d}: {v:.2f}°C" for i, v in enumerate(forecast_c)
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="72-row CSV (raw climate features)"),
    outputs=gr.Textbox(label=f"{_horizon}h Forecast (°C)", lines=26),
    title="LSTM Temperature Forecaster",
    description=(
        f"Upload {_lookback} hours of raw Jena Climate feature data → "
        f"get a {_horizon}-step temperature forecast in °C.  "
        "All columns must match the training schema and include 'T (degC)'."
    ),
)

if __name__ == "__main__":
    demo.launch()
