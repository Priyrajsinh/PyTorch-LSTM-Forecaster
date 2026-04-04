"""Self-contained Hugging Face Space app for LSTM Temperature Forecaster.

Loads the model and scaler directly from hf_space/models/ — no external API
calls.  Before deploying copy the artefacts:

    cp models/lstm_checkpoint.pt hf_space/models/lstm_checkpoint.pt
    cp models/scaler.joblib      hf_space/models/scaler.joblib

The uploaded CSV must have exactly 72 rows and the same feature columns as
the training data (including "T (degC)") in raw unscaled units.
"""

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch

from src.models.lstm import LSTMForecaster

# ── Load artefacts once at module import ──────────────────────────────────────
_MODELS_DIR = Path("models")
_CHECKPOINT_PATH = _MODELS_DIR / "lstm_checkpoint.pt"
_SCALER_PATH = _MODELS_DIR / "scaler.joblib"

_ckpt: dict = torch.load(  # nosec B614
    str(_CHECKPOINT_PATH), map_location="cpu", weights_only=False
)
_hf_model = LSTMForecaster(_ckpt["config"])
_hf_model.load_state_dict(_ckpt["model_state"])
_hf_model.eval()

_hf_scaler = joblib.load(_SCALER_PATH)
_n_features: int = int(_hf_scaler.n_features_in_)
_target_col: str = _ckpt["config"]["data"]["target_col"]
_lookback: int = _ckpt["config"]["training"]["lookback"]
_horizon: int = _ckpt["config"]["training"]["horizon"]


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

    # Scale input with the training scaler
    arr = df.values.astype(np.float32)  # [lookback, n_features]
    arr_scaled = _hf_scaler.transform(arr)

    x = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(
        0
    )  # [1, lookback, n_features]

    with torch.no_grad():
        out, _ = _hf_model(x)  # [1, horizon]

    preds_scaled: np.ndarray = out.squeeze(0).numpy()  # [horizon]

    # Inverse-transform target column back to °C
    dummy = np.zeros((len(preds_scaled), _n_features), dtype=np.float32)
    dummy[:, target_idx] = preds_scaled
    forecast_c: np.ndarray = _hf_scaler.inverse_transform(dummy)[:, target_idx]

    return "\n".join(f"Hour +{i + 1:2d}: {v:.2f}°C" for i, v in enumerate(forecast_c))


demo = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="72-row CSV (raw climate features)"),
    outputs=gr.Textbox(label="24 h Forecast (°C)", lines=26),
    title="LSTM Temperature Forecaster",
    description=(
        "Upload 72 hours of raw Jena Climate feature data → "
        "get a multi-step temperature forecast in °C.  "
        "All columns must match the training schema and include 'T (degC)'."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
