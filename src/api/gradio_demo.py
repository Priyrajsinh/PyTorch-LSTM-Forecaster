"""Gradio local demo — calls the running FastAPI server at localhost:8000.

Usage:
    # 1. Start the API first:
    #    uvicorn src.api.app:app --reload --port 8000
    # 2. In a second terminal:
    #    python -m src.api.gradio_demo

The CSV uploaded must have exactly config.training.lookback (72) rows and
column names matching the training features, including "T (degC)".
"""

import gradio as gr
import httpx
import pandas as pd

API_URL = "http://localhost:8000/api/v1/forecast"
_TIMEOUT = 15.0


def run_forecast(csv_upload: str) -> tuple[str, str]:
    """Upload a CSV of 72 hourly rows and return a 24 h / 48 h temperature forecast.

    The CSV must contain the same feature columns as the training data
    (including "T (degC)") in unscaled °C and SI units.  The API handles
    standardisation internally.

    Args:
        csv_upload: File path provided by the gr.File component.

    Returns:
        Tuple of (horizon_label, forecast_lines) — both plain strings ready
        to display in Gradio Textbox components.
    """
    try:
        df = pd.read_csv(csv_upload)
    except Exception as exc:  # noqa: BLE001
        return "Error", f"Could not read CSV: {exc}"

    payload = {
        "values": df.values.tolist(),
        "feature_names": list(df.columns),
    }

    try:
        resp = httpx.post(API_URL, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        return "Error", f"API error {exc.response.status_code}: {detail}"
    except httpx.RequestError as exc:
        return "Error", f"Could not reach API at {API_URL}: {exc}"

    data = resp.json()
    forecast_lines = "\n".join(
        f"Hour +{i + 1:2d}: {v:.2f}°C" for i, v in enumerate(data["forecast"])
    )
    horizon_label = (
        f"Horizon: {data['horizon_hours']}h  |  trace: {data['trace_id'][:8]}"
    )
    return horizon_label, forecast_lines


demo = gr.Interface(
    fn=run_forecast,
    inputs=gr.File(label="Upload 72-row CSV (raw climate features)"),
    outputs=[
        gr.Textbox(label="Horizon"),
        gr.Textbox(label="Forecast (°C)", lines=26),
    ],
    title="LSTM Temperature Forecaster — Local Demo",
    description=(
        "Upload 72 hours of raw Jena Climate feature data (CSV) → "
        "get a multi-step temperature forecast in °C.  "
        "Requires the FastAPI server to be running on localhost:8000."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=True)
