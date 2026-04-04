"""Self-contained Hugging Face Space app for LSTM Temperature Forecaster.

LSTMForecaster is inlined here so the Space has zero dependency on the main
repo's src/ package — everything needed is in this single file.

Before deploying, copy the artefacts into hf_space/models/:
    cp models/lstm_checkpoint.pt hf_space/models/lstm_checkpoint.pt
    cp models/scaler.joblib      hf_space/models/scaler.joblib

Tab 1 — Live Forecast: type a city name, fetches last 72h from Open-Meteo,
engineers all 15 Jena Climate features, and returns a 48h forecast plot.
Tab 2 — Upload CSV: original developer interface (72-row raw CSV).
"""

from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import matplotlib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── Feature column order (must match scaler fit order) ───────────────────────
FEATURE_ORDER = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd_sin",
    "wd_cos",
]


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
_TARGET_IDX: int = FEATURE_ORDER.index(_target_col)


# ── Geocoding + weather fetch ─────────────────────────────────────────────────
def geocode(city: str) -> tuple[float, float, str]:
    """Return (lat, lon, display_name) for a city name via Open-Meteo geocoding.

    Raises:
        ValueError: If the city name is not found.
        requests.HTTPError: On non-2xx responses.
    """
    params: dict[str, str | int] = {"name": city, "count": 1}
    resp = requests.get(  # nosec B113
        "https://geocoding-api.open-meteo.com/v1/search",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city!r}")
    r = results[0]
    display = f"{r.get('name', city)}, {r.get('country', '')}"
    return float(r["latitude"]), float(r["longitude"]), display


def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    """Fetch the last 72 complete hourly rows from Open-Meteo.

    Raises:
        ValueError: If fewer than 72 non-NaN rows are returned.
        requests.HTTPError: On non-2xx responses.
    """
    wx_params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,dewpoint_2m,relativehumidity_2m,"
            "surface_pressure,windspeed_10m,windgusts_10m,winddirection_10m"
        ),
        "past_days": 3,
        "forecast_days": 1,
        "timezone": "auto",
    }
    resp = requests.get(  # nosec B113
        "https://api.open-meteo.com/v1/forecast",
        params=wx_params,
        timeout=15,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"]).dropna()
    if len(df) < 72:
        raise ValueError(
            f"Only {len(df)} complete hourly rows available; need at least 72."
        )
    return df.tail(72).reset_index(drop=True)


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Derive all 15 Jena Climate columns from 8 Open-Meteo variables.

    Formulas follow standard meteorological definitions so the result
    matches the distribution the scaler was fit on.

    Returns:
        DataFrame of shape (72, 15) with columns in FEATURE_ORDER.
    """
    t = df_raw["temperature_2m"].astype(float)
    tdew = df_raw["dewpoint_2m"].astype(float)
    rh = df_raw["relativehumidity_2m"].astype(float).clip(0, 100)
    p = df_raw["surface_pressure"].astype(float)  # already hPa == mbar
    ws = df_raw["windspeed_10m"].astype(float) / 3.6  # km/h → m/s
    wg = (
        df_raw["windgusts_10m"].astype(float) / 3.6
        if "windgusts_10m" in df_raw.columns
        else ws * 1.5
    )
    wd_deg = df_raw["winddirection_10m"].astype(float)

    # Potential temperature (Poisson formula)
    tpot = (t + 273.15) * (1000.0 / p) ** 0.286
    # Saturation vapour pressure (Magnus formula)
    vpmax = 6.1078 * np.exp(17.27 * t / (t + 237.3))
    # Actual / deficit vapour pressure
    vpact = (vpmax * rh / 100.0).clip(upper=vpmax)
    vpdef = vpmax - vpact
    # Specific humidity (g/kg) and water vapour mole fraction (mmol/mol)
    sh = 0.622 * vpact / (p - vpact) * 1000.0
    h2oc = sh * 1.608
    # Air density (g/m³) via ideal gas law
    rho = (p * 100.0 / (287.058 * (t + 273.15)) * 1000.0).clip(lower=1e-6)
    # Circular wind-direction encoding
    wd_sin = np.sin(wd_deg * np.pi / 180.0)
    wd_cos = np.cos(wd_deg * np.pi / 180.0)

    return pd.DataFrame(
        {
            "p (mbar)": p.values,
            "T (degC)": t.values,
            "Tpot (K)": tpot.values,
            "Tdew (degC)": tdew.values,
            "rh (%)": rh.values,
            "VPmax (mbar)": vpmax.values,
            "VPact (mbar)": vpact.values,
            "VPdef (mbar)": vpdef.values,
            "sh (g/kg)": sh.values,
            "H2OC (mmol/mol)": h2oc.values,
            "rho (g/m**3)": rho.values,
            "wv (m/s)": ws.values,
            "max. wv (m/s)": wg.values,
            "wd_sin": wd_sin.values,
            "wd_cos": wd_cos.values,
        }
    )[FEATURE_ORDER]


# ── Inference helpers ─────────────────────────────────────────────────────────
def _run_inference(arr: np.ndarray, target_idx: int) -> np.ndarray:
    """Scale → LSTM → inverse-transform; returns forecast_c of shape (horizon,)."""
    arr_scaled = _hf_scaler.transform(arr.astype(np.float32))
    x = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out, _ = _hf_model(x)
    preds_scaled: np.ndarray = out.squeeze(0).numpy()
    dummy = np.zeros((len(preds_scaled), _n_features), dtype=np.float32)
    dummy[:, target_idx] = preds_scaled
    return _hf_scaler.inverse_transform(dummy)[:, target_idx]


# ── CSV-based forecast (original developer interface) ─────────────────────────
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
    forecast_c = _run_inference(df.values, target_idx)
    return "\n".join(f"Hour +{i + 1:2d}: {v:.2f}°C" for i, v in enumerate(forecast_c))


# ── City-based live forecast ──────────────────────────────────────────────────
def predict_from_city(city: str) -> tuple[Figure | None, str]:
    """Geocode city → fetch 72h weather → engineer features → LSTM → plot.

    Returns:
        (matplotlib Figure, summary string) on success.
        (None, error message) on any failure.
    """
    if not city or not city.strip():
        return None, "Please enter a city name."
    try:
        lat, lon, display_name = geocode(city.strip())
        df_raw = fetch_weather(lat, lon)
        df_eng = engineer_features(df_raw)

        if df_eng.shape != (72, 15):
            raise ValueError(f"Feature engineering returned shape {df_eng.shape}.")

        forecast_c = _run_inference(df_eng.values, _TARGET_IDX)

        # Build forecast plot
        x = list(range(1, _horizon + 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, forecast_c, linewidth=2, color="steelblue")
        ax.set_xticks(range(0, _horizon + 1, 6))
        ax.set_xticklabels([f"+{h}h" for h in range(0, _horizon + 1, 6)], rotation=45)
        ax.set_title(f"48h Forecast for {display_name}", fontsize=13)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        avg_1_6 = float(np.mean(forecast_c[:6]))
        avg_7_24 = float(np.mean(forecast_c[6:24]))
        avg_25_48 = float(np.mean(forecast_c[24:48]))
        summary = (
            f"Hours +1 to +6:   avg {avg_1_6:.1f}°C\n"
            f"Hours +7 to +24:  avg {avg_7_24:.1f}°C\n"
            f"Hours +25 to +48: avg {avg_25_48:.1f}°C"
        )
        return fig, summary

    except Exception as exc:  # noqa: BLE001
        return None, f"Error: {exc}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="LSTM Temperature Forecaster") as demo:
    gr.Markdown("# LSTM Temperature Forecaster")

    with gr.Tab("Live Forecast"):
        gr.Markdown(
            "_Model trained on Jena, Germany — "
            "accuracy varies for tropical/arid climates._"
        )
        city_input = gr.Textbox(label="City", placeholder="e.g. London, Tokyo, Mumbai")
        forecast_btn = gr.Button("Get Forecast", variant="primary")
        forecast_plot = gr.Plot()
        forecast_summary = gr.Textbox(label="Summary", lines=3)
        forecast_btn.click(
            predict_from_city,
            inputs=city_input,
            outputs=[forecast_plot, forecast_summary],
        )

    with gr.Tab("Upload CSV (Developer)"):
        gr.Markdown(
            f"Upload {_lookback} hours of raw Jena Climate feature data "
            f"→ get a {_horizon}-step temperature forecast in °C. "
            "All columns must match the training schema and include 'T (degC)'."
        )
        csv_input = gr.File(label="72-row CSV (raw climate features)")
        csv_btn = gr.Button("Run Forecast")
        csv_output = gr.Textbox(label=f"{_horizon}h Forecast (°C)", lines=26)
        csv_btn.click(predict_from_csv, inputs=csv_input, outputs=csv_output)

if __name__ == "__main__":
    demo.launch()
