"""FastAPI production application for B3 LSTM Temperature Forecaster.

Exposes three endpoints:
  GET  /api/v1/health       — liveness + model status
  POST /api/v1/forecast     — multi-step temperature forecast (rate-limited)
  GET  /api/v1/model_info   — architecture metadata and evaluation results

Middleware stack (outermost → innermost):
  MaxBodySizeMiddleware  — rejects payloads > config.api.max_payload_mb
  CORSMiddleware         — cross-origin policy
  TrustedHostMiddleware  — host header validation
  Instrumentator         — Prometheus metrics at /metrics
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable
from uuid import uuid4

import joblib
import numpy as np
import psutil
import torch
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.exceptions import ModelLoadError, PredictionError
from src.logger import get_logger
from src.models.lstm import LSTMForecaster

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("config/config.yaml")
CHECKPOINT_PATH = Path("models/lstm_checkpoint.pt")
SCALER_PATH = Path("models/scaler.joblib")
METRICS_PATH = Path("reports/metrics.json")
RESULTS_PATH = Path("reports/results.json")

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as _f:
    CONFIG: dict[str, Any] = yaml.safe_load(_f)

# ── App State ─────────────────────────────────────────────────────────────────
_model: LSTMForecaster | None = None
_scaler: Any = None
_checkpoint: dict[str, Any] = {}
_model_loaded: bool = False
_start_time: float = 0.0
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Content-Length Middleware ─────────────────────────────────────────────────
class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject any request whose Content-Length header exceeds the configured limit."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Pass request through unless Content-Length exceeds the limit.

        Args:
            request: Incoming Starlette request.
            call_next: Next middleware or route handler.

        Returns:
            413 JSON response if oversized; otherwise the downstream response.
        """
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            limit_mb = self.max_bytes // 1_000_000
            return JSONResponse(
                status_code=413,
                content={"detail": f"Payload exceeds {limit_mb} MB limit."},
            )
        return await call_next(request)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model checkpoint and scaler on startup; log on shutdown."""
    global _model, _scaler, _checkpoint, _model_loaded, _start_time

    _start_time = time.time()
    try:
        _checkpoint = torch.load(  # nosec B614
            str(CHECKPOINT_PATH),
            map_location=_device,
            weights_only=False,
        )
        _model = LSTMForecaster(_checkpoint["config"])
        _model.load_state_dict(_checkpoint["model_state"])
        _model.eval()
        _model.to(_device)
        _scaler = joblib.load(SCALER_PATH)
        _model_loaded = True
        logger.info(
            "Model and scaler loaded",
            extra={"epoch": _checkpoint.get("epoch"), "device": str(_device)},
        )
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        logger.error("Startup failed — model not loaded", extra={"error": str(exc)})
        raise ModelLoadError(f"Startup failed: {exc}") from exc

    yield

    logger.info("Application shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LSTM Temperature Forecaster",
    description="Multi-step LSTM forecasting API for Jena Climate data.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware (added last-to-first — outermost first in execution) ───────────
_max_payload_bytes = int(CONFIG["api"]["max_payload_mb"]) * 1_000_000
app.add_middleware(MaxBodySizeMiddleware, max_bytes=_max_payload_bytes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["api"]["cors_origins"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# ── Rate-limit state and error handler ───────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type]
)


# ── PredictionError → 422 ─────────────────────────────────────────────────────
@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request, exc: PredictionError
) -> JSONResponse:
    """Map PredictionError to HTTP 422 Unprocessable Entity.

    Args:
        request: The incoming request (unused but required by FastAPI).
        exc: The PredictionError that was raised.

    Returns:
        JSONResponse with status 422 and the error message.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ── Prometheus metrics ────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ForecastInput(BaseModel):
    """Payload for POST /api/v1/forecast.

    Attributes:
        values: Raw (unscaled) climate readings — shape [lookback, n_features].
                All columns must be present in the same order as the training data.
        feature_names: Column names in the same order as `values` columns.
                       Must contain config.data.target_col (e.g. "T (degC)").
    """

    values: list[list[float]]
    feature_names: list[str]


class ForecastOutput(BaseModel):
    """Response from POST /api/v1/forecast.

    Attributes:
        forecast: Predicted temperatures in °C for each future hour.
        horizon_hours: Number of forecast steps (== config.training.horizon).
        trace_id: UUID identifying this specific inference call.
    """

    forecast: list[float]
    horizon_hours: int
    trace_id: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    """Return service liveness, model status, uptime, and memory usage.

    Returns:
        JSON dict with keys:
          - status (str): "ok" if model loaded, else "degraded".
          - model_loaded (bool): True after successful startup.
          - uptime_seconds (float): Seconds since app started.
          - memory_mb (float): RSS memory of this process in MB.
          - checkpoint_epoch (int | None): Epoch saved in the checkpoint.
    """
    mem_mb = round(psutil.Process().memory_info().rss / 1_000_000, 2)
    return {
        "status": "ok" if _model_loaded else "degraded",
        "model_loaded": _model_loaded,
        "uptime_seconds": round(time.time() - _start_time, 2),
        "memory_mb": mem_mb,
        "checkpoint_epoch": _checkpoint.get("epoch"),
    }


@app.post("/api/v1/forecast", response_model=ForecastOutput)
@limiter.limit(CONFIG["api"]["rate_limit_forecast"])
async def forecast(request: Request, body: ForecastInput) -> ForecastOutput:
    """Run multi-step LSTM forecast and return temperatures in °C.

    The API scales the input internally with the training StandardScaler,
    runs model inference, then inverse-transforms the output back to °C.
    Callers should supply raw (unscaled) climate readings.

    Args:
        request: FastAPI Request object (required by slowapi rate limiter).
        body: ForecastInput with `values` of shape [lookback, n_features]
              and `feature_names` listing each column in the same order.

    Returns:
        ForecastOutput containing `forecast` (list of floats in °C),
        `horizon_hours`, and a unique `trace_id`.

    Raises:
        PredictionError: If row count != lookback, target column is missing,
                         or model inference fails.
    """
    if _model is None:
        raise PredictionError("Model is not loaded — service is starting up.")

    lookback: int = CONFIG["training"]["lookback"]
    if len(body.values) != lookback:
        raise PredictionError(f"Expected {lookback} rows, got {len(body.values)}")

    target_col: str = CONFIG["data"]["target_col"]
    if target_col not in body.feature_names:
        raise PredictionError(
            f"feature_names must include '{target_col}' for inverse scaling."
        )

    try:
        arr = np.array(body.values, dtype=np.float32)  # [lookback, n_features]
        arr_scaled = _scaler.transform(arr)  # standardise in-place copy
        X = (
            torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0).to(_device)
        )  # [1, lookback, n_features]

        with torch.no_grad():
            out, _ = _model(X)  # [1, horizon]

        preds_scaled: np.ndarray = out.squeeze(0).cpu().numpy()  # [horizon]

        # Inverse-transform: rebuild a dummy full-feature array, fill target col
        n_features: int = int(_scaler.n_features_in_)
        target_idx: int = body.feature_names.index(target_col)
        dummy = np.zeros((len(preds_scaled), n_features), dtype=np.float32)
        dummy[:, target_idx] = preds_scaled
        forecast_c: list[float] = _scaler.inverse_transform(dummy)[
            :, target_idx
        ].tolist()

    except (ValueError, RuntimeError) as exc:
        raise PredictionError(f"Inference failed: {exc}") from exc

    logger.info(
        "Forecast complete",
        extra={"horizon": CONFIG["training"]["horizon"], "trace_id": str(uuid4())},
    )

    return ForecastOutput(
        forecast=forecast_c,
        horizon_hours=CONFIG["training"]["horizon"],
        trace_id=str(uuid4()),
    )


@app.get("/api/v1/model_info")
async def model_info() -> dict[str, Any]:
    """Return model architecture metadata and evaluation metrics.

    Reads reports/metrics.json (or reports/results.json as fallback).
    If neither file exists, the 'results' key is omitted from the response.

    Returns:
        JSON dict with keys:
          - model_type (str): Always "LSTMForecaster".
          - hidden_size (int): LSTM hidden dimension from config.
          - num_layers (int): Number of LSTM stacked layers from config.
          - lookback (int): Input window length in hours.
          - horizon (int): Output forecast length in hours.
          - results (dict | absent): Per-horizon MSE/MAE/RMSE if available.
    """
    results: dict[str, Any] | None = None
    for candidate in (METRICS_PATH, RESULTS_PATH):
        if candidate.exists():
            with open(candidate) as fh:
                results = json.load(fh)
            break

    info: dict[str, Any] = {
        "model_type": "LSTMForecaster",
        "hidden_size": CONFIG["model"]["hidden_size"],
        "num_layers": CONFIG["model"]["num_layers"],
        "lookback": CONFIG["training"]["lookback"],
        "horizon": CONFIG["training"]["horizon"],
    }
    if results is not None:
        info["results"] = results
    return info
