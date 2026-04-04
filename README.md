# PyTorch LSTM Temperature Forecaster

> Multi-step time-series forecasting on the Jena Climate dataset.
> Stacked LSTM trained from scratch in PyTorch, served via FastAPI, tracked with MLflow,
> and deployed as a live Gradio demo — type any city and get a 48-hour forecast.

[![CI](https://github.com/Priyrajsinh/PyTorch-LSTM-Forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/Priyrajsinh/PyTorch-LSTM-Forecaster/actions/workflows/ci.yml)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Priyrajsinh/PyTorch-LSTM-Forecaster)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Coverage](https://img.shields.io/badge/Coverage-86%25-green)

---

## Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/Priyrajsinh/PyTorch-LSTM-Forecaster)**

Type any city name → get a 48-hour temperature forecast plot, no CSV upload needed.
Real weather data is fetched live from [Open-Meteo](https://open-meteo.com) (no API key required),
all 15 Jena Climate features are reconstructed from thermodynamic formulas, and the LSTM runs inference in under a second.

---

## Results

| Horizon | MSE (°C²) | MAE (°C) | RMSE (°C) |
|:-------:|:---------:|:--------:|:---------:|
| 6 h     | 2.65      | 1.22     | 1.63      |
| 12 h    | 4.25      | 1.55     | 2.06      |
| 24 h    | 5.99      | 1.87     | 2.45      |
| **48 h**| **9.10**  | **2.31** | **3.02**  |

> Model predicts temperature 6 to 48 hours ahead using the last 72 hours of climate data as input.

![Forecast vs Actual](reports/figures/forecast_vs_actual.png)

---

## Architecture

```
Jena Climate CSV  (420K rows, 10-min intervals)
        │
        ▼
  Hourly resample + wind direction sin/cos encoding   ← src/data/dataset.py
        │
        ▼
  Chronological split  70 / 15 / 15                   ← src/data/preprocessing.py
  StandardScaler  (fit on train ONLY)
        │
        ▼
  SlidingWindowDataset  lookback=72h, horizon=48h     ← src/data/torch_dataset.py
        │
        ▼
  LSTMForecaster  (2 layers · hidden=64 · MIMO head)  ← src/models/lstm.py
        │
        ▼
  FastAPI  /forecast  +  Gradio Space                 ← src/api/  |  hf_space/
```

**MIMO (Multi-Input Multi-Output):** the model outputs all 48 forecast steps in a single forward pass via a linear projection head — no autoregressive decoding, no error accumulation.

---

## Stack

| Layer | Tool |
|---|---|
| Model | PyTorch `nn.LSTM` + linear head (MIMO) |
| Data validation | Pandera schema on all 15 Jena columns |
| Experiment tracking | MLflow (params, metrics, artefacts) |
| REST API | FastAPI · slowapi rate limiting · Prometheus `/metrics` |
| Live demo | Gradio Blocks · Open-Meteo weather API |
| CI/CD | GitHub Actions · black · flake8 · mypy · bandit · pytest-cov |
| Container | Docker multi-stage (non-root user, minimal image) |

---

## Project Structure

```
PyTorch-LSTM-Forecaster/
├── src/
│   ├── data/          — loading, Pandera validation, preprocessing, EDA
│   ├── models/        — LSTMForecaster (MIMO, stacked LSTM + linear head)
│   ├── training/      — train loop, gradient clipping, early stopping, MLflow
│   ├── evaluation/    — multi-horizon MSE/MAE/RMSE, forecast plot
│   └── api/           — FastAPI app (rate-limited, Prometheus metrics)
├── hf_space/          — self-contained Gradio Space (no FastAPI dependency)
│   ├── app.py         — geocode → Open-Meteo → feature engineering → LSTM
│   └── models/        — checkpoint + scaler (gitignored, copy before deploy)
├── tests/             — 50 tests · 86% coverage
├── config/            — config.yaml (all hyperparameters, never hardcoded)
├── Dockerfile         — multi-stage, non-root appuser
└── Makefile           — install / train / test / lint / serve / docker-build
```

---

## Quickstart

```bash
git clone https://github.com/Priyrajsinh/PyTorch-LSTM-Forecaster.git
cd PyTorch-LSTM-Forecaster
pip install -r requirements.txt

# Train the model
python -m src.training.train --config config/config.yaml

# Evaluate across all horizons
python -m src.evaluation.evaluate --config config/config.yaml

# Run the REST API (http://localhost:8000/docs)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t b3-lstm-forecaster .
docker run -p 8000:8000 b3-lstm-forecaster
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Liveness check + model status + uptime |
| `POST` | `/api/v1/forecast` | 72-row climate input → 48-step forecast |
| `GET`  | `/api/v1/model_info` | Architecture metadata + benchmark results |
| `GET`  | `/metrics` | Prometheus scrape endpoint |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"values": [[...72 rows × 15 cols...]], "feature_names": ["p (mbar)", ...]}'
```

---

## How the Live Forecast Works

The Gradio demo re-derives all 15 Jena Climate features from 8 freely available
Open-Meteo variables — no retraining needed:

| Open-Meteo variable | Jena column | How |
|---|---|---|
| `surface_pressure` | `p (mbar)` | Direct |
| `temperature_2m` | `T (degC)` | Direct |
| `dewpoint_2m` | `Tdew (degC)` | Direct |
| `relativehumidity_2m` | `rh (%)` | Direct |
| `windspeed_10m` | `wv (m/s)` | ÷ 3.6 (km/h → m/s) |
| `windgusts_10m` | `max. wv (m/s)` | ÷ 3.6 |
| `winddirection_10m` | `wd_sin`, `wd_cos` | sin/cos circular encoding |
| Derived | `Tpot`, `VPmax`, `VPact`, `VPdef`, `sh`, `H2OC`, `rho` | Poisson / Magnus / ideal gas |

---

## References

- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation 9(8).
- Chollet (2017). *Deep Learning with Python* — Jena Climate dataset introduced here.
- Open-Meteo — free, no-key weather API used for the live demo.
