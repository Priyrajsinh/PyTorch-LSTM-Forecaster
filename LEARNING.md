# B3: PyTorch LSTM Time Series Forecaster
# Owner: Priyrajsinh Parmar | github.com/Priyrajsinh
Stack: Python 3.12, torch, torchinfo, mlflow, fastapi, uvicorn, gradio,
       pandera, pydantic, slowapi, prometheus-fastapi-instrumentator,
       python-json-logger, bandit, scikit-learn, pandas, numpy, matplotlib
Config: ALL hyperparameters in config/config.yaml — never hardcode
Logging: get_logger from src/logger.py
Exceptions: all errors through src/exceptions.py
Validation: pandera CLIMATE_SCHEMA in src/data/validation.py before any split
Schemas: ForecastInput / ForecastOutput in src/data/schemas.py
Security: bandit -r src/ -ll must return zero findings
Dataset: Jena Climate (420K rows, 14 features, 10-min → resample 1H)
Target: T (degC) temperature prediction
Split: CHRONOLOGICAL ONLY — never shuffle time series
Scaler: StandardScaler fit on train ONLY, saved to models/scaler.joblib
Model: LSTMForecaster(nn.Module) — hidden_size=64, num_layers=2
Docker image: b3-lstm-forecaster (hardcoded everywhere)
HF Space: self-contained, loads model directly — no localhost calls
NEVER commit: models/lstm_checkpoint.pt data/raw/*.csv
Commands: make install/train/test/lint/serve/gradio/audit/docker-build

---

## Day 6 — 2026-04-04 — FastAPI /forecast, self-contained HF Space, Docker multi-stage
> Project: PyTorch-LSTM-Forecaster

### What was done
- Built `src/api/app.py`: production FastAPI v5 with slowapi, CORS, TrustedHost, MaxBodySize middleware, Prometheus /metrics, lifespan startup, scaler-aware inference.
- Built `src/api/gradio_demo.py`: local Gradio UI calling localhost:8000/api/v1/forecast.
- Built `hf_space/app.py`: self-contained Space loading model/scaler directly from `hf_space/models/`.
- Built multi-stage `Dockerfile` producing image `b3-lstm-forecaster` with a non-root `appuser`.
- Wrote `tests/test_api.py`: 8 hermetic tests (health, forecast shape/validation, model_info, /metrics) using `TestClient` with patched `torch.load` / `joblib.load`.

### Why it was done
- Expose the trained model as a REST API for programmatic consumption and as a Gradio demo for end-user exploration.
- Docker containerises the API so it runs identically on any host.
- HF Space gives a public shareable demo without requiring a running server.

### How it was done
- Used `@asynccontextmanager lifespan` (FastAPI >= 0.95 pattern) instead of deprecated `@on_event("startup")`.
- API scales raw input via `scaler.transform()`, runs model, then inverse-transforms predictions to °C — callers send raw climate data, get real temperatures back.
- Tests patch `torch.load` and `joblib.load` at the top level before `TestClient` triggers the lifespan — no real checkpoint files needed in CI.
- Dockerfile: Stage 1 `pip install --user` into builder; Stage 2 copies only `src/`, `models/`, `config/` and `~/.local` from builder.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `slowapi` | FastAPI-native rate limiter, reads config string "20/minute" | `fastapi-limiter` | Requires Redis for state; overkill for single-instance |
| `prometheus-fastapi-instrumentator` | One-line `.instrument(app).expose()` with zero config | Manual `prometheus_client` | More boilerplate, no auto-labelling of routes |
| `lifespan` context manager | Modern async startup/shutdown, no deprecation warnings | `@app.on_event("startup")` | Deprecated since FastAPI 0.95 |
| `BaseHTTPMiddleware` | Clean per-request dispatch, standard Starlette API | Custom ASGI middleware | Lower-level, more error-prone |
| Multi-stage Dockerfile | Final image has no build tools; minimises attack surface | Single-stage | Larger image, build deps in production |

### Definitions (plain English)
- **lifespan**: An async context manager run once when FastAPI starts/stops — the right place to load heavy resources like ML models.
- **slowapi**: A rate-limiter for FastAPI/Starlette that counts requests per key (e.g. IP) over a time window and returns 429 when exceeded.
- **Instrumentator**: A library that automatically records latency, status codes, and request counts for every FastAPI route and exposes them at `/metrics` for Prometheus to scrape.
- **Multi-stage Dockerfile**: A Dockerfile with multiple `FROM` stages; only artefacts explicitly `COPY --from=builder` reach the final image, keeping it lean.
- **Inverse transform**: Undoing StandardScaler's standardisation — multiplying by the training std and adding the training mean — to convert model outputs from z-score space back to original units (°C).

### Real-world use case
- Stripe and Uber serve ML models behind rate-limited FastAPI/Flask services; Prometheus scrapes `/metrics` and Grafana dashboards alert on latency spikes.
- HuggingFace Spaces uses exactly this self-contained Gradio pattern: model artefacts bundled in the repo, no external API dependencies.

### How to remember it
- Think of the API as a **pipeline in reverse**: training scaled data going *in*, so the API must scale input the same way → run model → un-scale output. Same scaler, same direction, opposite ends.

### Status
- [x] Done
- Next step: Day 7 — push HF Space, write README, final portfolio polish.

---

## Day 7 — 2026-04-04 — pytest suite, 86% coverage, CI gates green, README

> Project: PyTorch-LSTM-Forecaster

### What was done
- Wrote `tests/test_data.py`, `tests/test_model.py`, `tests/test_training.py`, `tests/test_evaluation.py` for 86% total branch coverage.
- Configured `pytest-cov` with `--cov-fail-under=70` as a CI hard gate.
- Added GitHub Actions `ci.yml`: install → lint (black/isort/flake8/mypy/bandit) → test with coverage.
- Wrote `README.md` with architecture diagram, benchmark results table, and HF Space badge.

### Why it was done
- Untested ML code silently degrades; coverage gates catch regressions before they reach prod.
- CI enforces code style and security checks on every push, not just locally.

### How it was done
- Patched `torch.load` and `joblib.load` at import time so unit tests never need real model artefacts.
- Used `pytest.fixture` with `tmp_path` for file I/O tests to keep them hermetic.
- `ci.yml` runs `python -m flake8 src/ tests/ --max-line-length=88` — matching black's line length to avoid false positives.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `pytest-cov` | Integrates directly with pytest runner; one flag adds coverage | `coverage.py` standalone | Requires separate `coverage run` + `coverage report` invocation |
| `unittest.mock.patch` | Standard library, no extra deps | `pytest-mock` | Extra dep; `patch` is already sufficient |
| GitHub Actions | Free for public repos; YAML-native matrix builds | CircleCI / Travis | Paid tiers required for scale; less native GitHub integration |

### Definitions (plain English)
- **Branch coverage**: Measures whether every `if`/`else` path was exercised — stricter than line coverage which only checks if a line ran at all.
- **CI gate**: A required check that must pass before a PR can merge — turns quality standards from guidelines into enforceable rules.
- **`--cov-fail-under`**: A pytest-cov flag that exits with a non-zero code if total coverage drops below the threshold, which makes the CI step fail.

### Real-world use case
- Google, Airbnb, and Stripe all require coverage thresholds as merge gates; GitHub Actions is the industry standard for open-source CI pipelines.

### How to remember it
- Tests are the **safety net under a tightrope walker**: you don't notice them until something falls — then they're the only thing that matters.

### Status
- [x] Done
- Next step: Day 8 — city-based live forecast in the Gradio demo.

---

## Day 8 — 2026-04-04 — City-based live forecast via Open-Meteo API

> Project: PyTorch-LSTM-Forecaster

### What was done
- Added `geocode(city)` — hits Open-Meteo geocoding API, returns `(lat, lon, display_name)`.
- Added `fetch_weather(lat, lon)` — fetches last 72 complete hourly rows from Open-Meteo forecast API (no API key required).
- Added `engineer_features(df_raw)` — derives all 15 Jena Climate columns from 8 raw Open-Meteo variables using thermodynamic formulas (Magnus, Poisson, ideal gas law).
- Added `predict_from_city(city)` — orchestrates the full pipeline and returns a matplotlib forecast plot + 3-bucket summary string.
- Replaced `gr.Interface` with `gr.Blocks` + 2 tabs: "Live Forecast" (new) and "Upload CSV (Developer)" (original, unchanged).

### Why it was done
- The original demo required a 72-row CSV with 15 specific columns — a format no real user has. The city tab makes the demo usable by anyone.
- No model retraining needed: all 15 features the scaler expects can be reconstructed from publicly available weather data.

### How it was done
- Open-Meteo provides 8 raw variables (T, Tdew, rh, pressure, wind speed/gust/direction); the remaining 7 Jena columns are derived from thermodynamic identities.
- Used `past_days=3&forecast_days=1` to guarantee 72+ complete hourly rows even for sparse stations.
- Safety clamps applied: `rh` clamped to [0, 100]; `VPact` clamped below `VPmax`; `rho` clamped above zero to prevent division by zero in downstream formulas.
- `matplotlib.use("Agg")` called before `import matplotlib.pyplot` to prevent display errors in headless HF Space containers.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| Open-Meteo | Free, no API key, HTTPS, returns hourly data matching Jena schema | OpenWeatherMap | Requires API key; free tier rate-limited to 60 calls/min |
| `gr.Blocks` | Enables multi-tab layout, granular component wiring | `gr.Interface` | Single-function only; can't nest multiple workflows |
| Thermodynamic formulas | Exact match to Jena dataset derivation; no extra ML model | A second model to predict missing features | Extra complexity, extra potential for distribution shift |

### Definitions (plain English)
- **Magnus formula**: `VPmax = 6.1078 × exp(17.27×T / (T+237.3))` — approximates the saturation vapour pressure of water in millibars given temperature T in °C.
- **Poisson formula for potential temperature**: `Tpot = (T+273.15) × (1000/p)^0.286` — the temperature air would have if brought adiabatically to 1000 mbar; used to compare air masses at different altitudes.
- **Circular encoding**: Replacing a cyclic angle θ with `(sin θ, cos θ)` so 359° and 1° are numerically close — essential for wind direction in ML inputs.
- **`gr.Blocks`**: Gradio's low-level layout API; lets you arrange any mix of inputs, outputs, and event handlers inside tabs, rows, and columns.

### Real-world use case
- Weather APIs + thermodynamic feature engineering is the standard pattern in NWP (numerical weather prediction) pre-processing pipelines at ECMWF and NOAA.
- HuggingFace Spaces demos use exactly this Blocks + multi-tab pattern to expose different model modes to end users.

### How to remember it
- Think of Open-Meteo as a **CSV generator**: it gives you 8 columns, and thermodynamics gives you the other 7 for free — no model needed.

### Status
- [x] Done
- Next step: Push updated HF Space; test with London, Mumbai, Denver, Dubai.

---
