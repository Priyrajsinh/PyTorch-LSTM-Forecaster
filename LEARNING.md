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
