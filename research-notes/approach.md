# Research Notes — B3 PyTorch LSTM
## Papers I Read Before Starting
- LSTM (Hochreiter & Schmidhuber, 1997) — long-range time dependencies via gating
- Sequence to Sequence Learning (Sutskever et al., 2014) — multi-step forecasting
## Architecture Decisions
- LSTM over GRU: slightly more expressive, standard choice for weather forecasting
- lookback=72h: captures daily + 3-day weather patterns
- horizon=24h: 24-step ahead forecast, matches P5 ETTh1 structure
- StandardScaler on train only: prevents data leakage from future statistics
- Chronological split: shuffling would leak future into training, completely invalid
## What I Would Do Differently
- Add attention mechanism over LSTM hidden states
- Try Temporal Fusion Transformer (TFT) for multi-horizon
## Surprising Findings
- [Fill in after training]
