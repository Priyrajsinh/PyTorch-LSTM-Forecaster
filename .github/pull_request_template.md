## Summary
<!-- What does this PR do? -->

## Changes
- [ ] Feature / Bug fix / Refactor / Docs

## Checklist
- [ ] `make lint` passes (black, flake8, isort, mypy, bandit zero findings)
- [ ] `make test` passes with ≥ 70% coverage
- [ ] `pip-audit` shows no known vulnerabilities
- [ ] No hardcoded hyperparameters (all in config/config.yaml)
- [ ] No `print()` statements — using `get_logger` from src/logger.py
- [ ] Time series data NOT shuffled anywhere
- [ ] Scaler fitted on train split ONLY

## Test plan
<!-- How was this tested? -->
