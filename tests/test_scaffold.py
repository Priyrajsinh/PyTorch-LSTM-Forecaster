"""Day 0 scaffold tests — no pre-built artifacts required."""

import pytest
import numpy as np


def test_climate_schema_exists():
    """Verify CLIMATE_SCHEMA is importable and not None."""
    from src.data.validation import CLIMATE_SCHEMA

    assert CLIMATE_SCHEMA is not None


def test_forecast_input_rejects_empty():
    """ForecastInput must raise on empty values list."""
    from src.data.schemas import ForecastInput

    with pytest.raises(Exception):
        ForecastInput(values=[], feature_names=[])


def test_exceptions_hierarchy():
    """DataLoadError must be a subclass of ProjectBaseError."""
    from src.exceptions import DataLoadError, ProjectBaseError

    assert issubclass(DataLoadError, ProjectBaseError)


def test_seed_sets_torch_determinism(tmp_path):
    """set_seed(42) must produce identical torch tensors across calls."""
    from utils.seed import set_seed
    import torch

    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b)
