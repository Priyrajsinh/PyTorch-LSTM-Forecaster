"""Day 0 scaffold tests — no pre-built artifacts required."""

import logging
from pathlib import Path

import pytest
import torch


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

    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b)


def test_get_logger_returns_logger():
    """get_logger must return a Logger with the correct name."""
    from src.logger import get_logger

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"
    assert logger.level == logging.INFO


def test_get_logger_no_duplicate_handlers():
    """Calling get_logger twice with the same name must not add extra handlers."""
    from src.logger import get_logger

    logger1 = get_logger("test.dedup")
    logger2 = get_logger("test.dedup")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1


def test_base_forecaster_requires_abstract_methods():
    """BaseForecaster cannot be instantiated without implementing abstract methods."""
    from src.models.base import BaseForecaster

    with pytest.raises(TypeError):
        BaseForecaster()  # type: ignore[abstract]


def test_base_forecaster_concrete_subclass():
    """A concrete subclass of BaseForecaster must run forward and from_checkpoint."""
    from src.models.base import BaseForecaster

    class DummyForecaster(BaseForecaster):
        def forward(self, x: torch.Tensor, hidden):  # type: ignore[override]
            return x, hidden

        @classmethod
        def from_checkpoint(cls, path: Path) -> "DummyForecaster":
            return cls()

    model = DummyForecaster()
    x = torch.zeros(1, 72, 15)
    out, h = model(x, None)
    assert out.shape == x.shape
    assert DummyForecaster.from_checkpoint(Path(".")) is not None
