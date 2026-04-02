"""Pydantic schemas for API request and response validation."""
import uuid
from typing import Optional

from pydantic import BaseModel, validator


class ForecastInput(BaseModel):
    """Input schema for the /api/v1/forecast endpoint.

    Attributes:
        values: 2-D list of shape [lookback, n_features] with scaled feature values.
        feature_names: Ordered list of feature column names (must match training order).
    """

    values: list[list[float]]
    feature_names: list[str]

    @validator("values")
    def check_shape(cls, v: list[list[float]], values: dict) -> list[list[float]]:
        """Validate that values is non-empty.

        Args:
            v: The values list to validate.
            values: Already-validated fields dict.

        Returns:
            The validated values list.

        Raises:
            ValueError: If values is empty.
        """
        if len(v) == 0:
            raise ValueError("values cannot be empty")
        return v


class ForecastOutput(BaseModel):
    """Output schema for the /api/v1/forecast endpoint.

    Attributes:
        forecast: Predicted T (degC) values for each horizon step.
        horizon_hours: Number of hours ahead forecast covers.
        mse: Optional mean squared error if ground truth was provided.
        trace_id: UUID for request tracing.
    """

    forecast: list[float]
    horizon_hours: int
    mse: Optional[float] = None
    trace_id: str = str(uuid.uuid4())
