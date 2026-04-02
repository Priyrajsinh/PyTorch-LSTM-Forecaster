"""Project-wide exception hierarchy for B3 LSTM Forecaster."""


class ProjectBaseError(Exception):
    """Base exception for all B3 project errors."""


class DataLoadError(ProjectBaseError):
    """Raised when raw data cannot be loaded or parsed."""


class DataValidationError(ProjectBaseError):
    """Raised when a pandera schema validation fails."""


class ModelLoadError(ProjectBaseError):
    """Raised when a model checkpoint cannot be loaded."""


class PredictionError(ProjectBaseError):
    """Raised when inference fails due to bad input or model error."""


class ConfigError(ProjectBaseError):
    """Raised when config/config.yaml is missing or malformed."""
