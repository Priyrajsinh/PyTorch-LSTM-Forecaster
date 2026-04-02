"""Structured JSON logger for B3 LSTM Forecaster."""

import logging
import sys

from pythonjsonlogger import json as jsonlogger


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-formatted logger with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance with JSON formatter on stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
