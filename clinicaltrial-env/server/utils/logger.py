"""Structured logging helpers."""

import json
import logging
from typing import Any


def configure_logger(level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("clinicaltrial-env")
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_json(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    """Emit a compact structured log line."""
    logger.info(json.dumps({"event": event, **payload}, default=str))

