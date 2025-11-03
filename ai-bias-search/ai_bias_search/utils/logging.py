"""Logging helpers with basic masking of sensitive data."""

from __future__ import annotations

import logging
from typing import Any, Mapping


SENSITIVE_KEYS = {"authorization", "api-key", "api_key", "apikey"}


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root logger and return a project-specific logger."""

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger("ai_bias_search")


def mask_sensitive(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *data* with sensitive header values replaced."""

    masked: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            masked[key] = "****"
        else:
            masked[key] = value
    return masked
