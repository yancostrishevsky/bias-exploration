"""Logging helpers with basic masking of sensitive data."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

SENSITIVE_KEYS = {
    "authorization",
    "api-key",
    "api_key",
    "apikey",
    "x-api-key",
    "x-els-apikey",
    "x-els-insttoken",
    "insttoken",
}


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure the root logger and return a project-specific logger."""

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
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
