"""Stub connector for the Perplexity API."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .base import ConnectorError


class PerplexityConnector:
    """Stub implementation that guards on missing API keys."""

    name = "perplexity"

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        pass

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not os.getenv("PERPLEXITY_API_KEY"):
            raise ConnectorError("Set PERPLEXITY_API_KEY in .env to enable this connector")
        raise ConnectorError("Perplexity connector is not implemented in this demo")


__all__ = ["PerplexityConnector"]
