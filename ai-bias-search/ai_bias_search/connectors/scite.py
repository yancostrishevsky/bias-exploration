"""Stub connector for Scite."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .base import ConnectorError


class SciteConnector:
    """Scite connector placeholder raising informative errors until configured."""

    name = "scite"

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        pass

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not os.getenv("SCITE_API_KEY"):
            raise ConnectorError("Set SCITE_API_KEY in .env to enable this connector")
        raise ConnectorError("Scite connector is not implemented in this demo")


__all__ = ["SciteConnector"]
