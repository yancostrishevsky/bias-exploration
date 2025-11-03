"""Base protocol and common helpers for search connectors."""

from __future__ import annotations

from typing import Protocol


class SearchConnector(Protocol):
    """Protocol describing a search connector."""

    name: str

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        """Return up to *k* records for *query*."""


class ConnectorError(RuntimeError):
    """Raised when a connector cannot fulfil a request."""


__all__ = ["SearchConnector", "ConnectorError"]
