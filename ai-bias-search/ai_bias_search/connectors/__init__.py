"""Connector registry and factory helpers."""

from __future__ import annotations

from typing import Dict, Type

from .base import SearchConnector
from .consensus import ConsensusConnector
from .core import CoreConnector
from .openalex import OpenAlexConnector
from .perplexity import PerplexityConnector
from .scopus import ScopusConnector
from .scite import SciteConnector
from .semanticscholar import SemanticScholarConnector

CONNECTOR_REGISTRY: Dict[str, Type[SearchConnector]] = {
    OpenAlexConnector.name: OpenAlexConnector,
    SemanticScholarConnector.name: SemanticScholarConnector,
    ScopusConnector.name: ScopusConnector,
    CoreConnector.name: CoreConnector,
    PerplexityConnector.name: PerplexityConnector,
    ConsensusConnector.name: ConsensusConnector,
    SciteConnector.name: SciteConnector,
}


def get_connector(name: str) -> Type[SearchConnector]:
    """Return the connector class for *name*."""

    try:
        return CONNECTOR_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unknown connector: {name}") from exc


__all__ = ["CONNECTOR_REGISTRY", "get_connector"]
