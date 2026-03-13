"""Backward-compatible import location for Scopus enrichment.

The implementation lives in `ai_bias_search.enrichment.scopus`.
"""

from __future__ import annotations

from ai_bias_search.enrichment.scopus import (  # noqa: F401
    BASE_URL,
    CACHE_DIR,
    ScopusAuthError,
    ScopusEnricher,
    ScopusEnrichError,
    ScopusPermissionError,
    enrich_with_scopus,
)
from ai_bias_search.enrichment.scopus_rankings import (  # noqa: F401
    ScopusRankingsEnricher,
    enrich_with_scopus_rankings,
)

__all__ = [
    "BASE_URL",
    "CACHE_DIR",
    "ScopusEnrichError",
    "ScopusAuthError",
    "ScopusPermissionError",
    "ScopusEnricher",
    "enrich_with_scopus",
    "ScopusRankingsEnricher",
    "enrich_with_scopus_rankings",
]
