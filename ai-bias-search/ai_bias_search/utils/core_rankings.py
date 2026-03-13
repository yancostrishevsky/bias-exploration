"""CORE conference ranking lookup helpers.

This module preserves the original public API (`lookup_core_rank`, `is_known_core_acronym`)
but is now backed by the unified rankings registry (`ai_bias_search.rankings`).
"""

from __future__ import annotations

import os
from pathlib import Path

from ai_bias_search.rankings.base import normalize_title
from ai_bias_search.rankings.registry import get_provider
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def set_core_rankings_path(path: Path | None) -> None:
    """Override the CORE rankings dataset path for this process (useful in tests)."""

    if path is None:
        os.environ.pop("CORE_RANKINGS_PATH", None)
    else:
        os.environ["CORE_RANKINGS_PATH"] = str(path)
    clear_core_rankings_cache()


def clear_core_rankings_cache() -> None:
    """Clear the in-memory CORE rankings cache."""

    try:
        provider = get_provider("core")
    except KeyError:
        return
    provider.reset()


def lookup_core_rank(venue_name: str | None, venue_acronym: str | None = None) -> str | None:
    """Lookup the CORE rank for a venue name/acronym (A*, A, B, C)."""

    try:
        provider = get_provider("core")
    except KeyError:
        return None

    if venue_acronym:
        result = provider.match(venue_acronym, None)
        value = result.rank_value
        if isinstance(value, str) and value.strip():
            return value.strip()

    if venue_name:
        result = provider.match(venue_name, None)
        value = result.rank_value
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def is_known_core_acronym(value: str | None) -> bool:
    """Return True when *value* appears as an acronym in the CORE dataset."""

    if not value:
        return False

    try:
        provider = get_provider("core")
    except KeyError:
        return False

    try:
        provider.load()
    except FileNotFoundError:
        return False

    engine = provider._engine  # type: ignore[attr-defined]
    if engine is None:
        return False

    key = normalize_title(value, provider.cfg.normalization)
    if not key:
        return False

    # Core acronyms are indexed as title aliases; use an exact-key membership check.
    return key in engine.title_index


__all__ = [
    "clear_core_rankings_cache",
    "is_known_core_acronym",
    "lookup_core_rank",
    "set_core_rankings_path",
]

