"""Ranking list providers (CORE, JIF, and future lists).

The rankings subsystem is configured via YAML files in `ai_bias_search/rankings/sources/`.
"""

from __future__ import annotations

from ai_bias_search.rankings.registry import get_provider, list_rankings, match_all

__all__ = ["get_provider", "list_rankings", "match_all"]

