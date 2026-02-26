"""Ranking providers registry (config discovery + match aggregation)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from diskcache import Cache

from ai_bias_search.rankings.base import MatchResult, RankingConfig, RankingEntry, RankingProvider
from ai_bias_search.rankings.base import iter_normalized_issns, normalize_title
from ai_bias_search.rankings.io import load_config, load_dataset, resolve_dataset_path
from ai_bias_search.rankings.match import MatchingEngine
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()

SOURCES_DIR = Path(__file__).resolve().parent / "sources"
CACHE_ROOT = (Path(__file__).resolve().parents[2] / "data" / "cache" / "rankings").resolve()


def _stable_signature(cfg: RankingConfig) -> str:
    dataset_path = resolve_dataset_path(cfg)
    try:
        dataset_mtime = int(dataset_path.stat().st_mtime)
    except FileNotFoundError:
        dataset_mtime = 0
    config_path = cfg.config_path
    try:
        config_mtime = int(config_path.stat().st_mtime) if config_path else 0
    except FileNotFoundError:
        config_mtime = 0

    payload = {
        "id": cfg.id,
        "dataset": str(dataset_path),
        "dataset_mtime": dataset_mtime,
        "config_mtime": config_mtime,
        "fuzzy_threshold": cfg.fuzzy_threshold,
        "allow_fuzzy": cfg.allow_fuzzy,
        "validate_issn_checksum": cfg.validate_issn_checksum,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


@dataclass(slots=True)
class DatasetRankingProvider(RankingProvider):
    cfg: RankingConfig
    _engine: MatchingEngine | None = None
    _cache: Cache | None = None
    _signature: str | None = None
    _cache_hits: int = 0
    _cache_misses: int = 0

    @property
    def id(self) -> str:  # type: ignore[override]
        return self.cfg.id

    @property
    def label(self) -> str:  # type: ignore[override]
        return self.cfg.label

    def load(self) -> None:
        if self._engine is not None:
            return
        entries = load_dataset(self.cfg)
        self._engine = MatchingEngine.build(self.cfg, entries)
        self._signature = _stable_signature(self.cfg)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache_dir = (CACHE_ROOT / self.cfg.id).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(cache_dir)

    def reset(self) -> None:
        """Drop in-memory indices and close the on-disk cache (useful in tests)."""

        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:
                pass
        self._engine = None
        self._cache = None
        self._signature = None
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, title_norm: str, issns_norm: Sequence[str]) -> str:
        issn_part = ",".join(issns_norm)
        signature = self._signature or "nosig"
        return f"{signature}|t:{title_norm}|i:{issn_part}"

    def match(self, venue_title: str | None, issn_list: list[str] | None) -> MatchResult:
        try:
            self.load()
        except FileNotFoundError as exc:
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=0.0,
                evidence={"error": str(exc)},
                matched=False,
            )

        if self._engine is None:
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=0.0,
                evidence={"error": "provider_not_loaded"},
                matched=False,
            )

        title_norm = normalize_title(venue_title, self.cfg.normalization) if venue_title else ""
        issns_norm = tuple(
            iter_normalized_issns(issn_list or [], validate_checksum=self.cfg.validate_issn_checksum)
        )

        cache = self._cache
        if cache is None:
            return self._engine.match(venue_title, issn_list)

        key = self._cache_key(title_norm, issns_norm)
        cached = cache.get(key)
        if isinstance(cached, MatchResult):
            self._cache_hits += 1
            return cached

        result = self._engine.match(venue_title, issn_list)
        cache.set(key, result, expire=int(self.cfg.cache_ttl_days) * 24 * 60 * 60)
        self._cache_misses += 1
        return result

    def resolve_entry(self, result: MatchResult) -> RankingEntry | None:
        """Resolve the matched `RankingEntry` for *result* when available."""

        engine = self._engine
        if engine is None or not result.matched:
            return None

        evidence = result.evidence
        if not isinstance(evidence, Mapping):
            return None

        if result.method == "issn_exact":
            issn = evidence.get("issn")
            return engine.issn_index.get(issn) if isinstance(issn, str) else None
        if result.method == "title_exact":
            key = evidence.get("title_norm")
            return engine.title_index.get(key) if isinstance(key, str) else None
        if result.method == "title_fuzzy":
            key = evidence.get("best_key")
            return engine.title_index.get(key) if isinstance(key, str) else None
        return None

    def stats(self) -> dict[str, Any]:
        engine = self._engine
        return {
            "id": self.cfg.id,
            "label": self.cfg.label,
            "enabled": self.cfg.enabled,
            "dataset_path": str(resolve_dataset_path(self.cfg)),
            "loaded": engine is not None,
            "entries": len(engine.entries) if engine else 0,
            "title_index": len(engine.title_index) if engine else 0,
            "issn_index": len(engine.issn_index) if engine else 0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }


_PROVIDERS: dict[str, DatasetRankingProvider] | None = None


def _load_providers() -> dict[str, DatasetRankingProvider]:
    providers: dict[str, DatasetRankingProvider] = {}
    if not SOURCES_DIR.exists():
        LOGGER.warning("Ranking sources directory missing: %s", SOURCES_DIR)
        return providers

    for cfg_path in sorted(SOURCES_DIR.glob("*.yaml")):
        try:
            cfg = load_config(cfg_path)
        except Exception as exc:
            LOGGER.warning("Failed to load ranking config %s: %s", cfg_path, exc)
            continue
        providers[cfg.id] = DatasetRankingProvider(cfg)
    return providers


def list_rankings(*, include_disabled: bool = True) -> list[str]:
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _load_providers()
    ids = sorted(_PROVIDERS.keys())
    if include_disabled:
        return ids
    return [rid for rid in ids if _PROVIDERS[rid].cfg.enabled]


def get_provider(ranking_id: str) -> DatasetRankingProvider:
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _load_providers()
    try:
        return _PROVIDERS[ranking_id]
    except KeyError as exc:
        raise KeyError(f"Unknown ranking provider: {ranking_id}") from exc


def match_all(
    venue_title: str | None,
    issn_list: list[str] | None,
    *,
    include_disabled: bool = False,
    skip_ids: set[str] | None = None,
) -> dict[str, MatchResult]:
    """Match *venue_title*/*issn_list* across all configured ranking providers."""

    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _load_providers()

    results: dict[str, MatchResult] = {}
    for rid, provider in _PROVIDERS.items():
        if skip_ids and rid in skip_ids:
            continue
        if not include_disabled and not provider.cfg.enabled:
            continue
        try:
            results[rid] = provider.match(venue_title, issn_list)
        except Exception as exc:
            LOGGER.warning("Ranking match failed id=%s error=%s", rid, exc)
            results[rid] = MatchResult(
                source_id=rid,
                rank_value=None,
                method="unmatched",
                score=0.0,
                evidence={"error": str(exc)},
                matched=False,
            )
    return results
