"""Shared matching engine for ranking lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from rapidfuzz import fuzz, process  # type: ignore[import-untyped]

from ai_bias_search.rankings.base import MatchResult, RankingConfig, RankingEntry, normalize_title
from ai_bias_search.rankings.base import iter_normalized_issns
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def _update_unique_index(
    index: Dict[str, RankingEntry],
    ambiguous: set[str],
    key: str,
    entry: RankingEntry,
    *,
    label: str,
) -> None:
    if not key:
        return
    if key in ambiguous:
        return
    existing = index.get(key)
    if existing is None:
        index[key] = entry
        return
    if existing.venue_key == entry.venue_key:
        return
    index.pop(key, None)
    ambiguous.add(key)
    LOGGER.warning("Duplicate %s key %r in %s; marking ambiguous", label, key, entry.source_id)


@dataclass(slots=True)
class MatchingEngine:
    cfg: RankingConfig
    entries: Sequence[RankingEntry]
    issn_index: Mapping[str, RankingEntry]
    title_index: Mapping[str, RankingEntry]
    title_keys: Sequence[str]

    @classmethod
    def build(cls, cfg: RankingConfig, entries: Sequence[RankingEntry]) -> "MatchingEngine":
        issn_index: dict[str, RankingEntry] = {}
        title_index: dict[str, RankingEntry] = {}
        ambiguous_issn: set[str] = set()
        ambiguous_title: set[str] = set()

        for entry in entries:
            for issn in (entry.issn_l, entry.issn_print, entry.issn_online):
                if issn:
                    _update_unique_index(
                        issn_index, ambiguous_issn, issn, entry, label="issn"
                    )

            if entry.title_norm:
                _update_unique_index(
                    title_index, ambiguous_title, entry.title_norm, entry, label="title"
                )

            aliases = entry.extra.get("_title_aliases") if isinstance(entry.extra, Mapping) else None
            if isinstance(aliases, list):
                for raw_alias in aliases:
                    if not isinstance(raw_alias, str):
                        continue
                    alias_norm = normalize_title(raw_alias, cfg.normalization)
                    if alias_norm:
                        _update_unique_index(
                            title_index, ambiguous_title, alias_norm, entry, label="title_alias"
                        )

        title_keys = tuple(title_index.keys())
        LOGGER.info(
            "Built ranking indices (id=%s entries=%d titles=%d issn=%d)",
            cfg.id,
            len(entries),
            len(title_index),
            len(issn_index),
        )
        return cls(
            cfg=cfg,
            entries=entries,
            issn_index=issn_index,
            title_index=title_index,
            title_keys=title_keys,
        )

    def match(self, venue_title: str | None, issn_list: list[str] | None) -> MatchResult:
        issns = iter_normalized_issns(
            issn_list or [], validate_checksum=self.cfg.validate_issn_checksum
        )
        for key in issns:
            entry = self.issn_index.get(key)
            if entry is not None:
                return MatchResult(
                    source_id=self.cfg.id,
                    rank_value=entry.rank_value,
                    method="issn_exact",
                    score=1.0,
                    evidence={
                        "issn": key,
                        "venue_key": entry.venue_key,
                        "title": entry.title,
                        "rank_year": entry.rank_year,
                    },
                    matched=True,
                )

        title_norm = normalize_title(venue_title, self.cfg.normalization) if venue_title else ""
        if title_norm:
            entry = self.title_index.get(title_norm)
            if entry is not None:
                return MatchResult(
                    source_id=self.cfg.id,
                    rank_value=entry.rank_value,
                    method="title_exact",
                    score=1.0,
                    evidence={
                        "title_norm": title_norm,
                        "venue_key": entry.venue_key,
                        "title": entry.title,
                        "rank_year": entry.rank_year,
                    },
                    matched=True,
                )

        if not self.cfg.allow_fuzzy:
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=0.0,
                evidence={"reason": "fuzzy_disabled", "title_norm": title_norm or None},
                matched=False,
            )

        if not title_norm or not self.title_keys:
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=0.0,
                evidence={"reason": "no_query_or_index", "title_norm": title_norm or None},
                matched=False,
            )

        matches = process.extract(
            title_norm,
            self.title_keys,
            scorer=fuzz.ratio,
            limit=max(2, int(self.cfg.fuzzy_top_k)),
        )
        if not matches:
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=0.0,
                evidence={"reason": "no_candidates", "title_norm": title_norm},
                matched=False,
            )

        best_key, best_score, _ = matches[0]
        best_key_str = str(best_key)
        best_score_f = float(best_score) / 100.0
        candidates = [
            {"key": str(key), "score": float(score) / 100.0} for key, score, _ in matches[: self.cfg.fuzzy_top_k]
        ]

        if best_score_f < float(self.cfg.fuzzy_threshold):
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=best_score_f,
                evidence={"reason": "below_threshold", "title_norm": title_norm, "candidates": candidates},
                matched=False,
            )

        if self.cfg.reject_ambiguous_fuzzy and len(matches) > 1:
            second_score_f = float(matches[1][1]) / 100.0
            if best_score_f - second_score_f < float(self.cfg.fuzzy_ambiguity_delta):
                return MatchResult(
                    source_id=self.cfg.id,
                    rank_value=None,
                    method="unmatched",
                    score=best_score_f,
                    evidence={
                        "reason": "ambiguous",
                        "title_norm": title_norm,
                        "candidates": candidates,
                    },
                    matched=False,
                )

        entry = self.title_index.get(best_key_str)
        if entry is None:
            return MatchResult(
                source_id=self.cfg.id,
                rank_value=None,
                method="unmatched",
                score=best_score_f,
                evidence={"reason": "index_miss", "title_norm": title_norm, "best_key": best_key_str},
                matched=False,
            )

        return MatchResult(
            source_id=self.cfg.id,
            rank_value=entry.rank_value,
            method="title_fuzzy",
            score=best_score_f,
            evidence={
                "title_norm": title_norm,
                "best_key": best_key_str,
                "candidates": candidates,
                "venue_key": entry.venue_key,
                "title": entry.title,
                "rank_year": entry.rank_year,
            },
            matched=True,
        )

