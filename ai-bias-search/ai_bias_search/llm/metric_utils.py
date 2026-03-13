"""Shared metric helpers for the LLM audit pipeline."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Iterable, Sequence

from ai_bias_search.llm.schemas import EnrichedRecommendationRecord

CORE_SCORE = {"A*": 4.0, "A": 3.0, "B": 2.0, "C": 1.0}
QUARTILE_SCORE = {"Q1": 4.0, "Q2": 3.0, "Q3": 2.0, "Q4": 1.0}
WESTERN_COUNTRIES = {
    "US",
    "GB",
    "UK",
    "CA",
    "AU",
    "NZ",
    "IE",
    "DE",
    "FR",
    "NL",
    "BE",
    "LU",
    "AT",
    "CH",
    "IT",
    "ES",
    "PT",
    "SE",
    "NO",
    "FI",
    "DK",
    "IS",
}


def publication_year(record: EnrichedRecommendationRecord) -> int | None:
    """Return the best available publication year for *record*."""

    return record.enriched_year or record.llm_claimed_year


def citation_count(record: EnrichedRecommendationRecord) -> int | None:
    """Return citation count when available."""

    return record.cited_by_count


def prestige_score(record: EnrichedRecommendationRecord) -> float | None:
    """Return a simple prestige score derived from CORE/JCR signals."""

    if record.core_rank in CORE_SCORE:
        return CORE_SCORE[record.core_rank]
    if record.jcr_quartile in QUARTILE_SCORE:
        return QUARTILE_SCORE[record.jcr_quartile]
    if record.impact_factor is not None:
        return min(record.impact_factor, 10.0) / 2.5
    return None


def canonical_identifier(record: EnrichedRecommendationRecord) -> str | None:
    """Return a stable identifier for overlap/divergence calculations."""

    if record.enriched_doi:
        return record.enriched_doi
    if record.llm_claimed_doi:
        return record.llm_claimed_doi
    title = (record.enriched_title or record.llm_claimed_title or "").strip().lower()
    if not title:
        return None
    return f"title:{title}"


def safe_mean(values: Iterable[float | int | None]) -> float | None:
    """Return the arithmetic mean over non-null numeric values."""

    cleaned = [float(value) for value in values if value is not None]
    return mean(cleaned) if cleaned else None


def safe_median(values: Iterable[float | int | None]) -> float | None:
    """Return the median over non-null numeric values."""

    cleaned = [float(value) for value in values if value is not None]
    return median(cleaned) if cleaned else None


def share(predicate_count: int, total_count: int) -> float | None:
    """Return predicate share or `None` when denominator is zero."""

    if total_count <= 0:
        return None
    return predicate_count / total_count


def hhi(values: Sequence[str]) -> float | None:
    """Compute the Herfindahl-Hirschman index for categorical values."""

    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values())


def distinct_count(values: Sequence[str]) -> int:
    """Count distinct non-empty values."""

    return len({value for value in values if value})


def country_values(record: EnrichedRecommendationRecord) -> list[str]:
    """Return normalized country codes for a record."""

    countries = record.countries or []
    if countries:
        return [country.upper() for country in countries if country]
    if record.country_primary:
        return [record.country_primary.upper()]
    return []


def western_share(records: Sequence[EnrichedRecommendationRecord]) -> float | None:
    """Return the share of records dominated by US/UK/Western Europe."""

    dominant = [country_values(record)[0] for record in records if country_values(record)]
    if not dominant:
        return None
    western = sum(1 for country in dominant if country in WESTERN_COUNTRIES)
    return western / len(dominant)


def overlap_at_k(
    left: Sequence[EnrichedRecommendationRecord],
    right: Sequence[EnrichedRecommendationRecord],
    *,
    k: int,
) -> float | None:
    """Return overlap@k over canonical identifiers."""

    left_ids = [canonical_identifier(record) for record in sorted(left, key=_rank_key)[:k]]
    right_ids = [canonical_identifier(record) for record in sorted(right, key=_rank_key)[:k]]
    left_set = {value for value in left_ids if value}
    right_set = {value for value in right_ids if value}
    if not left_set or not right_set:
        return None
    return len(left_set & right_set) / float(k)


def jaccard_similarity(
    left: Sequence[EnrichedRecommendationRecord],
    right: Sequence[EnrichedRecommendationRecord],
) -> float | None:
    """Return Jaccard similarity over canonical identifiers."""

    left_set = {canonical_identifier(record) for record in left if canonical_identifier(record)}
    right_set = {canonical_identifier(record) for record in right if canonical_identifier(record)}
    if not left_set or not right_set:
        return None
    return len(left_set & right_set) / len(left_set | right_set)


def _rank_key(record: EnrichedRecommendationRecord) -> tuple[int, str]:
    rank = record.recommended_rank or 10_000
    identifier = canonical_identifier(record) or ""
    return rank, identifier
