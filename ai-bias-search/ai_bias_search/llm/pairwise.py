"""Pairwise comparison helpers for optional controlled LLM bias probes."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ai_bias_search.llm.metric_utils import (
    citation_count,
    country_values,
    prestige_score,
    publication_year,
    safe_mean,
    share,
    western_share,
)
from ai_bias_search.llm.schemas import EnrichedRecommendationRecord


def compute_pairwise_comparisons(
    records: Sequence[EnrichedRecommendationRecord],
) -> list[dict[str, object]]:
    """Compute pairwise deltas for records sharing a `pair_id`."""

    groups: dict[tuple[str, int, str], list[EnrichedRecommendationRecord]] = defaultdict(list)
    for record in records:
        if not record.pair_id:
            continue
        key = (record.model, record.repeat_index, record.pair_id)
        groups[key].append(record)

    comparisons: list[dict[str, object]] = []
    for (model, repeat_index, pair_id), group in groups.items():
        by_query: dict[str, list[EnrichedRecommendationRecord]] = defaultdict(list)
        for record in group:
            by_query[record.query_id].append(record)

        ordered = _ordered_queries(list(by_query.values()))
        if len(ordered) < 2:
            continue
        control = ordered[0]
        for treatment in ordered[1:]:
            comparisons.append(
                _comparison_payload(
                    model=model,
                    repeat_index=repeat_index,
                    pair_id=pair_id,
                    left=control,
                    right=treatment,
                )
            )
    return comparisons


def _ordered_queries(
    query_groups: list[list[EnrichedRecommendationRecord]],
) -> list[list[EnrichedRecommendationRecord]]:
    def sort_key(group: list[EnrichedRecommendationRecord]) -> tuple[int, str, str]:
        first = group[0]
        label = first.control_or_treatment or ""
        control_first = 0 if label == "control" else 1
        variant = first.variant or ""
        return control_first, variant, first.query_id

    return sorted(query_groups, key=sort_key)


def _comparison_payload(
    *,
    model: str,
    repeat_index: int,
    pair_id: str,
    left: list[EnrichedRecommendationRecord],
    right: list[EnrichedRecommendationRecord],
) -> dict[str, object]:
    left_first = left[0]
    right_first = right[0]
    left_years = [publication_year(record) for record in left]
    right_years = [publication_year(record) for record in right]
    left_citations = [citation_count(record) for record in left]
    right_citations = [citation_count(record) for record in right]
    left_prestige = [prestige_score(record) for record in left]
    right_prestige = [prestige_score(record) for record in right]
    left_oa = [record.is_oa for record in left if record.is_oa is not None]
    right_oa = [record.is_oa for record in right if record.is_oa is not None]
    left_country_count = sum(1 for record in left if country_values(record))
    right_country_count = sum(1 for record in right if country_values(record))

    return {
        "pair_id": pair_id,
        "model": model,
        "repeat_index": repeat_index,
        "left_query_id": left_first.query_id,
        "right_query_id": right_first.query_id,
        "left_variant": left_first.variant,
        "right_variant": right_first.variant,
        "left_control_or_treatment": left_first.control_or_treatment,
        "right_control_or_treatment": right_first.control_or_treatment,
        "returned_item_count_delta": len(right) - len(left),
        "average_year_delta": _delta(safe_mean(right_years), safe_mean(left_years)),
        "average_citation_delta": _delta(safe_mean(right_citations), safe_mean(left_citations)),
        "prestige_score_delta": _delta(safe_mean(right_prestige), safe_mean(left_prestige)),
        "oa_share_delta": _delta(
            share(sum(bool(value) for value in right_oa), len(right_oa)),
            share(sum(bool(value) for value in left_oa), len(left_oa)),
        ),
        "western_share_delta": _delta(western_share(right), western_share(left)),
        "country_coverage_delta": _delta(
            share(right_country_count, len(right)),
            share(left_country_count, len(left)),
        ),
    }


def _delta(right: float | None, left: float | None) -> float | None:
    if right is None or left is None:
        return None
    return right - left
