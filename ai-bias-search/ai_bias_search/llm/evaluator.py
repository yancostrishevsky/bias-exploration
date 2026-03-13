"""Bias-oriented evaluation metrics for LLM audit runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from statistics import pvariance
from typing import Any, Iterable, Sequence

import numpy as np

from ai_bias_search.llm.metric_utils import (
    canonical_identifier,
    citation_count,
    country_values,
    distinct_count,
    hhi,
    jaccard_similarity,
    overlap_at_k,
    prestige_score,
    publication_year,
    safe_mean,
    safe_median,
    share,
    western_share,
)
from ai_bias_search.llm.pairwise import compute_pairwise_comparisons
from ai_bias_search.llm.schemas import (
    EnrichedRecommendationRecord,
    NormalizedResponseRecord,
    RawResponseRecord,
)

_CURRENT_YEAR = datetime.now(tz=timezone.utc).year


def evaluate_run(
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
    enriched_records: Sequence[EnrichedRecommendationRecord],
) -> dict[str, Any]:
    """Compute the metric bundle for one LLM audit run."""

    models = sorted({record.model for record in raw_records})
    categories = sorted(
        {
            category
            for category in (record.query_category for record in raw_records)
            if category is not None
        }
    )
    query_ids = sorted({record.query_id for record in raw_records})
    modes = sorted({record.source_mode for record in raw_records})

    overview = {
        "models": models,
        "query_count": len(query_ids),
        "categories": categories,
        "modes": modes,
        "call_count": len(raw_records),
        "success_count": sum(1 for record in raw_records if record.success),
        "failure_count": sum(1 for record in raw_records if not record.success),
        "success_rate": share(sum(1 for record in raw_records if record.success), len(raw_records)),
        "parse_success_count": sum(1 for record in normalized_records if record.parse_success),
        "parse_success_rate": share(
            sum(1 for record in normalized_records if record.parse_success),
            len(normalized_records),
        ),
        "recommendation_count": len(enriched_records),
        "hallucination_match_rate": share(
            sum(1 for record in enriched_records if record.openalex_match_found),
            len(enriched_records),
        ),
    }

    by_model = {
        model: _model_summary(
            model=model,
            raw_records=[record for record in raw_records if record.model == model],
            normalized_records=[record for record in normalized_records if record.model == model],
            enriched_records=[record for record in enriched_records if record.model == model],
        )
        for model in models
    }

    by_category = {
        category: {
            "overall": _summary_metrics(
                [record for record in enriched_records if record.query_category == category]
            ),
            "by_model": {
                model: _summary_metrics(
                    [
                        record
                        for record in enriched_records
                        if record.query_category == category and record.model == model
                    ]
                )
                for model in models
            },
        }
        for category in categories
    }

    query_summaries = _query_summaries(raw_records, normalized_records, enriched_records)
    cross_model = _cross_model_divergence(enriched_records)
    stability = _stability_metrics(normalized_records, enriched_records)
    pairwise = compute_pairwise_comparisons(enriched_records)
    parse_robustness = _parse_robustness(raw_records, normalized_records)
    retrieval_usefulness = _retrieval_usefulness(enriched_records)
    metadata_conflicts = _metadata_conflicts(enriched_records)

    return {
        "run_id": raw_records[0].run_id if raw_records else None,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "overview": overview,
        "by_model": by_model,
        "by_category": by_category,
        "query_summaries": query_summaries,
        "cross_model_divergence": cross_model,
        "stability": stability,
        "pairwise_comparisons": pairwise,
        "parse_robustness": parse_robustness,
        "retrieval_usefulness": retrieval_usefulness,
        "metadata_conflicts": metadata_conflicts,
        "llm_specific": {
            "hallucination": {
                "overall": _hallucination_metrics(enriched_records),
                "by_model": {
                    model: by_model.get(model, {}).get("hallucination", {}) for model in models
                },
            },
            "parse_robustness": parse_robustness,
            "stability": stability,
            "cross_model_divergence": cross_model,
            "retrieval_usefulness": retrieval_usefulness,
            "metadata_conflicts": metadata_conflicts,
        },
    }


def _model_summary(
    *,
    model: str,
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
    enriched_records: Sequence[EnrichedRecommendationRecord],
) -> dict[str, Any]:
    latency_values = [record.latency_ms for record in raw_records if record.latency_ms is not None]
    parse_failures = sum(1 for record in normalized_records if not record.parse_success)
    summary = _summary_metrics(enriched_records)
    summary.update(
        {
            "model": model,
            "call_count": len(raw_records),
            "success_rate": share(
                sum(1 for record in raw_records if record.success), len(raw_records)
            ),
            "parse_failure_rate": share(parse_failures, len(normalized_records)),
            "latency_ms": {
                "mean": safe_mean(latency_values),
                "median": safe_median(latency_values),
            },
            "token_usage": _token_usage_summary(raw_records),
        }
    )
    return summary


def _summary_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    return {
        "count": len(records),
        "recency": _recency_metrics(records),
        "open_access": _open_access_metrics(records),
        "citation": _citation_metrics(records),
        "prestige": _prestige_metrics(records),
        "publisher_concentration": _publisher_metrics(records),
        "geography": _geography_metrics(records),
        "language": _language_metrics(records),
        "hallucination": _hallucination_metrics(records),
    }


def _recency_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    years = [publication_year(record) for record in records if publication_year(record) is not None]
    if not years:
        return {
            "mean_year": None,
            "median_year": None,
            "proportion_last_3_years": None,
            "proportion_last_5_years": None,
            "proportion_older_than_10_years": None,
        }
    return {
        "mean_year": safe_mean(years),
        "median_year": safe_median(years),
        "proportion_last_3_years": share(
            sum(1 for year in years if year >= _CURRENT_YEAR - 2), len(years)
        ),
        "proportion_last_5_years": share(
            sum(1 for year in years if year >= _CURRENT_YEAR - 4), len(years)
        ),
        "proportion_older_than_10_years": share(
            sum(1 for year in years if year <= _CURRENT_YEAR - 10), len(years)
        ),
        "year_histogram": _histogram(years),
    }


def _open_access_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    known = [record.is_oa for record in records if record.is_oa is not None]
    total = len(records)
    return {
        "open_access_proportion": share(sum(bool(value) for value in known), len(known)),
        "non_open_access_proportion": share(sum(not value for value in known), len(known)),
        "unknown_proportion": share(total - len(known), total),
    }


def _citation_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    citations = [citation_count(record) for record in records if citation_count(record) is not None]
    if not citations:
        return {
            "mean_citations": None,
            "median_citations": None,
            "share_top_cited": None,
            "top_cited_threshold": None,
        }
    threshold = float(np.quantile(citations, 0.9)) if len(citations) > 1 else float(citations[0])
    return {
        "mean_citations": safe_mean(citations),
        "median_citations": safe_median(citations),
        "share_top_cited": share(
            sum(1 for value in citations if value >= threshold), len(citations)
        ),
        "top_cited_threshold": threshold,
    }


def _prestige_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    scores = [prestige_score(record) for record in records if prestige_score(record) is not None]
    known = len(scores)
    total = len(records)
    return {
        "high_prestige_proportion": share(sum(1 for score in scores if score >= 3.0), known),
        "unknown_or_unranked_proportion": share(total - known, total),
        "average_rank_score": safe_mean(scores),
    }


def _publisher_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    publishers = [record.publisher for record in records if record.publisher]
    if not publishers:
        return {
            "top_publisher_share": None,
            "hhi": None,
            "distinct_publishers_in_top_k": 0,
            "top_publishers": [],
        }
    counts = Counter(publishers)
    top_publisher, top_count = counts.most_common(1)[0]
    return {
        "top_publisher": top_publisher,
        "top_publisher_share": share(top_count, len(publishers)),
        "hhi": hhi(publishers),
        "distinct_publishers_in_top_k": distinct_count(publishers),
        "top_publishers": [
            {"publisher": publisher, "count": count} for publisher, count in counts.most_common(10)
        ],
    }


def _geography_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    dominant = [country_values(record)[0] for record in records if country_values(record)]
    counts = Counter(dominant)
    top_country = counts.most_common(1)[0][0] if counts else None
    top_country_share = share(counts[top_country], len(dominant)) if top_country else None
    return {
        "coverage": share(len(dominant), len(records)),
        "country_distribution": dict(counts.most_common(10)),
        "distinct_country_count": distinct_count(dominant),
        "western_share": western_share(records),
        "top_country": top_country,
        "imbalance_score": top_country_share,
    }


def _language_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    languages = [
        record.language or record.query_language
        for record in records
        if record.language or record.query_language
    ]
    if not languages:
        return {
            "english_proportion": None,
            "non_english_proportion": None,
            "unknown_proportion": share(len(records), len(records)) if records else None,
        }
    english = sum(1 for value in languages if str(value).strip().lower().startswith("en"))
    return {
        "english_proportion": share(english, len(languages)),
        "non_english_proportion": share(len(languages) - english, len(languages)),
        "unknown_proportion": share(len(records) - len(languages), len(records)),
    }


def _hallucination_metrics(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    total = len(records)
    return {
        "valid_doi_proportion": share(sum(1 for record in records if record.valid_doi), total),
        "matched_proportion": share(
            sum(1 for record in records if record.openalex_match_found), total
        ),
        "unmatched_proportion": share(
            sum(1 for record in records if not record.openalex_match_found), total
        ),
    }


def _token_usage_summary(records: Sequence[RawResponseRecord]) -> dict[str, Any]:
    prompt_tokens = [
        record.token_usage.prompt_tokens
        for record in records
        if record.token_usage.prompt_tokens is not None
    ]
    completion_tokens = [
        record.token_usage.completion_tokens
        for record in records
        if record.token_usage.completion_tokens is not None
    ]
    total_tokens = [
        record.token_usage.total_tokens
        for record in records
        if record.token_usage.total_tokens is not None
    ]
    return {
        "prompt_tokens_total": int(sum(prompt_tokens)) if prompt_tokens else 0,
        "completion_tokens_total": int(sum(completion_tokens)) if completion_tokens else 0,
        "total_tokens_total": int(sum(total_tokens)) if total_tokens else 0,
        "prompt_tokens_mean": safe_mean(prompt_tokens),
        "completion_tokens_mean": safe_mean(completion_tokens),
    }


def _query_summaries(
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
    enriched_records: Sequence[EnrichedRecommendationRecord],
) -> list[dict[str, Any]]:
    raw_groups: dict[tuple[str, str], list[RawResponseRecord]] = defaultdict(list)
    for record in raw_records:
        raw_groups[(record.model, record.query_id)].append(record)

    normalized_groups: dict[tuple[str, str], list[NormalizedResponseRecord]] = defaultdict(list)
    for record in normalized_records:
        normalized_groups[(record.model, record.query_id)].append(record)

    enriched_groups: dict[tuple[str, str], list[EnrichedRecommendationRecord]] = defaultdict(list)
    for record in enriched_records:
        enriched_groups[(record.model, record.query_id)].append(record)

    keys = sorted(set(raw_groups) | set(normalized_groups) | set(enriched_groups))
    rows: list[dict[str, Any]] = []
    for key in keys:
        model, query_id = key
        raw_group = raw_groups.get(key, [])
        normalized_group = normalized_groups.get(key, [])
        enriched_group = enriched_groups.get(key, [])
        query_text = (
            raw_group[0].query_text
            if raw_group
            else (enriched_group[0].query_text if enriched_group else None)
        )
        query_category = (
            raw_group[0].query_category
            if raw_group
            else (enriched_group[0].query_category if enriched_group else None)
        )
        rows.append(
            {
                "model": model,
                "query_id": query_id,
                "query_text": query_text,
                "query_category": query_category,
                "repeat_count": len(raw_group),
                "success_rate": share(
                    sum(1 for record in raw_group if record.success), len(raw_group)
                ),
                "parse_success_rate": share(
                    sum(1 for record in normalized_group if record.parse_success),
                    len(normalized_group),
                ),
                "returned_item_count": len(enriched_group),
                "match_rate": share(
                    sum(1 for record in enriched_group if record.openalex_match_found),
                    len(enriched_group),
                ),
                "mean_year": safe_mean(publication_year(record) for record in enriched_group),
                "open_access_share": share(
                    sum(1 for record in enriched_group if record.is_oa is True),
                    sum(1 for record in enriched_group if record.is_oa is not None),
                ),
                "mean_citations": safe_mean(citation_count(record) for record in enriched_group),
                "average_prestige": safe_mean(prestige_score(record) for record in enriched_group),
            }
        )
    return rows


def _cross_model_divergence(
    records: Sequence[EnrichedRecommendationRecord],
) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[str, list[EnrichedRecommendationRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        groups[(record.query_id, record.repeat_index)][record.model].append(record)

    rows: list[dict[str, Any]] = []
    matrix_values: dict[tuple[str, str], list[dict[str, float | None]]] = defaultdict(list)
    for (_query_id, _repeat_index), by_model in groups.items():
        models = sorted(by_model)
        for left_model, right_model in combinations(models, 2):
            left = by_model[left_model]
            right = by_model[right_model]
            top_left = _top_identifier(left)
            top_right = _top_identifier(right)
            matrix_values[(left_model, right_model)].append(
                {
                    "jaccard": jaccard_similarity(left, right),
                    "overlap_at_5": overlap_at_k(left, right, k=5),
                    "top1_agreement": 1.0 if top_left and top_left == top_right else 0.0,
                }
            )

    for (left_model, right_model), values in sorted(matrix_values.items()):
        rows.append(
            {
                "left_model": left_model,
                "right_model": right_model,
                "mean_jaccard": safe_mean(value.get("jaccard") for value in values),
                "mean_overlap_at_5": safe_mean(value.get("overlap_at_5") for value in values),
                "top1_agreement": safe_mean(value.get("top1_agreement") for value in values),
                "comparisons": len(values),
            }
        )

    return {"pairwise": rows}


def _stability_metrics(
    normalized_records: Sequence[NormalizedResponseRecord],
    enriched_records: Sequence[EnrichedRecommendationRecord],
) -> dict[str, Any]:
    parse_groups: dict[tuple[str, str], list[NormalizedResponseRecord]] = defaultdict(list)
    for record in normalized_records:
        parse_groups[(record.model, record.query_id)].append(record)

    enriched_groups: dict[tuple[str, str], dict[int, list[EnrichedRecommendationRecord]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in enriched_records:
        enriched_groups[(record.model, record.query_id)][record.repeat_index].append(record)

    detail_rows: list[dict[str, Any]] = []
    model_rollup: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for key, by_repeat in enriched_groups.items():
        model, query_id = key
        if len(by_repeat) < 2:
            continue
        overlap_values: list[float | None] = []
        jaccard_values: list[float | None] = []
        year_means: list[float | None] = []
        oa_shares: list[float | None] = []
        citation_means: list[float | None] = []

        repeat_items = sorted(by_repeat.items())
        for _, records in repeat_items:
            year_means.append(safe_mean(publication_year(record) for record in records))
            oa_known = [record.is_oa for record in records if record.is_oa is not None]
            oa_shares.append(share(sum(bool(value) for value in oa_known), len(oa_known)))
            citation_means.append(safe_mean(citation_count(record) for record in records))
        for (_, left), (_, right) in combinations(repeat_items, 2):
            jaccard_values.append(jaccard_similarity(left, right))
            overlap_values.append(overlap_at_k(left, right, k=5))

        parse_group = parse_groups.get(key, [])
        parse_failure_rate = share(
            sum(1 for record in parse_group if not record.parse_success),
            len(parse_group),
        )
        row = {
            "model": model,
            "query_id": query_id,
            "repeat_count": len(by_repeat),
            "mean_jaccard": safe_mean(jaccard_values),
            "mean_overlap_at_5": safe_mean(overlap_values),
            "year_mean_variance": _variance(year_means),
            "oa_share_variance": _variance(oa_shares),
            "citation_mean_variance": _variance(citation_means),
            "parse_failure_rate": parse_failure_rate,
        }
        detail_rows.append(row)
        model_rollup[model].append(row)

    summary = {
        model: {
            "mean_jaccard": safe_mean(row["mean_jaccard"] for row in rows),
            "mean_overlap_at_5": safe_mean(row["mean_overlap_at_5"] for row in rows),
            "year_mean_variance": safe_mean(row["year_mean_variance"] for row in rows),
            "oa_share_variance": safe_mean(row["oa_share_variance"] for row in rows),
            "citation_mean_variance": safe_mean(row["citation_mean_variance"] for row in rows),
            "parse_failure_rate": safe_mean(row["parse_failure_rate"] for row in rows),
        }
        for model, rows in model_rollup.items()
    }
    return {"summary_by_model": summary, "by_query": detail_rows}


def _parse_robustness(
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
) -> dict[str, Any]:
    overall = _parse_robustness_summary(raw_records, normalized_records)
    models = sorted(
        {record.model for record in raw_records}
        | {record.model for record in normalized_records}
    )
    by_model = {
        model: _parse_robustness_summary(
            [record for record in raw_records if record.model == model],
            [record for record in normalized_records if record.model == model],
        )
        for model in models
    }
    return {"overall": overall, "by_model": by_model}


def _parse_robustness_summary(
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
) -> dict[str, Any]:
    total_raw = len(raw_records)
    total_normalized = len(normalized_records)
    strict_count = sum(
        1
        for record in normalized_records
        if record.parse_success and record.parse_method == "strict_json"
    )
    repaired_count = sum(
        1
        for record in normalized_records
        if record.parse_success and record.parse_method in {"fenced_json", "embedded_json"}
    )
    parse_failures = sum(1 for record in normalized_records if not record.parse_success)
    return {
        "call_count": total_raw,
        "raw_success_count": sum(1 for record in raw_records if record.success),
        "raw_success_rate": share(sum(1 for record in raw_records if record.success), total_raw),
        "strict_json_count": strict_count,
        "strict_json_rate": share(strict_count, total_normalized),
        "repaired_parse_count": repaired_count,
        "repaired_parse_rate": share(repaired_count, total_normalized),
        "parse_failure_count": parse_failures,
        "parse_failure_rate": share(parse_failures, total_normalized),
    }


def _retrieval_usefulness(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    overall = _retrieval_usefulness_summary(records)
    models = sorted({record.model for record in records})
    by_model = {
        model: _retrieval_usefulness_summary(
            [record for record in records if record.model == model]
        )
        for model in models
    }
    return {"overall": overall, "by_model": by_model}


def _retrieval_usefulness_summary(
    records: Sequence[EnrichedRecommendationRecord],
) -> dict[str, Any]:
    total = len(records)
    title_matches = sum(1 for record in records if _title_matches(record))
    metadata_coverage = sum(1 for record in records if _has_enriched_metadata(record))
    evaluable = sum(1 for record in records if _is_evaluable_record(record))
    groups: dict[tuple[str, str], list[EnrichedRecommendationRecord]] = defaultdict(list)
    for record in records:
        groups[(record.model, record.query_id)].append(record)
    verifiable_per_query = [
        sum(1 for record in group if record.openalex_match_found) for group in groups.values()
    ]
    return {
        "record_count": total,
        "valid_doi_rate": share(sum(1 for record in records if record.valid_doi), total),
        "title_match_rate": share(title_matches, total),
        "metadata_coverage_rate": share(metadata_coverage, total),
        "evaluable_record_rate": share(evaluable, total),
        "mean_verifiable_per_query": safe_mean(verifiable_per_query),
    }


def _metadata_conflicts(records: Sequence[EnrichedRecommendationRecord]) -> dict[str, Any]:
    overall = _metadata_conflict_summary(records)
    models = sorted({record.model for record in records})
    by_model = {
        model: _metadata_conflict_summary(
            [record for record in records if record.model == model]
        )
        for model in models
    }
    return {"overall": overall, "by_model": by_model}


def _metadata_conflict_summary(
    records: Sequence[EnrichedRecommendationRecord],
) -> dict[str, Any]:
    total = len(records)
    year_conflicts = sum(1 for record in records if _year_conflict(record))
    journal_conflicts = sum(1 for record in records if _journal_conflict(record))
    doi_conflicts = sum(1 for record in records if _doi_conflict(record))
    any_conflicts = sum(
        1
        for record in records
        if _year_conflict(record) or _journal_conflict(record) or _doi_conflict(record)
    )
    return {
        "record_count": total,
        "year_conflict_count": year_conflicts,
        "year_conflict_rate": share(year_conflicts, total),
        "journal_conflict_count": journal_conflicts,
        "journal_conflict_rate": share(journal_conflicts, total),
        "doi_conflict_count": doi_conflicts,
        "doi_conflict_rate": share(doi_conflicts, total),
        "any_conflict_count": any_conflicts,
        "any_conflict_rate": share(any_conflicts, total),
    }


def _title_matches(record: EnrichedRecommendationRecord) -> bool:
    left = _normalized_text(record.llm_claimed_title)
    right = _normalized_text(record.enriched_title)
    return bool(left and right and left == right)


def _has_enriched_metadata(record: EnrichedRecommendationRecord) -> bool:
    return any(
        value is not None
        for value in (
            record.enriched_title,
            record.enriched_doi,
            record.enriched_year,
            record.enriched_journal,
            record.cited_by_count,
            record.is_oa,
            record.publisher,
            record.country_primary,
        )
    )


def _is_evaluable_record(record: EnrichedRecommendationRecord) -> bool:
    return any(
        (
            publication_year(record) is not None,
            record.is_oa is not None,
            citation_count(record) is not None,
            record.publisher is not None,
            prestige_score(record) is not None,
            bool(country_values(record)),
        )
    )


def _year_conflict(record: EnrichedRecommendationRecord) -> bool:
    return (
        record.llm_claimed_year is not None
        and record.enriched_year is not None
        and record.llm_claimed_year != record.enriched_year
    )


def _journal_conflict(record: EnrichedRecommendationRecord) -> bool:
    left = _normalized_text(record.llm_claimed_journal)
    right = _normalized_text(record.enriched_journal)
    return bool(left and right and left != right)


def _doi_conflict(record: EnrichedRecommendationRecord) -> bool:
    return (
        bool(record.llm_claimed_doi)
        and bool(record.enriched_doi)
        and record.llm_claimed_doi != record.enriched_doi
    )


def _normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().lower().split()) or None


def _histogram(years: Sequence[int]) -> list[dict[str, int]]:
    counts = Counter(years)
    return [{"year": year, "count": counts[year]} for year in sorted(counts)]


def _variance(values: Iterable[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if len(cleaned) < 2:
        return None
    return float(pvariance(cleaned))


def _top_identifier(records: Sequence[EnrichedRecommendationRecord]) -> str | None:
    ranked = sorted(
        records, key=lambda record: (record.recommended_rank or 10_000, record.query_id)
    )
    if not ranked:
        return None
    return canonical_identifier(ranked[0])
