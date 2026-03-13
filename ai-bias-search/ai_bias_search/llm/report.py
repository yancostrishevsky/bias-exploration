"""Shared-template HTML reporting for LLM audit runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ai_bias_search.llm.schemas import (
    EnrichedRecommendationRecord,
    NormalizedResponseRecord,
    RawResponseRecord,
)
from ai_bias_search.report.make_report import (
    build_report_context,
    render_report_context,
)


def render_llm_report(
    *,
    run_dir: Path,
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
    enriched_records: Sequence[EnrichedRecommendationRecord],
    metrics: dict[str, Any],
) -> Path:
    """Render the LLM audit report through the shared report template."""

    output_path = run_dir / "report.html"
    frame = _frame_from_enriched(enriched_records)
    shared_metrics = _build_shared_metrics(metrics=metrics, frame=frame)
    llm_specific = _build_llm_specific_context(
        raw_records=raw_records,
        normalized_records=normalized_records,
        enriched_records=enriched_records,
        metrics=metrics,
    )
    summary = {
        "total_records": len(frame),
        "platforms": sorted({record.model for record in raw_records}),
        "query_count": metrics.get("overview", {}).get("query_count"),
        "call_count": metrics.get("overview", {}).get("call_count"),
        "run_id": run_dir.name,
    }
    context = build_report_context(
        frame=frame,
        latest_metrics=shared_metrics,
        metrics_timestamp=run_dir.name,
        diagnostics={},
        output_path=output_path,
        metrics_path=run_dir / "metrics.json",
        enriched_download_path=run_dir / "enriched_recommendations.jsonl",
        entity_label_singular="Model",
        entity_label_plural="Models",
        enriched_download_label="Enriched Recommendations JSONL",
        summary_override=summary,
        llm_specific=llm_specific,
    )
    return render_report_context(output_path=output_path, context=context)


def _frame_from_enriched(records: Sequence[EnrichedRecommendationRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "platform": record.model,
                "rank": record.recommended_rank,
                "title": record.enriched_title or record.llm_claimed_title,
                "doi": record.enriched_doi or record.llm_claimed_doi,
                "year": record.enriched_year or record.llm_claimed_year,
                "journal_title": record.enriched_journal or record.llm_claimed_journal,
                "source": record.enriched_journal or record.llm_claimed_journal,
                "authors": record.enriched_authors or record.llm_claimed_authors,
                "publisher": record.publisher,
                "is_oa": record.is_oa,
                "is_open_access": record.is_oa,
                "oa": record.is_oa,
                "citations": record.cited_by_count,
                "cited_by_count": record.cited_by_count,
                "language": record.language or record.query_language,
                "country_primary": record.country_primary,
                "country_dominant": record.country_primary,
                "countries": record.countries,
                "core_rank": record.core_rank,
                "impact_factor": record.impact_factor,
                "jcr_quartile": record.jcr_quartile,
                "rankings": record.rankings,
                "query_id": record.query_id,
                "query_text": record.query_text,
                "query_category": record.query_category,
                "repeat_index": record.repeat_index,
                "source_mode": record.source_mode,
                "pair_id": record.pair_id,
                "variant": record.variant,
                "control_or_treatment": record.control_or_treatment,
                "openalex_match_found": record.openalex_match_found,
                "valid_doi": record.valid_doi,
                "llm_claimed_title": record.llm_claimed_title,
                "llm_claimed_doi": record.llm_claimed_doi,
                "llm_claimed_year": record.llm_claimed_year,
                "llm_claimed_journal": record.llm_claimed_journal,
                "enriched_title": record.enriched_title,
                "enriched_doi": record.enriched_doi,
                "enriched_year": record.enriched_year,
                "enriched_journal": record.enriched_journal,
                "parse_status": record.parse_status,
                "extra": record.extra,
            }
        )
    return pd.DataFrame(rows)


def _build_shared_metrics(*, metrics: Mapping[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    shared_metrics = dict(metrics)
    ks = _top_ks(frame)
    overall_block = _bias_block(frame, ks=ks)
    platforms = (
        frame.get("platform", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    by_platform = {
        model: _bias_block(frame[frame["platform"].astype(str) == model], ks=ks)
        for model in sorted(platforms)
    }
    shared_metrics["biases"] = {**overall_block, "by_platform": by_platform}
    shared_metrics["pairwise"] = {
        f"{row['left_model']}_vs_{row['right_model']}": {"jaccard": row.get("mean_jaccard")}
        for row in metrics.get("cross_model_divergence", {}).get("pairwise", [])
        if isinstance(row, dict) and row.get("left_model") and row.get("right_model")
    }
    return shared_metrics


def _bias_block(frame: pd.DataFrame, *, ks: Sequence[int]) -> dict[str, Any]:
    years = _numeric_values(frame, "year")
    oa_values = [
        value
        for value in frame.get("is_oa", pd.Series(dtype=object)).tolist()
        if value is not None
    ]
    publishers = [
        str(value)
        for value in frame.get("publisher", pd.Series(dtype=object)).dropna().tolist()
        if str(value).strip()
    ]
    citations = _numeric_values(frame, "cited_by_count")
    return {
        "recency": {
            "mean_year": _mean(years),
            "median_year": _median(years),
        },
        "open_access": {
            "share_open_access": (
                sum(1 for value in oa_values if bool(value)) / len(oa_values) if oa_values else None
            ),
        },
        "publisher_hhi": {
            "hhi": _hhi(publishers),
            "available": bool(publishers),
        },
        "top_k_bias": {
            "ks": list(ks),
            "oa": {"per_k": _oa_top_k(frame, ks=ks)},
            "country": _country_top_k(frame, ks=ks),
            "doc_type": {
                "per_k": {},
                "reason": "Document type metadata unavailable for most LLM outputs.",
            },
            "citations": {
                "spearman_rank_vs_citations": _spearman_rank_vs_citations(frame),
                "per_k": _citations_top_k(frame, ks=ks),
            },
        },
        "rank_vs_citations": {
            "spearman": _spearman_rank_vs_citations(frame),
            "reliability": "high" if citations else "low",
            "reason": None if citations else "Citation metadata unavailable.",
            "note": None,
        },
    }


def _oa_top_k(frame: pd.DataFrame, *, ks: Sequence[int]) -> dict[str, dict[str, Any]]:
    ranked = _ranked_frame(frame)
    series = _oa_series(frame)
    overall_share = _series_share(series)
    payload: dict[str, dict[str, Any]] = {}
    for k in ks:
        top = ranked.head(k)
        top_series = _oa_series(top)
        top_share = _series_share(top_series)
        missing_pct = _missing_pct(top_series)
        payload[str(k)] = {
            "top_k_share": top_share,
            "overall_share": overall_share,
            "delta_share": (
                top_share - overall_share
                if top_share is not None and overall_share is not None
                else None
            ),
            "reliability": "high" if top_share is not None else "low",
            "missing_pct": missing_pct,
        }
    return payload


def _country_top_k(frame: pd.DataFrame, *, ks: Sequence[int]) -> dict[str, Any]:
    country_lists = [_country_list(row) for row in frame.to_dict(orient="records")]
    covered = [countries for countries in country_lists if countries]
    overall_fractional = _distribution(covered, fractional=True)
    overall_dominant = _distribution(covered, fractional=False)
    ranked = _ranked_frame(frame)
    per_k: dict[str, dict[str, Any]] = {}
    for k in ks:
        top_lists = [_country_list(row) for row in ranked.head(k).to_dict(orient="records")]
        top_covered = [countries for countries in top_lists if countries]
        top_fractional = _distribution(top_covered, fractional=True)
        top_dominant = _distribution(top_covered, fractional=False)
        per_k[str(k)] = {
            "reliability": "high" if top_covered else "low",
            "variants": {
                "countries": {
                    "fractional": {
                        "top_distribution": top_fractional,
                        "overrepresentation_ratio": _over_ratio(top_fractional, overall_fractional),
                    },
                    "dominant": {
                        "top_distribution": top_dominant,
                        "overrepresentation_ratio": _over_ratio(top_dominant, overall_dominant),
                    },
                },
                "regions": {"fractional": {}, "dominant": {}},
            },
        }
    return {
        "available_share": (len(covered) / len(country_lists)) if country_lists else None,
        "multi_country_share": (
            sum(1 for countries in covered if len(countries) > 1) / len(covered)
            if covered
            else None
        ),
        "enabled_for_bias_metrics": bool(covered),
        "reason": None if covered else "Country metadata unavailable.",
        "min_coverage_threshold": None,
        "overall_distribution_variants": {
            "countries": {
                "fractional": overall_fractional,
                "dominant": overall_dominant,
            },
            "regions": {"fractional": {}, "dominant": {}},
        },
        "per_k": per_k,
    }


def _citations_top_k(frame: pd.DataFrame, *, ks: Sequence[int]) -> dict[str, dict[str, Any]]:
    ranked = _ranked_frame(frame)
    payload: dict[str, dict[str, Any]] = {}
    for k in ks:
        top = _numeric_values(ranked.head(k), "cited_by_count")
        rest = _numeric_values(ranked.iloc[k:], "cited_by_count")
        payload[str(k)] = {
            "median_top_k": _median(top),
            "median_rest": _median(rest),
            "reliability": "high" if top or rest else "low",
        }
    return payload


def _spearman_rank_vs_citations(frame: pd.DataFrame) -> float | None:
    ranked = _ranked_frame(frame)
    if ranked.empty:
        return None
    pairs = ranked[["rank", "cited_by_count"]].copy()
    pairs["rank"] = pd.to_numeric(pairs["rank"], errors="coerce")
    pairs["cited_by_count"] = pd.to_numeric(pairs["cited_by_count"], errors="coerce")
    pairs = pairs.dropna()
    if len(pairs) < 2:
        return None
    return float(pairs["rank"].corr(pairs["cited_by_count"], method="spearman"))


def _build_llm_specific_context(
    *,
    raw_records: Sequence[RawResponseRecord],
    normalized_records: Sequence[NormalizedResponseRecord],
    enriched_records: Sequence[EnrichedRecommendationRecord],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_by_request = {record.request_id: record for record in normalized_records}
    enriched_by_request: dict[str, list[dict[str, Any]]] = {}
    for record in enriched_records:
        enriched_by_request.setdefault(record.request_id, []).append(record.model_dump())

    query_map: dict[str, dict[str, Any]] = {}
    for raw in sorted(
        raw_records,
        key=lambda record: (
            record.query_category or "",
            record.query_id,
            record.model,
            record.repeat_index,
        ),
    ):
        entry = query_map.setdefault(
            raw.query_id,
            {
                "query_id": raw.query_id,
                "query_text": raw.query_text,
                "query_category": raw.query_category,
                "query_language": raw.query_language,
                "source_mode": raw.source_mode,
                "pair_id": raw.pair_id,
                "variant": raw.variant,
                "control_or_treatment": raw.control_or_treatment,
                "prompt_text": raw.prompt_text,
                "responses": [],
            },
        )
        normalized = normalized_by_request.get(raw.request_id)
        entry["responses"].append(
            {
                "request_id": raw.request_id,
                "model": raw.model,
                "repeat_index": raw.repeat_index,
                "success": raw.success,
                "latency_ms": raw.latency_ms,
                "token_usage": raw.token_usage.model_dump(),
                "error_message": raw.error_message,
                "raw_response_text": raw.raw_response_text,
                "raw_response_json": raw.raw_response_json,
                "parse_success": normalized.parse_success if normalized else False,
                "parse_status": normalized.parse_status if normalized else "missing",
                "parse_method": normalized.parse_method if normalized else None,
                "parse_error": normalized.parse_error if normalized else None,
                "article_recommendations": [
                    item.model_dump()
                    for item in (normalized.article_recommendations if normalized else [])
                ],
                "enriched_recommendations": enriched_by_request.get(raw.request_id, []),
            }
        )

    hallucination = metrics.get("llm_specific", {}).get("hallucination", {})
    parse_robustness = metrics.get("parse_robustness", {})
    retrieval_usefulness = metrics.get("retrieval_usefulness", {})
    metadata_conflicts = metrics.get("metadata_conflicts", {})
    stability = metrics.get("stability", {})
    cross_model = metrics.get("cross_model_divergence", {})
    hallucination_rows = _llm_metric_rows(
        overall=hallucination.get("overall", {}),
        by_model=hallucination.get("by_model", {}),
        metadata_conflicts=metadata_conflicts.get("by_model", {}),
    )
    if hallucination_rows and metadata_conflicts.get("overall"):
        hallucination_rows[0].update(
            {
                "any_conflict_rate": metadata_conflicts["overall"].get("any_conflict_rate"),
                "any_conflict_count": metadata_conflicts["overall"].get("any_conflict_count"),
            }
        )

    return {
        "present": True,
        "query_count": metrics.get("overview", {}).get("query_count"),
        "call_count": metrics.get("overview", {}).get("call_count"),
        "query_details": list(query_map.values()),
        "hallucination_rows": hallucination_rows,
        "parse_rows": _llm_metric_rows(
            overall=parse_robustness.get("overall", {}),
            by_model=parse_robustness.get("by_model", {}),
        ),
        "usefulness_rows": _llm_metric_rows(
            overall=retrieval_usefulness.get("overall", {}),
            by_model=retrieval_usefulness.get("by_model", {}),
        ),
        "conflict_rows": _llm_metric_rows(
            overall=metadata_conflicts.get("overall", {}),
            by_model=metadata_conflicts.get("by_model", {}),
        ),
        "stability_rows": stability.get("by_query", []),
        "stability_summary_rows": [
            {"model": model, **payload}
            for model, payload in sorted(stability.get("summary_by_model", {}).items())
        ],
        "cross_model_rows": cross_model.get("pairwise", []),
        "pairwise_rows": metrics.get("pairwise_comparisons", []),
    }


def _llm_metric_rows(
    *,
    overall: Mapping[str, Any],
    by_model: Mapping[str, Any],
    metadata_conflicts: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if overall:
        rows.append({"model": "overall", **overall})
    for model, payload in sorted(by_model.items()):
        row = {"model": model, **payload}
        if metadata_conflicts and model in metadata_conflicts:
            row.update(
                {
                    "any_conflict_rate": metadata_conflicts[model].get("any_conflict_rate"),
                    "any_conflict_count": metadata_conflicts[model].get("any_conflict_count"),
                }
            )
        rows.append(row)
    return rows


def _top_ks(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "rank" not in frame.columns:
        return [5]
    ranks = pd.to_numeric(frame["rank"], errors="coerce").dropna()
    if ranks.empty:
        return [5]
    max_rank = int(ranks.max())
    ks = [k for k in (5, 10, 20) if k <= max_rank]
    return ks or [max_rank]


def _ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["rank"] = pd.to_numeric(ranked["rank"], errors="coerce")
    ranked = ranked.dropna(subset=["rank"]).sort_values("rank", kind="mergesort")
    return ranked


def _numeric_values(frame: pd.DataFrame, column: str) -> list[float]:
    if frame.empty or column not in frame.columns:
        return []
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return [float(value) for value in series.tolist()]


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _hhi(values: Sequence[str]) -> float | None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    total = sum(counts.values())
    return float(sum((count / total) ** 2 for count in counts.values()))


def _oa_series(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    if "is_oa" in frame.columns:
        return frame["is_oa"]
    return None


def _series_share(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _missing_pct(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    total = len(series)
    if total == 0:
        return None
    return float(series.isna().mean())


def _country_list(row: Mapping[str, Any]) -> list[str]:
    countries = row.get("countries")
    if isinstance(countries, list):
        cleaned = [str(value).strip().upper() for value in countries if str(value).strip()]
        if cleaned:
            return cleaned
    country = row.get("country_primary") or row.get("country_dominant")
    if country:
        return [str(country).strip().upper()]
    return []


def _distribution(country_lists: Iterable[list[str]], *, fractional: bool) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total_weight = 0.0
    for countries in country_lists:
        if not countries:
            continue
        if fractional:
            weight = 1.0 / len(countries)
            for country in countries:
                counts[country] += weight
                total_weight += weight
        else:
            counts[countries[0]] += 1.0
            total_weight += 1.0
    if total_weight <= 0:
        return {}
    return {
        country: float(count / total_weight)
        for country, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    }


def _over_ratio(
    top_distribution: Mapping[str, float],
    overall_distribution: Mapping[str, float],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for label, value in top_distribution.items():
        baseline = overall_distribution.get(label)
        if baseline and baseline > 0:
            ratios[label] = float(value / baseline)
    return ratios
