"""Metadata enrichment helpers for normalized LLM recommendations."""

from __future__ import annotations

from typing import Any, Iterable

from ai_bias_search.llm.schemas import EnrichedRecommendationRecord, NormalizedResponseRecord
from ai_bias_search.normalization.openalex_enrich import enrich_with_openalex
from ai_bias_search.utils.config import ImpactFactorConfig, RetryConfig
from ai_bias_search.utils.ids import normalise_doi
from ai_bias_search.utils.rate_limit import RateLimiter


def enrich_recommendations(
    records: Iterable[NormalizedResponseRecord],
    *,
    enabled: bool,
    openalex_mailto: str | None,
    impact_factor: ImpactFactorConfig | None,
    retries: RetryConfig,
    rate_limiter: RateLimiter | None,
) -> list[EnrichedRecommendationRecord]:
    """Enrich article recommendation items with OpenAlex metadata when enabled."""

    base_rows = _flatten_article_recommendations(records)
    if not base_rows:
        return []

    if not enabled:
        return [_map_without_enrichment(row) for row in base_rows]

    openalex_inputs = [row["openalex_input"] for row in base_rows]
    enriched_rows = enrich_with_openalex(
        openalex_inputs,
        openalex_mailto,
        impact_factor,
        rate_limiter=rate_limiter,
        retries=retries,
    )
    return [
        _map_enriched(base, enriched)
        for base, enriched in zip(base_rows, enriched_rows, strict=False)
    ]


def _flatten_article_recommendations(
    records: Iterable[NormalizedResponseRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record.article_recommendations:
            rows.append(
                {
                    "normalized": record,
                    "item": item,
                    "openalex_input": {
                        "title": item.title,
                        "doi": item.doi,
                        "rank": item.rank or 1,
                        "source": item.journal,
                        "year": item.year,
                        "authors": item.authors,
                        "extra": {
                            "llm": {
                                "request_id": record.request_id,
                                "query_id": record.query_id,
                                "model": record.model,
                            }
                        },
                    },
                }
            )
    return rows


def _map_without_enrichment(row: dict[str, Any]) -> EnrichedRecommendationRecord:
    record: NormalizedResponseRecord = row["normalized"]
    item = row["item"]
    return EnrichedRecommendationRecord(
        run_id=record.run_id,
        request_id=record.request_id,
        source_mode=record.source_mode,
        query_id=record.query_id,
        query_text=record.query_text,
        query_category=record.query_category,
        query_language=record.query_language,
        model=record.model,
        provider=record.provider,
        repeat_index=record.repeat_index,
        pair_id=record.pair_id,
        variant=record.variant,
        control_or_treatment=record.control_or_treatment,
        topic=record.topic,
        recommended_rank=item.rank,
        llm_claimed_title=item.title,
        llm_claimed_doi=item.doi,
        llm_claimed_year=item.year,
        llm_claimed_journal=item.journal,
        llm_claimed_authors=item.authors,
        rationale=item.rationale,
        parse_status=record.parse_status,
        parse_confidence=item.parse_confidence,
        valid_doi=bool(normalise_doi(item.doi)),
        openalex_match_found=False,
        provenance={
            "title": "llm_output",
            "doi": "llm_output",
            "year": "llm_output",
            "journal": "llm_output",
        },
        extra={"raw_item": item.raw_item},
    )


def _map_enriched(
    base_row: dict[str, Any], enriched_row: dict[str, Any]
) -> EnrichedRecommendationRecord:
    record: NormalizedResponseRecord = base_row["normalized"]
    item = base_row["item"]

    extra = enriched_row.get("extra") if isinstance(enriched_row.get("extra"), dict) else {}
    openalex_payload = (
        extra.get("openalex_enrich") if isinstance(extra.get("openalex_enrich"), dict) else {}
    )
    match_found = bool(openalex_payload)
    enriched_title = _coerce_text(enriched_row.get("title"))
    enriched_doi = normalise_doi(enriched_row.get("doi"))
    enriched_year = _coerce_int(enriched_row.get("publication_year") or enriched_row.get("year"))
    enriched_journal = _coerce_text(
        enriched_row.get("journal_title")
        or enriched_row.get("host_venue")
        or enriched_row.get("source")
    )

    return EnrichedRecommendationRecord(
        run_id=record.run_id,
        request_id=record.request_id,
        source_mode=record.source_mode,
        query_id=record.query_id,
        query_text=record.query_text,
        query_category=record.query_category,
        query_language=record.query_language,
        model=record.model,
        provider=record.provider,
        repeat_index=record.repeat_index,
        pair_id=record.pair_id,
        variant=record.variant,
        control_or_treatment=record.control_or_treatment,
        topic=record.topic,
        recommended_rank=item.rank,
        llm_claimed_title=item.title,
        llm_claimed_doi=item.doi,
        llm_claimed_year=item.year,
        llm_claimed_journal=item.journal,
        llm_claimed_authors=item.authors,
        rationale=item.rationale,
        parse_status=record.parse_status,
        parse_confidence=item.parse_confidence,
        enriched_title=enriched_title,
        enriched_doi=enriched_doi,
        enriched_year=enriched_year,
        enriched_journal=enriched_journal,
        enriched_authors=_coerce_authors(enriched_row.get("authors")),
        openalex_id=_coerce_text(enriched_row.get("openalex_id")),
        openalex_match_found=match_found,
        valid_doi=bool(normalise_doi(item.doi)),
        is_oa=_coerce_bool(enriched_row.get("is_oa")),
        cited_by_count=_coerce_int(
            enriched_row.get("cited_by_count") or enriched_row.get("citations")
        ),
        publisher=_coerce_text(enriched_row.get("publisher")),
        core_rank=_coerce_text(enriched_row.get("core_rank")),
        impact_factor=_coerce_float(enriched_row.get("impact_factor")),
        jcr_quartile=_coerce_text(enriched_row.get("jcr_quartile")),
        language=_coerce_text(enriched_row.get("language")),
        country_primary=_coerce_text(
            enriched_row.get("country_primary") or enriched_row.get("country_dominant")
        ),
        countries=_coerce_authors(enriched_row.get("countries")),
        rankings=(
            enriched_row.get("rankings") if isinstance(enriched_row.get("rankings"), dict) else {}
        ),
        provenance={
            "title": "openalex" if match_found and enriched_title else "llm_output",
            "doi": "openalex" if match_found and enriched_doi else "llm_output",
            "year": "openalex" if match_found and enriched_year is not None else "llm_output",
            "journal": "openalex" if match_found and enriched_journal else "llm_output",
        },
        extra={
            "raw_item": item.raw_item,
            "enrich_trace": extra.get("enrich_trace"),
            "openalex": openalex_payload,
        },
    )


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    return None


def _coerce_authors(value: Any) -> list[str] | None:
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or None
    if isinstance(value, str):
        out = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        return out or None
    return None
