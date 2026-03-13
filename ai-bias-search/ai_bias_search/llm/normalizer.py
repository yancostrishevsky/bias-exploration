"""Normalization of raw LLM responses into structured records."""

from __future__ import annotations

from typing import Any, Iterable

from ai_bias_search.llm.parser import parse_json_response
from ai_bias_search.llm.schemas import (
    ArticleRecommendationItem,
    NormalizedResponseRecord,
    ParsedPayload,
    RankingItem,
    RawResponseRecord,
)

_PARSE_CONFIDENCE = {
    "strict_json": 1.0,
    "fenced_json": 0.8,
    "embedded_json": 0.6,
}


def normalize_responses(records: Iterable[RawResponseRecord]) -> list[NormalizedResponseRecord]:
    """Normalize raw response records into typed, structured outputs."""

    return [normalize_response_record(record) for record in records]


def normalize_response_record(record: RawResponseRecord) -> NormalizedResponseRecord:
    """Normalize one raw response record."""

    parsed = (
        parse_json_response(record.raw_response_text)
        if record.success
        else ParsedPayload(
            success=False,
            parse_error=record.error_message or "request_failed",
        )
    )

    normalized_kind = "generic"
    article_items: list[ArticleRecommendationItem] = []
    ranking_items: list[RankingItem] = []

    if parsed.success:
        confidence = _PARSE_CONFIDENCE.get(parsed.parse_method or "", 0.5)
        if record.mode == "article_recommendation":
            normalized_kind = "article_recommendation"
            article_items = _normalize_article_items(parsed.parsed_json, confidence=confidence)
        elif record.mode == "ranking":
            normalized_kind = "ranking"
            ranking_items = _normalize_ranking_items(parsed.parsed_json, confidence=confidence)
        else:
            normalized_kind = "generic"
        parse_status = "parsed"
    else:
        confidence = None
        parse_status = "request_failed" if not record.success else "parse_failed"

    return NormalizedResponseRecord(
        run_id=record.run_id,
        request_id=record.request_id,
        timestamp=record.timestamp,
        source_mode=record.source_mode,
        query_id=record.query_id,
        query_text=record.query_text,
        query_category=record.query_category,
        query_language=record.query_language,
        mode=record.mode,
        model=record.model,
        provider=record.provider,
        repeat_index=record.repeat_index,
        pair_id=record.pair_id,
        variant=record.variant,
        control_or_treatment=record.control_or_treatment,
        expected_format=record.expected_format,
        topic=record.topic,
        input_metadata=record.input_metadata,
        parse_success=parsed.success,
        parse_status=parse_status,
        parse_method=parsed.parse_method,
        parse_error=parsed.parse_error,
        parse_confidence=confidence,
        normalized_kind=normalized_kind,
        item_count=len(article_items) if article_items else len(ranking_items),
        article_recommendations=article_items,
        ranking_items=ranking_items,
        parsed_payload=parsed.parsed_json,
        raw_response_text=record.raw_response_text,
        success=record.success,
        error_message=record.error_message,
    )


def _normalize_article_items(payload: Any, *, confidence: float) -> list[ArticleRecommendationItem]:
    items = _extract_item_list(
        payload,
        keys=("articles", "recommendations", "results", "items", "papers", "records"),
    )
    normalized: list[ArticleRecommendationItem] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            normalized.append(
                ArticleRecommendationItem(
                    rank=_coerce_int(item.get("rank"))
                    or _coerce_int(item.get("position"))
                    or index,
                    title=_coerce_text(item.get("title"))
                    or _coerce_text(item.get("paper_title"))
                    or _coerce_text(item.get("name")),
                    doi=_coerce_text(item.get("doi")),
                    year=_coerce_int(item.get("year")) or _coerce_int(item.get("publication_year")),
                    journal=_coerce_text(item.get("journal"))
                    or _coerce_text(item.get("venue"))
                    or _coerce_text(item.get("journal_title"))
                    or _coerce_text(item.get("source")),
                    authors=_coerce_authors(item.get("authors")),
                    rationale=_coerce_text(item.get("rationale"))
                    or _coerce_text(item.get("explanation"))
                    or _coerce_text(item.get("reason"))
                    or _coerce_text(item.get("justification"))
                    or _coerce_text(item.get("why")),
                    parse_confidence=confidence,
                    raw_item=item,
                )
            )
            continue
        if isinstance(item, str) and item.strip():
            normalized.append(
                ArticleRecommendationItem(
                    rank=index,
                    title=item.strip(),
                    parse_confidence=min(confidence, 0.5),
                    raw_item=item,
                )
            )
    return normalized


def _normalize_ranking_items(payload: Any, *, confidence: float) -> list[RankingItem]:
    items = _extract_item_list(
        payload,
        keys=("ranking", "rankings", "results", "items", "judgments"),
    )
    normalized: list[RankingItem] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            normalized.append(
                RankingItem(
                    rank=_coerce_int(item.get("rank"))
                    or _coerce_int(item.get("position"))
                    or index,
                    identifier=_coerce_text(item.get("id"))
                    or _coerce_text(item.get("candidate_id"))
                    or _coerce_text(item.get("doi")),
                    title=_coerce_text(item.get("title"))
                    or _coerce_text(item.get("name"))
                    or _coerce_text(item.get("label")),
                    score=_coerce_float(item.get("score"))
                    or _coerce_float(item.get("relevance_score")),
                    rationale=_coerce_text(item.get("rationale"))
                    or _coerce_text(item.get("reason"))
                    or _coerce_text(item.get("justification")),
                    parse_confidence=confidence,
                    raw_item=item,
                )
            )
            continue
        if isinstance(item, str) and item.strip():
            normalized.append(
                RankingItem(
                    rank=index,
                    title=item.strip(),
                    parse_confidence=min(confidence, 0.5),
                    raw_item=item,
                )
            )
    return normalized


def _extract_item_list(payload: Any, *, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


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
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_authors(value: Any) -> list[str] | None:
    if isinstance(value, list):
        authors = [str(item).strip() for item in value if str(item).strip()]
        return authors or None
    if isinstance(value, str):
        authors = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        return authors or None
    return None
