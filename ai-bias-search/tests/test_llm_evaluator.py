from __future__ import annotations

from ai_bias_search.llm.evaluator import evaluate_run
from ai_bias_search.llm.schemas import (
    EnrichedRecommendationRecord,
    NormalizedResponseRecord,
    RawResponseRecord,
    TokenUsage,
)


def _raw(model: str, request_id: str, repeat_index: int, success: bool = True) -> RawResponseRecord:
    return RawResponseRecord(
        run_id="run1",
        timestamp="20260101T000000Z",
        request_id=request_id,
        query_id="q1",
        query_text="heart failure treatment",
        query_category="cardiology",
        query_language="en",
        mode="article_recommendation",
        model=model,
        provider="openrouter",
        repeat_index=repeat_index,
        prompt_text="prompt",
        success=success,
        latency_ms=100 + repeat_index,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        raw_response_text='{"articles":[]}' if success else None,
        error_message=None if success else "boom",
    )


def _normalized(
    model: str, request_id: str, repeat_index: int, parse_success: bool = True
) -> NormalizedResponseRecord:
    return NormalizedResponseRecord(
        run_id="run1",
        request_id=request_id,
        timestamp="20260101T000000Z",
        query_id="q1",
        query_text="heart failure treatment",
        query_category="cardiology",
        query_language="en",
        mode="article_recommendation",
        model=model,
        provider="openrouter",
        repeat_index=repeat_index,
        parse_success=parse_success,
        parse_status="parsed" if parse_success else "parse_failed",
        parse_method="strict_json" if parse_success else None,
        parse_error=None if parse_success else "json_parse_failed",
        parse_confidence=1.0 if parse_success else None,
        normalized_kind="article_recommendation",
        item_count=1 if parse_success else 0,
        success=True,
    )


def _enriched(
    model: str,
    request_id: str,
    repeat_index: int,
    doi: str,
    year: int,
    citations: int,
    matched: bool = True,
) -> EnrichedRecommendationRecord:
    return EnrichedRecommendationRecord(
        run_id="run1",
        request_id=request_id,
        query_id="q1",
        query_text="heart failure treatment",
        query_category="cardiology",
        query_language="en",
        model=model,
        provider="openrouter",
        repeat_index=repeat_index,
        recommended_rank=1,
        llm_claimed_title=f"{doi} title",
        llm_claimed_doi=doi,
        llm_claimed_year=year,
        llm_claimed_journal="Journal of Tests",
        enriched_title=f"{doi} title",
        enriched_doi=doi,
        enriched_year=year,
        enriched_journal="Journal of Tests",
        openalex_match_found=matched,
        valid_doi=True,
        cited_by_count=citations,
        is_oa=True,
        publisher="Publisher A",
        country_primary="US",
    )


def test_evaluate_run_computes_model_and_cross_model_metrics() -> None:
    raw_records = [
        _raw("model-a", "a-r1", 0),
        _raw("model-a", "a-r2", 1),
        _raw("model-b", "b-r1", 0),
    ]
    normalized_records = [
        _normalized("model-a", "a-r1", 0),
        _normalized("model-a", "a-r2", 1),
        _normalized("model-b", "b-r1", 0),
    ]
    enriched_records = [
        _enriched("model-a", "a-r1", 0, "10.1000/a", 2024, 50),
        _enriched("model-a", "a-r2", 1, "10.1000/a", 2024, 55),
        _enriched("model-b", "b-r1", 0, "10.1000/b", 2018, 500),
    ]

    metrics = evaluate_run(raw_records, normalized_records, enriched_records)

    assert metrics["overview"]["call_count"] == 3
    assert metrics["overview"]["query_count"] == 1
    assert metrics["overview"]["parse_success_rate"] == 1.0
    assert metrics["by_model"]["model-a"]["hallucination"]["matched_proportion"] == 1.0
    assert metrics["by_model"]["model-b"]["citation"]["mean_citations"] == 500.0
    assert metrics["cross_model_divergence"]["pairwise"][0]["mean_jaccard"] == 0.0
    assert metrics["stability"]["by_query"][0]["mean_jaccard"] == 1.0
