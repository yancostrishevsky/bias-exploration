from __future__ import annotations

from ai_bias_search.llm.normalizer import normalize_response_record
from ai_bias_search.llm.schemas import RawResponseRecord


def test_normalize_article_recommendation_response() -> None:
    record = RawResponseRecord(
        run_id="run1",
        timestamp="20260101T000000Z",
        request_id="req1",
        query_id="q1",
        query_text="heart failure treatment",
        query_category="cardiology",
        query_language="en",
        mode="article_recommendation",
        model="openai/gpt-4o-mini",
        provider="openrouter",
        repeat_index=0,
        prompt_text="prompt",
        success=True,
        raw_response_text=(
            '{"articles":[{"rank":1,"title":"Paper","doi":"10.1000/test",'
            '"year":2024,"journal":"Nature","authors":["Alice"],"rationale":"Why"}]}'
        ),
    )

    normalized = normalize_response_record(record)

    assert normalized.parse_success is True
    assert normalized.parse_status == "parsed"
    assert normalized.item_count == 1
    item = normalized.article_recommendations[0]
    assert item.title == "Paper"
    assert item.doi == "10.1000/test"
    assert item.journal == "Nature"


def test_normalize_ranking_response() -> None:
    record = RawResponseRecord(
        run_id="run1",
        timestamp="20260101T000000Z",
        request_id="req2",
        query_id="q2",
        query_text="explainable ai in medicine",
        query_category="prestige_bias",
        query_language="en",
        mode="ranking",
        model="openai/gpt-4o-mini",
        provider="openrouter",
        repeat_index=0,
        prompt_text="prompt",
        success=True,
        raw_response_text=(
            '{"ranking":[{"rank":1,"title":"Venue A","score":0.9,'
            '"rationale":"best"}]}'
        ),
    )

    normalized = normalize_response_record(record)

    assert normalized.parse_success is True
    assert normalized.normalized_kind == "ranking"
    assert normalized.ranking_items[0].title == "Venue A"
    assert normalized.ranking_items[0].score == 0.9


def test_normalize_failed_request_preserves_failure() -> None:
    record = RawResponseRecord(
        run_id="run1",
        timestamp="20260101T000000Z",
        request_id="req3",
        query_id="q3",
        query_text="digital public health interventions",
        query_category="public_health",
        query_language="en",
        mode="article_recommendation",
        model="openai/gpt-4o-mini",
        provider="openrouter",
        repeat_index=0,
        prompt_text="prompt",
        success=False,
        error_message="boom",
    )

    normalized = normalize_response_record(record)

    assert normalized.parse_success is False
    assert normalized.parse_status == "request_failed"
    assert normalized.parse_error == "boom"
