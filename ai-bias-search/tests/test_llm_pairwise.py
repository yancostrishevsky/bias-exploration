from __future__ import annotations

from ai_bias_search.llm.pairwise import compute_pairwise_comparisons
from ai_bias_search.llm.schemas import EnrichedRecommendationRecord


def _record(
    *, query_id: str, variant: str, control: str, year: int, citations: int
) -> EnrichedRecommendationRecord:
    return EnrichedRecommendationRecord(
        run_id="run1",
        request_id=f"{query_id}-req",
        query_id=query_id,
        query_text=f"Query {query_id}",
        query_category="geography_bias",
        query_language="en",
        model="openai/gpt-4o-mini",
        provider="openrouter",
        repeat_index=0,
        pair_id="geo_pair",
        variant=variant,
        control_or_treatment=control,
        recommended_rank=1,
        llm_claimed_title=f"{query_id} paper",
        enriched_title=f"{query_id} paper",
        enriched_doi=f"10.1000/{query_id}",
        enriched_year=year,
        cited_by_count=citations,
        openalex_match_found=True,
        valid_doi=True,
        is_oa=True,
        country_primary="US" if control == "control" else "NG",
    )


def test_compute_pairwise_comparisons_returns_deltas() -> None:
    rows = compute_pairwise_comparisons(
        [
            _record(
                query_id="q_control",
                variant="us",
                control="control",
                year=2020,
                citations=100,
            ),
            _record(
                query_id="q_treat",
                variant="africa",
                control="treatment",
                year=2023,
                citations=20,
            ),
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["pair_id"] == "geo_pair"
    assert row["left_query_id"] == "q_control"
    assert row["average_year_delta"] == 3.0
    assert row["average_citation_delta"] == -80.0
