from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_bias_search.normalize.records import normalize_record, normalize_records

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name,expected_issn,expected_publisher",
    [
        (
            "openalex_record_with_issn_publisher.json",
            ["1234-5679", "8765-4321"],
            "ACM",
        ),
        (
            "semanticscholar_record_with_issn_publisher.json",
            ["2049-3630", "1476-4687"],
            "S2 Publisher",
        ),
        (
            "core_record_with_issn_publisher.json",
            ["1749-4885", "1749-4893"],
            "CORE Publisher",
        ),
    ],
)
def test_normalize_record_extracts_issn_and_publisher(
    fixture_name: str, expected_issn: list[str], expected_publisher: str
) -> None:
    raw = _fixture(fixture_name)
    normalized = normalize_record(raw, platform=str(raw["platform"]))

    assert normalized["issn"] == expected_issn
    assert normalized["publisher"] == expected_publisher
    assert normalized["journal_match"]["confidence"] == 1.0


def test_normalize_scopus_citations_missing_vs_zero_and_wrong_field() -> None:
    missing_raw = _fixture("scopus_record_missing_citations.json")
    zero_raw = _fixture("scopus_record_zero_citations.json")

    missing = normalize_record(missing_raw, platform="scopus")
    zero = normalize_record(zero_raw, platform="scopus")
    wrong_field = normalize_record(
        {
            "platform": "scopus",
            "title": "Wrong Field",
            "rank": 1,
            "citationCount": 11,
            "extra": {"scopus": {"raw": {"publishername": "Elsevier"}}},
        },
        platform="scopus",
    )

    assert missing["citations"] is None
    assert missing["metrics_quality"]["citations"] == "missing"

    assert zero["citations"] == 0
    assert zero["metrics_quality"]["citations"] == "ok"

    assert wrong_field["citations"] is None
    assert wrong_field["metrics_quality"]["citations"] == "missing"


def test_normalize_records_flags_suspicious_zero_citations() -> None:
    records = [
        {
            "platform": "scopus",
            "title": f"paper-{i}",
            "rank": i + 1,
            "year": 2018 + (i % 4),
            "citedby-count": 0,
            "extra": {"scopus": {"raw": {"prism:issn": "1234-5678"}}},
        }
        for i in range(30)
    ]

    normalized = normalize_records(records)
    assert all(item["citations"] is None for item in normalized)
    assert all(item["metrics_quality"]["citations"] == "suspicious" for item in normalized)
