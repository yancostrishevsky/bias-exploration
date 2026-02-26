from __future__ import annotations

from ai_bias_search.diagnostics.sanity import run_sanity_checks


def test_run_sanity_checks_computes_platform_rates() -> None:
    records = [
        {
            "platform": "openalex",
            "title": "OpenAlex paper",
            "rank": 1,
            "publisher": "ACM",
            "issn_list": ["1234-5678"],
            "cited_by_count": 10,
        },
        {
            "platform": "openalex",
            "title": "OpenAlex missing",
            "rank": 2,
            "cited_by_count": None,
        },
        {
            "platform": "scopus",
            "title": "Scopus missing",
            "rank": 1,
        },
    ]

    summary = run_sanity_checks(records)

    assert summary["total_records"] == 3
    assert "openalex" in summary["by_platform"]
    assert "scopus" in summary["by_platform"]

    openalex = summary["by_platform"]["openalex"]
    assert openalex["issn_missing_rate"] == 0.5
    assert openalex["publisher_missing_rate"] == 0.5
    assert openalex["citations_missing_rate"] == 0.5


def test_run_sanity_checks_warns_on_suspicious_citations() -> None:
    records = [
        {
            "platform": "scopus",
            "title": f"paper-{i}",
            "rank": i + 1,
            "year": 2016 + (i % 5),
            "citedby-count": 0,
            "extra": {"scopus": {"raw": {"prism:issn": "1234-5678"}}},
        }
        for i in range(30)
    ]

    summary = run_sanity_checks(records)
    warnings = summary["warnings"]
    assert any("suspicious citations detected" in warning for warning in warnings)
