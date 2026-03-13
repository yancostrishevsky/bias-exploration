from __future__ import annotations

from datetime import datetime

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


def test_run_sanity_checks_marks_core_structural_citation_limitation() -> None:
    current_year = datetime.utcnow().year
    records = [
        {
            "platform": "core",
            "title": f"core-paper-{idx}",
            "rank": idx + 1,
            "year": None,
            "extra": {"core": {"citationCount": 0, "acceptedDate": f"{current_year - 6}-01-01"}},
        }
        for idx in range(40)
    ]

    summary = run_sanity_checks(records)
    core_caps = summary["platform_capabilities"]["core"]
    assert core_caps["citations_available"] is False
    assert (
        core_caps["citations_reason"]
        == "CORE citationCount unreliable (structural limitation)"
    )
    assert any(
        warning == "core: citations unavailable (structural limitation), citation-based metrics disabled"
        for warning in summary["warnings"]
    )
    assert not any("suspicious citations detected" in warning for warning in summary["warnings"])


def test_run_sanity_checks_extracts_openalex_host_org_publisher() -> None:
    records = [
        {
            "platform": "openalex",
            "title": "OpenAlex host org publisher",
            "rank": 1,
            "extra": {
                "openalex": {
                    "primary_location": {
                        "source": {
                            "host_organization_name": "Elsevier BV",
                            "host_organization_lineage_names": ["Elsevier Group", "Elsevier BV"],
                            "display_name": "Journal Name",
                        }
                    },
                    "host_venue": {},
                }
            },
        }
    ]
    summary = run_sanity_checks(records)
    openalex = summary["by_platform"]["openalex"]
    assert openalex["publisher_missing_rate"] == 0.0
    assert openalex["publisher_parser_bug_rate"] == 0.0


def test_run_sanity_checks_tracks_openalex_structural_missing_separately() -> None:
    records = [
        {
            "platform": "openalex",
            "title": "With publisher",
            "rank": 1,
            "extra": {
                "openalex": {
                    "primary_location": {
                        "source": {"host_organization_name": "Elsevier BV"}
                    }
                }
            },
        },
        {
            "platform": "openalex",
            "title": "Without publisher fields",
            "rank": 2,
            "extra": {"openalex": {"primary_location": {"source": {}}}},
        },
    ]

    summary = run_sanity_checks(records)
    openalex = summary["by_platform"]["openalex"]
    assert openalex["publisher_parser_bug_rate"] == 0.0
    assert openalex["publisher_structural_missing_rate"] == 0.5


def test_run_sanity_checks_reports_year_discrepancies_for_semanticscholar() -> None:
    records = [
        {
            "platform": "semanticscholar",
            "title": "Mismatch",
            "rank": 1,
            "publication_year": 2012,
            "extra": {
                "semanticscholar": {
                    "paperId": "s2-1",
                    "year": 1897,
                    "publicationDate": "1897-03-01",
                    "publicationVenue": {"name": "Venue"},
                },
                "openalex": {"publication_year": 2012},
            },
        }
    ]
    summary = run_sanity_checks(records)
    discrepancies = summary["year_discrepancies"]["semanticscholar"]
    assert discrepancies["count"] == 1
    assert discrepancies["share"] == 1.0
    assert discrepancies["examples"][0]["year_raw"] == 1897
    assert discrepancies["examples"][0]["year_enriched"] == 2012
    assert discrepancies["examples"][0]["year_provenance"] == "mixed"


def test_run_sanity_checks_marks_connector_error_when_no_records_due_to_http_errors() -> None:
    summary = run_sanity_checks(
        [],
        request_logs={
            "core": [
                {"status_code": 404, "method": "GET", "endpoint": "https://api.core.ac.uk/v3/search/works"},
            ]
        },
    )
    assert summary["platform_health"]["core"]["enabled"] is False
    assert summary["platform_health"]["core"]["error_rate"] == 1.0
    assert summary["by_platform"]["core"]["result_status"] == "connector_error"
    assert any("connector_error" in warning for warning in summary["warnings"])


def test_run_sanity_checks_includes_failing_field_samples() -> None:
    summary = run_sanity_checks(
        [
            {
                "platform": "semanticscholar",
                "title": "Missing fields",
                "rank": 1,
                "extra": {"semanticscholar": {"paperId": "s2"}},
            }
        ]
    )
    failing = summary["failing_field_samples"]["semanticscholar"]
    assert "publisher" in failing
    assert "year" in failing
    assert "issn" in failing
    assert "citations" in failing
    assert isinstance(failing["year"]["field_paths"], list)


def test_run_sanity_checks_includes_geo_capabilities_coverage_and_samples() -> None:
    records = [
        {
            "platform": "openalex",
            "title": "Geo OA",
            "rank": 1,
            "extra": {
                "openalex": {
                    "authorships": [
                        {"institutions": [{"country_code": "PL"}, {"country_code": "DE"}]}
                    ]
                }
            },
        },
        {
            "platform": "semanticscholar",
            "title": "Geo missing",
            "rank": 1,
            "extra": {"semanticscholar": {"paperId": "s2-geo"}},
        },
    ]
    summary = run_sanity_checks(records)
    geo = summary["geo"]
    assert geo["capabilities"]["openalex"]["available"] is True
    assert geo["coverage"]["openalex"]["available_share"] == 1.0
    assert geo["coverage"]["openalex"]["multi_country_share"] == 1.0
    assert geo["capabilities"]["semanticscholar"]["available"] is False
    assert geo["coverage"]["semanticscholar"]["reason"] == "structural_unavailability"
    openalex_sample = geo["samples"]["openalex"][0]["canonical"]
    assert openalex_sample["countries"] == ["DE", "PL"]
    assert openalex_sample["country_primary"] == "MULTI"
