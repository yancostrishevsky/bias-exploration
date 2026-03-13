from __future__ import annotations

from datetime import datetime

from ai_bias_search.diagnostics.sanity import run_sanity_checks
from ai_bias_search.utils.config import DiagnosticsConfig


def test_semanticscholar_publisher_structural_unavailability_is_classified() -> None:
    records = [
        {
            "platform": "semanticscholar",
            "title": "S2 paper",
            "rank": 1,
            "extra": {
                "semanticscholar": {
                    "paperId": "abc",
                    "journal": {"name": "Journal Name"},
                    "publicationVenue": {"name": "Venue Name"},
                    "citationCount": 3,
                }
            },
        }
    ]

    summary = run_sanity_checks(
        records,
        diagnostics=DiagnosticsConfig(
            enabled=True,
            capture_samples=False,
            capture_requests=False,
        ),
    )

    caps = summary["platform_capabilities"]["semanticscholar"]
    assert caps["publisher_available"] is False
    assert summary["by_platform"]["semanticscholar"]["publisher_structurally_unavailable"] is True
    assert any("structural API limitation" in warning for warning in summary["warnings"])


def test_diagnostics_samples_and_request_logs_are_redacted() -> None:
    records = [
        {
            "platform": "openalex",
            "title": "OpenAlex paper",
            "rank": 1,
            "extra": {
                "openalex": {
                    "id": "https://openalex.org/W1",
                    "primary_location": {"source": {"display_name": "Journal of Tests"}},
                    "host_venue": {"display_name": "Journal of Tests"},
                    "publication_year": 2021,
                    "apiKey": "secret",
                },
                "enrich_trace": [
                    {
                        "platform": "openalex",
                        "strategy": "doi_filter",
                        "status": 200,
                        "result_count": 1,
                        "Authorization": "Bearer hidden",
                    }
                ],
            },
        }
    ]
    request_logs = {
        "openalex": [
            {
                "stage": "collect",
                "endpoint": "https://api.openalex.org/works?apiKey=secret",
                "method": "GET",
                "params": {"apiKey": "secret", "search": "llm"},
                "headers": {"Authorization": "Bearer secret", "X-ELS-APIKey": "top-secret"},
                "status_code": 200,
                "duration_ms": 12,
                "response_keys": ["results"],
                "response_snippet": "{\"results\":[]}",
                "ts": "2026-02-26T00:00:00Z",
            }
        ]
    }

    summary = run_sanity_checks(
        records,
        diagnostics=DiagnosticsConfig(
            enabled=True,
            capture_samples=True,
            capture_requests=True,
            max_sample_records=1,
            max_enrich_trace_entries=2,
            redact_fields=["apiKey", "Authorization", "X-ELS-APIKey"],
        ),
        request_logs=request_logs,
    )

    samples = summary["samples"]["openalex"]
    assert len(samples) == 1
    raw_sample = samples[0]["raw_snippet"]
    assert "apiKey" not in raw_sample
    assert samples[0]["record_id"] is not None
    assert "doi" in samples[0]["mapping_attempts"]
    assert isinstance(samples[0]["canonical"], dict)
    assert samples[0]["enrich_trace"][0]["Authorization"] == "****"

    logged = summary["requests"]["openalex"][0]
    assert logged["params"]["apiKey"] == "****"
    assert logged["headers"]["Authorization"] == "****"
    assert logged["headers"]["X-ELS-APIKey"] == "****"
    assert logged["endpoint"] == "https://api.openalex.org/works"


def test_core_samples_include_structural_citation_note() -> None:
    current_year = datetime.utcnow().year
    records = [
        {
            "platform": "core",
            "title": "CORE sample",
            "rank": 1,
            "extra": {
                "core": {
                    "id": "core:1",
                    "title": "CORE sample",
                    "citationCount": 0,
                    "acceptedDate": f"{current_year - 7}-01-01",
                }
            },
        }
        for _ in range(30)
    ]
    summary = run_sanity_checks(
        records,
        diagnostics=DiagnosticsConfig(
            enabled=True,
            capture_samples=True,
            capture_requests=False,
            max_sample_records=1,
        ),
    )
    sample = summary["samples"]["core"][0]
    assert sample["raw_snippet"]["citationCount"] == 0
    assert sample["canonical"]["citations"] is None
    assert sample["canonical"]["metrics_quality"]["citations"] == "structurally_unavailable"
    assert "citations" in sample["mapping_attempts"]
    assert any("structurally unavailable" in note for note in sample["notes"])
