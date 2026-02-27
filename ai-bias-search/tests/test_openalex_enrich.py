from pathlib import Path
import json
from typing import Any

import httpx
import pytest

from ai_bias_search.normalization import openalex_enrich


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_enrich_with_openalex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect cache to temporary test directory
    monkeypatch.setattr(openalex_enrich, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(openalex_enrich, "lookup_core_rank", lambda **_: "A*")
    openalex_work = _fixture("openalex_work_with_publishers.json")
    requests: list[tuple[str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.host == "api.openalex.org":
            requests.append((request.url.path, dict(request.url.params)))
            if request.url.path == "/works" and request.url.params.get("filter") == "doi:10.1234/example":
                return httpx.Response(200, json={"results": [openalex_work]})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.Client

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)

    records: list[dict[str, Any]] = [
        {
            "title": "A paper",
            "doi": "10.1234/example",
            "raw_id": "0195be9b2cf9cd7052ef359ff4167d63d0e80161",
            "url": None,
            "rank": 1,
            "source": None,
            "year": 2023,
            "authors": ["Doe"],
            "extra": {},
        }
    ]

    enriched = openalex_enrich.enrich_with_openalex(records, mailto="test@example.com")
    enriched_record = enriched[0]
    assert enriched_record["language"] == "en"
    assert enriched_record["is_oa"] is True
    assert enriched_record["host_venue"] == "Fallback Venue"
    assert enriched_record["venue_type"] == "conference"
    assert enriched_record["is_core_listed"] is True
    assert enriched_record["core_rank"] == "A*"
    assert enriched_record["publisher"] == "ACM"
    assert enriched_record["cited_by_count"] == 42
    assert any(
        path == "/works" and params.get("filter") == "doi:10.1234/example"
        for path, params in requests
    )
    assert all(params.get("search") != "0195be9b2cf9cd7052ef359ff4167d63d0e80161" for _, params in requests)
    assert enriched_record["extra"]["enrich_trace"][0]["strategy"] == "doi_filter"


def test_extract_venue_candidates_ignores_roman_acronym() -> None:
    metadata = {
        "host_venue": {"display_name": "Example Conference", "abbreviated_title": "XII"},
        "primary_location": {"source": {"display_name": "Example Conference"}},
    }
    record: dict[str, Any] = {"extra": {}}
    result = openalex_enrich.extract_venue_candidates(metadata, record)
    assert result["venue_acronym"] is None


def test_extract_venue_candidates_dblp_sigir() -> None:
    metadata = {"host_venue": {"display_name": "Example Conference"}}
    record: dict[str, Any] = {
        "extra": {
            "semanticscholar": {
                "venue": (
                    "ACM SIGIR Conference on Research and Development in " "Information Retrieval"
                ),
                "externalIds": {"DBLP": "conf/sigir/2022"},
            }
        }
    }
    result = openalex_enrich.extract_venue_candidates(metadata, record)
    assert result["venue_acronym"] == "SIGIR"


def test_extract_venue_candidates_url_ineligible() -> None:
    metadata = {
        "host_venue": {"display_name": "https://doi.org/10.1234/example"},
        "primary_location": {"source": {"display_name": "https://doi.org/10.1234/example"}},
    }
    record: dict[str, Any] = {"extra": {}}
    result = openalex_enrich.extract_venue_candidates(metadata, record)
    assert result["eligible"] is False
    assert any("venue_name_url" in reason for reason in result["reasons"])
