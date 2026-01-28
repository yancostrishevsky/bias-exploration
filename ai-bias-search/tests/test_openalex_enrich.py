from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_bias_search.normalization import openalex_enrich


def test_enrich_with_openalex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect cache to temporary test directory
    monkeypatch.setattr(openalex_enrich, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(openalex_enrich, "lookup_core_rank", lambda **_: "A*")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.host == "api.openalex.org":
            if request.url.path == "/works/doi:10.1234/example":
                return httpx.Response(200, json={"id": "https://openalex.org/W123"})
            if request.url.path == "/works/W123":
                return httpx.Response(
                    200,
                    json={
                        "language": "en",
                        "open_access": {"is_oa": True},
                        "publication_year": 2023,
                        "primary_location": {
                            "source": {
                                "display_name": (
                                    "ACM Conference on Computer and Communications Security"
                                ),
                                "type": "conference",
                                "publisher": "ACM",
                                "is_core": True,
                                "abbreviated_title": "CCS",
                            },
                            "raw_source_name": "ACM CCS",
                        },
                        "host_venue": {
                            "display_name": "Fallback Venue",
                            "publisher": "Fallback Publisher",
                        },
                        "cited_by_count": 42,
                    },
                )
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
            "url": None,
            "rank": 1,
            "raw_id": None,
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
