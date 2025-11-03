from pathlib import Path

import httpx
import pytest
import respx

from ai_bias_search.normalization import openalex_enrich


@respx.mock
def test_enrich_with_openalex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect cache to temporary test directory
    monkeypatch.setattr(openalex_enrich, "CACHE_DIR", tmp_path / "cache")

    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"id": "https://openalex.org/W123"}]},
        )
    )
    respx.get("https://api.openalex.org/works/W123").mock(
        return_value=httpx.Response(
            200,
            json={
                "language": "en",
                "is_oa": True,
                "publication_year": 2023,
                "host_venue": {"display_name": "Journal"},
                "publisher": "Publisher",
                "cited_by_count": 42,
            },
        )
    )

    records = [
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
    assert enriched_record["publisher"] == "Publisher"
    assert enriched_record["cited_by_count"] == 42
