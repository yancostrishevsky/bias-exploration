from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_bias_search.connectors.base import ConnectorError
from ai_bias_search.connectors.scopus import ScopusConnector
from ai_bias_search.utils.config import AppConfig, RetryConfig, ScopusEnrichConfig
from ai_bias_search.utils.rate_limit import RateLimiter

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "scopus"


def _snapshot(name: str) -> dict[str, Any]:
    return json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


def test_scopus_connector_paginates_and_maps_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    page_1 = _snapshot("search_page_1.json")
    page_2 = _snapshot("search_page_2.json")
    starts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/content/search/scopus"
        starts.append(request.url.params.get("start", ""))
        assert request.headers.get("X-ELS-APIKey") == "test-key"
        assert request.headers.get("X-ELS-Insttoken") is None
        if request.url.params.get("start") == "0":
            return httpx.Response(200, json=page_1)
        if request.url.params.get("start") == "2":
            return httpx.Response(200, json=page_2)
        return httpx.Response(200, json={"search-results": {"entry": []}})

    client = httpx.Client(
        base_url="https://api.elsevier.com",
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )
    connector = ScopusConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
        config=ScopusEnrichConfig(enabled=True, page_size=2, max_records_per_query=10),
    )

    records = connector.search("heart", k=3, params={"count": 2})
    assert len(records) == 3
    assert starts == ["0", "2"]

    first = records[0]
    assert first["rank"] == 1
    assert first["scopus_id"] == "85012345678"
    assert first["eid"] == "2-s2.0-85012345678"
    assert first["doi"] == "10.1234/example.one"
    assert first["title"] == "First Scopus Paper"
    assert first["creator"] == "Doe, Jane"
    assert first["publicationName"] == "Journal of Testing"
    assert first["issn"] == "1234-5678"
    assert first["eIssn"] == "8765-4321"
    assert first["coverDate"] == "2021-01-15"
    assert first["citedby-count"] == 17
    assert first["openaccessFlag"] is True
    assert first["source-id"] == "12345"
    assert first["extra"]["scopus"]["raw"]["dc:identifier"] == "SCOPUS_ID:85012345678"

    third = records[2]
    assert third["rank"] == 3
    assert third["title"] == "Third Scopus Paper"
    assert third["year"] == 2019

    client.close()


def test_scopus_connector_builds_params_and_headers_with_insttoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.setenv("SCOPUS_INSTTOKEN", "inst-token")

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"search-results": {"entry": []}})

    client = httpx.Client(
        base_url="https://api.elsevier.com",
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )
    connector = ScopusConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
        config=ScopusEnrichConfig(
            enabled=True,
            search_view="COMPLETE",
            search_fields="dc:identifier,dc:title",
            search_sort="-citedby-count",
        ),
    )

    _ = connector.search("llm bias", k=1)

    headers = captured["headers"]
    params = captured["params"]
    assert headers["x-els-apikey"] == "test-key"
    assert headers["x-els-insttoken"] == "inst-token"
    assert params["view"] == "COMPLETE"
    assert params["field"] == "dc:identifier,dc:title"
    assert params["sort"] == "-citedby-count"
    assert params["count"] == "1"
    assert params["start"] == "0"

    client.close()


def test_scopus_connector_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.delenv("ELSEVIER_API_KEY", raising=False)

    with pytest.raises(ConnectorError):
        _ = ScopusConnector(
            rate_limiter=RateLimiter(rate=1000, burst=1000),
            retries=RetryConfig(max=1, backoff=1.0),
            client=httpx.Client(base_url="https://api.elsevier.com"),
            config=ScopusEnrichConfig(enabled=True),
        )


def test_config_accepts_scopus_alias_block() -> None:
    cfg = AppConfig.model_validate(
        {
            "queries_file": "queries/queries.csv",
            "platforms": ["scopus"],
            "rate_limit": {"scopus": {"rps": 1, "burst": 2}},
            "scopus": {"enabled": True, "fail_open": False},
        }
    )
    assert cfg.scopus.enabled is True
    assert cfg.scopus.fail_open is False
    assert cfg.scopus_enrich.enabled is True
