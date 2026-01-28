from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from ai_bias_search.connectors.core import CoreConnector
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.rate_limit import RateLimiter


def test_core_connector_paginates_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")
    monkeypatch.setenv("CORE_SEARCH_PATH", "/search/works")
    monkeypatch.setenv("CORE_QUERY_PARAM", "q")
    monkeypatch.setenv("CORE_LIMIT_PARAM", "limit")
    monkeypatch.setenv("CORE_OFFSET_PARAM", "offset")
    monkeypatch.setenv("CORE_AUTH_HEADER", "Authorization")
    monkeypatch.setenv("CORE_AUTH_PREFIX", "Bearer")

    calls: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-key"
        params = dict(request.url.params)
        calls.append({"path": request.url.path, "params": params})
        offset = int(params.get("offset", "0"))
        limit = int(params.get("limit", "0"))

        if offset == 0:
            body = {
                "results": [
                    {
                        "id": "core:1",
                        "title": "Paper One",
                        "doi": "10.1234/ONE",
                        "year": 2020,
                        "authors": [{"name": "Alice"}, {"name": "Bob"}],
                        "url": "https://example.org/one",
                        "venue": "Test Venue",
                    }
                ]
            }
            return httpx.Response(200, json=body, request=request)

        if offset == 1 and limit >= 1:
            body = {
                "results": [
                    {
                        "id": "core:2",
                        "title": "Paper Two",
                        "identifiers": {"doi": "10.9999/two"},
                        "publishedDate": "2019-01-02",
                        "authors": ["Carol"],
                        "landingPageUrl": "https://example.org/two",
                        "publisher": "Test Publisher",
                    }
                ]
            }
            return httpx.Response(200, json=body, request=request)

        return httpx.Response(200, json={"results": []}, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.core.example", transport=transport)
    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
    )

    records = connector.search("test query", k=2)
    assert len(records) == 2

    first = records[0]
    assert first["rank"] == 1
    assert first["title"] == "Paper One"
    assert first["doi"] == "10.1234/one"
    assert first["year"] == 2020
    assert first["authors"] == ["Alice", "Bob"]
    assert first["source"] == "Test Venue"
    assert first["url"] == "https://example.org/one"

    second = records[1]
    assert second["rank"] == 2
    assert second["doi"] == "10.9999/two"
    assert second["year"] == 2019
    assert second["authors"] == ["Carol"]
    assert second["source"] == "Test Publisher"
    assert second["url"] == "https://example.org/two"

    assert calls[0]["params"]["q"] == "test query"
    assert calls[0]["params"]["offset"] == "0"
    assert calls[1]["params"]["offset"] == "1"


def test_core_connector_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")
    monkeypatch.setenv("CORE_SEARCH_PATH", "/search/works")
    try:
        _ = CoreConnector(
            rate_limiter=RateLimiter(rate=1000, burst=1000),
            retries=RetryConfig(max=1, backoff=1.0),
            client=httpx.Client(base_url="https://api.core.example"),
        )
    except Exception as exc:
        assert "CORE_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected connector initialization to fail without CORE_API_KEY")
