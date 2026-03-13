from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from ai_bias_search.connectors.core import CoreConnector, join_url
from ai_bias_search.diagnostics.capture import (
    configure_request_capture,
    request_capture_snapshot,
    reset_request_capture,
)
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.rate_limit import RateLimiter


def _query_params(request: httpx.Request) -> dict[str, str]:
    return dict(request.url.params)


def test_core_connector_paginates_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    calls: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers.get("Authorization") == "Bearer test-key"
        assert request.headers.get("Accept") == "application/json"
        assert request.url.path == "/v3/search/works"
        assert request.content == b""
        params = _query_params(request)
        calls.append({"path": request.url.path, "params": params})
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 0))
        assert params.get("scroll") == "false"
        assert params.get("stats") == "false"

        if offset == 0:
            payload = {
                "total_hits": 2,
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
                ],
            }
            return httpx.Response(200, json=payload, request=request)

        if offset == 1 and limit >= 1:
            payload = {
                "total_hits": 2,
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
                ],
            }
            return httpx.Response(200, json=payload, request=request)

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


def test_core_connector_retries_on_es_overload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(
                200,
                json={
                    "total": 20,
                    "successful": 15,
                    "failed": 5,
                    "message": (
                        "es_rejected_execution_exception: rejected execution "
                        "on search thread pool"
                    ),
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "core:1", "title": "Recovered", "doi": "10.1/x", "year": 2020}
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.core.example", transport=transport)
    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=2, backoff=1.0),
        client=client,
    )

    records = connector.search("test query", k=1)
    assert attempts["count"] == 2
    assert len(records) == 1
    assert records[0]["title"] == "Recovered"


def test_core_connector_preserves_zero_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "core:zero",
                        "title": "Zero citations",
                        "doi": "10.1/zero",
                        "year": 2020,
                        "citationCount": 0,
                    }
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.core.example", transport=transport)
    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
    )

    records = connector.search("zero", k=1)
    assert len(records) == 1
    assert records[0]["citations"] == 0
    assert records[0]["cited_by_count"] == 0


def test_core_connector_uses_get_only_and_canonical_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example/v3/")

    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url).split("?", 1)[0]))
        params = _query_params(request)
        assert params["q"] == "test query"
        assert params["scroll"] == "false"
        assert params["stats"] == "false"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "core:1", "title": "OK", "doi": "10.1/x", "yearPublished": 2020}
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.core.example/v3/", transport=transport)
    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
    )

    records = connector.search("test query", k=1)
    assert len(records) == 1
    assert records[0]["title"] == "OK"
    assert calls == [("GET", "https://api.core.example/v3/search/works")]


def test_core_join_url_avoids_double_v3() -> None:
    final_url = join_url("https://api.core.ac.uk/v3", "/search/works")
    assert final_url == "https://api.core.ac.uk/v3/search/works"

    double_url = join_url("https://api.core.ac.uk/v3", "/v3/search/works")
    assert double_url == "https://api.core.ac.uk/v3/search/works"


def test_core_connector_uses_query_params_not_json_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v3/search/works"
        params = _query_params(request)
        assert params == {
            "q": "query params query",
            "limit": "1",
            "offset": "0",
            "scroll": "false",
            "stats": "false",
        }
        assert request.content == b""
        return httpx.Response(
            200,
            json={"results": [{"id": "core:1", "title": "One"}]},
            request=request,
        )

    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=httpx.Client(
            base_url="https://api.core.example",
            transport=httpx.MockTransport(handler),
        ),
    )
    records = connector.search("query params query", k=1)
    assert len(records) == 1


def test_core_connector_disables_on_404_and_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")
    monkeypatch.setenv("CORE_FAIL_OPEN", "true")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(404, json={"error": "No route found"}, request=request)

    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=httpx.Client(
            base_url="https://api.core.example",
            transport=httpx.MockTransport(handler),
        ),
    )

    first = connector.search("test query", k=5)
    second = connector.search("test query", k=5)
    health = connector.platform_health()

    assert first == []
    assert second == []
    assert calls["count"] == 1
    assert health["enabled"] is False
    assert "status=404" in str(health["reason"])
    assert health["error_rate"] == 1.0


def test_core_connector_returns_partial_results_when_second_page_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(_query_params(request).get("offset", 0))
        if offset == 0:
            return httpx.Response(
                200,
                json={
                    "total_hits": 10,
                    "results": [
                        {"id": "core:1", "title": "One", "doi": "10.1/one", "year": 2020},
                        {"id": "core:2", "title": "Two", "doi": "10.1/two", "year": 2020},
                    ],
                },
                request=request,
            )
        return httpx.Response(500, json={"error": "boom"}, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.core.example", transport=transport)
    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
    )

    caplog.set_level("WARNING")
    records = connector.search("test query", k=5)
    assert [record["title"] for record in records] == ["One", "Two"]
    assert any("transient backend error" in record.message for record in caplog.records)


def test_core_connector_stops_on_short_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(_query_params(request).get("offset", 0))
        calls.append(offset)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "core:1", "title": "One", "doi": "10.1/one", "year": 2020},
                    {"id": "core:2", "title": "Two", "doi": "10.1/two", "year": 2020},
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.core.example", transport=transport)
    connector = CoreConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
    )

    records = connector.search("test query", k=5)
    assert [record["title"] for record in records] == ["One", "Two"]
    assert calls == [0]


def test_core_connector_retries_500_and_captures_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setenv("CORE_API_BASE_URL", "https://api.core.example")

    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(500, json={"error": "transient"}, request=request)
        return httpx.Response(
            200,
            json={"results": [{"id": "core:1", "title": "Recovered", "doi": "10.1/r"}]},
            request=request,
        )

    configure_request_capture(enabled=True, max_logs=10, redact_fields=[])
    reset_request_capture()
    try:
        transport = httpx.MockTransport(handler)
        client = httpx.Client(base_url="https://api.core.example", transport=transport)
        connector = CoreConnector(
            rate_limiter=RateLimiter(rate=1000, burst=1000),
            retries=RetryConfig(max=2, backoff=1.0),
            client=client,
        )
        records = connector.search("test query", k=1)
    finally:
        logs = request_capture_snapshot()
        configure_request_capture(enabled=False, max_logs=0, redact_fields=[])
        reset_request_capture()

    assert len(records) == 1
    assert attempts["count"] == 2
    assert "core" in logs
    entries = logs["core"]
    assert len(entries) >= 2
    assert entries[0]["method"] == "GET"
    assert entries[0]["endpoint"] == "https://api.core.example/v3/search/works"
    assert entries[0]["status_code"] == 500
    assert entries[1]["method"] == "GET"
    assert entries[1]["endpoint"] == "https://api.core.example/v3/search/works"
    assert entries[1]["status_code"] == 200
