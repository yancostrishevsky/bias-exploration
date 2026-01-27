from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import httpx
import pytest

from ai_bias_search.normalization import openalex_enrich
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.rate_limit import RateLimiter


@dataclass
class FakeOpenAlexClient:
    handler: Any
    calls: List[Tuple[str, Dict[str, Any]]]

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls = []

    def get(self, path: str, *, params: Dict[str, Any] | None = None) -> httpx.Response:
        params = params or {}
        self.calls.append((path, dict(params)))
        return self.handler(path, params)


def _response(status_code: int, path: str, *, json_body: Any = None) -> httpx.Response:
    request = httpx.Request("GET", f"https://api.openalex.org{path}")
    if json_body is None:
        return httpx.Response(status_code, request=request)
    return httpx.Response(status_code, request=request, json=json_body)


def test_direct_doi_404_is_not_retried() -> None:
    def handler(path: str, params: Dict[str, Any]) -> httpx.Response:
        if path.startswith("/works/doi:"):
            return _response(404, path, json_body={"error": "not found"})
        if path == "/works":
            return _response(200, path, json_body={"results": []})
        return _response(500, path, json_body={"error": "unexpected"})

    client = FakeOpenAlexClient(handler)
    limiter = RateLimiter(rate=1000, burst=1000)
    retrying = openalex_enrich._openalex_retrying(
        RetryConfig(max=3, backoff=1.5), sleep=lambda _: None
    )

    resolved = openalex_enrich._resolve_openalex_id(
        "10.1234/example",
        client=client,  # type: ignore[arg-type]
        limiter=limiter,
        retrying=retrying,
        mailto=None,
    )

    assert resolved is None
    doi_calls = [path for path, _ in client.calls if path.startswith("/works/doi:")]
    assert len(doi_calls) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_retryable_statuses_are_retried(status: int) -> None:
    attempts = {"count": 0}

    def handler(path: str, params: Dict[str, Any]) -> httpx.Response:
        if path.startswith("/works/doi:"):
            attempts["count"] += 1
            if attempts["count"] < 3:
                return _response(status, path, json_body={"error": "retry me"})
            return _response(200, path, json_body={"id": "https://openalex.org/W123"})
        return _response(500, path, json_body={"error": "unexpected"})

    client = FakeOpenAlexClient(handler)
    limiter = RateLimiter(rate=1000, burst=1000)
    retrying = openalex_enrich._openalex_retrying(
        RetryConfig(max=3, backoff=1.5), sleep=lambda _: None
    )

    resolved = openalex_enrich._resolve_openalex_id(
        "10.1234/example",
        client=client,  # type: ignore[arg-type]
        limiter=limiter,
        retrying=retrying,
        mailto=None,
    )

    assert resolved == "W123"
    doi_calls = [path for path, _ in client.calls if path.startswith("/works/doi:")]
    assert len(doi_calls) == 3
