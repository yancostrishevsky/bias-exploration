from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_bias_search.enrichment import scopus_rankings as rankings_impl
from ai_bias_search.utils.config import RetryConfig, ScopusRankingConfig
from ai_bias_search.utils.rate_limit import RateLimiter

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_scopus_rankings_happy_path_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rankings_impl, "CACHE_DIR", tmp_path / "cache")

    enhanced_payload = _fixture("scopus_serial_title_enhanced.json")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        assert request.url.params.get("view") == "ENHANCED"
        return httpx.Response(200, json=enhanced_payload)

    client = httpx.Client(
        base_url=rankings_impl.BASE_URL,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    enricher = rankings_impl.ScopusRankingsEnricher(
        ScopusRankingConfig(enabled=True, api_key="test-key"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record = {
        "title": "Paper",
        "rank": 1,
        "issn": "1234-5678",
        "rankings": {"core": {"value": "A"}},
    }
    out = enricher.enrich([record])[0]
    scopus = out["rankings"]["scopus"]
    assert scopus["citescore"] == {"value": 4.2, "year": 2023}
    assert scopus["citescore_tracker"] == {"value": 4.8, "year": 2024}
    assert scopus["sjr"] == {"value": 0.72, "year": 2023}
    assert scopus["snip"] == {"value": 1.24, "year": 2023}
    assert scopus["source_issn_used"] == "1234-5678"
    assert isinstance(scopus["retrieved_at"], str)
    assert any(item.get("kind") == "current" for item in scopus["series"]["citescore"])
    assert any(item.get("kind") == "tracker" for item in scopus["series"]["citescore"])
    assert out["rankings"]["core"]["value"] == "A"

    # Second run should come from disk cache and avoid additional calls.
    out_cached = enricher.enrich([record])[0]
    assert calls["count"] == 1
    assert out_cached["rankings"]["scopus"]["sjr"]["year"] == 2023
    client.close()


def test_scopus_rankings_falls_back_to_standard_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rankings_impl, "CACHE_DIR", tmp_path / "cache")

    standard_payload = _fixture("scopus_serial_title_standard_only.json")
    seen_views: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        view = str(request.url.params.get("view"))
        seen_views.append(view)
        if view == "ENHANCED":
            return httpx.Response(403, json={"error": "not entitled"})
        return httpx.Response(200, json=standard_payload)

    client = httpx.Client(
        base_url=rankings_impl.BASE_URL,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    out = rankings_impl.ScopusRankingsEnricher(
        ScopusRankingConfig(enabled=True, api_key="test-key"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    ).enrich([{"title": "Paper", "rank": 1, "issn": "1234-5678"}])[0]
    client.close()

    assert seen_views == ["ENHANCED", "STANDARD"]
    scopus = out["rankings"]["scopus"]
    assert scopus["citescore"] == {"value": 2.2, "year": 2022}
    assert scopus["citescore_tracker"] == {"value": None, "year": None}
    assert scopus["sjr"] == {"value": 0.41, "year": 2022}
    assert scopus["snip"] == {"value": 0.88, "year": 2022}


def test_scopus_rankings_missing_metrics_fields_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rankings_impl, "CACHE_DIR", tmp_path / "cache")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"serial-metadata-response": {"entry": [{"dc:title": "X"}]}},
        )

    client = httpx.Client(
        base_url=rankings_impl.BASE_URL,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    out = rankings_impl.ScopusRankingsEnricher(
        ScopusRankingConfig(enabled=True, api_key="test-key"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    ).enrich([{"title": "Paper", "rank": 1, "issn": "1234-5678"}])[0]
    client.close()

    scopus = out["rankings"]["scopus"]
    assert scopus["citescore"]["value"] is None
    assert scopus["sjr"]["value"] is None
    assert scopus["snip"]["value"] is None
    assert scopus["series"] == {"citescore": [], "sjr": [], "snip": []}
    assert scopus["source_issn_used"] is None


def test_scopus_rankings_issn_not_found_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rankings_impl, "CACHE_DIR", tmp_path / "cache")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = httpx.Client(
        base_url=rankings_impl.BASE_URL,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    out = rankings_impl.ScopusRankingsEnricher(
        ScopusRankingConfig(enabled=True, api_key="test-key", fail_open=False),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    ).enrich([{"title": "Paper", "rank": 1, "issn": "1234-5678"}])[0]
    client.close()

    assert out["rankings"]["scopus"]["citescore"]["value"] is None
    assert out["rankings"]["scopus"]["source_issn_used"] is None


def test_scopus_rankings_multiple_issns_first_fails_second_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rankings_impl, "CACHE_DIR", tmp_path / "cache")
    enhanced_payload = _fixture("scopus_serial_title_enhanced.json")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path.endswith("/0000-0000"):
            return httpx.Response(404, json={"error": "missing"})
        return httpx.Response(200, json=enhanced_payload)

    client = httpx.Client(
        base_url=rankings_impl.BASE_URL,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    out = rankings_impl.ScopusRankingsEnricher(
        ScopusRankingConfig(enabled=True, api_key="test-key"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    ).enrich(
        [{"title": "Paper", "rank": 1, "issn": "0000-0000", "eissn": "1234-5678"}]
    )[0]
    client.close()

    assert any(path.endswith("/0000-0000") for path in requested)
    assert any(path.endswith("/1234-5678") for path in requested)
    assert out["rankings"]["scopus"]["source_issn_used"] == "1234-5678"


def test_scopus_rankings_parses_object_and_list_variations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rankings_impl, "CACHE_DIR", tmp_path / "cache")
    payload = _fixture("scopus_serial_title_object_variation.json")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(
        base_url=rankings_impl.BASE_URL,
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )
    out = rankings_impl.ScopusRankingsEnricher(
        ScopusRankingConfig(enabled=True, api_key="test-key"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    ).enrich([{"title": "Paper", "rank": 1, "issn": "1234-5678"}])[0]
    client.close()

    scopus = out["rankings"]["scopus"]
    assert scopus["citescore"] == {"value": 1.1, "year": 2021}
    assert scopus["citescore_tracker"] == {"value": 1.3, "year": 2022}
    assert scopus["sjr"] == {"value": 0.22, "year": 2021}
    assert scopus["snip"] == {"value": 0.73, "year": 2021}
