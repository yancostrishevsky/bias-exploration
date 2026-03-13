from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_bias_search.enrichment import scopus as scopus_impl
from ai_bias_search.normalization import scopus_enrich
from ai_bias_search.utils.config import RetryConfig, ScopusEnrichConfig
from ai_bias_search.utils.rate_limit import RateLimiter

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "scopus"
FIXTURES = Path(__file__).parent / "fixtures"


def _snapshot(name: str) -> dict[str, Any]:
    return json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_scopus_enrich_doi_fallback_search_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(404, json={"error": "not found"})
        if request.url.path == "/content/search/scopus":
            return httpx.Response(
                200,
                json={
                    "search-results": {
                        "entry": [
                            {
                                "dc:identifier": "SCOPUS_ID:85012345678",
                            }
                        ]
                    }
                },
            )
        if request.url.path == "/content/abstract/scopus_id/85012345678":
            return httpx.Response(
                200,
                json={
                    "abstracts-retrieval-response": {
                        "coredata": {
                            "dc:identifier": "SCOPUS_ID:85012345678",
                            "prism:issn": "12345678",
                            "prism:eIssn": "87654321",
                            "prism:coverDate": "2021-05-01",
                            "subtypeDescription": "Article",
                            "citedby-count": "7",
                            "dc:description": "An abstract",
                        },
                        "authors": {"author": [{"ce:indexed-name": "Doe, Jane"}]},
                        "authkeywords": {
                            "author-keyword": [{"$": "keyword1"}, {"$": "keyword2"}]
                        },
                        "affiliation": [{"affiliation-country": "Poland"}],
                    }
                },
            )
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url=scopus_enrich.BASE_URL, timeout=30.0)

    cfg = ScopusEnrichConfig(enabled=True)
    enricher = scopus_enrich.ScopusEnricher(
        cfg,
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    records: list[dict[str, Any]] = [
        {
            "title": "A paper",
            "doi": "10.1234/example",
            "url": None,
            "rank": 1,
            "raw_id": None,
            "source": None,
            "year": None,
            "authors": None,
            "publication_year": None,
            "cited_by_count": None,
            "extra": {},
        }
    ]

    enriched = enricher.enrich(records)
    assert calls["count"] == 3
    assert enriched[0]["scopus_id"] == "85012345678"
    assert enriched[0]["issn"] == "1234-5678"
    assert enriched[0]["eissn"] == "8765-4321"
    assert enriched[0]["publication_year"] == 2021
    assert enriched[0]["cited_by_count"] == 7
    assert enriched[0]["authors"] == ["Doe, Jane"]
    assert enriched[0]["extra"]["scopus_enrich"]["subtype"] == "Article"

    # Second run should hit DiskCache and avoid network calls.
    enriched_again = enricher.enrich(records)
    assert calls["count"] == 3
    assert enriched_again[0]["scopus_id"] == "85012345678"

    client.close()


def test_scopus_enrich_does_not_overwrite_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(
                200,
                json={
                    "abstracts-retrieval-response": {
                        "coredata": {
                            "dc:identifier": "SCOPUS_ID:850999",
                            "prism:coverDate": "2021-01-01",
                        }
                    }
                },
            )
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )

    record: dict[str, Any] = {
        "title": "A paper",
        "doi": "10.1234/example",
        "url": None,
        "rank": 1,
        "raw_id": None,
        "source": None,
        "year": None,
        "authors": None,
        "publication_year": 2020,
        "cited_by_count": None,
        "extra": {},
    }

    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, overwrite_existing=False),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    out = enricher.enrich([record])[0]
    assert out["publication_year"] == 2020

    enricher_overwrite = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, overwrite_existing=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    out_overwrite = enricher_overwrite.enrich([record])[0]
    assert out_overwrite["publication_year"] == 2021

    client.close()


def test_scopus_enrich_partial_failures_return_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/10.1111/bad"):
            return httpx.Response(500, json={"error": "server error"})
        if request.url.path.startswith("/content/abstract/doi/10.1234/good"):
            return httpx.Response(
                200,
                json={
                    "abstracts-retrieval-response": {
                        "coredata": {
                            "dc:identifier": "SCOPUS_ID:850777",
                            "prism:coverDate": "2019-01-01",
                        }
                    }
                },
            )
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )

    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    records: list[dict[str, Any]] = [
        {"title": "Bad", "doi": "10.1111/bad", "rank": 1, "extra": {}},
        {"title": "Good", "doi": "10.1234/good", "rank": 1, "extra": {}},
    ]

    out = enricher.enrich(records)
    assert len(out) == 2
    assert out[0].get("publication_year") is None
    assert out[1]["publication_year"] == 2019

    client.close()


def test_scopus_enrich_caches_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(404, json={"error": "not found"})
        if request.url.path == "/content/search/scopus":
            return httpx.Response(200, json={"search-results": {"entry": []}})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )

    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    records: list[dict[str, Any]] = [{"title": "A paper", "doi": "10.1234/example", "rank": 1}]
    out = enricher.enrich(records)
    assert len(out) == 1
    assert calls["count"] == 2

    out_again = enricher.enrich(records)
    assert len(out_again) == 1
    assert calls["count"] == 2

    client.close()


def test_scopus_enrich_builds_headers_and_abstract_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.setenv("SCOPUS_INSTTOKEN", "inst-token")

    abstract_payload = _snapshot("abstract_full.json")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["params"] = dict(request.url.params)
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(200, json=abstract_payload)
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )

    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(
            enabled=True,
            abstract_view="FULL",
            abstract_fields="dc:identifier,citedby-count",
            fail_open=False,
        ),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record: dict[str, Any] = {"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}
    out = enricher.enrich([record])[0]
    assert out["scopus_id"] == "85012345678"
    assert captured["headers"]["x-els-apikey"] == "test-key"
    assert captured["headers"]["x-els-insttoken"] == "inst-token"
    assert captured["params"]["view"] == "FULL"
    assert captured["params"]["field"] == "dc:identifier,citedby-count"

    client.close()


def test_scopus_enrich_401_fail_open_true_returns_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTTOKEN", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(401, json={"service-error": {"status": {"statusCode": "401"}}})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, fail_open=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    out = enricher.enrich([{"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}])
    assert len(out) == 1
    assert out[0].get("scopus_id") is None
    assert out[0]["scopus_meta"]["status"] == "auth_error"
    assert out[0]["scopus_meta"]["status_code"] == 401
    assert isinstance(out[0]["scopus_meta"]["attempts"], list)

    client.close()


def test_scopus_enrich_abstract_fallback_full_401_meta_200(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    caplog.set_level("INFO")

    abstract_payload = _snapshot("abstract_full.json")
    attempted_views: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            view = request.url.params.get("view")
            attempted_views.append(view)
            if view == "FULL":
                return httpx.Response(401, json={"service-error": {"status": {"statusCode": "401"}}})
            if view == "META":
                return httpx.Response(200, json=abstract_payload)
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(
            enabled=True,
            fail_open=False,
            abstract_view="FULL",
            abstract_fallback_views=["META", None],
        ),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    out = enricher.enrich([{"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}])[0]
    assert attempted_views == ["FULL", "META"]
    assert out["scopus_id"] == "85012345678"
    assert out["scopus_enrich_view_used"] == "META"
    assert out["scopus_enrich_downgraded"] is True
    assert out["scopus_meta"]["abstract_downgraded"] is True
    assert any(
        "scopus: FULL view not authorized, falling back to META" in record.message
        for record in caplog.records
    )

    client.close()


def test_scopus_enrich_abstract_fallback_full_meta_401_then_field_minimal_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    abstract_payload = _snapshot("abstract_full.json")
    attempts: list[tuple[str | None, str | None]] = []
    minimal = "dc:title,prism:doi,citedby-count"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            view = request.url.params.get("view")
            field = request.url.params.get("field")
            attempts.append((view, field))
            if field == minimal:
                return httpx.Response(200, json=abstract_payload)
            return httpx.Response(401, json={"service-error": {"status": {"statusCode": "401"}}})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(
            enabled=True,
            fail_open=False,
            abstract_view="FULL",
            abstract_fallback_views=["META"],
            abstract_fields_minimal=minimal,
        ),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    out = enricher.enrich([{"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}])[0]
    assert attempts[:4] == [("FULL", None), ("META", None), (None, None), ("FULL", minimal)]
    assert out["scopus_enrich_field_used"] == minimal
    assert out["scopus_enrich_downgraded"] is True
    assert out["scopus_meta"]["abstract_field_used"] == minimal

    client.close()


def test_scopus_enrich_abstract_fallback_full_then_meta_then_no_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    abstract_payload = _snapshot("abstract_full.json")
    attempts: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            view = request.url.params.get("view")
            attempts.append(view)
            if view in {"FULL", "META"}:
                return httpx.Response(401, json={"service-error": {"status": {"statusCode": "401"}}})
            if view is None:
                return httpx.Response(200, json=abstract_payload)
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(
            enabled=True,
            fail_open=False,
            abstract_view="FULL",
            abstract_fallback_views=[],
            abstract_fields_minimal=None,
        ),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    out = enricher.enrich([{"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}])[0]
    assert attempts == ["FULL", "META", None]
    assert out["scopus_id"] == "85012345678"
    assert "scopus_enrich_view_used" not in out
    assert out["scopus_meta"]["abstract_view_used"] is None
    assert out["scopus_enrich_downgraded"] is True

    client.close()


def test_scopus_enrich_401_fail_open_false_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(401, json={"service-error": {"status": {"statusCode": "401"}}})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, fail_open=False),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    record = {"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}
    with pytest.raises(scopus_impl.ScopusAuthError):
        _ = enricher.enrich([record])

    client.close()


def test_scopus_enrich_403_fail_open_false_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(403, json={"service-error": {"status": {"statusCode": "403"}}})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, fail_open=False),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    record = {"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}
    with pytest.raises(scopus_impl.ScopusPermissionError):
        _ = enricher.enrich([record])

    client.close()


def test_scopus_enrich_disables_after_auth_error_when_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            calls["count"] += 1
            return httpx.Response(401, json={"service-error": {"status": {"statusCode": "401"}}})
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(
            enabled=True,
            fail_open=True,
            abstract_fallback_views=[],
            abstract_fields_minimal=None,
        ),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    records = [
        {"title": "A", "doi": "10.1234/a", "rank": 1, "extra": {}},
        {"title": "B", "doi": "10.1234/b", "rank": 2, "extra": {}},
    ]
    out = enricher.enrich(records)
    assert len(out) == 2
    assert calls["count"] == 3

    client.close()


def test_scopus_enrich_cache_key_includes_abstract_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    calls = {"count": 0}
    abstract_payload = _snapshot("abstract_full.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            calls["count"] += 1
            return httpx.Response(200, json=abstract_payload)
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )

    record = {"title": "A paper", "doi": "10.1234/example", "rank": 1, "extra": {}}
    enricher_a = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, abstract_fields="dc:identifier"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    _ = enricher_a.enrich([record])
    assert calls["count"] == 1

    enricher_b = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, abstract_fields="dc:identifier,citedby-count"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )
    _ = enricher_b.enrich([record])
    assert calls["count"] == 2

    client.close()


def test_clean_title_for_scopus_preserves_normal_title() -> None:
    title = "Bridging the gap - how the gap is changing in clinical research"
    cleaned = scopus_impl.clean_title_for_scopus(title)
    assert cleaned == title
    assert "  " not in cleaned


def test_scopus_enrich_skips_char_spaced_title_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    calls = {"search": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/content/search/scopus":
            calls["search"] += 1
        return httpx.Response(404, json={"error": "unexpected"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, title_search_enabled=True, title_search_min_len=10),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record = {
        "title": _fixture("scopus_mangled_title.json")["title"],
        "rank": 1,
        "extra": {},
    }
    out = enricher.enrich([record])[0]
    assert calls["search"] == 0
    assert "title_search_disabled_char_spaced" in (out.get("scopus_meta") or {}).get("notes", [])

    client.close()


def test_scopus_enrich_title_500_is_disabled_after_one_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    calls = {"search": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/content/search/scopus":
            calls["search"] += 1
            return httpx.Response(
                500,
                json={"service-error": {"status": {"statusText": "Error transforming XML with XSL"}}},
            )
        return httpx.Response(404, json={"error": "unexpected"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, title_search_enabled=True, title_search_min_len=10),
        retries=RetryConfig(max=3, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    out = enricher.enrich([{"title": "Bridging the gap in clinical work", "rank": 1, "extra": {}}])[0]
    assert calls["search"] == 2
    assert "disabled_title_after_500" in (out.get("scopus_meta") or {}).get("notes", [])
    trace = (out.get("scopus_meta") or {}).get("enrich_trace") or []
    assert any(item.get("note") == "disabled_title_after_500" for item in trace)

    client.close()


def test_scopus_enrich_uses_title_after_doi_search_zero_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    scopus_search_payload = _fixture("scopus_search_payload_with_fields.json")

    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/content/abstract/doi/"):
            return httpx.Response(404, json={"error": "doi abstract missing"})
        if request.url.path == "/content/search/scopus":
            query = request.url.params.get("query", "")
            queries.append(query)
            if query.startswith("DOI("):
                return httpx.Response(200, json={"search-results": {"opensearch:totalResults": "0", "entry": []}})
            if query.startswith("TITLE("):
                return httpx.Response(
                    200,
                    json=scopus_search_payload,
                )
        if request.url.path == "/content/abstract/scopus_id/85012345678":
            return httpx.Response(200, json=_snapshot("abstract_full.json"))
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=scopus_enrich.BASE_URL,
        timeout=30.0,
    )
    enricher = scopus_enrich.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, title_search_enabled=True, title_search_min_len=10),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    out = enricher.enrich(
        [{"title": "Bridging the gap in clinical work", "doi": "10.1234/example", "rank": 1, "extra": {}}]
    )[0]
    assert out["scopus_id"] == "85012345678"
    assert queries[0].startswith("DOI(")
    assert queries[1].startswith("TITLE(")
    trace = (out.get("scopus_meta") or {}).get("enrich_trace") or []
    assert any(item.get("strategy") == "doi_search" for item in trace)
    assert any(item.get("strategy") == "title_search" for item in trace)

    client.close()
