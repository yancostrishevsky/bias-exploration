from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_bias_search.enrichment import scopus as scopus_impl
from ai_bias_search.utils.config import RetryConfig, ScopusEnrichConfig
from ai_bias_search.utils.rate_limit import RateLimiter


def _abstract_response(*, scopus_id: str = "85012345678", issn: str = "12345679") -> dict[str, Any]:
    return {
        "abstracts-retrieval-response": {
            "coredata": {
                "dc:identifier": f"SCOPUS_ID:{scopus_id}",
                "eid": f"2-s2.0-{scopus_id}",
                "prism:issn": issn,
                "prism:eIssn": "87654321",
                "prism:publicationName": "Journal of Testing",
                "prism:coverDate": "2021-05-01",
                "subtypeDescription": "Article",
                "citedby-count": "7",
                "dc:description": "An abstract",
            },
            "authors": {"author": [{"ce:indexed-name": "Doe, Jane"}]},
        }
    }


def _transport(routes: list[dict[str, object]]) -> tuple[httpx.MockTransport, dict[str, int]]:
    counts: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        for route in routes:
            method = route["method"]
            path = route["path"]
            key = route["key"]
            params_contains = route.get("params_contains")

            if request.method != method:
                continue
            if request.url.path != path:
                continue
            if isinstance(params_contains, dict):
                for expected_key, expected_value in params_contains.items():
                    if request.url.params.get(str(expected_key)) != str(expected_value):
                        break
                else:
                    counts[str(key)] = counts.get(str(key), 0) + 1
                    return httpx.Response(
                        int(route.get("status_code", 200)), json=route.get("json")
                    )
                continue

            counts[str(key)] = counts.get(str(key), 0) + 1
            return httpx.Response(int(route.get("status_code", 200)), json=route.get("json"))

        return httpx.Response(500, json={"error": "unmocked request", "url": str(request.url)})

    return httpx.MockTransport(handler), counts


def test_scopus_abstract_retrieval_parses_citedby_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    doi = "10.1234/example"
    doi_path = f"/content/abstract/doi/{doi}"

    transport, counts = _transport(
        [
            {
                "key": "abstract",
                "method": "GET",
                "path": doi_path,
                "params_contains": {"view": "FULL"},
                "json": _abstract_response(),
            }
        ]
    )

    client = httpx.Client(base_url=scopus_impl.BASE_URL, timeout=30.0, transport=transport)
    enricher = scopus_impl.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, abstract_view="FULL"),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record: dict[str, Any] = {"title": "A paper", "doi": doi, "rank": 1, "extra": {}}
    out = enricher.enrich([record])[0]
    assert counts.get("abstract") == 1
    assert out["cited_by_count"] == 7
    assert out["scopus"]["abstract"]["citedby_count"] == 7
    assert out["scopus_meta"]["method"] == "doi"
    client.close()


def test_scopus_serial_title_metrics_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    doi = "10.1234/example"
    abstract_path = f"/content/abstract/doi/{doi}"
    serial_path = "/content/serial/title/issn/1234-5679"
    transport, counts = _transport(
        [
            {
                "key": "abstract",
                "method": "GET",
                "path": abstract_path,
                "params_contains": {"view": "FULL"},
                "json": _abstract_response(),
            },
            {
                "key": "serial",
                "method": "GET",
                "path": serial_path,
                "params_contains": {"view": "STANDARD"},
                "json": {
                    "serial-metadata-response": {
                        "entry": [
                            {
                                "SNIPList": {"SNIP": [{"@year": "2022", "$": "0.924"}]},
                                "SJRList": {"SJR": [{"@year": "2022", "$": "0.570"}]},
                                "citeScoreYearInfoList": {
                                    "citeScoreYearInfo": [
                                        {
                                            "citeScoreCurrentMetric": "1.3",
                                            "citeScoreCurrentMetricYear": "2022",
                                            "citeScoreTracker": "1.4",
                                            "citeScoreTrackerYear": "2023",
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                },
            },
        ]
    )

    client = httpx.Client(base_url=scopus_impl.BASE_URL, timeout=30.0, transport=transport)
    enricher = scopus_impl.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, enable_serial_title_metrics=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record: dict[str, Any] = {"title": "A paper", "doi": doi, "rank": 1, "extra": {}}
    out = enricher.enrich([record])[0]
    assert counts.get("serial") == 1
    assert out["scopus"]["serial_metrics"]["snip"]["year"] == 2022
    assert out["scopus"]["serial_metrics"]["snip"]["value"] == 0.924
    assert out["scopus"]["serial_metrics"]["sjr"]["year"] == 2022
    assert out["scopus"]["serial_metrics"]["citescore"]["year"] == 2023
    client.close()


def test_scopus_citation_overview_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    doi = "10.1234/example"
    scopus_id = "85012345678"

    abstract_path = f"/content/abstract/doi/{doi}"
    citations_path = "/content/abstract/citations"
    transport, counts = _transport(
        [
            {
                "key": "abstract",
                "method": "GET",
                "path": abstract_path,
                "params_contains": {"view": "FULL"},
                "json": _abstract_response(scopus_id=scopus_id),
            },
            {
                "key": "citations",
                "method": "GET",
                "path": citations_path,
                "params_contains": {"scopus_id": scopus_id, "citation": "exclude-self"},
                "json": {
                    "abstract-citations-response": {
                        "citeColumnTotalXML": {
                            "columnHeading": ["2019", "2020"],
                            "columnTotal": ["1", "2"],
                            "rangeColumnTotal": "3",
                            "grandTotal": "3",
                        }
                    }
                },
            },
        ]
    )

    client = httpx.Client(base_url=scopus_impl.BASE_URL, timeout=30.0, transport=transport)
    enricher = scopus_impl.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, enable_citation_overview=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record: dict[str, Any] = {"title": "A paper", "doi": doi, "rank": 1, "extra": {}}
    out = enricher.enrich([record])[0]
    assert counts.get("citations") == 1
    assert out["scopus"]["citation_overview"]["exclude_self"] is True
    assert out["scopus"]["citation_overview"]["by_year"] == {2019: 1, 2020: 2}
    assert out["scopus"]["citation_overview"]["grand_total"] == 3
    client.close()


def test_scopus_plumx_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    doi = "10.1234/example"
    abstract_path = f"/content/abstract/doi/{doi}"
    plumx_path = f"/analytics/plumx/doi/{doi}"
    transport, counts = _transport(
        [
            {
                "key": "abstract",
                "method": "GET",
                "path": abstract_path,
                "params_contains": {"view": "FULL"},
                "json": _abstract_response(),
            },
            {
                "key": "plumx",
                "method": "GET",
                "path": plumx_path,
                "json": {
                    "id_type": "doi",
                    "id_value": doi,
                    "count_categories": [
                        {"name": "Usage", "total": 17},
                        {"name": "Captures", "total": 2},
                    ],
                },
            },
        ]
    )

    client = httpx.Client(base_url=scopus_impl.BASE_URL, timeout=30.0, transport=transport)
    enricher = scopus_impl.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, enable_plumx=True),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    record: dict[str, Any] = {"title": "A paper", "doi": doi, "rank": 1, "extra": {}}
    out = enricher.enrich([record])[0]
    assert counts.get("plumx") == 1
    assert out["scopus"]["plumx"]["categories"]["Usage"] == 17
    assert out["scopus"]["plumx"]["categories"]["Captures"] == 2
    client.close()


def test_scopus_abstract_maps_bias_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scopus_impl, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")

    doi = "10.1234/example"
    abstract_path = f"/content/abstract/doi/{doi}"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == abstract_path:
            return httpx.Response(
                200,
                json={
                    "abstracts-retrieval-response": {
                        "coredata": {
                            "dc:identifier": "SCOPUS_ID:85012345678",
                            "prism:publicationName": "Journal of Bias Testing",
                            "prism:issn": "1234-5678;8765 4321",
                            "prism:coverDate": "2020-04-11",
                            "subtype": "ar",
                            "subtypeDescription": "Article",
                            "citedby-count": "12",
                            "openaccessFlag": "1",
                            "source-id": "998877",
                        },
                        "authors": {
                            "author": [
                                {
                                    "@auid": "111",
                                    "ce:indexed-name": "Doe, Jane",
                                    "affiliation": {"@id": "1"},
                                },
                                {
                                    "@auid": "222",
                                    "ce:indexed-name": "Smith, John",
                                    "affiliation": {"affiliation-country": "Germany", "affilname": "TU Berlin"},
                                },
                            ]
                        },
                        "affiliation": [
                            {
                                "@id": "1",
                                "affiliation-country": "Poland",
                                "affiliation-city": "Warsaw",
                                "affilname": "UW",
                            }
                        ],
                    }
                },
            )
        return httpx.Response(404, json={"error": "unmocked request", "url": str(request.url)})

    client = httpx.Client(base_url=scopus_impl.BASE_URL, timeout=30.0, transport=httpx.MockTransport(handler))
    enricher = scopus_impl.ScopusEnricher(
        ScopusEnrichConfig(enabled=True, fail_open=False),
        retries=RetryConfig(max=1, backoff=1.0),
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        client=client,
        sleep=lambda _: None,
    )

    out = enricher.enrich([{"title": "A paper", "doi": doi, "rank": 1, "extra": {}}])[0]
    assert out["is_open_access"] is True
    assert out["is_oa"] is True
    assert out["doc_type"] == "Article"
    assert out["journal_title"] == "Journal of Bias Testing"
    assert out["source_id"] == "998877"
    assert out["issn_list"] == ["12345678", "87654321"]
    assert out["affiliation_countries"] == ["Poland", "Germany"]
    assert out["affiliation_institutions"] == ["UW", "TU Berlin"]
    assert out["affiliation_cities"] == ["Warsaw"]
    assert out["author_ids"] == ["111", "222"]
    assert out["author_count"] == 2

    client.close()
