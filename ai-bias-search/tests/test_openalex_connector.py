from __future__ import annotations

import httpx

from ai_bias_search.connectors.openalex import OpenAlexConnector
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.rate_limit import RateLimiter


def test_openalex_connector_maps_publisher_with_fallbacks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "Primary publisher",
                        "doi": "10.1000/one",
                        "id": "https://openalex.org/W1",
                        "primary_location": {
                            "source": {
                                "publisher": "Primary Publisher",
                                "display_name": "Journal A",
                            }
                        },
                        "host_venue": {"publisher": "Host Publisher"},
                        "publication_year": 2023,
                        "authorships": [],
                    },
                    {
                        "display_name": "Host fallback",
                        "doi": "10.1000/two",
                        "id": "https://openalex.org/W2",
                        "primary_location": {"source": {"display_name": "Journal B"}},
                        "host_venue": {"publisher": "Host Publisher"},
                        "publication_year": 2022,
                        "authorships": [],
                    },
                    {
                        "display_name": "Location fallback",
                        "doi": "10.1000/three",
                        "id": "https://openalex.org/W3",
                        "primary_location": {"source": {"display_name": "Journal C"}},
                        "host_venue": {},
                        "locations": [{"source": {"publisher": "Location Publisher"}}],
                        "publication_year": 2021,
                        "authorships": [],
                    },
                ]
            },
        )

    client = httpx.Client(
        base_url="https://api.openalex.org",
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )
    connector = OpenAlexConnector(
        rate_limiter=RateLimiter(rate=1000, burst=1000),
        retries=RetryConfig(max=1, backoff=1.0),
        client=client,
    )

    records = connector.search("llm", k=3)
    assert records[0]["publisher"] == "Primary Publisher"
    assert records[1]["publisher"] == "Host Publisher"
    assert records[2]["publisher"] == "Location Publisher"

    client.close()
