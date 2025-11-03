"""Enrichment of records with OpenAlex metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from diskcache import Cache
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import best_identifier, normalise_doi
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.models import EnrichedRecord
from ai_bias_search.utils.rate_limit import RateLimiter


LOGGER = configure_logging()
CACHE_DIR = (Path(__file__).resolve().parents[2] / "data" / "cache" / "openalex").resolve()


def enrich_with_openalex(records: List[Dict[str, Any]], mailto: str | None) -> List[Dict[str, Any]]:
    """Augment *records* with OpenAlex metadata."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(rate=2, burst=5)
    retries = RetryConfig()
    retrying = Retrying(
        stop=stop_after_attempt(retries.max),
        wait=wait_exponential(multiplier=1, exp_base=retries.backoff, min=1),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )

    enriched: List[Dict[str, Any]] = []
    with Cache(CACHE_DIR) as cache:
        with httpx.Client(base_url="https://api.openalex.org", timeout=30.0) as client:
            for record in records:
                identifier = best_identifier(record)
                if not identifier:
                    enriched.append(record)
                    continue
                cache_key = identifier.lower()
                cached = cache.get(cache_key)
                if cached is None:
                    try:
                        metadata = _fetch_openalex_metadata(
                            identifier=identifier,
                            client=client,
                            limiter=limiter,
                            retrying=retrying,
                            mailto=mailto,
                        )
                    except httpx.HTTPError as exc:
                        LOGGER.warning("OpenAlex enrichment failed id=%s error=%s", identifier, exc)
                        metadata = None
                    cache.set(cache_key, metadata, expire=60 * 60 * 24 * 7)  # one week
                else:
                    metadata = cached

                if metadata:
                    merged = EnrichedRecord(**record)
                    merged.language = metadata.get("language")
                    merged.is_oa = metadata.get("is_oa")
                    merged.publication_year = metadata.get("publication_year")
                    host = metadata.get("host_venue") or {}
                    if isinstance(host, dict):
                        merged.host_venue = host.get("display_name")
                    merged.publisher = metadata.get("publisher")
                    merged.cited_by_count = metadata.get("cited_by_count")
                    merged.extra = {**record.get("extra", {}), "openalex_enrich": metadata}
                    enriched.append(merged.model_dump())
                else:
                    enriched.append(record)
    return enriched


def _fetch_openalex_metadata(
    *,
    identifier: str,
    client: httpx.Client,
    limiter: RateLimiter,
    retrying: Retrying,
    mailto: str | None,
) -> Optional[Dict[str, Any]]:
    """Retrieve OpenAlex metadata for *identifier*."""

    openalex_id = _resolve_openalex_id(identifier, client=client, limiter=limiter, retrying=retrying, mailto=mailto)
    if not openalex_id:
        return None

    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto

    def execute() -> Dict[str, Any]:
        limiter.acquire()
        response = client.get(f"/works/{openalex_id}", params=params)
        response.raise_for_status()
        return response.json()

    return retrying(execute)


def _resolve_openalex_id(
    identifier: str,
    *,
    client: httpx.Client,
    limiter: RateLimiter,
    retrying: Retrying,
    mailto: str | None,
) -> Optional[str]:
    """Resolve a DOI or URL to an OpenAlex work ID."""

    if identifier.startswith("https://openalex.org/"):
        return identifier.removeprefix("https://openalex.org/")
    doi = normalise_doi(identifier)
    params: Dict[str, Any]
    if doi:
        params = {"filter": f"doi:{doi}"}
    else:
        params = {"search": identifier}
    if mailto:
        params["mailto"] = mailto

    def execute() -> Dict[str, Any]:
        limiter.acquire()
        response = client.get("/works", params=params)
        response.raise_for_status()
        return response.json()

    payload = retrying(execute)
    results = payload.get("results") or []
    if not results:
        return None
    first = results[0]
    openalex_id = first.get("id")
    if isinstance(openalex_id, str) and openalex_id.startswith("https://openalex.org/"):
        return openalex_id.split("/")[-1]
    if isinstance(openalex_id, str) and openalex_id:
        return openalex_id
    return None
