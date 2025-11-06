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
                identifier = best_identifier(record)  # powinien preferować DOI
                if not identifier:
                    LOGGER.debug("OpenAlex: no identifier for record (title=%r)", record.get("title"))
                    enriched.append(record)
                    continue

                cache_key = identifier.lower()
                metadata = cache.get(cache_key)
                if metadata is None:
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
                    # cache zarówno hit jak i miss (None), żeby nie mielić w kółko
                    cache.set(cache_key, metadata, expire=60 * 60 * 24 * 7)

                if not metadata:
                    enriched.append(record)
                    continue

                merged = EnrichedRecord(**record)
                # proste mapowanie pól
                merged.language = metadata.get("language")
                merged.publication_year = metadata.get("publication_year")
                merged.cited_by_count = metadata.get("cited_by_count")

                # is_oa z obiektu open_access
                oa = metadata.get("open_access") or {}
                merged.is_oa = bool(oa.get("is_oa"))

                # host_venue i publisher z bezpiecznym fallbackiem
                host = metadata.get("host_venue") or {}
                if isinstance(host, dict):
                    merged.host_venue = host.get("display_name")
                publisher = host.get("publisher")
                if not publisher:
                    # alternatywna ścieżka przez primary_location.source.publisher
                    pl = metadata.get("primary_location") or {}
                    src = pl.get("source") or {}
                    if isinstance(src, dict):
                        publisher = src.get("publisher")
                merged.publisher = publisher

                # do extra dokładamy cały surowy payload z OpenAlex
                prev_extra = record.get("extra") or {}
                merged.extra = {**prev_extra, "openalex_enrich": metadata}

                enriched.append(merged.model_dump())
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
    openalex_id_or_path = _resolve_openalex_id(
        identifier, client=client, limiter=limiter, retrying=retrying, mailto=mailto
    )
    if not openalex_id_or_path:
        return None

    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto

    def execute() -> Dict[str, Any]:
        limiter.acquire()
        # openalex_id_or_path jest już w formie 'works/<key>' lub sam 'W...' -> normalizujemy
        path = openalex_id_or_path
        if not path.startswith("works/"):
            path = f"works/{path}"
        resp = client.get(f"/{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    return retrying(execute)


def _resolve_openalex_id(
    identifier: str,
    *,
    client: httpx.Client,
    limiter: RateLimiter,
    retrying: Retrying,
    mailto: str | None,
) -> Optional[str]:
    """Resolve a DOI or URL to an OpenAlex work path/key.

    Preferujemy bezpośrednie ścieżki:
      - /works/doi:{doi}
      - /works/https://doi.org/{doi}
      - /works/{openalex_id}
    Fallback: /works?filter=doi:... lub /works?search=...
    """
    # 1) OpenAlex URL → zwróć ID
    if identifier.startswith("https://openalex.org/"):
        return identifier.removeprefix("https://openalex.org/")

    # 2) DOI → użyj bezpośredniej ścieżki /works/doi:{doi}
    doi = normalise_doi(identifier)
    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto

    if doi:
        def execute_direct() -> httpx.Response:
            limiter.acquire()
            # OpenAlex akceptuje i 'doi:10.123/abc' i pełny URL doi
            return client.get(f"/works/doi:{doi}", params=params)

        try:
            resp = retrying(lambda: (r := execute_direct(), r.raise_for_status(), r)[0])  # noqa: E731
            # jeśli jest 200, mamy komplet
            payload = resp.json()
            work_id = payload.get("id")
            if isinstance(work_id, str) and work_id.startswith("https://openalex.org/"):
                return work_id.split("/")[-1]
        except httpx.HTTPStatusError as exc:
            # 404 przy direct → spróbuj filter=doi:...
            if exc.response.status_code != 404:
                raise

        # Fallback na filter=doi
        def execute_filter() -> Dict[str, Any]:
            limiter.acquire()
            r = client.get("/works", params={**params, "filter": f"doi:{doi}"})
            r.raise_for_status()
            return r.json()

        payload = retrying(execute_filter)
        results = payload.get("results") or []
        if results:
            openalex_id = results[0].get("id")
            if isinstance(openalex_id, str) and openalex_id.startswith("https://openalex.org/"):
                return openalex_id.split("/")[-1]
            if isinstance(openalex_id, str) and openalex_id:
                return openalex_id

    # 3) Brak DOI → spróbuj wyszukiwania (głośne, ale ostatnia deska ratunku)
    def execute_search() -> Dict[str, Any]:
        limiter.acquire()
        r = client.get("/works", params={**params, "search": identifier})
        r.raise_for_status()
        return r.json()

    payload = retrying(execute_search)
    results = payload.get("results") or []
    if not results:
        LOGGER.debug("OpenAlex: no results for identifier=%r", identifier)
        return None
    openalex_id = results[0].get("id")
    if isinstance(openalex_id, str) and openalex_id.startswith("https://openalex.org/"):
        return openalex_id.split("/")[-1]
    if isinstance(openalex_id, str) and openalex_id:
        return openalex_id
    return None
