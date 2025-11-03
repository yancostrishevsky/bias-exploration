"""Connector for the OpenAlex works endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError, SearchConnector


LOGGER = configure_logging()


class OpenAlexConnector:
    """Implementation of the OpenAlex API search connector."""

    name = "openalex"
    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        retries: Optional[RetryConfig] = None,
        mailto: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.rate_limiter = rate_limiter or RateLimiter(rate=2, burst=5)
        self.retries = retries or RetryConfig()
        self.client = client or httpx.Client(base_url=self.BASE_URL, timeout=30.0)
        self.mailto = mailto
        self.retrying = Retrying(
            stop=stop_after_attempt(self.retries.max),
            wait=wait_exponential(multiplier=1, exp_base=self.retries.backoff, min=1),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return top *k* OpenAlex works for *query*."""

        request_params: Dict[str, Any] = {"search": query, "per_page": k}
        if params:
            request_params.update(params)
        if self.mailto:
            request_params["mailto"] = self.mailto

        LOGGER.info("openalex.search query=%s k=%s", query, k)
        try:
            payload = self._get("/works", request_params)
        except httpx.HTTPError as exc:
            raise ConnectorError(f"OpenAlex request failed: {exc}") from exc

        results = payload.get("results", [])
        records: List[Dict[str, Any]] = []
        for idx, item in enumerate(results[:k], start=1):
            authors = [
                authorship.get("author", {}).get("display_name")
                for authorship in item.get("authorships", [])
                if authorship.get("author") and authorship.get("author", {}).get("display_name")
            ]
            raw_doi = item.get("doi")
            doi = normalise_doi(raw_doi)
            if not doi:
                doi = doi_from_url(raw_doi)
            if not doi:
                primary_location = item.get("primary_location") or {}
                if isinstance(primary_location, dict):
                    doi = doi_from_url(primary_location.get("landing_page_url"))
            record = Record(
                title=item.get("display_name", ""),
                doi=doi,
                url=(item.get("primary_location", {}) or {}).get("landing_page_url") or item.get("id"),
                rank=idx,
                raw_id=item.get("id"),
                source=(item.get("host_venue", {}) or {}).get("display_name"),
                year=item.get("publication_year"),
                authors=authors if authors else None,
                extra={"openalex": item},
            )
            records.append(record.model_dump())
        return records

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        def execute() -> Dict[str, Any]:
            self.rate_limiter.acquire()
            response = self.client.get(path, params=params, headers={"User-Agent": "ai-bias-search/0.1"})
            response.raise_for_status()
            return response.json()

        return self.retrying(execute)


__all__ = ["OpenAlexConnector"]
