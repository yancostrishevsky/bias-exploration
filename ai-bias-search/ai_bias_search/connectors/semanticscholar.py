"""Connector for the Semantic Scholar Graph search API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError


LOGGER = configure_logging()


class SemanticScholarConnector:
    """Semantic Scholar bulk search connector."""

    name = "semanticscholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        retries: Optional[RetryConfig] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.rate_limiter = rate_limiter or RateLimiter(rate=1, burst=2)
        self.retries = retries or RetryConfig()
        self.client = client or httpx.Client(timeout=30.0)
        self.retrying = Retrying(
            stop=stop_after_attempt(self.retries.max),
            wait=wait_exponential(multiplier=1, exp_base=self.retries.backoff, min=1),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Call the Semantic Scholar search endpoint."""

        payload: Dict[str, Any] = {"query": query, "limit": k}
        if params:
            payload.update(params)
        fields = [
            "title",
            "year",
            "url",
            "venue",
            "authors.name",
            "externalIds",
            "paperId",
        ]
        payload.setdefault("fields", ",".join(fields))

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        LOGGER.info("semanticscholar.search query=%s k=%s", query, k)
        LOGGER.debug("semanticscholar payload=%s headers=%s", payload, mask_sensitive(headers))

        try:
            data = self._post(self.BASE_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Semantic Scholar request failed: {exc}") from exc

        papers = data.get("data", []) if isinstance(data, dict) else []
        records: List[Dict[str, Any]] = []
        for idx, paper in enumerate(papers[:k], start=1):
            authors = [author.get("name") for author in paper.get("authors", []) if author.get("name")]
            doi = None
            external_ids = paper.get("externalIds") or {}
            if isinstance(external_ids, dict):
                doi = external_ids.get("DOI")
            record = Record(
                title=paper.get("title", ""),
                doi=doi,
                url=paper.get("url"),
                rank=idx,
                raw_id=paper.get("paperId"),
                source=paper.get("venue"),
                year=paper.get("year"),
                authors=authors if authors else None,
                extra={"semanticscholar": paper},
            )
            records.append(record.model_dump())
        return records

    def _post(self, url: str, json: Dict[str, Any], headers: Dict[str, Any]) -> Dict[str, Any]:
        def execute() -> Dict[str, Any]:
            self.rate_limiter.acquire()
            response = self.client.post(url, json=json, headers=headers)
            response.raise_for_status()
            return response.json()

        return self.retrying(execute)


__all__ = ["SemanticScholarConnector"]
