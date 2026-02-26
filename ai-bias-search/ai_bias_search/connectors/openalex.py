"""Connector for the OpenAlex works endpoint."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError

LOGGER = configure_logging()
_SPLIT_TOKENS_RE = re.compile(r"[,\s;|]+")


def _coerce_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _extract_issn_list(item: dict[str, Any]) -> list[str]:
    host = _coerce_mapping(item.get("host_venue"))
    primary_location = _coerce_mapping(item.get("primary_location"))
    source = _coerce_mapping(primary_location.get("source"))
    out: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for entry in value:
                add(entry)
            return
        text = str(value).strip()
        if not text:
            return
        for token in _SPLIT_TOKENS_RE.split(text):
            normalized = normalize_issn(token)
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)

    add(host.get("issn_l"))
    add(host.get("issn"))
    add(source.get("issn_l"))
    add(source.get("issn"))
    return out


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
            host_venue = _coerce_mapping(item.get("host_venue"))
            primary_location = _coerce_mapping(item.get("primary_location"))
            source_obj = _coerce_mapping(primary_location.get("source"))
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
            publisher = source_obj.get("publisher") or host_venue.get("publisher")
            publisher = publisher.strip() if isinstance(publisher, str) and publisher.strip() else None
            journal_title = source_obj.get("display_name") or host_venue.get("display_name")
            journal_title = (
                journal_title.strip() if isinstance(journal_title, str) and journal_title.strip() else None
            )
            cited_by_count = _coerce_int(item.get("cited_by_count"))
            is_oa = bool(item.get("open_access", {}).get("is_oa")) if isinstance(item.get("open_access"), dict) else None

            record = Record(
                title=item.get("display_name", ""),
                doi=doi,
                url=primary_location.get("landing_page_url") or item.get("id"),
                rank=idx,
                raw_id=item.get("id"),
                source=host_venue.get("display_name"),
                year=item.get("publication_year"),
                authors=authors if authors else None,
                extra={"openalex": item},
            )
            normalized = record.model_dump()
            normalized.update(
                {
                    "publisher": publisher,
                    "journal_title": journal_title,
                    "issn_list": _extract_issn_list(item) or None,
                    "cited_by_count": cited_by_count,
                    "citations": cited_by_count,
                    "is_oa": is_oa,
                }
            )
            records.append(normalized)
        return records

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        def execute() -> Dict[str, Any]:
            self.rate_limiter.acquire()
            response = self.client.get(
                path, params=params, headers={"User-Agent": "ai-bias-search/0.1"}
            )
            response.raise_for_status()
            return response.json()

        return self.retrying(execute)


__all__ = ["OpenAlexConnector"]
