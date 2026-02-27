"""Connector for the Semantic Scholar Graph search API."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_bias_search.diagnostics.capture import capture_request
from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import normalise_doi
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
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


def _extract_issn_values(paper: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                add(item)
            return
        text = str(value).strip()
        if not text:
            return
        for token in _SPLIT_TOKENS_RE.split(text):
            normalized = normalize_issn(token)
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)

    journal = _coerce_mapping(paper.get("journal"))
    publication_venue = _coerce_mapping(paper.get("publicationVenue"))
    external_ids = _coerce_mapping(paper.get("externalIds"))
    add(publication_venue.get("issn"))
    add(journal.get("issn"))
    add(external_ids.get("ISSN"))
    return out


def _clean_publisher(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text in {"0", "0.0"}:
        return None
    return text


def _extract_publisher(paper: dict[str, Any]) -> str | None:
    venue_obj = _coerce_mapping(paper.get("venue"))
    journal = _coerce_mapping(paper.get("journal"))
    publication_venue = _coerce_mapping(paper.get("publicationVenue"))
    return (
        _clean_publisher(venue_obj.get("publisher"))
        or _clean_publisher(journal.get("publisher"))
        or _clean_publisher(publication_venue.get("publisher"))
        or _clean_publisher(paper.get("publisher"))
    )


class SemanticScholarConnector:
    """Semantic Scholar search connector with bulk+ranked and paging."""

    name = "semanticscholar"
    BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    RANKED_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    DEFAULT_FIELDS = [
        "title",
        "abstract",
        "year",
        "publicationDate",
        "url",
        "venue",
        "authors.name",
        "externalIds",
        "paperId",
        "isOpenAccess",
        "openAccessPdf",
        "s2FieldsOfStudy",
        "citationCount",
        "referenceCount",
        "journal",
        "publicationVenue",
    ]
    PER_PAGE = 200

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
        self.user_agent = os.getenv(
            "AI_BIAS_USER_AGENT", "ai-bias-search/0.1 (+contact@example.com)"
        )

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Call the Semantic Scholar search endpoint and return normalized records."""
        if not query or k <= 0:
            return []

        # query params
        qparams: Dict[str, Any] = {"query": query}
        if params:
            qparams.update(params)
        qparams.setdefault("fields", ",".join(self.DEFAULT_FIELDS))

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        LOGGER.info("semanticscholar.search query=%s k=%s", query, k)
        LOGGER.debug("semanticscholar params=%s headers=%s", qparams, mask_sensitive(headers))

        try:
            papers = self._collect_bulk(qparams, headers, want=k)
        except ConnectorError as bulk_err:
            LOGGER.warning("Bulk failed (%s). Falling back to ranked /paper/search.", bulk_err)
            papers = self._collect_ranked(qparams, headers, want=k)

        records: List[Dict[str, Any]] = []
        for idx, paper in enumerate(papers[:k], start=1):
            authors = [a.get("name") for a in paper.get("authors", []) if a.get("name")]
            external_ids = paper.get("externalIds") or {}
            doi = normalise_doi(external_ids.get("DOI")) if isinstance(external_ids, dict) else None
            journal = _coerce_mapping(paper.get("journal"))
            publication_venue = _coerce_mapping(paper.get("publicationVenue"))
            venue_obj = _coerce_mapping(paper.get("venue"))
            publisher = _extract_publisher(paper)
            venue_name = (
                venue_obj.get("name") if isinstance(venue_obj.get("name"), str) else paper.get("venue")
            )
            journal_title = publication_venue.get("name") or journal.get("name") or venue_name
            journal_title = (
                journal_title.strip()
                if isinstance(journal_title, str) and journal_title.strip()
                else None
            )
            citations = _coerce_int(paper.get("citationCount"))
            is_oa = paper.get("isOpenAccess")
            if not isinstance(is_oa, bool):
                is_oa = None
            record = Record(
                title=paper.get("title", ""),
                doi=doi,
                url=paper.get("url"),
                rank=idx,
                raw_id=paper.get("paperId"),
                source=paper.get("venue"),
                year=paper.get("year"),
                authors=authors or None,
                extra={"semanticscholar": paper},
            )
            normalized = record.model_dump()
            normalized.update(
                {
                    "publisher": publisher,
                    "journal_title": journal_title,
                    "issn_list": _extract_issn_values(paper) or None,
                    "cited_by_count": citations,
                    "citations": citations,
                    "is_oa": is_oa,
                }
            )
            records.append(normalized)
        return records

    # ---------- helpers ----------

    def _collect_bulk(
        self, qparams: Dict[str, Any], headers: Dict[str, Any], want: int
    ) -> List[Dict[str, Any]]:
        """Iterate /paper/search/bulk with token pagination until we collect `want` items."""
        collected: List[Dict[str, Any]] = []
        token: Optional[str] = None

        while len(collected) < want:
            self.rate_limiter.acquire()
            params = dict(qparams)
            params["limit"] = min(self.PER_PAGE, want - len(collected))
            if token:
                params["token"] = token

            try:
                page = self._get(self.BULK_URL, params=params, headers=headers)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (403,):
                    raise ConnectorError(f"Bulk 403: {exc}") from exc
                raise ConnectorError(f"Bulk failed: {exc}") from exc

            data = page.get("data", []) if isinstance(page, dict) else []
            if not data:
                break
            collected.extend(data)
            token = page.get("token")
            if not token:
                break
        return collected

    def _collect_ranked(
        self, qparams: Dict[str, Any], headers: Dict[str, Any], want: int
    ) -> List[Dict[str, Any]]:
        """Single-page /paper/search (ranked)."""
        params = dict(qparams)
        params["limit"] = min(self.PER_PAGE, want)
        try:
            page = self._get(self.RANKED_URL, params=params, headers=headers)
        except httpx.HTTPStatusError as exc:
            raise ConnectorError(f"Ranked failed: {exc}") from exc
        return page.get("data", []) if isinstance(page, dict) else []

    def _get(self, url: str, params: Dict[str, Any], headers: Dict[str, Any]) -> Dict[str, Any]:
        def execute() -> Dict[str, Any]:
            response: httpx.Response | None = None
            payload: Dict[str, Any] | None = None
            started = time.perf_counter()
            try:
                response = self.client.get(url, params=params, headers=headers)
                if response.status_code == 429:
                    ra = response.headers.get("Retry-After")
                    if ra:
                        try:
                            time.sleep(int(ra))
                        except (ValueError, TypeError):
                            pass
                response.raise_for_status()
                payload = response.json()
                return payload
            finally:
                duration_ms = int((time.perf_counter() - started) * 1000)
                capture_request(
                    platform=self.name,
                    stage="collect",
                    endpoint=url,
                    method="GET",
                    params=params,
                    headers=headers,
                    status_code=(response.status_code if response is not None else None),
                    duration_ms=duration_ms,
                    response_payload=payload,
                )

        return self.retrying(execute)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


__all__ = ["SemanticScholarConnector"]
