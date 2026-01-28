"""Connector for the CORE v3 search API.

This connector is intentionally configurable via environment variables because CORE's
API surface and authentication header conventions can vary between deployments.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError

LOGGER = configure_logging()


def _is_retryable_core_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403, 404):
            return False
        if status in (408, 429):
            return True
        return 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _parse_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, (datetime, date)):
        return value.year
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt).year
        except ValueError:
            continue
    # last resort: first 4 digits
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        try:
            return int(digits[:4])
        except ValueError:
            return None
    return None


def _extract_items(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ("results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            # sometimes nested
            nested = _extract_items(value)
            if nested:
                return nested

    hits = payload.get("hits")
    if isinstance(hits, dict):
        hits_items = hits.get("hits")
        if isinstance(hits_items, list):
            return [item for item in hits_items if isinstance(item, dict)]
    if isinstance(hits, list):
        return [item for item in hits if isinstance(item, dict)]

    return []


def _authors_from_item(item: dict) -> list[str]:
    authors: list[str] = []
    raw = item.get("authors") or item.get("author") or item.get("creators")
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str) and entry.strip():
                authors.append(entry.strip())
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("display_name") or entry.get("fullName")
                if isinstance(name, str) and name.strip():
                    authors.append(name.strip())
    elif isinstance(raw, dict):
        name = raw.get("name") if isinstance(raw.get("name"), str) else None
        if name and name.strip():
            authors.append(name.strip())
    elif isinstance(raw, str) and raw.strip():
        authors.append(raw.strip())
    return authors


def _doi_from_item(item: dict) -> str | None:
    for key in ("doi", "DOI"):
        doi = normalise_doi(item.get(key))
        if doi:
            return doi

    identifiers = item.get("identifiers")
    if isinstance(identifiers, dict):
        doi = normalise_doi(identifiers.get("doi") or identifiers.get("DOI"))
        if doi:
            return doi

    external_ids = item.get("externalIds") or item.get("external_ids")
    if isinstance(external_ids, dict):
        doi = normalise_doi(external_ids.get("DOI") or external_ids.get("doi"))
        if doi:
            return doi

    # last resort: DOI hidden inside a URL-like field
    for key in ("url", "landingPageUrl", "landing_page_url", "downloadUrl", "download_url"):
        maybe_url = item.get(key)
        doi = doi_from_url(maybe_url) if isinstance(maybe_url, str) else None
        if doi:
            return doi
    return None


def _url_from_item(item: dict) -> str | None:
    for key in ("url", "landingPageUrl", "landing_page_url", "downloadUrl", "download_url"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    urls = item.get("urls")
    if isinstance(urls, list):
        for entry in urls:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
            if isinstance(entry, dict):
                maybe = entry.get("url")
                if isinstance(maybe, str) and maybe.strip():
                    return maybe.strip()
    return None


class CoreConnector:
    """Implementation of the CORE v3 search connector."""

    name = "core"

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        retries: Optional[RetryConfig] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.rate_limiter = rate_limiter or RateLimiter(rate=1, burst=2)
        self.retries = retries or RetryConfig()

        api_key = os.getenv("CORE_API_KEY")
        if not api_key:
            raise ConnectorError("Set CORE_API_KEY in .env to enable this connector")
        self.api_key = api_key

        base_url = os.getenv("CORE_API_BASE_URL", "https://api.core.ac.uk/v3")
        self.search_path = os.getenv("CORE_SEARCH_PATH", "/search/works")
        self.query_param = os.getenv("CORE_QUERY_PARAM", "q")
        self.limit_param = os.getenv("CORE_LIMIT_PARAM", "limit")
        self.offset_param = os.getenv("CORE_OFFSET_PARAM", "offset")
        self.auth_header = os.getenv("CORE_AUTH_HEADER", "Authorization")
        self.auth_prefix = os.getenv("CORE_AUTH_PREFIX", "Bearer")
        self.user_agent = os.getenv(
            "AI_BIAS_USER_AGENT",
            "ai-bias-search/0.1 (+contact@example.com)",
        )

        self.client = client or httpx.Client(base_url=base_url, timeout=30.0)
        self.retrying = Retrying(
            stop=stop_after_attempt(self.retries.max),
            wait=wait_exponential(multiplier=1, exp_base=self.retries.backoff, min=1),
            retry=retry_if_exception(_is_retryable_core_error),
            reraise=True,
        )

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not query or k <= 0:
            return []

        request_params: Dict[str, Any] = dict(params or {})
        request_params[self.query_param] = query

        per_page = min(int(request_params.pop(self.limit_param, k)), 100, k)
        offset = int(request_params.pop(self.offset_param, 0))

        LOGGER.info("core.search query=%s k=%s", query, k)

        records: list[dict] = []
        seen_keys: set[str] = set()
        while len(records) < k:
            page_params = dict(request_params)
            page_params[self.limit_param] = min(per_page, k - len(records))
            page_params[self.offset_param] = offset

            payload = self._get(self.search_path, page_params)
            items = _extract_items(payload)
            if not items:
                break

            added = 0
            for item in items:
                if len(records) >= k:
                    break
                raw_key = (
                    item.get("id")
                    or item.get("coreId")
                    or item.get("_id")
                    or item.get("doi")
                    or item.get("title")
                    or ""
                )
                key = str(raw_key).strip()
                if not key:
                    key = f"offset:{offset}:idx:{len(records)}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                record = self._normalize_item(item, rank=len(records) + 1)
                records.append(record)
                added += 1

            offset += len(items)
            if added == 0:
                break

        return records

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.auth_header.lower() == "authorization":
            headers[self.auth_header] = f"{self.auth_prefix} {self.api_key}".strip()
        else:
            headers[self.auth_header] = self.api_key

        def execute() -> Dict[str, Any]:
            self.rate_limiter.acquire()
            response = self.client.get(path, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ConnectorError("CORE response is not a JSON object")
            return data

        try:
            return self.retrying(execute)
        except httpx.HTTPError as exc:
            raise ConnectorError(f"CORE request failed: {exc}") from exc

    def _normalize_item(self, item: dict, *, rank: int) -> Dict[str, Any]:
        title = item.get("title") or item.get("display_name") or item.get("displayName")
        if not isinstance(title, str):
            title = ""

        year = _parse_year(
            item.get("year") or item.get("publicationYear") or item.get("publishedDate")
        )

        source = item.get("venue") or item.get("journal") or item.get("publisher")
        source_value = source.strip() if isinstance(source, str) and source.strip() else None

        record = Record(
            title=title,
            doi=_doi_from_item(item),
            url=_url_from_item(item),
            rank=rank,
            raw_id=str(item.get("id") or item.get("coreId") or item.get("_id") or "") or None,
            source=source_value,
            year=year,
            authors=_authors_from_item(item) or None,
            extra={"core": item},
        )
        return record.model_dump()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


__all__ = ["CoreConnector"]
