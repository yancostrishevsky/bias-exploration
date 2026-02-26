"""Connector for the Elsevier Scopus Search API."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.config import RetryConfig, ScopusEnrichConfig
from ai_bias_search.utils.ids import normalise_doi
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError

LOGGER = configure_logging()

_SCOPUS_ID_RE = re.compile(r"SCOPUS_ID:(?P<id>\d+)")
_YEAR_RE = re.compile(r"(?P<year>\d{4})")
_SPLIT_TOKENS_RE = re.compile(r"[,\s;|]+")


def _safe_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _extract_scopus_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _SCOPUS_ID_RE.search(text)
    if match:
        return match.group("id")
    if text.isdigit():
        return text
    return None


def _parse_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and 1800 <= value <= 2100:
        return value
    text = str(value).strip()
    if not text:
        return None
    match = _YEAR_RE.search(text)
    if not match:
        return None
    try:
        year = int(match.group("year"))
    except ValueError:
        return None
    if 1800 <= year <= 2100:
        return year
    return None


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None


def _parse_openaccess_flag(value: object) -> bool | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    parsed = _parse_int(text)
    return parsed


def _extract_issn_list(*values: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            text = str(item).strip()
            if not text:
                continue
            for token in _SPLIT_TOKENS_RE.split(text):
                normalized = normalize_issn(token)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    out.append(normalized)
    return out


def _is_retryable_scopus_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (400, 401, 403, 404):
            return False
        if status in (408, 429):
            return True
        return 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _resolve_api_key() -> str | None:
    value = os.getenv("SCOPUS_API_KEY") or os.getenv("ELSEVIER_API_KEY")
    if not value:
        return None
    return value.strip() or None


def _resolve_inst_token() -> str | None:
    value = (
        os.getenv("SCOPUS_INSTTOKEN")
        or os.getenv("SCOPUS_INST_TOKEN")
        or os.getenv("SCOPUS_INSTTOKEN".lower())
    )
    if not value:
        return None
    return value.strip() or None


def _user_agent() -> str:
    return os.getenv("AI_BIAS_USER_AGENT", "ai-bias-search/0.1 (+contact@example.com)")


def _trace_headers(response: httpx.Response) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in (
        "X-ELS-Request-ID",
        "X-ELS-ReqId",
        "X-ELS-Reqid",
        "X-ELS-Trans-Id",
        "X-ELS-TransId",
    ):
        value = response.headers.get(key)
        if value:
            result[key] = value
    return result


def _extract_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("search-results")
    if not isinstance(results, dict):
        return []
    entries = results.get("entry")
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _extract_url(entry: dict[str, Any]) -> str | None:
    url = _safe_str(entry.get("prism:url") or entry.get("dc:identifier"))
    if url:
        return url
    links = entry.get("link")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            href = _safe_str(link.get("@href"))
            if href:
                return href
    return None


class ScopusConnector:
    """Implementation of the Scopus Search API connector."""

    name = "scopus"

    def __init__(
        self,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        retries: Optional[RetryConfig] = None,
        client: Optional[httpx.Client] = None,
        config: Optional[ScopusEnrichConfig] = None,
    ) -> None:
        self.config = config or ScopusEnrichConfig(enabled=True)

        self.base_url = (os.getenv("SCOPUS_BASE_URL") or self.config.base_url).strip()
        if not self.base_url:
            self.base_url = "https://api.elsevier.com"
        self.timeout = _env_float("SCOPUS_TIMEOUT", self.config.timeout)

        rate = _env_float("SCOPUS_RPS", self.config.rps)
        burst_default = max(int(self.config.burst), 1)
        burst = _env_int("SCOPUS_BURST", burst_default)
        self.rate_limiter = rate_limiter or RateLimiter(rate=rate, burst=max(burst, 1))

        effective_retries = retries or RetryConfig()
        default_max = (
            self.config.max_retries
            if isinstance(self.config.max_retries, int)
            else effective_retries.max
        )
        max_retries = _env_int("SCOPUS_MAX_RETRIES", default_max)
        self.retries = effective_retries.model_copy(update={"max": max_retries})
        self.retrying = Retrying(
            stop=stop_after_attempt(self.retries.max),
            wait=wait_exponential(multiplier=1, exp_base=self.retries.backoff, min=1, max=30),
            retry=retry_if_exception(_is_retryable_scopus_error),
            reraise=True,
        )

        self.api_key = _resolve_api_key()
        if not self.api_key:
            raise ConnectorError("Set SCOPUS_API_KEY (or ELSEVIER_API_KEY) in .env")
        self.inst_token = _resolve_inst_token()

        self.search_view = os.getenv("SCOPUS_SEARCH_VIEW", self.config.search_view).strip()
        self.search_fields = _safe_str(os.getenv("SCOPUS_SEARCH_FIELDS", self.config.search_fields))
        self.search_sort = _safe_str(os.getenv("SCOPUS_SEARCH_SORT", self.config.search_sort))

        max_records_default = max(int(self.config.max_records_per_query), 1)
        self.max_records_per_query = _env_int("SCOPUS_MAX_RECORDS_PER_QUERY", max_records_default)
        page_size_default = min(max(int(self.config.page_size), 1), 100)
        self.page_size = min(max(_env_int("SCOPUS_PAGE_SIZE", page_size_default), 1), 100)

        self.client = client or httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
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

        payload_params: dict[str, Any] = dict(params or {})
        start = max(_parse_int(payload_params.pop("start", 0)) or 0, 0)
        requested_count = _parse_int(payload_params.pop("count", self.page_size)) or self.page_size
        requested_count = min(max(requested_count, 1), 100)
        requested_max = _parse_int(payload_params.pop("max_records", self.max_records_per_query))
        hard_limit = max(requested_max or self.max_records_per_query, 1)
        limit = max(0, min(k, hard_limit))

        sort = _safe_str(payload_params.pop("sort", None)) or self.search_sort
        view = _safe_str(payload_params.pop("view", None)) or self.search_view
        fields = _safe_str(payload_params.pop("field", None)) or self.search_fields

        LOGGER.info("scopus.search query=%s k=%s limit=%s", query, k, limit)

        records: list[dict[str, Any]] = []
        total_results: int | None = None
        while len(records) < limit:
            count = min(requested_count, limit - len(records), 100)
            if count <= 0:
                break
            request_params: dict[str, Any] = {
                "query": query,
                "start": start,
                "count": count,
            }
            if sort:
                request_params["sort"] = sort
            if view:
                request_params["view"] = view
            if fields:
                request_params["field"] = fields
            request_params.update(payload_params)

            try:
                payload = self._get("/content/search/scopus", params=request_params)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 401:
                    raise ConnectorError(
                        "Scopus returned 401 Unauthorized. Verify SCOPUS_API_KEY and "
                        "X-ELS-APIKey header; some APIs also require institutional access."
                    ) from exc
                if status == 403:
                    raise ConnectorError(
                        "Scopus returned 403 Forbidden (insufficient privileges/configuration). "
                        "Use institutional IP/VPN, SCOPUS_INSTTOKEN, and confirm service-level "
                        "entitlements are enabled by Elsevier."
                    ) from exc
                raise ConnectorError(f"Scopus request failed: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ConnectorError(f"Scopus request failed: {exc}") from exc

            entries = _extract_entries(payload)
            if total_results is None:
                search_results = payload.get("search-results")
                if isinstance(search_results, dict):
                    total_results = _parse_int(search_results.get("opensearch:totalResults"))

            if not entries:
                break

            for entry in entries:
                if len(records) >= limit:
                    break
                records.append(
                    self._normalize_entry(
                        entry,
                        rank=len(records) + 1,
                        query=query,
                        request_params=request_params,
                    )
                )

            start += len(entries)
            if len(entries) < count:
                break
            if total_results is not None and start >= total_results:
                break

        return records

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-ELS-APIKey": self.api_key or "",
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        }
        if self.inst_token:
            headers["X-ELS-Insttoken"] = self.inst_token
        return headers

    def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers()

        def execute() -> dict[str, Any]:
            self.rate_limiter.acquire()
            LOGGER.debug(
                "scopus request path=%s params=%s headers=%s",
                path,
                params,
                mask_sensitive(headers),
            )
            response = self.client.get(path, params=params, headers=headers)
            traces = _trace_headers(response)
            if traces:
                LOGGER.debug("scopus response status=%s traces=%s", response.status_code, traces)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ConnectorError("Scopus response is not a JSON object")
            return payload

        return self.retrying(execute)

    def _normalize_entry(
        self,
        entry: dict[str, Any],
        *,
        rank: int,
        query: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        scopus_id = _extract_scopus_id(entry.get("dc:identifier") or entry.get("identifier"))
        eid = _safe_str(entry.get("eid") or entry.get("prism:eid"))
        doi = normalise_doi(_safe_str(entry.get("prism:doi") or entry.get("doi")))
        title = _safe_str(entry.get("dc:title") or entry.get("title")) or ""
        creator = _safe_str(entry.get("dc:creator") or entry.get("creator"))
        publication_name = _safe_str(
            entry.get("prism:publicationName") or entry.get("publicationName")
        )
        issn = normalize_issn(entry.get("prism:issn") or entry.get("issn"))
        eissn = normalize_issn(entry.get("prism:eIssn") or entry.get("prism:eissn"))
        cover_date = _safe_str(entry.get("prism:coverDate") or entry.get("coverDate"))
        citedby_count = _parse_int(entry.get("citedby-count"))
        openaccess_flag = _parse_openaccess_flag(entry.get("openaccessFlag"))
        source_id = _safe_str(entry.get("source-id") or entry.get("source_id"))
        publisher = _safe_str(
            entry.get("publishername")
            or entry.get("dc:publisher")
            or entry.get("publisher")
            or entry.get("prism:publisher")
        )

        authors: list[str] | None = [creator] if creator else None
        record = Record(
            title=title,
            doi=doi,
            url=_extract_url(entry),
            rank=rank,
            raw_id=scopus_id or eid,
            source=publication_name,
            year=_parse_year(cover_date),
            authors=authors,
            extra={
                "scopus": {
                    "raw": entry,
                    "query": query,
                    "request_params": request_params,
                    "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            },
        )
        normalized = record.model_dump()
        normalized.update(
            {
                "scopus_id": scopus_id,
                "eid": eid,
                "creator": creator,
                "publication_name": publication_name,
                "publicationName": publication_name,
                "journal_title": publication_name,
                "issn": issn,
                "eissn": eissn,
                "eIssn": eissn,
                "issn_list": _extract_issn_list(issn, eissn) or None,
                "cover_date": cover_date,
                "coverDate": cover_date,
                "cited_by_count": citedby_count,
                "citedby-count": citedby_count,
                "citations": citedby_count,
                "openaccess_flag": openaccess_flag,
                "openaccessFlag": openaccess_flag,
                "source_id": source_id,
                "source-id": source_id,
                "publisher": publisher,
            }
        )
        return normalized

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


__all__ = ["ScopusConnector"]
