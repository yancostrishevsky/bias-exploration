"""Connector for the CORE v3 search API."""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from ai_bias_search.diagnostics.capture import capture_request
from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError

LOGGER = configure_logging()
_SPLIT_TOKENS_RE = re.compile(r"[,\s;|]+")


class CoreTransientError(ConnectorError):
    """Retryable error for transient CORE backend failures."""


class CorePermanentError(ConnectorError):
    """Non-retryable error that disables CORE for the current run."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_retryable_core_error(exc: BaseException) -> bool:
    if isinstance(exc, CoreTransientError):
        return True
    if isinstance(exc, CorePermanentError):
        return False
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
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        try:
            return int(digits[:4])
        except ValueError:
            return None
    return None


def _clean_api_key(value: str) -> str:
    key = value.strip()
    if len(key) >= 2 and ((key[0] == key[-1] == '"') or (key[0] == key[-1] == "'")):
        key = key[1:-1].strip()
    return key


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_core_base_url(base_url: str) -> str:
    cleaned = (base_url or "").strip()
    if not cleaned:
        cleaned = "https://api.core.ac.uk/v3"
    cleaned = cleaned.rstrip("/")

    try:
        parts = urlsplit(cleaned)
    except Exception:
        return "https://api.core.ac.uk"

    path = (parts.path or "").rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def join_url(base_url: str, path: str) -> str:
    """Join base URL and path while preventing duplicated `/v3` segments."""

    normalized_base = _normalize_core_base_url(base_url)
    base_parts = urlsplit(normalized_base)
    base_path = (base_parts.path or "").rstrip("/")

    target_path = "/" + str(path or "").strip().lstrip("/")
    target_path = "/" + re.sub(r"/{2,}", "/", target_path).lstrip("/")

    if base_path.endswith("/v3") and target_path.startswith("/v3/"):
        target_path = target_path[3:]
    elif base_path.endswith("/v3") and target_path == "/v3":
        target_path = "/"

    merged_path = f"{base_path}{target_path}" if base_path else target_path
    merged_path = "/" + merged_path.lstrip("/")
    merged_path = re.sub(r"/{2,}", "/", merged_path)
    return urlunsplit((base_parts.scheme, base_parts.netloc, merged_path, "", ""))


def _extract_items(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ("results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
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


def _is_transient_overload_payload(payload: dict) -> bool:
    """Detect 200-OK responses that still indicate a transient backend overload."""

    message = payload.get("message")
    if isinstance(message, str):
        lowered = message.lower()
        if "es_rejected_execution_exception" in lowered:
            return True
        if "rejected execution" in lowered:
            return True

    for key in ("failures", "errors"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return True

    total = payload.get("total")
    successful = payload.get("successful")
    if isinstance(total, int) and isinstance(successful, int) and successful < total:
        return True

    return False


def _has_results(payload: dict) -> bool:
    return bool(_extract_items(payload))


def _total_hits(payload: dict) -> int | None:
    for key in ("total_hits", "totalHits"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


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


def _extract_issn_values(item: dict[str, Any]) -> list[str]:
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

    identifiers = item.get("identifiers")
    if not isinstance(identifiers, dict):
        identifiers = {}

    add(item.get("issn"))
    add(item.get("eissn"))
    add(item.get("journalIssn"))
    add(identifiers.get("issn"))
    add(identifiers.get("eissn"))
    return out


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
    """Implementation of the CORE search connector."""

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
        self.api_key = _clean_api_key(api_key)
        self.fail_open = _env_bool("CORE_FAIL_OPEN", True)

        base_url = _normalize_core_base_url(os.getenv("CORE_API_BASE_URL", "https://api.core.ac.uk/v3"))
        if not urlsplit(base_url).path.endswith("/v3"):
            base_url = join_url(base_url, "/v3")
        self.search_url = join_url(base_url, "/search/works")
        self.max_page_size = int(os.getenv("CORE_MAX_PAGE_SIZE", "25"))
        self.user_agent = os.getenv(
            "AI_BIAS_USER_AGENT",
            "ai-bias-search/0.1 (+contact@example.com)",
        )

        self.client = client or httpx.Client(timeout=30.0, follow_redirects=True)
        self._disabled = False
        self._disable_reason: str | None = None
        self._request_count = 0
        self._error_count = 0

        max_attempts = max(1, int(self.retries.max))
        self.retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, exp_base=self.retries.backoff, min=1, max=10)
            + wait_random(0, 0.5),
            retry=retry_if_exception(_is_retryable_core_error),
            reraise=True,
        )

    def _disable_for_run(self, reason: str) -> None:
        self._disabled = True
        self._disable_reason = reason

    def platform_health(self) -> dict[str, Any]:
        error_rate = (self._error_count / self._request_count) if self._request_count > 0 else 0.0
        return {
            "enabled": not self._disabled,
            "reason": self._disable_reason,
            "error_rate": error_rate,
        }

    def search(
        self,
        query: str,
        k: int,
        prompt_template: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not query or k <= 0:
            return []
        if self._disabled:
            LOGGER.warning(
                "CORE connector disabled for this run; returning empty results (reason=%s)",
                self._disable_reason,
            )
            return []

        request_params: Dict[str, Any] = dict(params or {})
        per_page = min(int(request_params.pop("limit", k)), self.max_page_size, k)
        offset = int(request_params.pop("offset", 0))

        LOGGER.info("core.search query=%s k=%s", query, k)

        records: list[dict] = []
        seen_keys: set[str] = set()
        while len(records) < k:
            page_params: dict[str, Any] = {
                "q": query,
                "limit": min(per_page, k - len(records)),
                "offset": offset,
                "scroll": "false",
                "stats": "false",
            }

            try:
                payload = self._get_search(page_params)
            except CorePermanentError as exc:
                self._disable_for_run(str(exc))
                if records:
                    LOGGER.warning(
                        "CORE disabled after collecting %d records; returning partial results",
                        len(records),
                    )
                    break
                if self.fail_open:
                    LOGGER.warning("CORE disabled and fail-open enabled; returning empty results")
                    return []
                raise ConnectorError(str(exc)) from exc
            except CoreTransientError as exc:
                LOGGER.warning(
                    "CORE page fetch failed with transient backend error; returning partial "
                    "results (offset=%s limit=%s collected=%d): %s",
                    offset,
                    page_params["limit"],
                    len(records),
                    exc,
                )
                break
            except ConnectorError as exc:
                if records:
                    LOGGER.warning(
                        "CORE page fetch failed after collecting %d records; returning partial "
                        "results (offset=%s limit=%s): %s",
                        len(records),
                        offset,
                        page_params["limit"],
                        exc,
                    )
                    break
                raise

            if _is_transient_overload_payload(payload):
                LOGGER.warning(
                    "CORE backend reported shard failures (successful=%r total=%r failed=%r); "
                    "using any returned results",
                    payload.get("successful"),
                    payload.get("total"),
                    payload.get("failed"),
                )
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

            requested = int(page_params["limit"])
            total_hits = _total_hits(payload)
            if len(items) < requested and (
                total_hits is None or (offset + len(items)) >= total_hits
            ):
                break

            offset += len(items)
            if added == 0:
                break

        return records

    def _get_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        def execute() -> Dict[str, Any]:
            response: httpx.Response | None = None
            data: dict[str, Any] | None = None
            started = time.perf_counter()
            try:
                self.rate_limiter.acquire()
                LOGGER.debug(
                    "core request method=%s endpoint=%s params=%s headers=%s",
                    "GET",
                    self.search_url,
                    params,
                    mask_sensitive(headers),
                )
                response = self.client.get(self.search_url, params=params, headers=headers)
                response.raise_for_status()
                payload_raw = response.json()
                data = payload_raw if isinstance(payload_raw, dict) else None
                if not isinstance(data, dict):
                    raise ConnectorError("CORE response is not a JSON object")
                if _is_transient_overload_payload(data) and not _has_results(data):
                    raise CoreTransientError("CORE backend overloaded (no results returned)")
                return data
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                try:
                    payload_raw = exc.response.json()
                    if isinstance(payload_raw, dict):
                        data = payload_raw
                except Exception:
                    data = {
                        "error": (
                            exc.response.text[:300] if isinstance(exc.response.text, str) else str(exc)
                        )
                    }
                if 400 <= status <= 499 and status not in (408, 429):
                    message = f"CORE request failed status={status} endpoint={self.search_url}"
                    raise CorePermanentError(message, status_code=status) from exc
                if status in (408, 429) or 500 <= status <= 599:
                    raise CoreTransientError(
                        f"CORE transient error status={status} endpoint={self.search_url}"
                    ) from exc
                snippet: str | None = None
                try:
                    text = exc.response.text
                    snippet = text[:300] if text else None
                except Exception:
                    snippet = None
                details = f"CORE request failed (GET): {exc}"
                if snippet:
                    details = f"{details} (body={snippet!r})"
                raise ConnectorError(details) from exc
            finally:
                self._request_count += 1
                if response is not None and response.status_code >= 400:
                    self._error_count += 1
                duration_ms = int((time.perf_counter() - started) * 1000)
                capture_request(
                    platform=self.name,
                    stage="collect",
                    endpoint=self.search_url,
                    method="GET",
                    params=params,
                    headers=headers,
                    status_code=(response.status_code if response is not None else None),
                    duration_ms=duration_ms,
                    response_payload=data,
                )

        try:
            return self.retrying(execute)
        except CoreTransientError:
            raise
        except CorePermanentError:
            raise
        except httpx.HTTPError as exc:
            raise ConnectorError(f"CORE request failed: {exc}") from exc

    def _normalize_item(self, item: dict, *, rank: int) -> Dict[str, Any]:
        title = item.get("title") or item.get("display_name") or item.get("displayName")
        if not isinstance(title, str):
            title = ""

        year = _parse_year(
            item.get("year")
            or item.get("yearPublished")
            or item.get("publicationYear")
            or item.get("publishedDate")
            or item.get("publicationDate")
        )

        source = item.get("venue") or item.get("journal") or item.get("publisher")
        source_value = source.strip() if isinstance(source, str) and source.strip() else None
        publisher = item.get("publisher")
        publisher_value = (
            publisher.strip() if isinstance(publisher, str) and publisher.strip() else None
        )
        citation_raw = item.get("citationCount")
        if citation_raw is None:
            citation_raw = item.get("citations")
        citations = _coerce_int(citation_raw)

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
        normalized = record.model_dump()
        normalized.update(
            {
                "publisher": publisher_value,
                "journal_title": source_value,
                "issn_list": _extract_issn_values(item) or None,
                "cited_by_count": citations,
                "citations": citations,
            }
        )
        return normalized

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


__all__ = ["CoreConnector", "join_url"]
