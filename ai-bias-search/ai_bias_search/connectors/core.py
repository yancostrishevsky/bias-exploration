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
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
from ai_bias_search.utils.models import Record
from ai_bias_search.utils.rate_limit import RateLimiter

from .base import ConnectorError

LOGGER = configure_logging()


class CoreTransientError(ConnectorError):
    """Retryable error for transient CORE backend failures."""


def _is_retryable_core_error(exc: BaseException) -> bool:
    if isinstance(exc, CoreTransientError):
        return True
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


def _clean_api_key(value: str) -> str:
    key = value.strip()
    if len(key) >= 2 and ((key[0] == key[-1] == '"') or (key[0] == key[-1] == "'")):
        key = key[1:-1].strip()
    return key


def _toggle_trailing_slash(path: str) -> str:
    if not path:
        return path
    if path.endswith("/"):
        return path.rstrip("/")
    return f"{path}/"


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
        self.api_key = _clean_api_key(api_key)

        base_url = os.getenv("CORE_API_BASE_URL", "https://api.core.ac.uk/v3").strip()
        raw_search_path = os.getenv("CORE_SEARCH_PATH", "/search/works").strip()
        # Swagger docs use no trailing slash; normalise to avoid redirect edge-cases.
        self.search_path = (raw_search_path.rstrip("/") or "/search/works").strip()
        self.query_param = os.getenv("CORE_QUERY_PARAM", "q")
        self.limit_param = os.getenv("CORE_LIMIT_PARAM", "limit")
        self.offset_param = os.getenv("CORE_OFFSET_PARAM", "offset")
        self.max_page_size = int(os.getenv("CORE_MAX_PAGE_SIZE", "25"))
        self.search_method = os.getenv("CORE_SEARCH_METHOD", "AUTO").strip().upper()
        if self.search_method not in {"AUTO", "GET", "POST"}:
            LOGGER.warning("Unknown CORE_SEARCH_METHOD=%r; defaulting to AUTO", self.search_method)
            self.search_method = "AUTO"
        self.auth_header = os.getenv("CORE_AUTH_HEADER", "Authorization")
        self.auth_prefix = os.getenv("CORE_AUTH_PREFIX", "Bearer")
        self.user_agent = os.getenv(
            "AI_BIAS_USER_AGENT",
            "ai-bias-search/0.1 (+contact@example.com)",
        )

        self.client = client or httpx.Client(
            base_url=base_url,
            timeout=30.0,
            follow_redirects=True,
        )
        self.retrying = Retrying(
            stop=stop_after_attempt(self.retries.max),
            wait=wait_exponential(multiplier=1, exp_base=self.retries.backoff, min=1, max=10),
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

        per_page = min(int(request_params.pop(self.limit_param, k)), self.max_page_size, k)
        offset = int(request_params.pop(self.offset_param, 0))

        LOGGER.info("core.search query=%s k=%s", query, k)

        records: list[dict] = []
        seen_keys: set[str] = set()
        while len(records) < k:
            page_params = dict(request_params)
            page_params[self.limit_param] = min(per_page, k - len(records))
            page_params[self.offset_param] = offset

            try:
                payload = self._get(self.search_path, page_params)
            except ConnectorError as exc:
                if records:
                    LOGGER.warning(
                        "CORE page fetch failed after collecting %d records; returning partial "
                        "results (offset=%s limit=%s): %s",
                        len(records),
                        offset,
                        page_params[self.limit_param],
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

            requested = int(page_params[self.limit_param])
            total_hits = _total_hits(payload)
            if len(items) < requested and (
                total_hits is None or (offset + len(items)) >= total_hits
            ):
                break

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
            methods: list[str]
            if self.search_method == "AUTO":
                methods = ["GET", "POST"]
            else:
                methods = [self.search_method]

            candidate_paths = [path]
            alt_path = _toggle_trailing_slash(path)
            if alt_path and alt_path != path:
                candidate_paths.append(alt_path)

            retryable_exc: BaseException | None = None
            non_retryable_exc: BaseException | None = None
            for method in methods:
                if method == "POST":
                    # CORE routes commonly only support POST without a trailing slash.
                    post_path = (path.rstrip("/") or path).strip()
                    method_paths = [post_path]
                else:
                    method_paths = candidate_paths

                for candidate_path in method_paths:
                    try:
                        self.rate_limiter.acquire()
                        LOGGER.debug(
                            "core request method=%s path=%s params=%s headers=%s",
                            method,
                            candidate_path,
                            params,
                            mask_sensitive(headers),
                        )
                        if method == "GET":
                            query_params = dict(params)
                            query_params.setdefault("scroll", "false")
                            query_params.setdefault("stats", "false")
                            response = self.client.get(
                                candidate_path, params=query_params, headers=headers
                            )
                        elif method == "POST":
                            body = dict(params)
                            body.setdefault("scroll", False)
                            body.setdefault("stats", False)
                            response = self.client.post(candidate_path, json=body, headers=headers)
                        else:  # pragma: no cover - defensive guard
                            raise ConnectorError(f"Unsupported CORE search method: {method}")

                        response.raise_for_status()
                        data = response.json()
                        if method == "GET" and candidate_path != path and path == self.search_path:
                            self.search_path = candidate_path
                        break
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        if status in (408, 429) or 500 <= status <= 599:
                            retryable_exc = exc
                        else:
                            non_retryable_exc = exc

                        if status in (404, 405) or 500 <= status <= 599:
                            continue
                        raise
                else:
                    continue
                break
            else:
                if retryable_exc is not None:
                    raise retryable_exc
                if non_retryable_exc is not None:
                    raise non_retryable_exc
                raise ConnectorError("CORE request failed without a response")

            if not isinstance(data, dict):
                raise ConnectorError("CORE response is not a JSON object")
            if _is_transient_overload_payload(data) and not _has_results(data):
                raise CoreTransientError("CORE backend overloaded (no results returned)")
            return data

        try:
            return self.retrying(execute)
        except httpx.HTTPStatusError as exc:
            snippet: str | None = None
            try:
                text = exc.response.text
                snippet = text[:300] if text else None
            except Exception:
                snippet = None
            method = getattr(exc.request, "method", None)
            details = f"CORE request failed: {exc}"
            if method:
                details = f"CORE request failed ({method}): {exc}"
            if snippet:
                details = f"{details} (body={snippet!r})"
            raise ConnectorError(details) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"CORE request failed: {exc}") from exc
        except CoreTransientError as exc:
            raise ConnectorError(str(exc)) from exc

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
