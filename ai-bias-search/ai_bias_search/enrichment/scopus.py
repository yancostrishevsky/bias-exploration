"""Optional enrichment of records with Elsevier Scopus metadata.

This enrichment is designed to be:
- optional (OFF by default; gated by YAML config + env vars),
- resilient to partial failures (best-effort; never aborts the whole run),
- cache-backed (DiskCache; caches both hits and misses), and
- non-destructive by default (fills missing fields only).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
from urllib.parse import quote

import httpx
from diskcache import Cache
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.config import RetryConfig, ScopusEnrichConfig
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
from ai_bias_search.utils.rate_limit import RateLimiter

LOGGER = configure_logging()

BASE_URL = "https://api.elsevier.com"
CACHE_DIR = (Path(__file__).resolve().parents[2] / "data" / "cache" / "scopus").resolve()
_CACHE_MISSING = object()

_SCOPUS_ID_RE = re.compile(r"SCOPUS_ID:(?P<id>\d+)")
_YEAR_RE = re.compile(r"(?P<year>\d{4})")
_ISSN_ALLOWED_RE = re.compile(r"^[0-9X]{8}$")


class ScopusEnrichError(RuntimeError):
    """Base exception for Scopus enrichment failures."""


class ScopusResponseError(ScopusEnrichError):
    """Scopus API error with request/response context."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        path: str | None = None,
        params: dict[str, Any] | None = None,
        traces: dict[str, str] | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.path = path
        self.params = dict(params or {})
        self.traces = dict(traces or {})
        self.attempts = list(attempts or [])


class ScopusAuthError(ScopusResponseError):
    """Raised for 401/invalid authentication responses."""


class ScopusPermissionError(ScopusResponseError):
    """Raised for 403/insufficient privileges responses."""


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = str(value).strip().lower()
    if parsed in {"1", "true", "yes", "y", "on"}:
        return True
    if parsed in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _clean_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_view(value: object) -> str | None:
    text = _clean_optional_str(value)
    if text is None:
        return None
    if text.lower() in {"none", "null", "(none)", "off"}:
        return None
    return text


def _parse_fallback_views(
    raw: str | None,
    default: list[str | None] | None,
) -> list[str | None]:
    if raw is None:
        source = default if default is not None else ["META", None]
    else:
        tokens = [token.strip() for token in raw.split(",")]
        source = [token for token in tokens if token.strip()]
        if not source:
            source = default if default is not None else ["META", None]

    parsed: list[str | None] = []
    seen: set[str | None] = set()
    for item in source:
        view = _normalize_view(item)
        if view in seen:
            continue
        seen.add(view)
        parsed.append(view)
    return parsed


def _trace_headers(response: httpx.Response) -> dict[str, str]:
    traces: dict[str, str] = {}
    req_id = (
        response.headers.get("X-ELS-ReqId")
        or response.headers.get("X-ELS-Reqid")
        or response.headers.get("X-ELS-Request-ID")
    )
    trans_id = response.headers.get("X-ELS-TransId") or response.headers.get("X-ELS-Trans-Id")
    if req_id:
        traces["x-els-reqid"] = req_id
    if trans_id:
        traces["x-els-transid"] = trans_id
    return traces


def _log_scopus_response(path: str, status_code: int, traces: dict[str, str]) -> None:
    LOGGER.debug(
        "scopus enrich response endpoint=%s status=%s x-els-reqid=%s x-els-transid=%s",
        path,
        status_code,
        traces.get("x-els-reqid", "-"),
        traces.get("x-els-transid", "-"),
    )


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


def _is_retryable_scopus_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (400, 401, 403, 404):
            return False
        if status in (408, 429):
            return True
        return 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = int(text)
        return max(float(seconds), 0.0)
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return max((dt - now).total_seconds(), 0.0)
    except Exception:
        return None


class _ScopusWait:
    def __init__(self, backoff: float) -> None:
        self._fallback = wait_exponential(multiplier=1, exp_base=backoff, min=1, max=60)

    def __call__(self, retry_state: Any) -> float:
        exc = None
        if retry_state.outcome is not None:
            exc = retry_state.outcome.exception()
        if isinstance(exc, httpx.HTTPStatusError):
            ra = exc.response.headers.get("Retry-After")
            retry_after = _parse_retry_after(ra)
            if retry_after is not None:
                return retry_after
        return float(self._fallback(retry_state))


def _scopus_retrying(
    retries: RetryConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(retries.max),
        wait=_ScopusWait(retries.backoff),
        retry=retry_if_exception(_is_retryable_scopus_error),
        reraise=True,
        sleep=sleep,
    )


def _extract_scopus_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
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
    if isinstance(value, int):
        if 1800 <= value <= 2100:
            return value
        return None
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


def _normalize_title_for_cache(value: str) -> str | None:
    if not value or not isinstance(value, str):
        return None
    text = " ".join(value.strip().lower().split())
    return text or None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _abstract_response(payload: dict) -> dict[str, Any]:
    response = payload.get("abstracts-retrieval-response")
    if not isinstance(response, dict):
        return payload
    return response


def _dedupe_non_empty_str(values: Iterable[object]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _iter_author_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    authors = response.get("authors")
    if isinstance(authors, dict):
        for entry in _ensure_list(authors.get("author")):
            if isinstance(entry, dict):
                entries.append(entry)

    dc_creator = response.get("dc:creator")
    if isinstance(dc_creator, dict):
        for entry in _ensure_list(dc_creator.get("author")):
            if isinstance(entry, dict):
                entries.append(entry)

    return entries


def _author_names_from_abstract(payload: dict) -> list[str]:
    response = _abstract_response(payload)
    names: list[str] = []
    for entry in _iter_author_entries(response):
        for key in ("ce:indexed-name", "indexed-name", "authname"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
                break
        else:
            surname = entry.get("ce:surname") or entry.get("surname")
            given = entry.get("ce:given-name") or entry.get("given-name")
            if isinstance(surname, str) and isinstance(given, str):
                combined = f"{given.strip()} {surname.strip()}".strip()
                if combined:
                    names.append(combined)
    return _dedupe_non_empty_str(names)


def _author_ids_from_abstract(payload: dict) -> list[str]:
    response = _abstract_response(payload)
    ids: list[str] = []
    for entry in _iter_author_entries(response):
        for key in ("@auid", "auid", "author-id", "authid"):
            value = entry.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                ids.append(text)
                break
    return _dedupe_non_empty_str(ids)


def _author_count_from_abstract(payload: dict, *, fallback: int = 0) -> int | None:
    response = _abstract_response(payload)
    entries = _iter_author_entries(response)
    if entries:
        return len(entries)
    return fallback or None


def _parse_open_access(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "open"}:
        return True
    if text in {"0", "false", "no", "closed"}:
        return False
    parsed = _coerce_int(text)
    if parsed in (0, 1):
        return bool(parsed)
    return None


def _normalize_issn_token(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).upper()
    cleaned = re.sub(r"[^0-9X]", "", text)
    if not cleaned:
        return None
    if len(cleaned) == 8 and _ISSN_ALLOWED_RE.match(cleaned):
        return cleaned
    return None


def _issn_list_from_abstract(payload: dict) -> list[str]:
    response = _abstract_response(payload)
    coredata = response.get("coredata")
    if not isinstance(coredata, dict):
        coredata = {}

    tokens: list[str] = []
    raw_values = (
        coredata.get("prism:issn"),
        coredata.get("issn"),
        coredata.get("prism:eIssn"),
        coredata.get("prism:eissn"),
        response.get("prism:issn"),
        response.get("issn"),
    )
    for raw in raw_values:
        if raw is None:
            continue
        if isinstance(raw, list):
            iterable = raw
        else:
            iterable = [raw]
        for item in iterable:
            if item is None:
                continue
            text = str(item)
            for part in re.split(r"[,;\s]+", text):
                normalized = _normalize_issn_token(part)
                if normalized:
                    tokens.append(normalized)
            for raw_match in re.findall(r"[0-9Xx]{4}\s*[-]?\s*[0-9Xx]{4}", text):
                normalized = _normalize_issn_token(raw_match)
                if normalized:
                    tokens.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _affiliation_metadata_from_abstract(payload: dict) -> dict[str, list[str]]:
    response = _abstract_response(payload)
    countries: list[str] = []
    institutions: list[str] = []
    cities: list[str] = []

    def collect(entry: dict[str, Any]) -> None:
        country = entry.get("affiliation-country") or entry.get("country")
        if isinstance(country, str) and country.strip():
            countries.append(country.strip())
        institution = entry.get("affilname") or entry.get("preferred-name") or entry.get("name")
        if isinstance(institution, str) and institution.strip():
            institutions.append(institution.strip())
        city = entry.get("affiliation-city") or entry.get("city")
        if isinstance(city, str) and city.strip():
            cities.append(city.strip())

    top_level_affiliations = [
        entry
        for entry in _ensure_list(response.get("affiliation"))
        if isinstance(entry, dict)
    ]
    affiliation_lookup: dict[str, dict[str, Any]] = {}
    for entry in top_level_affiliations:
        collect(entry)
        aff_id = _clean_optional_str(entry.get("@id") or entry.get("affiliation-id") or entry.get("id"))
        if aff_id:
            affiliation_lookup[aff_id] = entry

    for author in _iter_author_entries(response):
        affiliations = _ensure_list(author.get("affiliation"))
        for affiliation in affiliations:
            if not isinstance(affiliation, dict):
                continue
            collect(affiliation)
            ref_id = _clean_optional_str(
                affiliation.get("@id") or affiliation.get("affiliation-id") or affiliation.get("id")
            )
            if ref_id and ref_id in affiliation_lookup:
                collect(affiliation_lookup[ref_id])

    return {
        "countries": _dedupe_non_empty_str(countries),
        "institutions": _dedupe_non_empty_str(institutions),
        "cities": _dedupe_non_empty_str(cities),
    }


def _affiliation_countries_from_abstract(payload: dict) -> list[str]:
    return _affiliation_metadata_from_abstract(payload)["countries"]


def _keywords_from_abstract(payload: dict) -> list[str]:
    response = _abstract_response(payload)

    authkeywords = response.get("authkeywords") or {}
    if not isinstance(authkeywords, dict):
        return []

    raw = authkeywords.get("author-keyword")
    entries: list[object]
    if raw is None:
        return []
    if isinstance(raw, list):
        entries = raw
    else:
        entries = [raw]

    keywords: list[str] = []
    for entry in entries:
        if isinstance(entry, str) and entry.strip():
            keywords.append(entry.strip())
            continue
        if isinstance(entry, dict):
            value = entry.get("$") or entry.get("text") or entry.get("value")
            if isinstance(value, str) and value.strip():
                keywords.append(value.strip())

    return _dedupe_non_empty_str(keywords)


def _extract_enrichment(payload: dict) -> dict[str, Any] | None:
    response = _abstract_response(payload)

    coredata = response.get("coredata") or {}
    if not isinstance(coredata, dict):
        coredata = {}

    scopus_id = _extract_scopus_id(coredata.get("dc:identifier") or response.get("dc:identifier"))
    if not scopus_id:
        scopus_id = _extract_scopus_id(response.get("scopus_id"))

    eid = coredata.get("eid") or coredata.get("prism:eid") or response.get("eid")
    eid = eid.strip() if isinstance(eid, str) and eid.strip() else None

    cover_date = coredata.get("prism:coverDate") or coredata.get("coverDate")
    publication_year = _parse_year(cover_date)

    publication_name = _safe_str(coredata.get("prism:publicationName") or coredata.get("publicationName"))
    journal_title = publication_name

    issn = normalize_issn(coredata.get("prism:issn") or coredata.get("issn"))
    eissn = normalize_issn(coredata.get("prism:eIssn") or coredata.get("prism:eissn"))
    issn_list = _issn_list_from_abstract(payload)
    source_id = _safe_str(coredata.get("source-id") or response.get("source-id"))
    publisher = _safe_str(
        coredata.get("publishername")
        or coredata.get("dc:publisher")
        or coredata.get("publisher")
        or response.get("publishername")
        or response.get("dc:publisher")
        or response.get("publisher")
    )
    open_access_raw: object = None
    for mapping, keys in (
        (coredata, ("openaccessFlag", "openaccess")),
        (response, ("openaccessFlag", "openaccess")),
    ):
        for key in keys:
            if key in mapping:
                open_access_raw = mapping.get(key)
                break
        if open_access_raw is not None:
            break
    open_access = _parse_open_access(open_access_raw)

    cited_by_count_raw: object = None
    for mapping, keys in (
        (coredata, ("citedby-count", "citedby_count")),
        (response, ("citedby-count", "citedby_count")),
    ):
        for key in keys:
            if key in mapping:
                cited_by_count_raw = mapping.get(key)
                break
        if cited_by_count_raw is not None:
            break
    cited_by_count = _coerce_int(cited_by_count_raw)

    abstract = coredata.get("dc:description") or coredata.get("description")
    if not isinstance(abstract, str) or not abstract.strip():
        abstract = None

    subtype_code = _safe_str(coredata.get("subtype") or response.get("subtype"))
    subtype_description = _safe_str(
        coredata.get("subtypeDescription")
        or coredata.get("subtype-description")
        or response.get("subtypeDescription")
        or response.get("subtype-description")
    )
    doc_type = subtype_description or subtype_code

    authors = _author_names_from_abstract(payload)
    author_ids = _author_ids_from_abstract(payload)
    author_count = _author_count_from_abstract(payload, fallback=max(len(authors), len(author_ids)))
    keywords = _keywords_from_abstract(payload)
    affiliation = _affiliation_metadata_from_abstract(payload)
    affiliation_countries = affiliation["countries"]
    affiliation_institutions = affiliation["institutions"]
    affiliation_cities = affiliation["cities"]

    has_any = any(
        value is not None
        for value in (
            scopus_id,
            issn,
            eissn,
            publication_year,
            cited_by_count,
            abstract,
            doc_type,
            open_access,
            source_id,
            publisher,
            journal_title,
            author_count,
        )
    ) or bool(
        authors
        or author_ids
        or keywords
        or affiliation_countries
        or affiliation_institutions
        or affiliation_cities
        or issn_list
    )
    if not has_any:
        return None

    abstract_block = {
        "scopus_id": scopus_id,
        "eid": eid,
        "publication_name": publication_name,
        "journal_title": journal_title,
        "issn": issn,
        "eissn": eissn,
        "issn_list": issn_list or None,
        "publication_year": publication_year,
        "citedby_count": cited_by_count,
        "doc_type": doc_type,
        "subtype": subtype_description,
        "subtype_code": subtype_code,
        "source_id": source_id,
        "publisher": publisher,
        "is_open_access": open_access,
        "keywords": keywords or None,
        "abstract": abstract,
        "authors": authors or None,
        "author_ids": author_ids or None,
        "author_count": author_count,
        "affiliation_countries": affiliation_countries or None,
        "affiliation_institutions": affiliation_institutions or None,
        "affiliation_cities": affiliation_cities or None,
    }

    # Minimal payload for cache/merge; avoid storing whole raw JSON.
    return {
        "scopus_id": scopus_id,
        "source_id": source_id,
        "issn": issn,
        "eissn": eissn,
        "issn_list": issn_list or None,
        "publication_year": publication_year,
        "cited_by_count": cited_by_count,
        "is_oa": open_access,
        "is_open_access": open_access,
        "doc_type": doc_type,
        "journal_title": journal_title,
        "publisher": publisher,
        "authors": authors or None,
        "author_ids": author_ids or None,
        "author_count": author_count,
        "affiliation_countries": affiliation_countries or None,
        "affiliation_institutions": affiliation_institutions or None,
        "affiliation_cities": affiliation_cities or None,
        "scopus": {
            "abstract": abstract_block,
        },
        "scopus_enrich": {
            "scopus_id": scopus_id,
            "cover_date": (
                cover_date if isinstance(cover_date, str) and cover_date.strip() else None
            ),
            "subtype": subtype_description,
            "keywords": keywords or None,
            "abstract": abstract,
            "doc_type": doc_type,
            "is_open_access": open_access,
            "journal_title": journal_title,
            "source_id": source_id,
            "publisher": publisher,
            "issn_list": issn_list or None,
            "author_ids": author_ids or None,
            "author_count": author_count,
            "affiliation_countries": affiliation_countries or None,
            "affiliation_institutions": affiliation_institutions or None,
            "affiliation_cities": affiliation_cities or None,
        },
    }


def _safe_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _first_scopus_entry(payload: dict) -> dict | None:
    results = payload.get("search-results") or {}
    if not isinstance(results, dict):
        return None
    entries = results.get("entry") or []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    return first if isinstance(first, dict) else None


def _ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _latest_year_value(entries: object) -> dict[str, object] | None:
    best_year: int | None = None
    best_value: float | None = None
    for item in _ensure_list(entries):
        if not isinstance(item, dict):
            continue
        year = _parse_year(item.get("@year") or item.get("year"))
        value = _coerce_float(item.get("$") or item.get("value"))
        if year is None or value is None:
            continue
        if best_year is None or year > best_year:
            best_year = year
            best_value = value
    if best_year is None or best_value is None:
        return None
    return {"year": best_year, "value": best_value}


def _extract_serial_metrics(payload: dict) -> dict[str, Any] | None:
    response = payload.get("serial-metadata-response")
    if not isinstance(response, dict):
        response = payload

    entry = response.get("entry")
    first = None
    for item in _ensure_list(entry):
        if isinstance(item, dict):
            first = item
            break
    if not isinstance(first, dict):
        return None

    metrics: dict[str, Any] = {}

    snip_list = first.get("SNIPList")
    if isinstance(snip_list, dict):
        snip = _latest_year_value(snip_list.get("SNIP"))
        if snip:
            metrics["snip"] = snip

    sjr_list = first.get("SJRList")
    if isinstance(sjr_list, dict):
        sjr = _latest_year_value(sjr_list.get("SJR"))
        if sjr:
            metrics["sjr"] = sjr

    citescore_list = first.get("citeScoreYearInfoList")
    citescore_best: dict[str, object] | None = None
    if isinstance(citescore_list, dict):
        infos = citescore_list.get("citeScoreYearInfo")
        for info in _ensure_list(infos):
            if not isinstance(info, dict):
                continue
            pairs = [
                (info.get("citeScoreCurrentMetricYear"), info.get("citeScoreCurrentMetric")),
                (info.get("citeScoreTrackerYear"), info.get("citeScoreTracker")),
            ]
            for year_raw, value_raw in pairs:
                year = _parse_year(year_raw)
                value = _coerce_float(value_raw)
                if year is None or value is None:
                    continue
                current_year = citescore_best.get("year") if citescore_best else None
                if citescore_best is None or (
                    isinstance(current_year, int) and current_year < year
                ):
                    citescore_best = {"year": year, "value": value}
    if citescore_best:
        metrics["citescore"] = citescore_best

    return metrics or None


def _extract_citation_overview(payload: dict, *, exclude_self: bool) -> dict[str, Any] | None:
    response = payload.get("abstract-citations-response")
    if not isinstance(response, dict):
        response = payload

    cct = response.get("citeColumnTotalXML")
    if not isinstance(cct, dict):
        return None

    years_raw = cct.get("columnHeading")
    totals_raw = cct.get("columnTotal")
    years = [str(item).strip() for item in _ensure_list(years_raw)]
    totals = [_coerce_int(item) for item in _ensure_list(totals_raw)]

    by_year: dict[int, int] = {}
    for year_text, count in zip(years, totals, strict=False):
        year = _parse_year(year_text)
        if year is None or count is None:
            continue
        by_year[int(year)] = int(count)

    return {
        "exclude_self": bool(exclude_self),
        "by_year": by_year or None,
        "range_total": _coerce_int(cct.get("rangeColumnTotal")),
        "grand_total": _coerce_int(cct.get("grandTotal")),
    }


def _extract_plumx(payload: dict) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    categories = payload.get("count_categories")
    if categories is None:
        return None

    category_totals: dict[str, int] = {}
    for entry in _ensure_list(categories):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        total = _coerce_int(entry.get("total"))
        if total is None:
            continue
        category_totals[name.strip()] = int(total)

    if not category_totals:
        return None
    return {
        "id_type": payload.get("id_type"),
        "id_value": payload.get("id_value"),
        "categories": category_totals,
    }


class ScopusEnricher:
    def __init__(
        self,
        config: ScopusEnrichConfig,
        *,
        retries: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_base_url = (os.getenv("SCOPUS_BASE_URL") or config.base_url).strip() or BASE_URL
        resolved_timeout = _env_float("SCOPUS_TIMEOUT", config.timeout)
        resolved_rps = _env_float("SCOPUS_RPS", config.rps)
        resolved_fail_open = _env_bool("SCOPUS_FAIL_OPEN", bool(config.fail_open))
        resolved_abstract_view = _normalize_view(
            os.getenv("SCOPUS_ABSTRACT_VIEW", config.abstract_view)
        ) or "FULL"
        resolved_search_view = _clean_optional_str(
            os.getenv("SCOPUS_SEARCH_VIEW", config.search_view)
        ) or "STANDARD"
        resolved_search_fields = os.getenv("SCOPUS_SEARCH_FIELDS", config.search_fields)
        resolved_abstract_fields = os.getenv("SCOPUS_ABSTRACT_FIELDS", config.abstract_fields)
        resolved_abstract_fields_minimal = os.getenv(
            "SCOPUS_ABSTRACT_FIELDS_MINIMAL", config.abstract_fields_minimal
        )
        resolved_fallback_views = _parse_fallback_views(
            os.getenv("SCOPUS_ABSTRACT_FALLBACK_VIEWS"),
            list(config.abstract_fallback_views),
        )
        resolved_search_fields = _clean_optional_str(resolved_search_fields)
        resolved_abstract_fields = _clean_optional_str(resolved_abstract_fields)
        resolved_abstract_fields_minimal = _clean_optional_str(resolved_abstract_fields_minimal)

        max_retries = (
            config.max_retries
            if isinstance(config.max_retries, int)
            else (retries or RetryConfig()).max
        )
        resolved_max_retries = _env_int("SCOPUS_MAX_RETRIES", max_retries)

        self.config = config.model_copy(
            update={
                "fail_open": resolved_fail_open,
                "base_url": resolved_base_url,
                "timeout": resolved_timeout,
                "rps": resolved_rps,
                "max_retries": resolved_max_retries,
                "abstract_view": resolved_abstract_view,
                "search_view": resolved_search_view,
                "search_fields": resolved_search_fields,
                "abstract_fields": resolved_abstract_fields,
                "abstract_fallback_views": resolved_fallback_views,
                "abstract_fields_minimal": resolved_abstract_fields_minimal,
            }
        )
        effective_retries = retries or RetryConfig()
        self.retries = effective_retries.model_copy(update={"max": resolved_max_retries})
        self.limiter = rate_limiter or RateLimiter(rate=self.config.rps, burst=self.config.burst)
        self.client = client
        self.sleep = sleep
        self._cache_hits = 0
        self._cache_misses = 0
        self._disabled_reason: str | None = None
        self._disabled_records = 0

    def enrich(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        if not self.config.enabled:
            return records

        api_key = _resolve_api_key()
        if not api_key:
            message = (
                "Scopus enrichment enabled but SCOPUS_API_KEY/ELSEVIER_API_KEY is missing. "
                "Set X-ELS-APIKey credentials in environment."
            )
            if self.config.fail_open:
                LOGGER.warning("%s Skipping due to fail_open=true.", message)
                return records
            raise ScopusAuthError(message)

        inst_token = _resolve_inst_token()
        headers: dict[str, str] = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        }
        if inst_token:
            headers["X-ELS-Insttoken"] = inst_token

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ttl_seconds = int(self.config.cache_ttl_days) * 24 * 60 * 60
        retrying = _scopus_retrying(self.retries, sleep=self.sleep)

        enriched: list[dict[str, Any]] = []
        with Cache(CACHE_DIR) as cache:
            client = self.client or httpx.Client(
                base_url=self.config.base_url or BASE_URL,
                timeout=self.config.timeout,
                follow_redirects=True,
            )
            owns_client = self.client is None
            try:
                for record in records:
                    if self._disabled_reason and self.config.fail_open:
                        self._disabled_records += 1
                        enriched.append(
                            self._record_with_error_meta(
                                record,
                                error_type="skipped_after_auth_error",
                                status_code=None,
                                message=self._disabled_reason,
                                path=None,
                                params=None,
                                traces=None,
                            )
                        )
                        continue
                    try:
                        enriched.append(
                            self._enrich_one(
                                record,
                                cache=cache,
                                client=client,
                                retrying=retrying,
                                headers=headers,
                                ttl_seconds=ttl_seconds,
                            )
                        )
                    except (ScopusEnrichError, httpx.HTTPError) as exc:
                        if self.config.fail_open:
                            if isinstance(exc, (ScopusAuthError, ScopusPermissionError)):
                                enriched.append(
                                    self._record_with_error_meta(
                                        record,
                                        error_type=(
                                            "auth_error"
                                            if isinstance(exc, ScopusAuthError)
                                            else "permission_error"
                                        ),
                                        status_code=exc.status_code,
                                        message=str(exc),
                                        path=exc.path,
                                        params=exc.params,
                                        traces=exc.traces,
                                        attempts=exc.attempts,
                                    )
                                )
                                if not self._disabled_reason:
                                    self._disabled_reason = str(exc)
                                    LOGGER.warning(
                                        "Scopus enrichment disabled for remaining records "
                                        "in this run due to auth/permission error: %s",
                                        exc,
                                    )
                            else:
                                LOGGER.warning(
                                    "Scopus enrichment failed record rank=%s error=%s",
                                    record.get("rank"),
                                    exc,
                                )
                                enriched.append(
                                    self._record_with_error_meta(
                                        record,
                                        error_type="request_error",
                                        status_code=None,
                                        message=str(exc),
                                        path=None,
                                        params=None,
                                        traces=None,
                                    )
                                )
                            continue
                        raise
            finally:
                if owns_client:
                    try:
                        client.close()
                    except Exception:
                        pass
        LOGGER.info(
            "Scopus enrichment finished "
            "(records=%d cache_hits=%d cache_misses=%d disabled_records=%d)",
            len(records),
            self._cache_hits,
            self._cache_misses,
            self._disabled_records,
        )
        return enriched

    def _record_with_error_meta(
        self,
        record: Dict[str, Any],
        *,
        error_type: str,
        status_code: int | None,
        message: str | None,
        path: str | None,
        params: dict[str, Any] | None,
        traces: dict[str, str] | None,
        attempts: list[dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        updated = dict(record)
        current_meta = updated.get("scopus_meta")
        if not isinstance(current_meta, dict):
            current_meta = {}
        error_payload: dict[str, Any] = {
            "status": error_type,
            "status_code": status_code,
            "message": message,
            "endpoint": path,
            "params": params or None,
            "x-els-reqid": (traces or {}).get("x-els-reqid"),
            "x-els-transid": (traces or {}).get("x-els-transid"),
            "attempts": attempts or None,
        }
        updated["scopus_meta"] = {**current_meta, **error_payload}
        return updated

    def _options_cache_key(self) -> str:
        return "|".join(
            [
                f"view:{self.config.abstract_view}",
                f"abstract_fields:{self.config.abstract_fields or ''}",
                "abstract_fallback_views:"
                + ",".join([value if value is not None else "None" for value in self.config.abstract_fallback_views]),
                f"abstract_fields_minimal:{self.config.abstract_fields_minimal or ''}",
                f"search_view:{self.config.search_view}",
                f"search_fields:{self.config.search_fields or ''}",
                f"search_sort:{self.config.search_sort or ''}",
                f"citov:{int(bool(self.config.enable_citation_overview))}",
                f"citov_exclself:{int(bool(self.config.citation_overview_exclude_self))}",
                f"serial:{int(bool(self.config.enable_serial_title_metrics))}",
                f"serial_view:{self.config.serial_title_view}",
                f"plumx:{int(bool(self.config.enable_plumx))}",
            ]
        )

    def _enrich_one(
        self,
        record: Dict[str, Any],
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> Dict[str, Any]:
        doi = normalise_doi(record.get("doi"))  # type: ignore[arg-type]
        if not doi:
            url = record.get("url")
            doi = doi_from_url(url) if isinstance(url, str) else None

        existing_scopus_id = _extract_scopus_id(record.get("scopus_id"))
        if not existing_scopus_id:
            scopus_block = record.get("scopus")
            if isinstance(scopus_block, dict):
                abstract = scopus_block.get("abstract")
                if isinstance(abstract, dict):
                    existing_scopus_id = _extract_scopus_id(abstract.get("scopus_id"))

        cache_key = None
        if doi:
            cache_key = f"doi:{doi}"
        elif existing_scopus_id:
            cache_key = f"scopus_id:{existing_scopus_id}"
        else:
            title = record.get("title")
            if self.config.title_search_enabled and isinstance(title, str):
                title = title.strip()
                if len(title) >= self.config.title_search_min_len:
                    normalized = _normalize_title_for_cache(title)
                    if normalized:
                        cache_key = f"title:{normalized}"

        if not cache_key:
            return record

        cache_key = f"{cache_key}|{self._options_cache_key()}"

        cached = cache.get(cache_key, default=_CACHE_MISSING)
        if cached is not _CACHE_MISSING:
            payload = cached
            cache_hit = True
            self._cache_hits += 1
        else:
            try:
                payload = self._lookup(
                    record,
                    doi=doi,
                    scopus_id=existing_scopus_id,
                    cache=cache,
                    client=client,
                    retrying=retrying,
                    headers=headers,
                    ttl_seconds=ttl_seconds,
                )
            except (ScopusEnrichError, httpx.HTTPError):
                raise
            cache.set(cache_key, payload, expire=ttl_seconds)
            cache_hit = False
            self._cache_misses += 1

        if not isinstance(payload, dict) or not payload:
            return record
        payload = dict(payload)
        meta = payload.get("scopus_meta")
        if not isinstance(meta, dict):
            meta = {}
        payload["scopus_meta"] = {**meta, "cache_hit": cache_hit, "cache_key": cache_key}
        return self._merge(record, payload)

    def _apply_abstract_fetch_meta(
        self,
        payload: dict[str, Any] | None,
        fetch_meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return payload

        view_used = fetch_meta.get("view_used")
        if isinstance(view_used, str):
            payload["scopus_enrich_view_used"] = view_used
        else:
            payload["scopus_enrich_view_used"] = None

        field_used = fetch_meta.get("field_used")
        payload["scopus_enrich_field_used"] = field_used if isinstance(field_used, str) else None
        payload["scopus_enrich_downgraded"] = bool(fetch_meta.get("downgraded"))

        meta = payload.get("scopus_meta")
        if not isinstance(meta, dict):
            meta = {}
        payload["scopus_meta"] = {
            **meta,
            "abstract_view_used": payload.get("scopus_enrich_view_used"),
            "abstract_field_used": payload.get("scopus_enrich_field_used"),
            "abstract_downgraded": payload.get("scopus_enrich_downgraded"),
            "abstract_attempts": fetch_meta.get("attempts"),
        }
        return payload

    def _lookup(
        self,
        record: Dict[str, Any],
        *,
        doi: str | None,
        scopus_id: str | None,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> dict[str, Any] | None:
        if scopus_id:
            abstract, fetch_meta = self._get_abstract_by_scopus_id(
                scopus_id,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            payload = _extract_enrichment(abstract) if abstract else None
            payload = self._apply_abstract_fetch_meta(payload, fetch_meta)
            if isinstance(payload, dict):
                meta = payload.get("scopus_meta")
                if not isinstance(meta, dict):
                    meta = {}
                payload["scopus_meta"] = {
                    **meta,
                    "method": "scopus_id",
                    "identifiers_used": ["scopus_id"],
                }
            return self._extend_payload(
                payload,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
                doi=doi,
            )

        title = record.get("title")
        title_text = title.strip() if isinstance(title, str) else None

        if doi:
            abstract, fetch_meta = self._get_abstract_by_doi(
                doi,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            if abstract:
                payload = _extract_enrichment(abstract)
                payload = self._apply_abstract_fetch_meta(payload, fetch_meta)
                if isinstance(payload, dict):
                    meta = payload.get("scopus_meta")
                    if not isinstance(meta, dict):
                        meta = {}
                    payload["scopus_meta"] = {**meta, "method": "doi", "identifiers_used": ["doi"]}
                return self._extend_payload(
                    payload,
                    cache=cache,
                    client=client,
                    retrying=retrying,
                    headers=headers,
                    ttl_seconds=ttl_seconds,
                    doi=doi,
                )

            found_id = self._search_scopus_id(
                query=f"DOI({doi})",
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            if not found_id:
                return None
            abstract, fetch_meta = self._get_abstract_by_scopus_id(
                found_id,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            payload = _extract_enrichment(abstract) if abstract else None
            payload = self._apply_abstract_fetch_meta(payload, fetch_meta)
            if isinstance(payload, dict):
                meta = payload.get("scopus_meta")
                if not isinstance(meta, dict):
                    meta = {}
                payload["scopus_meta"] = {
                    **meta,
                    "method": "doi_search",
                    "identifiers_used": ["doi", "scopus_search"],
                }
            return self._extend_payload(
                payload,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
                doi=doi,
            )

        if (
            self.config.title_search_enabled
            and title_text
            and len(title_text) >= self.config.title_search_min_len
        ):
            safe_title = re.sub(r"[\\r\\n\\t\\\"]+", " ", title_text).strip()
            found_id = self._search_scopus_id(
                query=f'TITLE("{safe_title}")',
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            if not found_id:
                return None
            abstract, fetch_meta = self._get_abstract_by_scopus_id(
                found_id,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            payload = _extract_enrichment(abstract) if abstract else None
            payload = self._apply_abstract_fetch_meta(payload, fetch_meta)
            if isinstance(payload, dict):
                meta = payload.get("scopus_meta")
                if not isinstance(meta, dict):
                    meta = {}
                payload["scopus_meta"] = {
                    **meta,
                    "method": "title_search",
                    "identifiers_used": ["title_search"],
                }
            return self._extend_payload(
                payload,
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
                doi=None,
            )

        return None

    def _extend_payload(
        self,
        payload: dict[str, Any] | None,
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
        doi: str | None,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict) or not payload:
            return None

        scopus = payload.get("scopus")
        if not isinstance(scopus, dict):
            scopus = {}

        abstract = scopus.get("abstract")
        scopus_id = payload.get("scopus_id")
        if not isinstance(scopus_id, str) or not scopus_id.strip():
            if isinstance(abstract, dict):
                scopus_id = abstract.get("scopus_id")

        issn = payload.get("issn")
        if not isinstance(issn, str) or not issn.strip():
            if isinstance(abstract, dict):
                issn = abstract.get("issn")

        if (
            self.config.enable_citation_overview
            and isinstance(scopus_id, str)
            and scopus_id.strip()
        ):
            overview = self._get_citation_overview(
                scopus_id.strip(),
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            if overview:
                parsed = _extract_citation_overview(
                    overview, exclude_self=bool(self.config.citation_overview_exclude_self)
                )
                if parsed:
                    scopus["citation_overview"] = parsed

        if self.config.enable_serial_title_metrics and isinstance(issn, str) and issn.strip():
            serial = self._get_serial_title(
                issn.strip(),
                cache=cache,
                client=client,
                retrying=retrying,
                headers=headers,
                ttl_seconds=ttl_seconds,
            )
            if serial:
                parsed = _extract_serial_metrics(serial)
                if parsed:
                    scopus["serial_metrics"] = parsed

        if self.config.enable_plumx:
            id_type = None
            id_value = None
            if doi:
                id_type = "doi"
                id_value = doi
            elif isinstance(scopus_id, str) and scopus_id.strip():
                id_type = "scopus-id"
                id_value = scopus_id.strip()

            if id_type and id_value:
                plumx = self._get_plumx(
                    id_type,
                    id_value,
                    cache=cache,
                    client=client,
                    retrying=retrying,
                    headers=headers,
                    ttl_seconds=ttl_seconds,
                )
                if plumx:
                    parsed = _extract_plumx(plumx)
                    if parsed:
                        scopus["plumx"] = parsed

        payload["scopus"] = scopus
        return payload

    def _get_json_optional(
        self,
        path: str,
        *,
        params: dict[str, Any] | None,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        allow_statuses: Iterable[int],
    ) -> dict[str, Any] | None:
        allow = set(allow_statuses)

        def execute() -> dict[str, Any] | None:
            self.limiter.acquire()
            LOGGER.debug(
                "scopus enrich request path=%s params=%s headers=%s",
                path,
                params,
                mask_sensitive(headers),
            )
            resp = client.get(path, params=params, headers=headers)
            traces = _trace_headers(resp)
            _log_scopus_response(path, resp.status_code, traces)
            if resp.status_code in allow:
                return None
            if resp.status_code == 401:
                has_insttoken = bool(headers.get("X-ELS-Insttoken"))
                inst_hint = (
                    "X-ELS-Insttoken header was sent."
                    if has_insttoken
                    else "Set SCOPUS_INSTTOKEN if your account requires institutional token auth."
                )
                LOGGER.warning(
                    "Scopus auth failure endpoint=%s status=401 "
                    "diagnostic='permission/entitlement or missing InstToken/IP' "
                    "view=%s field=%s x-els-reqid=%s x-els-transid=%s",
                    path,
                    (params or {}).get("view"),
                    (params or {}).get("field"),
                    traces.get("x-els-reqid", "-"),
                    traces.get("x-els-transid", "-"),
                )
                raise ScopusAuthError(
                    "Scopus returned 401 Unauthorized (permission/entitlement or missing "
                    "InstToken/IP). Verify SCOPUS_API_KEY and institutional access. "
                    f"{inst_hint}",
                    status_code=401,
                    path=path,
                    params=params,
                    traces=traces,
                )
            if resp.status_code == 403:
                LOGGER.warning(
                    "Scopus permission failure endpoint=%s status=403 "
                    "diagnostic='permission/entitlement or missing InstToken/IP' "
                    "view=%s field=%s x-els-reqid=%s x-els-transid=%s",
                    path,
                    (params or {}).get("view"),
                    (params or {}).get("field"),
                    traces.get("x-els-reqid", "-"),
                    traces.get("x-els-transid", "-"),
                )
                raise ScopusPermissionError(
                    "Scopus returned 403 Forbidden (permission/entitlement or missing "
                    "InstToken/IP). Confirm institutional access/service level and, if needed, "
                    "configure SCOPUS_INSTTOKEN.",
                    status_code=403,
                    path=path,
                    params=params,
                    traces=traces,
                )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None

        return retrying(execute)

    def _get_json_optional_cached(
        self,
        cache: Cache,
        *,
        cache_key: str,
        path: str,
        params: dict[str, Any] | None,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        allow_statuses: Iterable[int],
        ttl_seconds: int,
    ) -> dict[str, Any] | None:
        cached = cache.get(cache_key, default=_CACHE_MISSING)
        if cached is not _CACHE_MISSING:
            return cached if isinstance(cached, dict) else None
        data = self._get_json_optional(
            path,
            params=params,
            client=client,
            retrying=retrying,
            headers=headers,
            allow_statuses=allow_statuses,
        )
        cache.set(cache_key, data, expire=ttl_seconds)
        return data

    def _search_scopus_id(
        self,
        *,
        query: str,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> str | None:
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
        params: dict[str, Any] = {
            "query": query,
            "count": 1,
            "start": 0,
            "view": self.config.search_view,
        }
        if self.config.search_fields:
            params["field"] = self.config.search_fields
        if self.config.search_sort:
            params["sort"] = self.config.search_sort
        payload = self._get_json_optional_cached(
            cache,
            cache_key=(
                f"search:scopus:{digest}|view:{self.config.search_view}|"
                f"field:{self.config.search_fields or ''}|sort:{self.config.search_sort or ''}"
            ),
            path="/content/search/scopus",
            params=params,
            client=client,
            retrying=retrying,
            headers=headers,
            allow_statuses=(404,),
            ttl_seconds=ttl_seconds,
        )
        if not payload:
            return None
        entry = _first_scopus_entry(payload)
        if not entry:
            return None
        return _extract_scopus_id(
            entry.get("dc:identifier") or entry.get("eid") or entry.get("scopus_id")
        )

    def _abstract_attempts(self) -> list[dict[str, Any]]:
        primary_view = _normalize_view(self.config.abstract_view)
        primary_field = _clean_optional_str(self.config.abstract_fields)
        fallback_views = [_normalize_view(value) for value in self.config.abstract_fallback_views]
        minimal_field = _clean_optional_str(self.config.abstract_fields_minimal)

        attempts: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None]] = set()

        def add(view: str | None, field: str | None) -> None:
            key = (view, field)
            if key in seen:
                return
            seen.add(key)
            attempts.append(
                {
                    "view": view,
                    "field": field,
                    "downgraded": key != (primary_view, primary_field),
                }
            )

        add(primary_view, primary_field)
        for fallback_view in fallback_views:
            add(fallback_view, primary_field)
        if minimal_field:
            add(primary_view, minimal_field)
            for fallback_view in fallback_views:
                add(fallback_view, minimal_field)

        return attempts

    def _abstract_params(self, *, view: str | None, field: str | None) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if view is not None:
            params["view"] = view
        if field:
            params["field"] = field
        return params or None

    def _get_abstract_with_fallback(
        self,
        *,
        cache: Cache,
        cache_key_prefix: str,
        path: str,
        allow_statuses: Iterable[int],
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        attempts = self._abstract_attempts()
        attempt_history: list[dict[str, Any]] = []
        last_auth_error: ScopusResponseError | None = None

        for attempt in attempts:
            view = attempt.get("view")
            field = attempt.get("field")
            params = self._abstract_params(view=view, field=field)
            cache_key = f"{cache_key_prefix}|view:{view or ''}|field:{field or ''}"
            try:
                payload = self._get_json_optional_cached(
                    cache,
                    cache_key=cache_key,
                    path=path,
                    params=params,
                    client=client,
                    retrying=retrying,
                    headers=headers,
                    allow_statuses=allow_statuses,
                    ttl_seconds=ttl_seconds,
                )
            except (ScopusAuthError, ScopusPermissionError) as exc:
                last_auth_error = exc
                attempt_history.append(
                    {
                        "view": view,
                        "field": field,
                        "status_code": exc.status_code,
                        "x-els-reqid": exc.traces.get("x-els-reqid"),
                        "x-els-transid": exc.traces.get("x-els-transid"),
                    }
                )
                continue

            attempt_history.append({"view": view, "field": field, "status_code": 200})
            return (
                payload,
                {
                    "view_used": view,
                    "field_used": field,
                    "downgraded": bool(attempt.get("downgraded")),
                    "attempts": attempt_history,
                },
            )

        if isinstance(last_auth_error, ScopusResponseError):
            summary = ", ".join(
                [
                    f"view={entry.get('view')!r}|field={entry.get('field')!r}|status={entry.get('status_code')}"
                    for entry in attempt_history
                ]
            )
            message = (
                "Scopus Abstract Retrieval failed after fallback attempts "
                "(permission/entitlement or missing InstToken/IP). "
                f"Tried: {summary}"
            )
            error_cls: type[ScopusResponseError] = (
                ScopusPermissionError
                if isinstance(last_auth_error, ScopusPermissionError)
                else ScopusAuthError
            )
            raise error_cls(
                message,
                status_code=last_auth_error.status_code,
                path=last_auth_error.path,
                params=last_auth_error.params,
                traces=last_auth_error.traces,
                attempts=attempt_history,
            ) from last_auth_error

        return (
            None,
            {
                "view_used": None,
                "field_used": None,
                "downgraded": False,
                "attempts": attempt_history,
            },
        )

    def _get_abstract_by_scopus_id(
        self,
        scopus_id: str,
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return self._get_abstract_with_fallback(
            cache=cache,
            cache_key_prefix=f"abstract:scopus_id:{scopus_id}",
            path=f"/content/abstract/scopus_id/{quote(scopus_id, safe='')}",
            allow_statuses=(404,),
            client=client,
            retrying=retrying,
            headers=headers,
            ttl_seconds=ttl_seconds,
        )

    def _get_abstract_by_doi(
        self,
        doi: str,
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        # DOI contains a slash, so it must be percent-encoded in the URL path.
        doi_path = quote(doi, safe="")
        return self._get_abstract_with_fallback(
            cache=cache,
            cache_key_prefix=f"abstract:doi:{doi}",
            path=f"/content/abstract/doi/{doi_path}",
            allow_statuses=(400, 404),
            client=client,
            retrying=retrying,
            headers=headers,
            ttl_seconds=ttl_seconds,
        )

    def _get_citation_overview(
        self,
        scopus_id: str,
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {"scopus_id": scopus_id}
        if self.config.citation_overview_exclude_self:
            params["citation"] = "exclude-self"
        return self._get_json_optional_cached(
            cache,
            cache_key=f"citation_overview:scopus_id:{scopus_id}|exclude_self:{int(bool(self.config.citation_overview_exclude_self))}",
            path="/content/abstract/citations",
            params=params,
            client=client,
            retrying=retrying,
            headers=headers,
            allow_statuses=(400, 403, 404),
            ttl_seconds=ttl_seconds,
        )

    def _get_serial_title(
        self,
        issn: str,
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> dict[str, Any] | None:
        safe_issn = quote(issn, safe="")
        return self._get_json_optional_cached(
            cache,
            cache_key=f"serial_title:issn:{issn}|view:{self.config.serial_title_view}",
            path=f"/content/serial/title/issn/{safe_issn}",
            params={"view": self.config.serial_title_view},
            client=client,
            retrying=retrying,
            headers=headers,
            allow_statuses=(400, 403, 404),
            ttl_seconds=ttl_seconds,
        )

    def _get_plumx(
        self,
        id_type: str,
        id_value: str,
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> dict[str, Any] | None:
        safe_value = quote(id_value, safe="")
        return self._get_json_optional_cached(
            cache,
            cache_key=f"plumx:{id_type}:{id_value}",
            path=f"/analytics/plumx/{quote(id_type, safe='')}/{safe_value}",
            params=None,
            client=client,
            retrying=retrying,
            headers=headers,
            allow_statuses=(400, 403, 404),
            ttl_seconds=ttl_seconds,
        )

    def _merge(self, record: Dict[str, Any], payload: dict[str, Any]) -> Dict[str, Any]:
        overwrite = bool(self.config.overwrite_existing)
        prefer_issn = bool(self.config.prefer_issn_from_scopus)

        updated: dict[str, Any] = dict(record)

        for key in (
            "scopus_id",
            "publication_year",
            "cited_by_count",
            "authors",
            "source_id",
            "publisher",
            "journal_title",
            "doc_type",
            "is_oa",
            "is_open_access",
            "author_ids",
            "author_count",
            "affiliation_countries",
            "affiliation_institutions",
            "affiliation_cities",
            "issn_list",
            "scopus_enrich_view_used",
            "scopus_enrich_field_used",
            "scopus_enrich_downgraded",
        ):
            value = payload.get(key)
            if value is None:
                continue
            if overwrite or _is_missing(updated.get(key)):
                updated[key] = value

        for key in ("issn", "eissn"):
            value = payload.get(key)
            if value is None:
                continue
            current = updated.get(key)
            if _is_missing(current):
                updated[key] = value
            elif overwrite and prefer_issn:
                updated[key] = value

        scopus_payload = payload.get("scopus")
        if isinstance(scopus_payload, dict):
            current_scopus = updated.get("scopus")
            if not isinstance(current_scopus, dict):
                current_scopus = {}
            merged_scopus = dict(current_scopus)
            for section, section_payload in scopus_payload.items():
                if section_payload is None:
                    continue
                current_section = merged_scopus.get(section)
                if overwrite or _is_missing(current_section):
                    merged_scopus[section] = section_payload
                    continue
                if isinstance(current_section, dict) and isinstance(section_payload, dict):
                    merged_section = dict(current_section)
                    for key, value in section_payload.items():
                        if overwrite or _is_missing(merged_section.get(key)):
                            merged_section[key] = value
                    merged_scopus[section] = merged_section
            updated["scopus"] = merged_scopus

        meta_payload = payload.get("scopus_meta")
        if isinstance(meta_payload, dict):
            current_meta = updated.get("scopus_meta")
            if not isinstance(current_meta, dict):
                current_meta = {}
            updated["scopus_meta"] = {**current_meta, **meta_payload}

        prev_extra = updated.get("extra") or {}
        if not isinstance(prev_extra, dict):
            prev_extra = {}
        scopus_enrich = payload.get("scopus_enrich")
        if isinstance(scopus_enrich, dict):
            if overwrite or "scopus_enrich" not in prev_extra:
                prev_extra = {**prev_extra, "scopus_enrich": scopus_enrich}
        updated["extra"] = prev_extra

        return updated


def enrich_with_scopus(
    records: List[Dict[str, Any]],
    config: ScopusEnrichConfig,
    *,
    retries: RetryConfig | None = None,
    rate_limiter: RateLimiter | None = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper mirroring the OpenAlex enrichment API."""

    enricher = ScopusEnricher(config, retries=retries, rate_limiter=rate_limiter)
    return enricher.enrich(records)
