"""Scopus Serial Title enrichment for journal ranking metrics."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote

import httpx
from diskcache import Cache
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.config import RetryConfig, ScopusRankingConfig
from ai_bias_search.utils.logging import configure_logging, mask_sensitive
from ai_bias_search.utils.rate_limit import RateLimiter

LOGGER = configure_logging()

BASE_URL = "https://api.elsevier.com"
CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "cache" / "scopus_rankings"
).resolve()
_CACHE_MISSING = object()
_YEAR_RE = re.compile(r"(?P<year>\d{4})")


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
    return parsed if parsed > 0 else default


def _clean_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_api_key(cfg: ScopusRankingConfig) -> str | None:
    value = (
        os.getenv("SCOPUS_RANKINGS_API_KEY")
        or cfg.api_key
        or os.getenv("SCOPUS_API_KEY")
        or os.getenv("ELSEVIER_API_KEY")
    )
    return _clean_optional_str(value)


def _resolve_inst_token(cfg: ScopusRankingConfig) -> str | None:
    value = (
        os.getenv("SCOPUS_RANKINGS_INSTTOKEN")
        or cfg.insttoken
        or os.getenv("SCOPUS_INSTTOKEN")
        or os.getenv("SCOPUS_INST_TOKEN")
    )
    return _clean_optional_str(value)


def _resolve_view_preference(cfg: ScopusRankingConfig) -> list[str]:
    env_value = _clean_optional_str(os.getenv("SCOPUS_RANKINGS_VIEW_PREFERENCE"))
    source: Sequence[object]
    if env_value:
        source = [token for token in env_value.split(",") if token.strip()]
    else:
        source = cfg.view_preference

    seen: set[str] = set()
    views: list[str] = []
    for raw in source:
        view = str(raw).strip().upper()
        if view not in {"ENHANCED", "STANDARD"} or view in seen:
            continue
        seen.add(view)
        views.append(view)
    if not views:
        return ["ENHANCED", "STANDARD"]
    return views


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


def _ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _pick(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    if not mapping:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    lowered = {str(key).lower(): key for key in mapping.keys()}
    for key in keys:
        hit = lowered.get(str(key).lower())
        if hit is not None:
            return mapping[hit]
    return None


def _latest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {"value": None, "year": None}
    for entry in entries:
        year = entry.get("year")
        value = entry.get("value")
        if not isinstance(year, int) or not isinstance(value, (float, int)):
            continue
        if not isinstance(best.get("year"), int) or year > int(best["year"]):
            best = {"value": float(value), "year": year}
    return best


def _normalize_yearly_series(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    for item in entries:
        year = item.get("year")
        value = item.get("value")
        if not isinstance(year, int) or not isinstance(value, (int, float)):
            continue
        normalized: dict[str, Any] = {"year": int(year), "value": float(value)}
        kind = item.get("kind")
        kind_key = ""
        if isinstance(kind, str) and kind:
            normalized["kind"] = kind
            kind_key = kind
        keyed[(int(year), kind_key)] = normalized
    return [keyed[key] for key in sorted(keyed.keys(), key=lambda item: (item[0], item[1]))]


def _parse_year_value_entries(raw: object) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for entry in _ensure_list(raw):
        if not isinstance(entry, dict):
            continue
        year = _parse_year(
            _pick(
                entry,
                (
                    "@year",
                    "year",
                    "Year",
                    "publicationYear",
                ),
            )
        )
        value = _coerce_float(
            _pick(
                entry,
                (
                    "$",
                    "value",
                    "@value",
                    "Value",
                ),
            )
        )
        if year is None or value is None:
            continue
        parsed.append({"year": year, "value": value})
    return _normalize_yearly_series(parsed)


def _parse_sjr_series(entry: dict[str, Any]) -> list[dict[str, Any]]:
    block = _pick(entry, ("SJRList", "sjrList", "sjr_list"))
    if isinstance(block, dict):
        return _parse_year_value_entries(_pick(block, ("SJR", "sjr")) or block)
    return _parse_year_value_entries(block)


def _parse_snip_series(entry: dict[str, Any]) -> list[dict[str, Any]]:
    block = _pick(entry, ("SNIPList", "snipList", "snip_list"))
    if isinstance(block, dict):
        return _parse_year_value_entries(_pick(block, ("SNIP", "snip")) or block)
    return _parse_year_value_entries(block)


def _parse_citescore_series(
    entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    block = _pick(
        entry,
        (
            "citeScoreYearInfoList",
            "citescoreYearInfoList",
            "cite_score_year_info_list",
        ),
    )
    if isinstance(block, dict):
        infos = _pick(block, ("citeScoreYearInfo", "citescoreYearInfo")) or block
    else:
        infos = block

    current_series: list[dict[str, Any]] = []
    tracker_series: list[dict[str, Any]] = []
    merged_series: list[dict[str, Any]] = []

    for info in _ensure_list(infos):
        if not isinstance(info, dict):
            continue

        current_year = _parse_year(
            _pick(info, ("citeScoreCurrentMetricYear", "citescoreCurrentMetricYear"))
        )
        current_value = _coerce_float(
            _pick(info, ("citeScoreCurrentMetric", "citescoreCurrentMetric"))
        )
        if current_year is not None and current_value is not None:
            payload = {"year": current_year, "value": current_value, "kind": "current"}
            current_series.append(payload)
            merged_series.append(payload)

        tracker_year = _parse_year(_pick(info, ("citeScoreTrackerYear", "citescoreTrackerYear")))
        tracker_value = _coerce_float(_pick(info, ("citeScoreTracker", "citescoreTracker")))
        if tracker_year is not None and tracker_value is not None:
            payload = {"year": tracker_year, "value": tracker_value, "kind": "tracker"}
            tracker_series.append(payload)
            merged_series.append(payload)

    return (
        _normalize_yearly_series(current_series),
        _normalize_yearly_series(tracker_series),
        _normalize_yearly_series(merged_series),
    )


def _first_serial_entry(payload: dict[str, Any]) -> dict[str, Any] | None:
    response = payload.get("serial-metadata-response")
    if not isinstance(response, dict):
        response = payload
    entry_raw = _pick(response, ("entry", "entries"))
    for entry in _ensure_list(entry_raw):
        if isinstance(entry, dict):
            return entry
    return None


def _parse_serial_rankings(payload: dict[str, Any]) -> dict[str, Any]:
    entry = _first_serial_entry(payload)
    if not isinstance(entry, dict):
        return {
            "citescore": {"value": None, "year": None},
            "citescore_tracker": {"value": None, "year": None},
            "sjr": {"value": None, "year": None},
            "snip": {"value": None, "year": None},
            "series": {"citescore": [], "sjr": [], "snip": []},
        }

    sjr_series = _parse_sjr_series(entry)
    snip_series = _parse_snip_series(entry)
    citescore_series, tracker_series, merged_citescore_series = _parse_citescore_series(entry)

    return {
        "citescore": _latest(citescore_series),
        "citescore_tracker": _latest(tracker_series),
        "sjr": _latest(sjr_series),
        "snip": _latest(snip_series),
        "series": {
            "citescore": merged_citescore_series,
            "sjr": sjr_series,
            "snip": snip_series,
        },
    }


def _has_metrics(payload: dict[str, Any]) -> bool:
    for metric in ("citescore", "citescore_tracker", "sjr", "snip"):
        metric_payload = payload.get(metric)
        if not isinstance(metric_payload, dict):
            continue
        if metric_payload.get("value") is not None and metric_payload.get("year") is not None:
            return True
    series = payload.get("series")
    if isinstance(series, dict):
        for key in ("citescore", "sjr", "snip"):
            values = series.get(key)
            if isinstance(values, list) and values:
                return True
    return False


def _iso_utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _empty_rankings_payload() -> dict[str, Any]:
    return {
        "citescore": {"value": None, "year": None},
        "citescore_tracker": {"value": None, "year": None},
        "sjr": {"value": None, "year": None},
        "snip": {"value": None, "year": None},
        "series": {"citescore": [], "sjr": [], "snip": []},
        "source_issn_used": None,
        "retrieved_at": _iso_utc_now(),
    }


def _merge_scopus_rankings(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    rankings = updated.get("rankings")
    if not isinstance(rankings, dict):
        rankings = {}
    merged = dict(rankings)
    merged["scopus"] = payload
    updated["rankings"] = merged
    return updated


def _add_issn_candidate(candidates: list[str], seen: set[str], value: object) -> None:
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            _add_issn_candidate(candidates, seen, item)
        return

    text = str(value).strip()
    if not text:
        return

    tokens = re.split(r"[,\s;|]+", text)
    for token in tokens:
        candidate = normalize_issn(token)
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)


def _extract_issn_candidates(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    scopus = record.get("scopus")
    abstract = scopus.get("abstract") if isinstance(scopus, dict) else None

    _add_issn_candidate(candidates, seen, record.get("issn"))
    if isinstance(abstract, dict):
        _add_issn_candidate(candidates, seen, abstract.get("issn"))

    _add_issn_candidate(candidates, seen, record.get("issn_list"))
    if isinstance(abstract, dict):
        _add_issn_candidate(candidates, seen, abstract.get("issn_list"))

    _add_issn_candidate(candidates, seen, record.get("eissn"))
    if isinstance(abstract, dict):
        _add_issn_candidate(candidates, seen, abstract.get("eissn"))

    return candidates


def _is_retryable_scopus_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (400, 401, 403, 404):
            return False
        if status in (408, 429):
            return True
        return 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _retrying(
    retries: RetryConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Retrying:
    attempts = max(1, retries.max)
    return Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, exp_base=retries.backoff, min=1, max=60),
        retry=retry_if_exception(_is_retryable_scopus_error),
        reraise=True,
        sleep=sleep,
    )


class ScopusRankingsEnricher:
    """Enrich records with Scopus journal rankings from Serial Title API."""

    def __init__(
        self,
        cfg: ScopusRankingConfig,
        *,
        retries: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.retries = retries or RetryConfig()
        self.client = client
        self.sleep = sleep

        self.base_url = _clean_optional_str(os.getenv("SCOPUS_RANKINGS_BASE_URL")) or cfg.base_url
        self.timeout_s = float(_env_int("SCOPUS_RANKINGS_TIMEOUT_S", cfg.timeout_s))
        self.cache_ttl_days = int(_env_int("SCOPUS_RANKINGS_CACHE_TTL_DAYS", cfg.cache_ttl_days))
        self.view_preference = _resolve_view_preference(cfg)
        self.api_key = _resolve_api_key(cfg)
        self.insttoken = _resolve_inst_token(cfg)

        rps = _env_float("SCOPUS_RANKINGS_RPS", cfg.rate_limit)
        burst = max(1, int(round(rps)))
        self.limiter = rate_limiter or RateLimiter(rate=rps, burst=burst)

    def enrich(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        if not self.cfg.enabled:
            return records

        if not self.api_key:
            message = (
                "Scopus rankings enrichment enabled but SCOPUS_API_KEY/ELSEVIER_API_KEY is missing."
            )
            if self.cfg.fail_open:
                LOGGER.warning("%s Filling rankings.scopus with null metrics.", message)
                return [
                    _merge_scopus_rankings(record, _empty_rankings_payload()) for record in records
                ]
            raise ValueError(message)

        headers: dict[str, str] = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
            "User-Agent": os.getenv(
                "AI_BIAS_USER_AGENT",
                "ai-bias-search/0.1 (+contact@example.com)",
            ),
        }
        if self.insttoken:
            headers["X-ELS-Insttoken"] = self.insttoken

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ttl_seconds = self.cache_ttl_days * 24 * 60 * 60
        retrying = _retrying(self.retries, sleep=self.sleep)

        output: list[dict[str, Any]] = []
        with Cache(CACHE_DIR) as cache:
            client = self.client or httpx.Client(
                base_url=self.base_url or BASE_URL,
                timeout=self.timeout_s,
                follow_redirects=True,
            )
            owns_client = self.client is None
            try:
                for record in records:
                    try:
                        payload = self._enrich_one(
                            record,
                            cache=cache,
                            client=client,
                            retrying=retrying,
                            headers=headers,
                            ttl_seconds=ttl_seconds,
                        )
                    except (httpx.HTTPError, ValueError) as exc:
                        if not self.cfg.fail_open:
                            raise
                        LOGGER.warning(
                            "Scopus rankings enrichment failed for rank=%s doi=%s error=%s",
                            record.get("rank"),
                            record.get("doi"),
                            exc,
                        )
                        payload = _empty_rankings_payload()
                    output.append(_merge_scopus_rankings(record, payload))
            finally:
                if owns_client:
                    client.close()

        return output

    def _enrich_one(
        self,
        record: dict[str, Any],
        *,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> dict[str, Any]:
        issn_candidates = _extract_issn_candidates(record)
        if not issn_candidates:
            return _empty_rankings_payload()

        for issn in issn_candidates:
            for view in self.view_preference:
                serial_payload = self._get_serial_title(
                    issn,
                    view=view,
                    cache=cache,
                    client=client,
                    retrying=retrying,
                    headers=headers,
                    ttl_seconds=ttl_seconds,
                )
                if not isinstance(serial_payload, dict):
                    continue
                parsed = _parse_serial_rankings(serial_payload)
                if not _has_metrics(parsed):
                    continue
                return {
                    **parsed,
                    "source_issn_used": issn,
                    "retrieved_at": _iso_utc_now(),
                }

        payload = _empty_rankings_payload()
        payload["retrieved_at"] = _iso_utc_now()
        return payload

    def _get_serial_title(
        self,
        issn: str,
        *,
        view: str,
        cache: Cache,
        client: httpx.Client,
        retrying: Retrying,
        headers: dict[str, str],
        ttl_seconds: int,
    ) -> dict[str, Any] | None:
        cache_key = (
            f"serial_title_rankings:issn:{issn}|view:{view}|insttoken:{int(bool(self.insttoken))}"
        )
        cached = cache.get(cache_key, default=_CACHE_MISSING)
        if cached is not _CACHE_MISSING:
            return cached if isinstance(cached, dict) else None

        safe_issn = quote(issn, safe="")
        path = f"/content/serial/title/issn/{safe_issn}"
        params = {"view": view}

        def execute() -> dict[str, Any] | None:
            self.limiter.acquire()
            LOGGER.debug(
                "scopus rankings request path=%s params=%s headers=%s",
                path,
                params,
                mask_sensitive(headers),
            )
            response = client.get(path, params=params, headers=headers)
            if response.status_code in (400, 403, 404):
                return None
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None

        payload = retrying(execute)
        cache.set(cache_key, payload, expire=ttl_seconds)
        return payload


def enrich_with_scopus_rankings(
    records: list[dict[str, Any]],
    *,
    cfg: ScopusRankingConfig,
    retries: RetryConfig | None = None,
    rate_limiter: RateLimiter | None = None,
) -> list[dict[str, Any]]:
    """Public convenience wrapper for Scopus rankings enrichment."""

    enricher = ScopusRankingsEnricher(
        cfg,
        retries=retries,
        rate_limiter=rate_limiter,
    )
    return enricher.enrich(records)
