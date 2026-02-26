"""Enrichment of records with OpenAlex metadata."""

from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from diskcache import Cache
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ai_bias_search.rankings.base import MatchResult, normalize_issn, normalize_title
from ai_bias_search.rankings.registry import get_provider, match_all as match_rankings
from ai_bias_search.utils.config import ImpactFactorConfig, RetryConfig
from ai_bias_search.utils.core_rankings import is_known_core_acronym, lookup_core_rank
from ai_bias_search.utils.ids import best_identifier, normalise_doi
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.models import EnrichedRecord
from ai_bias_search.utils.rate_limit import RateLimiter

LOGGER = configure_logging()
CACHE_DIR = (Path(__file__).resolve().parents[2] / "data" / "cache" / "openalex").resolve()


def _is_retryable_openalex_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (404, 410):
            return False
        if status in (408, 429):
            return True
        return 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _openalex_retrying(
    retries: RetryConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(retries.max),
        wait=wait_exponential(multiplier=1, exp_base=retries.backoff, min=1),
        retry=retry_if_exception(_is_retryable_openalex_error),
        reraise=True,
        sleep=sleep,
    )


# --- acronym extraction helpers ---
# A) "... - EMNLP '09", "... - WWW '18"
_ACR_AFTER_DASH_RE = re.compile(r"\b-\s*([A-Z][A-Z0-9]{1,14})\b")

# B) "(SIGIR 2017)" "(KDD'16)"
_ACR_IN_PARENS_RE = re.compile(r"\(\s*([A-Z][A-Z0-9]{1,14})\s*['’]?\d{0,4}\s*\)")

# C) token w środku: "... ACM SIGIR conference ..."
#    łapiemy "SIGIR", "SIGKDD", "AAAI" itd.
_ACR_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,14})\b")

# typowe fałszywe trafienia
_ACR_STOP = {
    "ACM",
    "IEEE",
    "SPRINGER",
    "ELSEVIER",
    "NATURE",
    "SCIENCE",
    "PROC",
    "PROCEEDINGS",
    "VOLUME",
    "VOL",
    "NO",
    "INTERNATIONAL",
    "CONFERENCE",
    "JOURNAL",
    "TRANSACTIONS",
    "SYMPOSIUM",
    "WORKSHOP",
}
_ROMAN_NUMERAL_RE = re.compile(r"^(?=[IVXLCDM])[IVXLCDM]+$", re.IGNORECASE)
_NOISE_RE = re.compile(r"\s*\((?:print|online)\)\s*", re.IGNORECASE)
_GENERIC_ACRONYMS = {"BMC"}
_KNOWN_SHORT_VENUES = {"BMJ"}
_DBLP_PREFIX_MAP = {
    "conf/sigir/": "SIGIR",
    "conf/kdd/": "KDD",
    "conf/emnlp/": "EMNLP",
    "conf/www/": "WWW",
    "conf/icwsm/": "ICWSM",
}
_CORE_RANK_ORDER = {"A*": 4, "A": 3, "B": 2, "C": 1, None: 0}
_STRONG_ACRONYM_SOURCES = {
    "openalex_host_abbrev_title",
    "openalex_host_abbreviation",
    "openalex_source_abbrev_title",
    "s2_dblp",
}


def _is_url(s: str) -> bool:
    ss = s.strip().lower()
    return ss.startswith("http://") or ss.startswith("https://") or "doi.org" in ss


def _is_roman_numeral(value: str) -> bool:
    return bool(_ROMAN_NUMERAL_RE.match(value))


def _normalize_venue_name(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    v = html.unescape(value).strip()
    if not v:
        return None
    v = _NOISE_RE.sub("", v)
    v = re.sub(r"\s{2,}", " ", v).strip()
    return v or None


def _clean_acronym(value: str | None) -> str | None:
    if not value:
        return None
    v = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if len(v) < 2 or len(v) > 15:
        return None
    if v in _ACR_STOP:
        return None
    return v


def _normalize_acronym(value: str | None, *, source: str, reasons: List[str]) -> str | None:
    if not value or not isinstance(value, str):
        return None
    v = html.unescape(value).strip()
    if not v:
        return None
    cleaned = _clean_acronym(v)
    if not cleaned:
        return None
    if _is_roman_numeral(cleaned):
        reasons.append("acronym_roman_numeral")
        return None
    if cleaned in _GENERIC_ACRONYMS and source not in _STRONG_ACRONYM_SOURCES:
        reasons.append(f"acronym_generic_weak:{cleaned}")
    return cleaned


def _clean_acronym_token(value: str | None) -> str | None:
    cleaned = _clean_acronym(value)
    if cleaned and _is_roman_numeral(cleaned):
        return None
    return cleaned


def extract_acronym_from_venue_text(text: str | None) -> str | None:
    """Extract a conference acronym from venue text.

    Examples:
      - "Proceedings ... - EMNLP '09"
      - "(SIGIR 2017)"
    """
    if not text or not isinstance(text, str):
        return None

    t = html.unescape(text).strip()
    if not t or _is_url(t):
        return None

    # 1) najpewniejsze: po myślniku
    m = _ACR_AFTER_DASH_RE.search(t)
    if m:
        cand = _clean_acronym_token(m.group(1))
        if cand:
            return cand

    # 2) w nawiasach
    m = _ACR_IN_PARENS_RE.search(t)
    if m:
        cand = _clean_acronym_token(m.group(1))
        if cand:
            return cand

    # 3) tokeny w środku: zbierz wszystkie i wybierz "najbardziej sensowny"
    raw_tokens = [_clean_acronym_token(x) for x in _ACR_TOKEN_RE.findall(t)]
    tokens = [token for token in raw_tokens if token]
    if not tokens:
        return None

    # heurystyka: preferuj dłuższe (SIGKDD > KDD), ale nie absurdalnie długie
    tokens.sort(key=lambda x: (len(x), x), reverse=True)
    for token in tokens:
        if is_known_core_acronym(token):
            return token
    return None


def _rank_value(rank: str | None) -> int:
    return _CORE_RANK_ORDER.get(rank, 0)


def _choose_kdd_acronym() -> str:
    rank_sigkdd = lookup_core_rank(venue_name=None, venue_acronym="SIGKDD")
    rank_kdd = lookup_core_rank(venue_name=None, venue_acronym="KDD")
    if _rank_value(rank_sigkdd) > _rank_value(rank_kdd):
        return "SIGKDD"
    return "KDD"


def _acronym_from_dblp(dblp: Any) -> tuple[str | None, str | None]:
    if not isinstance(dblp, str) or not dblp.strip():
        return None, None
    dblp_norm = dblp.strip().lower()
    for prefix, acronym in _DBLP_PREFIX_MAP.items():
        if dblp_norm.startswith(prefix):
            if acronym == "KDD":
                return _choose_kdd_acronym(), f"s2_dblp:{prefix}"
            return acronym, f"s2_dblp:{prefix}"
    if "/kdd/" in dblp_norm:
        return _choose_kdd_acronym(), "s2_dblp:/kdd/"
    return None, None


def _acronym_from_s2_venue(venue: str | None) -> tuple[str | None, str | None]:
    if not venue or not isinstance(venue, str):
        return None, None
    v = html.unescape(venue).strip()
    if not v or _is_url(v):
        return None, None
    upper = v.upper()
    if "SIGIR" in upper:
        return "SIGIR", "s2_venue:sigir"
    if "SIGKDD" in upper:
        return "SIGKDD", "s2_venue:sigkdd"
    if "KDD" in upper and ("ACM" in upper or "SIGKDD" in upper):
        return "SIGKDD", "s2_venue:kdd_acm"
    if "EMNLP" in upper:
        return "EMNLP", "s2_venue:emnlp"
    if "WORLD WIDE WEB" in upper or re.search(r"\bWWW\b", upper):
        return "WWW", "s2_venue:www"
    return None, None


def _compact_raw(value: Any, *, max_len: int = 120) -> Any:
    if isinstance(value, str):
        if len(value) > max_len:
            return value[: max_len - 3] + "..."
        return value
    if isinstance(value, dict):
        return {k: _compact_raw(v, max_len=max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact_raw(v, max_len=max_len) for v in value]
    return value


def _select_impact_factor_title(metadata: dict, record: dict) -> str | None:
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(record, dict):
        record = {}

    primary_location = metadata.get("primary_location") or {}
    if not isinstance(primary_location, dict):
        primary_location = {}

    source = primary_location.get("source") or {}
    if not isinstance(source, dict):
        source = {}

    host = metadata.get("host_venue") or {}
    if not isinstance(host, dict):
        host = {}

    extra = record.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}
    s2 = extra.get("semanticscholar") or {}
    if not isinstance(s2, dict):
        s2 = {}

    candidates = (
        host.get("display_name"),
        source.get("display_name"),
        primary_location.get("raw_source_name"),
        record.get("source"),
        s2.get("venue"),
    )
    for value in candidates:
        if not isinstance(value, str):
            continue
        candidate = html.unescape(value).strip()
        if not candidate or _is_url(candidate):
            continue
        return candidate
    return None


def _collect_issn_candidates(metadata: dict, record: dict) -> list[str]:
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(record, dict):
        record = {}

    primary_location = metadata.get("primary_location") or {}
    if not isinstance(primary_location, dict):
        primary_location = {}

    source = primary_location.get("source") or {}
    if not isinstance(source, dict):
        source = {}

    host = metadata.get("host_venue") or {}
    if not isinstance(host, dict):
        host = {}

    candidates: list[str] = []
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
        tokens = (
            text.replace(";", ",").replace("|", ",").split(",")
            if ("," in text or ";" in text or "|" in text)
            else [text]
        )
        for token in tokens:
            normalized = normalize_issn(token.strip())
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)

    add(host.get("issn_l"))
    add(host.get("issn"))
    add(source.get("issn_l"))
    add(source.get("issn"))
    add(record.get("issn"))
    add(record.get("eissn"))

    return candidates


JIF_PAYLOAD_FIELDS = (
    "impact_factor",
    "impact_factor_year",
    "impact_factor_match",
    "impact_factor_title_raw",
    "impact_factor_title_key",
    "impact_factor_source",
    "jcr_year",
    "jcr_publisher",
    "jcr_issn",
    "jcr_eissn",
    "jcr_total_cites",
    "jcr_total_articles",
    "jcr_citable_items",
    "jcr_jif_5y",
    "jcr_jif_wo_self_cites",
    "jcr_jci",
    "jcr_quartile",
    "jcr_jif_rank",
    "jcr_match_type",
)


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
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


def _configure_jif_provider(impact_factor: ImpactFactorConfig) -> None:
    """Apply legacy ImpactFactorConfig overrides to the unified `jif` ranking provider."""

    provider = get_provider("jif")

    desired_path = impact_factor.xlsx_path
    if not desired_path.is_absolute():
        desired_path = (Path.cwd() / desired_path).resolve()

    changed = False
    if provider.cfg.dataset_path != desired_path:
        provider.cfg.dataset_path = desired_path
        changed = True

    if provider.cfg.sheet_name != impact_factor.sheet_name:
        provider.cfg.sheet_name = impact_factor.sheet_name
        changed = True

    desired_fields: dict[str, str | int] = {
        "title": impact_factor.title_column,
        "rank_value": impact_factor.jif_column,
    }
    if impact_factor.year_column:
        desired_fields["rank_year"] = impact_factor.year_column
    if impact_factor.issn_column:
        desired_fields["issn_print"] = impact_factor.issn_column
    if impact_factor.eissn_column:
        desired_fields["issn_online"] = impact_factor.eissn_column

    desired_extra: dict[str, str | int] = {}
    if impact_factor.publisher_column:
        desired_extra["publisher"] = impact_factor.publisher_column
    if impact_factor.total_cites_column:
        desired_extra["total_cites"] = impact_factor.total_cites_column
    if impact_factor.total_articles_column:
        desired_extra["total_articles"] = impact_factor.total_articles_column
    if impact_factor.citable_items_column:
        desired_extra["citable_items"] = impact_factor.citable_items_column
    if impact_factor.jif_5y_column:
        desired_extra["jif_5y"] = impact_factor.jif_5y_column
    if impact_factor.jif_wo_self_cites_column:
        desired_extra["jif_wo_self_cites"] = impact_factor.jif_wo_self_cites_column
    if impact_factor.jci_column:
        desired_extra["jci"] = impact_factor.jci_column
    if impact_factor.quartile_column:
        desired_extra["quartile"] = impact_factor.quartile_column
    if impact_factor.jif_rank_column:
        desired_extra["jif_rank"] = impact_factor.jif_rank_column

    if provider.cfg.fields != desired_fields:
        provider.cfg.fields = desired_fields
        changed = True
    if provider.cfg.extra_fields != desired_extra:
        provider.cfg.extra_fields = desired_extra
        changed = True

    desired_allow_fuzzy = bool(impact_factor.allow_fuzzy)
    desired_threshold = float(impact_factor.fuzzy_threshold) / 100.0
    if provider.cfg.allow_fuzzy != desired_allow_fuzzy:
        provider.cfg.allow_fuzzy = desired_allow_fuzzy
        changed = True
    if provider.cfg.fuzzy_threshold != desired_threshold:
        provider.cfg.fuzzy_threshold = desired_threshold
        changed = True
    if provider.cfg.reject_ambiguous_fuzzy != bool(impact_factor.reject_ambiguous):
        provider.cfg.reject_ambiguous_fuzzy = bool(impact_factor.reject_ambiguous)
        changed = True

    if changed:
        provider.reset()


def _jif_payload_from_match(
    *,
    result: MatchResult | None,
    title_raw: str | None,
    title_key: str | None,
    impact_enabled: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {key: None for key in JIF_PAYLOAD_FIELDS}
    if not impact_enabled:
        payload["impact_factor_match"] = "none"
        payload["jcr_match_type"] = "none"
        payload["impact_factor_source"] = None
        payload["impact_factor_title_raw"] = title_raw
        payload["impact_factor_title_key"] = title_key
        return payload

    if result is None or not result.matched or result.rank_value is None:
        payload["impact_factor_match"] = "none"
        payload["jcr_match_type"] = "none"
        payload["impact_factor_source"] = "rankings:jif"
        payload["impact_factor_title_raw"] = title_raw
        payload["impact_factor_title_key"] = title_key
        return payload

    provider = get_provider("jif")
    entry = provider.resolve_entry(result)

    match_type = result.method if result.method != "unmatched" else "none"
    payload["impact_factor_match"] = match_type
    payload["jcr_match_type"] = match_type
    payload["impact_factor_source"] = "rankings:jif"
    payload["impact_factor_title_raw"] = title_raw
    payload["impact_factor_title_key"] = title_key

    payload["impact_factor"] = _coerce_float(result.rank_value)
    payload["impact_factor_year"] = entry.rank_year if entry else None
    payload["jcr_year"] = entry.rank_year if entry else None
    payload["jcr_issn"] = entry.issn_print if entry else None
    payload["jcr_eissn"] = entry.issn_online if entry else None

    extra = entry.extra if entry else {}
    payload["jcr_publisher"] = str(extra.get("publisher")).strip() if extra.get("publisher") else None
    payload["jcr_total_cites"] = _coerce_int(extra.get("total_cites"))
    payload["jcr_total_articles"] = _coerce_int(extra.get("total_articles"))
    payload["jcr_citable_items"] = _coerce_int(extra.get("citable_items"))
    payload["jcr_jif_5y"] = _coerce_float(extra.get("jif_5y"))
    payload["jcr_jif_wo_self_cites"] = _coerce_float(extra.get("jif_wo_self_cites"))
    payload["jcr_jci"] = _coerce_float(extra.get("jci"))
    payload["jcr_quartile"] = str(extra.get("quartile")).strip() if extra.get("quartile") else None
    payload["jcr_jif_rank"] = str(extra.get("jif_rank")).strip() if extra.get("jif_rank") else None

    return payload


def _rankings_dict(results: dict[str, MatchResult]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for ranking_id, result in results.items():
        evidence = result.evidence if isinstance(result.evidence, dict) else dict(result.evidence)
        matched_key = evidence.get("issn") or evidence.get("title_norm") or evidence.get("best_key")
        source = None
        if result.method == "issn_exact":
            source = "issn"
        elif result.method == "title_exact":
            source = "title"
        elif result.method == "title_fuzzy":
            source = "fuzzy"
        payload[ranking_id] = {
            "matched": bool(result.matched),
            "value": result.rank_value,
            "method": result.method,
            "score": float(result.score),
            "evidence": evidence,
            "rank": result.rank_value if ranking_id == "core" else None,
            "source": source,
            "confidence": (float(result.score) if result.matched else None),
            "matched_key": str(matched_key) if matched_key is not None else None,
        }
    return payload


def extract_venue_candidates(metadata: dict, record: dict) -> dict:
    reasons: List[str] = []
    raw: Dict[str, Any] = {"openalex": {}, "semanticscholar": {}, "chosen": {}}

    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(record, dict):
        record = {}

    primary_location = metadata.get("primary_location") or {}
    if not isinstance(primary_location, dict):
        primary_location = {}

    source = primary_location.get("source") or {}
    if not isinstance(source, dict):
        source = {}

    host = metadata.get("host_venue") or {}
    if not isinstance(host, dict):
        host = {}

    raw_source_name = primary_location.get("raw_source_name")
    raw["openalex"] = {
        "host_display_name": host.get("display_name"),
        "source_display_name": source.get("display_name"),
        "raw_source_name": raw_source_name,
        "host_abbreviated_title": host.get("abbreviated_title"),
        "host_abbreviation": host.get("abbreviation"),
        "source_abbreviated_title": source.get("abbreviated_title"),
    }

    venue_name = None
    name_source = None
    for source_key, value in (
        ("openalex_host", host.get("display_name")),
        ("openalex_source", source.get("display_name")),
        ("openalex_raw", raw_source_name),
    ):
        candidate = _normalize_venue_name(value if isinstance(value, str) else None)
        if not candidate:
            continue
        if _is_url(candidate):
            reasons.append("venue_name_url")
            continue
        venue_name = candidate
        name_source = source_key
        reasons.append(f"venue_name:{source_key}")
        break

    extra = record.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}
    s2 = extra.get("semanticscholar") or {}
    if not isinstance(s2, dict):
        s2 = {}
    s2_external = s2.get("externalIds") or {}
    if not isinstance(s2_external, dict):
        s2_external = {}
    s2_venue_raw = s2.get("venue")
    s2_dblp = s2_external.get("DBLP")
    raw["semanticscholar"] = {
        "venue": s2_venue_raw,
        "dblp": s2_dblp,
    }

    if not venue_name:
        s2_venue_name = _normalize_venue_name(
            s2_venue_raw if isinstance(s2_venue_raw, str) else None
        )
        if s2_venue_name and not _is_url(s2_venue_name):
            venue_name = s2_venue_name
            name_source = "semanticscholar_venue"
            reasons.append("venue_name:semanticscholar")
        elif s2_venue_name and _is_url(s2_venue_name):
            reasons.append("venue_name_url")

    venue_acronym = None
    acronym_source = None
    for source_key, value in (
        ("openalex_host_abbrev_title", host.get("abbreviated_title")),
        ("openalex_host_abbreviation", host.get("abbreviation")),
        ("openalex_source_abbrev_title", source.get("abbreviated_title")),
    ):
        candidate = _normalize_acronym(
            value if isinstance(value, str) else None, source=source_key, reasons=reasons
        )
        if candidate and not is_known_core_acronym(candidate):
            reasons.append(f"venue_acronym_not_in_core:{candidate}")
            continue
        if candidate:
            venue_acronym = candidate
            acronym_source = source_key
            reasons.append(f"venue_acronym:{source_key}")
            break

    if not venue_acronym:
        dblp_candidate, dblp_source = _acronym_from_dblp(s2_dblp)
        if dblp_candidate:
            candidate = _normalize_acronym(dblp_candidate, source="s2_dblp", reasons=reasons)
            if candidate and not is_known_core_acronym(candidate):
                reasons.append(f"venue_acronym_not_in_core:{candidate}")
                candidate = None
            if candidate:
                venue_acronym = candidate
                acronym_source = dblp_source or "s2_dblp"
                reasons.append(f"venue_acronym:{acronym_source}")

    if not venue_acronym:
        s2_venue_name = _normalize_venue_name(
            s2_venue_raw if isinstance(s2_venue_raw, str) else None
        )
        heur_candidate, heur_source = _acronym_from_s2_venue(s2_venue_name)
        if heur_candidate:
            candidate = _normalize_acronym(heur_candidate, source="s2_venue", reasons=reasons)
            if candidate and not is_known_core_acronym(candidate):
                reasons.append(f"venue_acronym_not_in_core:{candidate}")
                candidate = None
            if candidate:
                venue_acronym = candidate
                acronym_source = heur_source or "s2_venue"
                reasons.append(f"venue_acronym:{acronym_source}")

    if not venue_acronym:
        fallback = extract_acronym_from_venue_text(raw_source_name)
        if not fallback:
            fallback = extract_acronym_from_venue_text(venue_name)
        candidate = _normalize_acronym(fallback, source="openalex_text", reasons=reasons)
        if candidate:
            venue_acronym = candidate
            acronym_source = "openalex_text"
            reasons.append("venue_acronym:openalex_text")

    eligible = True
    if not venue_name:
        eligible = False
        reasons.append("venue_name_missing")
    elif _is_url(venue_name):
        eligible = False
        reasons.append("venue_name_url")
    elif len(venue_name.strip()) < 4 and venue_name.strip().upper() not in _KNOWN_SHORT_VENUES:
        eligible = False
        reasons.append("venue_name_too_short")

    raw["chosen"] = {
        "venue_name_source": name_source,
        "venue_acronym_source": acronym_source,
    }

    return {
        "venue_name": venue_name,
        "venue_acronym": venue_acronym,
        "eligible": eligible,
        "reasons": reasons,
        "raw": raw,
    }


def enrich_with_openalex(
    records: List[Dict[str, Any]],
    mailto: str | None,
    impact_factor: ImpactFactorConfig | None = None,
    *,
    rate_limiter: RateLimiter | None = None,
    retries: RetryConfig | None = None,
) -> List[Dict[str, Any]]:
    """Augment *records* with OpenAlex metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limiter = rate_limiter or RateLimiter(rate=2, burst=5)
    retry_cfg = retries or RetryConfig()
    retrying = _openalex_retrying(retry_cfg)

    impact_enabled = bool(impact_factor and impact_factor.enabled)
    if impact_factor:
        _configure_jif_provider(impact_factor)

    enriched: List[Dict[str, Any]] = []
    with Cache(CACHE_DIR) as cache:
        with httpx.Client(base_url="https://api.openalex.org", timeout=30.0) as client:
            for record in records:
                identifier = best_identifier(record)  # powinien preferować DOI
                if not identifier:
                    LOGGER.debug(
                        "OpenAlex: no identifier for record (title=%r)", record.get("title")
                    )
                    title_raw = _select_impact_factor_title({}, record)
                    issn_candidates = _collect_issn_candidates({}, record)
                    skip_ids = {"jif"} if not impact_enabled else set()
                    results = match_rankings(title_raw, issn_candidates, skip_ids=skip_ids)
                    record = {**record, "rankings": _rankings_dict(results)}
                    if impact_enabled:
                        jif_title_key = (
                            normalize_title(title_raw, get_provider("jif").cfg.normalization)
                            if title_raw
                            else None
                        )
                        record = {
                            **record,
                            **_jif_payload_from_match(
                                result=results.get("jif"),
                                title_raw=title_raw,
                                title_key=jif_title_key,
                                impact_enabled=True,
                            ),
                        }
                    enriched.append(record)
                    continue

                cache_key = identifier.lower()
                metadata = cache.get(cache_key)
                if metadata is None:
                    try:
                        metadata = _fetch_openalex_metadata(
                            identifier=identifier,
                            client=client,
                            limiter=limiter,
                            retrying=retrying,
                            mailto=mailto,
                        )
                    except httpx.HTTPError as exc:
                        LOGGER.warning("OpenAlex enrichment failed id=%s error=%s", identifier, exc)
                        metadata = None
                    # cache zarówno hit jak i miss (None), żeby nie mielić w kółko
                    cache.set(cache_key, metadata, expire=60 * 60 * 24 * 7)

                if not metadata:
                    title_raw = _select_impact_factor_title({}, record)
                    issn_candidates = _collect_issn_candidates({}, record)
                    skip_ids = {"jif"} if not impact_enabled else set()
                    results = match_rankings(title_raw, issn_candidates, skip_ids=skip_ids)
                    record = {**record, "rankings": _rankings_dict(results)}
                    if impact_enabled:
                        jif_title_key = (
                            normalize_title(title_raw, get_provider("jif").cfg.normalization)
                            if title_raw
                            else None
                        )
                        record = {
                            **record,
                            **_jif_payload_from_match(
                                result=results.get("jif"),
                                title_raw=title_raw,
                                title_key=jif_title_key,
                                impact_enabled=True,
                            ),
                        }
                    enriched.append(record)
                    continue

                merged = EnrichedRecord(**record)
                # proste mapowanie pól
                merged.language = metadata.get("language")
                merged.publication_year = metadata.get("publication_year")
                merged.cited_by_count = metadata.get("cited_by_count")

                # is_oa z obiektu open_access
                oa = metadata.get("open_access") or {}
                merged.is_oa = bool(oa.get("is_oa"))

                primary_location = metadata.get("primary_location") or {}
                if not isinstance(primary_location, dict):
                    primary_location = {}

                source = primary_location.get("source") or {}
                if not isinstance(source, dict):
                    source = {}

                host = metadata.get("host_venue") or {}
                if not isinstance(host, dict):
                    host = {}

                venue_type = source.get("type")
                merged.venue_type = venue_type if isinstance(venue_type, str) else None

                is_core_listed = source.get("is_core")
                merged.is_core_listed = bool(is_core_listed) if is_core_listed is not None else None

                publisher = source.get("publisher") or host.get("publisher")
                merged.publisher = (
                    publisher.strip() if isinstance(publisher, str) and publisher.strip() else None
                )

                # --- venue & CORE ranking resolution (FIXED ORDER) ---
                core_lookup = extract_venue_candidates(metadata, record)
                venue_name = core_lookup.get("venue_name")
                venue_acronym = core_lookup.get("venue_acronym")
                eligible = core_lookup.get("eligible")
                reasons = core_lookup.get("reasons")
                raw = core_lookup.get("raw")

                merged.host_venue = venue_name

                LOGGER.debug(
                    "CORE query: venue_name=%r venue_acronym=%r eligible=%s reasons=%s raw=%s",
                    venue_name,
                    venue_acronym,
                    eligible,
                    reasons,
                    _compact_raw(raw),
                )

                title_raw = _select_impact_factor_title(metadata, record) or venue_name
                jif_title_key = None
                if impact_enabled and title_raw:
                    jif_title_key = normalize_title(
                        title_raw, get_provider("jif").cfg.normalization
                    )
                issn_candidates = _collect_issn_candidates(metadata, record)
                skip_ids = {"jif"} if not impact_enabled else set()
                results = match_rankings(title_raw, issn_candidates, skip_ids=skip_ids)

                if venue_acronym and eligible:
                    core_provider = get_provider("core")
                    core_res = core_provider.match(venue_acronym, None)
                    if core_res.matched:
                        results["core"] = core_res

                merged.rankings = _rankings_dict(results)
                merged.issn_list = issn_candidates or None
                if title_raw:
                    merged.journal_title = title_raw

                core_result = results.get("core")
                merged.core_rank = (
                    str(core_result.rank_value).strip()
                    if eligible and core_result and isinstance(core_result.rank_value, str)
                    else None
                )

                if impact_enabled:
                    merged = merged.model_copy(
                        update=_jif_payload_from_match(
                            result=results.get("jif"),
                            title_raw=title_raw,
                            title_key=jif_title_key,
                            impact_enabled=True,
                        )
                    )

                # do extra dokładamy cały surowy payload z OpenAlex
                prev_extra = record.get("extra") or {}
                if not isinstance(prev_extra, dict):
                    prev_extra = {}
                merged.extra = {
                    **prev_extra,
                    "openalex_enrich": metadata,
                    "core_lookup": core_lookup,
                }

                enriched.append(merged.model_dump())
    return enriched


def _fetch_openalex_metadata(
    *,
    identifier: str,
    client: httpx.Client,
    limiter: RateLimiter,
    retrying: Retrying,
    mailto: str | None,
) -> Optional[Dict[str, Any]]:
    """Retrieve OpenAlex metadata for *identifier*."""
    openalex_id_or_path = _resolve_openalex_id(
        identifier, client=client, limiter=limiter, retrying=retrying, mailto=mailto
    )
    if not openalex_id_or_path:
        return None

    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto

    def execute() -> Dict[str, Any]:
        limiter.acquire()
        # openalex_id_or_path jest już w formie 'works/<key>' lub sam 'W...' -> normalizujemy
        path = openalex_id_or_path
        if not path.startswith("works/"):
            path = f"works/{path}"
        resp = client.get(f"/{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    return retrying(execute)


def _resolve_openalex_id(
    identifier: str,
    *,
    client: httpx.Client,
    limiter: RateLimiter,
    retrying: Retrying,
    mailto: str | None,
) -> Optional[str]:
    """Resolve a DOI or URL to an OpenAlex work path/key.

    Preferujemy bezpośrednie ścieżki:
      - /works/doi:{doi}
      - /works/https://doi.org/{doi}
      - /works/{openalex_id}
    Fallback: /works?filter=doi:... lub /works?search=...
    """
    # 1) OpenAlex URL → zwróć ID
    if identifier.startswith("https://openalex.org/"):
        return identifier.removeprefix("https://openalex.org/")

    # 2) DOI → użyj bezpośredniej ścieżki /works/doi:{doi}
    doi = normalise_doi(identifier)
    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto

    if doi:

        def execute_direct() -> httpx.Response:
            limiter.acquire()
            # OpenAlex akceptuje i 'doi:10.123/abc' i pełny URL doi
            return client.get(f"/works/doi:{doi}", params=params)

        try:
            resp = retrying(
                lambda: (r := execute_direct(), r.raise_for_status(), r)[0]
            )  # noqa: E731
            # jeśli jest 200, mamy komplet
            payload = resp.json()
            work_id = payload.get("id")
            if isinstance(work_id, str) and work_id.startswith("https://openalex.org/"):
                return work_id.split("/")[-1]
        except httpx.HTTPStatusError as exc:
            # 404 przy direct → spróbuj filter=doi:...
            if exc.response.status_code not in (404, 410):
                raise

        # Fallback na filter=doi
        def execute_filter() -> Dict[str, Any]:
            limiter.acquire()
            r = client.get("/works", params={**params, "filter": f"doi:{doi}"})
            r.raise_for_status()
            return r.json()

        payload = retrying(execute_filter)
        results = payload.get("results") or []
        if results:
            openalex_id = results[0].get("id")
            if isinstance(openalex_id, str) and openalex_id.startswith("https://openalex.org/"):
                return openalex_id.split("/")[-1]
            if isinstance(openalex_id, str) and openalex_id:
                return openalex_id

    # 3) Brak DOI → spróbuj wyszukiwania (głośne, ale ostatnia deska ratunku)
    def execute_search() -> Dict[str, Any]:
        limiter.acquire()
        r = client.get("/works", params={**params, "search": identifier})
        r.raise_for_status()
        return r.json()

    payload = retrying(execute_search)
    results = payload.get("results") or []
    if not results:
        LOGGER.debug("OpenAlex: no results for identifier=%r", identifier)
        return None
    openalex_id = results[0].get("id")
    if isinstance(openalex_id, str) and openalex_id.startswith("https://openalex.org/"):
        return openalex_id.split("/")[-1]
    if isinstance(openalex_id, str) and openalex_id:
        return openalex_id
    return None
