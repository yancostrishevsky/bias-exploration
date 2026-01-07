"""Enrichment of records with OpenAlex metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import html
import httpx
from diskcache import Cache
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_bias_search.utils.core_rankings import lookup_core_rank
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.ids import best_identifier, normalise_doi
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.models import EnrichedRecord
from ai_bias_search.utils.rate_limit import RateLimiter


LOGGER = configure_logging()
CACHE_DIR = (Path(__file__).resolve().parents[2] / "data" / "cache" / "openalex").resolve()

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
    "ACM", "IEEE", "SPRINGER", "ELSEVIER", "NATURE", "SCIENCE",
    "PROC", "PROCEEDINGS", "VOLUME", "VOL", "NO", "INTERNATIONAL", "CONFERENCE",
    "JOURNAL", "TRANSACTIONS", "SYMPOSIUM", "WORKSHOP",
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
    """Wyciąga akronim konferencji z nazw typu 'Proceedings ... - EMNLP '09' albo 'ACM SIGIR conference...'."""
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
    tokens = [_clean_acronym_token(x) for x in _ACR_TOKEN_RE.findall(t)]
    tokens = [x for x in tokens if x]  # type: ignore[arg-type]
    if not tokens:
        return None

    # heurystyka: preferuj dłuższe (SIGKDD > KDD), ale nie absurdalnie długie
    tokens.sort(key=lambda x: (len(x), x), reverse=True)
    return tokens[0]

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
        s2_venue_name = _normalize_venue_name(s2_venue_raw if isinstance(s2_venue_raw, str) else None)
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
        candidate = _normalize_acronym(value if isinstance(value, str) else None, source=source_key, reasons=reasons)
        if candidate:
            venue_acronym = candidate
            acronym_source = source_key
            reasons.append(f"venue_acronym:{source_key}")
            break

    if not venue_acronym:
        dblp_candidate, dblp_source = _acronym_from_dblp(s2_dblp)
        if dblp_candidate:
            candidate = _normalize_acronym(dblp_candidate, source="s2_dblp", reasons=reasons)
            if candidate:
                venue_acronym = candidate
                acronym_source = dblp_source or "s2_dblp"
                reasons.append(f"venue_acronym:{acronym_source}")

    if not venue_acronym:
        s2_venue_name = _normalize_venue_name(s2_venue_raw if isinstance(s2_venue_raw, str) else None)
        heur_candidate, heur_source = _acronym_from_s2_venue(s2_venue_name)
        if heur_candidate:
            candidate = _normalize_acronym(heur_candidate, source="s2_venue", reasons=reasons)
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



def enrich_with_openalex(records: List[Dict[str, Any]], mailto: str | None) -> List[Dict[str, Any]]:
    """Augment *records* with OpenAlex metadata."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(rate=2, burst=5)
    retries = RetryConfig()
    retrying = Retrying(
        stop=stop_after_attempt(retries.max),
        wait=wait_exponential(multiplier=1, exp_base=retries.backoff, min=1),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )

    enriched: List[Dict[str, Any]] = []
    with Cache(CACHE_DIR) as cache:
        with httpx.Client(base_url="https://api.openalex.org", timeout=30.0) as client:
            for record in records:
                identifier = best_identifier(record)  # powinien preferować DOI
                if not identifier:
                    LOGGER.debug("OpenAlex: no identifier for record (title=%r)", record.get("title"))
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

                # --- venue & CORE ranking resolution (FIXED ORDER) ---
                core_lookup = extract_venue_candidates(metadata, record)
                venue_name = core_lookup.get("venue_name")
                venue_acronym = core_lookup.get("venue_acronym")
                eligible = core_lookup.get("eligible")
                reasons = core_lookup.get("reasons")
                raw = core_lookup.get("raw")

                merged.host_venue = venue_name

                LOGGER.info(
                    "CORE query: venue_name=%r venue_acronym=%r eligible=%s reasons=%s raw=%s",
                    venue_name,
                    venue_acronym,
                    eligible,
                    reasons,
                    _compact_raw(raw),
                )

                if eligible:
                    merged.core_rank = lookup_core_rank(
                        venue_name=venue_name,
                        venue_acronym=venue_acronym,
                    )
                else:
                    merged.core_rank = None

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
            resp = retrying(lambda: (r := execute_direct(), r.raise_for_status(), r)[0])  # noqa: E731
            # jeśli jest 200, mamy komplet
            payload = resp.json()
            work_id = payload.get("id")
            if isinstance(work_id, str) and work_id.startswith("https://openalex.org/"):
                return work_id.split("/")[-1]
        except httpx.HTTPStatusError as exc:
            # 404 przy direct → spróbuj filter=doi:...
            if exc.response.status_code != 404:
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
