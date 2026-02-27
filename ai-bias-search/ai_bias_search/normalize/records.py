"""Canonical metadata normalization for cross-platform bias analyses."""

from __future__ import annotations

import ast
from collections import Counter
import math
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

try:  # pragma: no cover - optional dependency in runtime environments.
    import pycountry  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    pycountry = None  # type: ignore[assignment]


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_publisher(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text in {"0", "0.0"}:
        return None
    return text


def extract_openalex_publisher(work: dict[str, Any]) -> tuple[str | None, str]:
    """Return OpenAlex publisher and provenance."""

    payload = work if isinstance(work, dict) else {}
    primary_location = _coerce_mapping(payload.get("primary_location"))
    source = _coerce_mapping(primary_location.get("source"))
    host = _coerce_mapping(payload.get("host_venue"))
    if not host:
        host = _coerce_mapping(source.get("host_venue"))

    host_org_name = _clean_publisher(source.get("host_organization_name"))
    if host_org_name:
        return host_org_name, "host_org_name"

    lineage = source.get("host_organization_lineage_names")
    if isinstance(lineage, (list, tuple)):
        for candidate in reversed(list(lineage)):
            lineage_name = _clean_publisher(candidate)
            if lineage_name:
                return lineage_name, "host_org_lineage_last"

    host_venue_publisher = _clean_publisher(host.get("publisher"))
    if host_venue_publisher:
        return host_venue_publisher, "host_venue_publisher"

    source_name = _clean_publisher(source.get("display_name"))
    if source_name:
        return source_name, "source_name_fallback"

    return None, "missing"


def _openalex_payload(raw: dict[str, Any]) -> dict[str, Any]:
    extra = _coerce_mapping(raw.get("extra"))
    return _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))


def _openalex_publisher_from_payload(payload: dict[str, Any]) -> str | None:
    publisher, _ = extract_openalex_publisher(payload)
    return publisher


def _semanticscholar_publisher_from_payload(payload: dict[str, Any]) -> str | None:
    venue = _coerce_mapping(payload.get("venue"))
    journal = _coerce_mapping(payload.get("journal"))
    publication_venue = _coerce_mapping(payload.get("publicationVenue"))
    return (
        _clean_publisher(venue.get("publisher"))
        or _clean_publisher(journal.get("publisher"))
        or _clean_publisher(publication_venue.get("publisher"))
        or _clean_publisher(payload.get("publisher"))
    )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


_YEAR_IN_TEXT_RE = re.compile(r"(18|19|20|21)\d{2}")


def _coerce_year_from_text(value: object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    match = _YEAR_IN_TEXT_RE.search(text)
    if not match:
        return None
    try:
        year = int(match.group(0))
    except ValueError:
        return None
    if 1800 <= year <= 2100:
        return year
    return None


def _coerce_year(value: object) -> int | None:
    year = _coerce_int(value)
    if year is None and isinstance(value, str):
        year = _coerce_year_from_text(value)
    if year is None:
        return None
    if 1800 <= year <= 2100:
        return year
    return None


def _is_plausible_year(
    value: int | None,
    *,
    min_year: int = 1800,
    max_year: int | None = None,
) -> bool:
    if value is None:
        return False
    if max_year is None:
        max_year = datetime.utcnow().year + 1
    return min_year <= value <= max_year


def _coerce_plausible_year(
    value: object,
    *,
    min_year: int = 1800,
    max_year: int | None = None,
) -> int | None:
    year = _coerce_year(value)
    if not _is_plausible_year(year, min_year=min_year, max_year=max_year):
        return None
    return year


def resolve_year(
    raw_year: object,
    raw_date: object,
    enriched_year: object,
    enriched_date: object,
    policy: dict[str, Any] | None = None,
) -> tuple[int | None, dict[str, Any]]:
    """Resolve canonical year with deterministic precedence and provenance."""

    cfg = policy or {}
    current_year = _coerce_int(cfg.get("current_year")) or datetime.utcnow().year
    min_year = _coerce_int(cfg.get("min_year")) or 1800
    max_year = _coerce_int(cfg.get("max_year")) or (current_year + 1)

    raw_year_source = str(cfg.get("raw_year_source") or "raw.year")
    raw_date_source = str(cfg.get("raw_date_source") or "raw.publicationDate")
    enriched_year_source = str(cfg.get("enriched_year_source") or "enriched.year")
    enriched_date_source = str(cfg.get("enriched_date_source") or "enriched.date")

    raw_year_value = _coerce_plausible_year(raw_year, min_year=min_year, max_year=max_year)
    raw_date_year = _coerce_plausible_year(raw_date, min_year=min_year, max_year=max_year)
    enriched_year_value = _coerce_plausible_year(
        enriched_year, min_year=min_year, max_year=max_year
    )
    enriched_date_year = _coerce_plausible_year(
        enriched_date, min_year=min_year, max_year=max_year
    )
    enriched_candidate = (
        enriched_year_value if enriched_year_value is not None else enriched_date_year
    )
    enriched_source = (
        enriched_year_source if enriched_year_value is not None else enriched_date_source
    )

    discrepancy = bool(
        raw_year_value is not None
        and enriched_candidate is not None
        and abs(raw_year_value - enriched_candidate) > 1
    )
    if raw_year_value is not None:
        value = raw_year_value
        source = raw_year_source
    elif raw_date_year is not None:
        value = raw_date_year
        source = raw_date_source
    elif enriched_candidate is not None:
        value = enriched_candidate
        source = enriched_source
    else:
        value = None
        source = "missing"

    provenance = {
        "value": value,
        "source": source,
        "enriched_candidate": enriched_candidate,
        "enriched_source": (enriched_source if enriched_candidate is not None else None),
        "discrepancy": discrepancy,
    }
    return value, provenance


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "open"}:
        return True
    if text in {"0", "false", "no", "closed"}:
        return False
    return None


def _coerce_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


_ISO2_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
_ISO3_COUNTRY_RE = re.compile(r"^[A-Z]{3}$")
_QUOTED_LIST_TOKEN_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    "argentina": "AR",
    "australia": "AU",
    "austria": "AT",
    "belgium": "BE",
    "brazil": "BR",
    "canada": "CA",
    "china": "CN",
    "czech republic": "CZ",
    "denmark": "DK",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "greece": "GR",
    "hungary": "HU",
    "india": "IN",
    "ireland": "IE",
    "italy": "IT",
    "japan": "JP",
    "luxembourg": "LU",
    "netherlands": "NL",
    "new zealand": "NZ",
    "norway": "NO",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "russian federation": "RU",
    "russia": "RU",
    "singapore": "SG",
    "south africa": "ZA",
    "south korea": "KR",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "taiwan": "TW",
    "turkey": "TR",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
}


def normalize_country_code(value: str) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text).strip()
    code = compact.upper()
    if _ISO2_COUNTRY_RE.fullmatch(code):
        return code
    mapped = _COUNTRY_NAME_TO_ISO2.get(compact.casefold())
    if mapped:
        return mapped

    if _ISO3_COUNTRY_RE.fullmatch(code) and pycountry is not None:
        match = pycountry.countries.get(alpha_3=code)
        if match is not None:
            alpha2 = _clean_text(getattr(match, "alpha_2", None))
            if alpha2 and _ISO2_COUNTRY_RE.fullmatch(alpha2.upper()):
                return alpha2.upper()

    if pycountry is None:
        return None

    for key in ("alpha_2", "alpha_3", "name"):
        match = pycountry.countries.get(**{key: compact})
        if match is None:
            continue
        alpha2 = _clean_text(getattr(match, "alpha_2", None))
        if alpha2 and _ISO2_COUNTRY_RE.fullmatch(alpha2.upper()):
            return alpha2.upper()

    lowered = compact.casefold()
    for country in pycountry.countries:
        common_name = _clean_text(getattr(country, "common_name", None))
        official_name = _clean_text(getattr(country, "official_name", None))
        if common_name and common_name.casefold() == lowered:
            alpha2 = _clean_text(getattr(country, "alpha_2", None))
            if alpha2 and _ISO2_COUNTRY_RE.fullmatch(alpha2.upper()):
                return alpha2.upper()
        if official_name and official_name.casefold() == lowered:
            alpha2 = _clean_text(getattr(country, "alpha_2", None))
            if alpha2 and _ISO2_COUNTRY_RE.fullmatch(alpha2.upper()):
                return alpha2.upper()

    try:
        fuzzy = pycountry.countries.search_fuzzy(compact)
    except Exception:
        fuzzy = []
    for candidate in fuzzy:
        alpha2 = _clean_text(getattr(candidate, "alpha_2", None))
        if alpha2 and _ISO2_COUNTRY_RE.fullmatch(alpha2.upper()):
            return alpha2.upper()
    return None


def _normalize_country_iso2(value: object) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    return normalize_country_code(text)


def _country_tokens_from_string(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    quoted: list[str] = []
    for first, second in _QUOTED_LIST_TOKEN_RE.findall(text):
        token = _clean_text(first or second)
        if token:
            quoted.append(token)
    if len(quoted) >= 2:
        return quoted

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        out: list[str] = []
        for item in parsed:
            cleaned = _clean_text(item)
            if cleaned:
                out.append(cleaned)
        return out

    if quoted:
        return quoted

    if text.startswith("[") and text.endswith("]"):
        middle = text[1:-1].strip()
        if not middle:
            return []
        if "," in middle or ";" in middle:
            out: list[str] = []
            for token in middle.replace(";", ",").split(","):
                cleaned = _clean_text(token.strip("'\""))
                if cleaned:
                    out.append(cleaned)
            return out
    return []


def _iter_country_values(value: object) -> Iterable[object]:
    for item in _ensure_sequence(value):
        if isinstance(item, str):
            tokens = _country_tokens_from_string(item)
            if tokens:
                for token in tokens:
                    yield token
                continue
        yield item


def _add_country(value: object, *, out: Counter[str]) -> None:
    if value is None:
        return
    for item in _iter_country_values(value):
        code = _normalize_country_iso2(item)
        if code:
            out[code] += 1


def _openalex_country_counts(raw: dict[str, Any]) -> Counter[str]:
    payload = _openalex_payload(raw)
    if not payload:
        payload = _coerce_mapping(_coerce_mapping(raw.get("extra")).get("openalex"))

    counts: Counter[str] = Counter()
    authorships = payload.get("authorships")
    if not isinstance(authorships, list):
        return counts
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        institutions = _ensure_sequence(authorship.get("institutions"))
        for institution in institutions:
            if not isinstance(institution, dict):
                continue
            _add_country(institution.get("country_code"), out=counts)
    return counts


def _scopus_country_counts(raw: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))

    # Preferred: values extracted from Abstract Retrieval response.
    _add_country(abstract.get("countries"), out=counts)
    _add_country(abstract.get("affiliation_countries"), out=counts)

    response = _coerce_mapping(abstract.get("response"))
    affiliations = [entry for entry in _ensure_sequence(response.get("affiliation")) if isinstance(entry, dict)]
    aff_lookup: dict[str, dict[str, Any]] = {}
    for entry in affiliations:
        _add_country(entry.get("affiliation-country") or entry.get("country"), out=counts)
        aff_id = _clean_text(entry.get("@id") or entry.get("affiliation-id") or entry.get("id"))
        if aff_id:
            aff_lookup[aff_id] = entry

    authors = _coerce_mapping(response.get("authors"))
    for author in _ensure_sequence(authors.get("author")):
        if not isinstance(author, dict):
            continue
        for affiliation in _ensure_sequence(author.get("affiliation")):
            if not isinstance(affiliation, dict):
                continue
            _add_country(affiliation.get("affiliation-country") or affiliation.get("country"), out=counts)
            ref_id = _clean_text(
                affiliation.get("@id") or affiliation.get("affiliation-id") or affiliation.get("id")
            )
            if ref_id and ref_id in aff_lookup:
                _add_country(
                    aff_lookup[ref_id].get("affiliation-country") or aff_lookup[ref_id].get("country"),
                    out=counts,
                )

    # Fallback to already materialized enrichment fields when response payload is absent.
    if not counts:
        _add_country(raw.get("countries"), out=counts)
    if not counts:
        _add_country(raw.get("affiliation_countries"), out=counts)
    if not counts:
        scopus_enrich = _coerce_mapping(_coerce_mapping(raw.get("extra")).get("scopus_enrich"))
        _add_country(scopus_enrich.get("countries"), out=counts)
    if not counts:
        _add_country(scopus_enrich.get("affiliation_countries"), out=counts)

    return counts


def _explicit_country_counts(raw: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    _add_country(raw.get("countries"), out=counts)
    _add_country(raw.get("affiliation_countries"), out=counts)
    return counts


def _resolve_country_fields(raw: dict[str, Any], *, platform: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    provenance = "missing"

    if platform == "openalex":
        counts = _openalex_country_counts(raw)
        if counts:
            provenance = "openalex.authorships.institutions.country_code"
    elif platform == "scopus":
        counts = _scopus_country_counts(raw)
        if counts:
            provenance = "scopus.abstract.affiliation-country"
    elif platform in {"semanticscholar", "core"}:
        counts = _explicit_country_counts(raw)
        if counts:
            if raw.get("countries") is not None:
                provenance = "record.countries"
            else:
                provenance = "record.affiliation_countries"
        else:
            provenance = f"{platform}.structural_unavailable"
    else:
        counts = _explicit_country_counts(raw)
        if counts:
            if raw.get("countries") is not None:
                provenance = "record.countries"
            else:
                provenance = "record.affiliation_countries"

    if not counts:
        return {
            "countries": None,
            "country_primary": None,
            "affiliation_countries": None,
            "country_dominant": None,
            "country_count": None,
            "country_provenance": provenance,
            "country_is_fractional": False,
            "country_counts": {},
        }

    countries = sorted(counts.keys())
    primary = countries[0] if len(countries) == 1 else "MULTI"
    return {
        "countries": countries,
        "country_primary": primary,
        "affiliation_countries": countries,
        "country_dominant": primary,
        "country_count": len(countries),
        "country_provenance": provenance,
        "country_is_fractional": False,
        "country_counts": dict(sorted(counts.items())),
    }


def _semanticscholar_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return _coerce_mapping(_coerce_mapping(raw.get("extra")).get("semanticscholar"))


def _priority_from_env(name: str, default: Sequence[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items = [item.strip().lower() for item in raw.split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    if not out:
        return list(default)
    return out


def _coerce_str_list(value: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in _ensure_sequence(value):
        text = _clean_text(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _ensure_sequence(value: object) -> Sequence[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if not isinstance(value, str) and hasattr(value, "tolist"):
        try:
            as_list = value.tolist()
        except Exception:
            as_list = None
        if isinstance(as_list, list):
            return as_list
        if isinstance(as_list, tuple):
            return list(as_list)
    return [value]


def _add_issn(value: object, out: list[str], seen: set[str]) -> None:
    if value is None:
        return
    for item in _ensure_sequence(value):
        text = _clean_text(item)
        if not text:
            continue
        for token in text.replace(";", ",").split(","):
            normalized = normalize_issn(_clean_text(token))
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)


def _add_issn_candidates(
    value: object,
    *,
    family: str,
    field_path: str,
    kind: str,
    out: list[dict[str, Any]],
) -> None:
    tokens: list[str] = []
    _add_issn(value, tokens, set())
    for token in tokens:
        out.append(
            {
                "value": token,
                "family": family,
                "field_path": field_path,
                "kind": kind,
            }
        )


def _collect_issn_candidates(raw: dict[str, Any], platform: str) -> list[dict[str, Any]]:
    extra = _coerce_mapping(raw.get("extra"))
    payload_openalex = _openalex_payload(raw)
    payload_scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
    payload_scopus_abs = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
    payload_sem = _semanticscholar_payload(raw)
    payload_core = _coerce_mapping(extra.get("core"))
    payload_core_identifiers = _coerce_mapping(payload_core.get("identifiers"))

    candidates: list[dict[str, Any]] = []

    # raw fallback
    _add_issn_candidates(raw.get("issn"), family="raw", field_path="issn", kind="print", out=candidates)
    _add_issn_candidates(raw.get("eissn"), family="raw", field_path="eissn", kind="electronic", out=candidates)
    _add_issn_candidates(
        raw.get("issn_list"),
        family="raw",
        field_path="issn_list",
        kind="unknown",
        out=candidates,
    )

    # openalex
    openalex_source = _coerce_mapping(
        _coerce_mapping(_coerce_mapping(payload_openalex.get("primary_location")).get("source"))
    )
    openalex_host = _coerce_mapping(payload_openalex.get("host_venue"))
    _add_issn_candidates(
        openalex_source.get("issn"), family="openalex", field_path="extra.openalex.source.issn", kind="unknown", out=candidates
    )
    _add_issn_candidates(
        openalex_source.get("issn_l"), family="openalex", field_path="extra.openalex.source.issn_l", kind="unknown", out=candidates
    )
    _add_issn_candidates(
        openalex_host.get("issn"), family="openalex", field_path="extra.openalex.host_venue.issn", kind="unknown", out=candidates
    )
    _add_issn_candidates(
        openalex_host.get("issn_l"), family="openalex", field_path="extra.openalex.host_venue.issn_l", kind="unknown", out=candidates
    )

    # scopus
    _add_issn_candidates(
        payload_scopus_raw.get("prism:issn"),
        family="scopus",
        field_path="extra.scopus.raw.prism:issn",
        kind="print",
        out=candidates,
    )
    _add_issn_candidates(
        payload_scopus_raw.get("prism:eIssn") or payload_scopus_raw.get("prism:eissn"),
        family="scopus",
        field_path="extra.scopus.raw.prism:eIssn",
        kind="electronic",
        out=candidates,
    )
    _add_issn_candidates(
        payload_scopus_abs.get("issn"),
        family="scopus",
        field_path="scopus.abstract.issn",
        kind="print",
        out=candidates,
    )
    _add_issn_candidates(
        payload_scopus_abs.get("eissn"),
        family="scopus",
        field_path="scopus.abstract.eissn",
        kind="electronic",
        out=candidates,
    )

    # semanticscholar
    sem_journal = _coerce_mapping(payload_sem.get("journal"))
    sem_venue = _coerce_mapping(payload_sem.get("publicationVenue"))
    sem_external = _coerce_mapping(payload_sem.get("externalIds"))
    _add_issn_candidates(
        sem_venue.get("issn"),
        family="semanticscholar",
        field_path="extra.semanticscholar.publicationVenue.issn",
        kind="unknown",
        out=candidates,
    )
    _add_issn_candidates(
        sem_journal.get("issn"),
        family="semanticscholar",
        field_path="extra.semanticscholar.journal.issn",
        kind="unknown",
        out=candidates,
    )
    _add_issn_candidates(
        sem_external.get("ISSN"),
        family="semanticscholar",
        field_path="extra.semanticscholar.externalIds.ISSN",
        kind="unknown",
        out=candidates,
    )

    # core
    _add_issn_candidates(
        payload_core.get("issn"),
        family="core",
        field_path="extra.core.issn",
        kind="print",
        out=candidates,
    )
    _add_issn_candidates(
        payload_core.get("eissn"),
        family="core",
        field_path="extra.core.eissn",
        kind="electronic",
        out=candidates,
    )
    _add_issn_candidates(
        payload_core_identifiers.get("issn"),
        family="core",
        field_path="extra.core.identifiers.issn",
        kind="unknown",
        out=candidates,
    )
    _add_issn_candidates(
        payload_core_identifiers.get("eissn"),
        family="core",
        field_path="extra.core.identifiers.eissn",
        kind="unknown",
        out=candidates,
    )

    for candidate in candidates:
        candidate["provenance"] = (
            "raw"
            if candidate["family"] in {"raw", platform}
            else "enriched"
        )
    return candidates


def canonical_issn_selection(
    raw: dict[str, Any],
    *,
    platform: str,
    source_priority: Sequence[str] | None = None,
) -> dict[str, Any]:
    priority = list(source_priority) if source_priority else _priority_from_env(
        "ISSN_SOURCE_PRIORITY",
        ("scopus", "openalex", "semanticscholar", "core", "raw"),
    )
    rank = {name: idx for idx, name in enumerate(priority)}
    candidates = _collect_issn_candidates(raw, platform)
    indexed = [
        (idx, candidate)
        for idx, candidate in enumerate(candidates)
    ]
    sorted_candidates = [
        candidate
        for _, candidate in sorted(
            indexed,
            key=lambda pair: (
                rank.get(str(pair[1].get("family")), len(priority) + 1),
                pair[0],
            ),
        )
    ]
    issn_list: list[str] = []
    source_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted_candidates:
        value = str(candidate.get("value") or "")
        if not value or value in seen:
            continue
        seen.add(value)
        issn_list.append(value)
        source_rows.append(candidate)

    issn = next(
        (row["value"] for row in source_rows if row.get("kind") == "print"),
        (issn_list[0] if issn_list else None),
    )
    eissn = next(
        (row["value"] for row in source_rows if row.get("kind") == "electronic" and row.get("value") != issn),
        None,
    )
    source_names: list[str] = []
    provenance_labels: list[str] = []
    for row in source_rows:
        family = str(row.get("family"))
        if family not in source_names:
            source_names.append(family)
        provenance = str(row.get("provenance"))
        if provenance not in provenance_labels:
            provenance_labels.append(provenance)

    if not issn_list:
        issn_provenance = "missing"
    elif len(provenance_labels) == 1:
        issn_provenance = provenance_labels[0]
    else:
        issn_provenance = "mixed"

    return {
        "issn": issn,
        "eissn": eissn,
        "issn_list": issn_list,
        "issn_source": (source_names[0] if source_names else None),
        "issn_provenance": issn_provenance,
        "sources_used": source_names,
        "source_rows": source_rows,
    }


def _add_issn_with_source(
    value: object,
    *,
    source: str,
    out: list[str],
    seen: set[str],
    sources_used: list[str],
) -> None:
    before = len(out)
    _add_issn(value, out, seen)
    if len(out) > before and source not in sources_used:
        sources_used.append(source)


def _extract_semanticscholar_issn(
    raw: dict[str, Any],
) -> tuple[list[str], str | None, dict[str, Any]]:
    payload = _semanticscholar_payload(raw)
    publication_venue = _coerce_mapping(payload.get("publicationVenue"))
    journal = _coerce_mapping(payload.get("journal"))
    external_ids = _coerce_mapping(payload.get("externalIds"))

    out: list[str] = []
    seen: set[str] = set()
    sources_used: list[str] = []

    _add_issn_with_source(
        publication_venue.get("issn"),
        source="publicationVenue.issn",
        out=out,
        seen=seen,
        sources_used=sources_used,
    )
    _add_issn_with_source(
        journal.get("issn"),
        source="journal.issn",
        out=out,
        seen=seen,
        sources_used=sources_used,
    )
    _add_issn_with_source(
        external_ids.get("ISSN"),
        source="externalIds.ISSN",
        out=out,
        seen=seen,
        sources_used=sources_used,
    )
    _add_issn_with_source(
        raw.get("issn"),
        source="record.issn",
        out=out,
        seen=seen,
        sources_used=sources_used,
    )
    _add_issn_with_source(
        raw.get("eissn"),
        source="record.eissn",
        out=out,
        seen=seen,
        sources_used=sources_used,
    )
    _add_issn_with_source(
        raw.get("issn_list"),
        source="record.issn_list",
        out=out,
        seen=seen,
        sources_used=sources_used,
    )

    provenance = {
        "preferred_source": (sources_used[0] if sources_used else None),
        "sources_used": sources_used,
        "count": len(out),
    }
    return out, ("direct" if out else None), provenance


def _extract_issn(raw: dict[str, Any], platform: str) -> tuple[list[str], str | None]:
    out: list[str] = []
    seen: set[str] = set()

    _add_issn(raw.get("issn"), out, seen)
    _add_issn(raw.get("eissn"), out, seen)
    _add_issn(raw.get("issn_list"), out, seen)

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        openalex_payload = _openalex_payload(raw)
        source = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(openalex_payload).get("primary_location")).get(
                "source"
            )
        )
        host = _coerce_mapping(_coerce_mapping(openalex_payload).get("host_venue"))
        _add_issn(source.get("issn"), out, seen)
        _add_issn(source.get("issn_l"), out, seen)
        _add_issn(host.get("issn"), out, seen)
        _add_issn(host.get("issn_l"), out, seen)
    elif platform == "scopus":
        scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
        _add_issn(scopus_raw.get("prism:issn"), out, seen)
        _add_issn(scopus_raw.get("prism:eIssn"), out, seen)
        _add_issn(scopus_raw.get("prism:eissn"), out, seen)
        abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
        _add_issn(abstract.get("issn"), out, seen)
        _add_issn(abstract.get("eissn"), out, seen)
        _add_issn(abstract.get("issn_list"), out, seen)
    elif platform == "semanticscholar":
        values, _, _ = _extract_semanticscholar_issn(raw)
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
    elif platform == "core":
        payload = _coerce_mapping(extra.get("core"))
        identifiers = _coerce_mapping(payload.get("identifiers"))
        _add_issn(payload.get("issn"), out, seen)
        _add_issn(payload.get("eissn"), out, seen)
        _add_issn(identifiers.get("issn"), out, seen)
        _add_issn(identifiers.get("eissn"), out, seen)

    if out:
        return out, "direct"
    return out, None


def _extract_publisher(raw: dict[str, Any], platform: str) -> str | None:
    publisher = _clean_publisher(raw.get("publisher"))
    if publisher:
        return publisher

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _openalex_payload(raw)
        return _openalex_publisher_from_payload(payload)
    if platform == "scopus":
        if _clean_publisher(raw.get("jcr_publisher")):
            return _clean_publisher(raw.get("jcr_publisher"))
        scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
        for key in ("publishername", "dc:publisher", "publisher", "prism:publisher"):
            value = _clean_publisher(scopus_raw.get(key))
            if value:
                return value
        scopus_abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
        for key in ("publisher", "publishername", "dc:publisher"):
            value = _clean_publisher(scopus_abstract.get(key))
            if value:
                return value
    if platform == "semanticscholar":
        payload = _coerce_mapping(extra.get("semanticscholar"))
        return _semanticscholar_publisher_from_payload(payload)
    if platform == "core":
        payload = _coerce_mapping(extra.get("core"))
        return _clean_publisher(payload.get("publisher"))

    return None


def _extract_journal_title(raw: dict[str, Any], platform: str) -> str | None:
    for key in ("journal_title", "publication_name", "publicationName", "host_venue", "source"):
        value = _clean_text(raw.get(key))
        if value:
            return value

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _openalex_payload(raw)
        source = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(payload.get("primary_location")).get("source"))
        )
        host = _coerce_mapping(payload.get("host_venue"))
        return _clean_text(source.get("display_name")) or _clean_text(host.get("display_name"))
    if platform == "scopus":
        scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
        return _clean_text(scopus_raw.get("prism:publicationName")) or _clean_text(
            scopus_raw.get("publicationName")
        )
    if platform == "semanticscholar":
        payload = _semanticscholar_payload(raw)
        journal = _coerce_mapping(payload.get("journal"))
        return _clean_text(journal.get("name")) or _clean_text(payload.get("venue"))
    if platform == "core":
        payload = _coerce_mapping(extra.get("core"))
        return _clean_text(payload.get("venue")) or _clean_text(payload.get("journal"))
    return None


def _extract_year(raw: dict[str, Any]) -> int | None:
    for key in ("publication_year", "year"):
        year = _coerce_year(raw.get(key))
        if year is not None:
            return year
    return None


def _collect_year_candidates(
    raw: dict[str, Any],
    *,
    platform: str,
) -> tuple[int | None, list[dict[str, Any]]]:
    sem_payload = _semanticscholar_payload(raw)

    # Semantic Scholar records often carry enrichment-updated top-level `year` values.
    # Treat the native payload year/date as raw first to avoid silent overrides.
    if platform == "semanticscholar":
        raw_year = _coerce_plausible_year(sem_payload.get("year"))
        if raw_year is None:
            raw_year = _coerce_plausible_year(sem_payload.get("publicationDate"))
        if raw_year is None:
            raw_year = _coerce_plausible_year(raw.get("year"))
        if raw_year is None:
            raw_year = _coerce_plausible_year(raw.get("publication_year"))
        if raw_year is None:
            raw_year = _coerce_plausible_year(raw.get("publicationDate"))
    else:
        raw_year = _coerce_plausible_year(raw.get("year"))
        if raw_year is None:
            raw_year = _coerce_plausible_year(raw.get("publication_year"))
        if raw_year is None:
            raw_year = _coerce_plausible_year(raw.get("publicationDate"))

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add(source: str, value: object, field_path: str) -> None:
        year_value = _coerce_plausible_year(value)
        if year_value is None:
            return
        key = (source, year_value)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "source": source,
                "value": year_value,
                "field_path": field_path,
            }
        )

    openalex_payload = _openalex_payload(raw)
    add("openalex", openalex_payload.get("publication_year"), "extra.openalex_enrich.publication_year")
    add(
        "openalex",
        _coerce_mapping(_coerce_mapping(raw.get("extra")).get("openalex")).get("publication_year"),
        "extra.openalex.publication_year",
    )

    scopus_abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
    add("scopus", scopus_abstract.get("publication_year"), "scopus.abstract.publication_year")
    scopus_enrich = _coerce_mapping(_coerce_mapping(raw.get("extra")).get("scopus_enrich"))
    add("scopus", scopus_enrich.get("cover_date"), "extra.scopus_enrich.cover_date")

    add("other", raw.get("publication_year"), "publication_year")
    add("other", sem_payload.get("publicationDate"), "extra.semanticscholar.publicationDate")

    return raw_year, candidates


def _resolve_year_with_provenance(
    *,
    raw: dict[str, Any],
    platform: str,
) -> dict[str, Any]:
    raw_year, candidates = _collect_year_candidates(raw, platform=platform)
    source_priority = _priority_from_env(
        "YEAR_TRUST_PRIORITY",
        ("scopus", "openalex", "other", "raw"),
    )
    rank = {name: idx for idx, name in enumerate(source_priority)}
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            rank.get(str(item.get("source")), len(source_priority) + 1),
            -int(item.get("value") or 0),
        ),
    )
    enriched_candidate = ranked_candidates[0] if ranked_candidates else None
    enriched_year = int(enriched_candidate["value"]) if enriched_candidate else None
    enriched_source = str(enriched_candidate["source"]) if enriched_candidate else None
    year_provenance = "missing"
    chosen_source = "missing"

    if raw_year is not None and enriched_year is not None:
        year_provenance = "mixed"
        if raw_year == enriched_year:
            final_year = raw_year
            chosen_source = "raw"
        else:
            raw_rank = rank.get("raw", len(source_priority) + 1)
            enriched_rank = rank.get(enriched_source or "other", len(source_priority) + 1)
            if enriched_rank <= raw_rank:
                final_year = enriched_year
                chosen_source = f"enriched:{enriched_source}"
            else:
                final_year = raw_year
                chosen_source = "raw"
    elif raw_year is not None:
        final_year = raw_year
        chosen_source = "raw"
        year_provenance = "raw"
    elif enriched_year is not None:
        final_year = enriched_year
        chosen_source = f"enriched:{enriched_source}"
        year_provenance = "enriched"
    else:
        final_year = None

    final_year = final_year if final_year not in (0, "0") else None
    return {
        "year": final_year,
        "year_raw": raw_year,
        "year_enriched": enriched_year,
        "year_provenance": year_provenance,
        "year_source": chosen_source,
        "year_candidates": ranked_candidates,
    }


def _resolve_semanticscholar_year(raw: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    payload = _semanticscholar_payload(raw)
    scopus_abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
    openalex_payload = _openalex_payload(raw)

    enriched_year = (
        _coerce_year(raw.get("publication_year"))
        or _coerce_year(openalex_payload.get("publication_year"))
        or _coerce_year(scopus_abstract.get("publication_year"))
    )
    enriched_date = (
        raw.get("cover_date")
        or scopus_abstract.get("cover_date")
        or _coerce_mapping(_coerce_mapping(raw.get("extra")).get("scopus_enrich")).get("cover_date")
    )
    return resolve_year(
        payload.get("year") if "year" in payload else raw.get("year"),
        payload.get("publicationDate") if "publicationDate" in payload else raw.get("publicationDate"),
        enriched_year,
        enriched_date,
        policy={
            "raw_year_source": "semantic_scholar.year",
            "raw_date_source": "semantic_scholar.publicationDate",
            "enriched_year_source": (
                "openalex.publication_year"
                if _coerce_year(openalex_payload.get("publication_year")) is not None
                else (
                    "scopus.publication_year"
                    if _coerce_year(scopus_abstract.get("publication_year")) is not None
                    else "record.publication_year"
                )
            ),
            "enriched_date_source": "scopus.cover_date",
        },
    )


def _resolve_semanticscholar_journal_title(raw: dict[str, Any]) -> tuple[str | None, str]:
    payload = _semanticscholar_payload(raw)
    publication_venue = _coerce_mapping(payload.get("publicationVenue"))
    journal = _coerce_mapping(payload.get("journal"))
    venue_value = payload.get("venue")

    publication_venue_name = _clean_text(publication_venue.get("name"))
    if publication_venue_name:
        return publication_venue_name, "publicationVenue.name"

    journal_name = _clean_text(journal.get("name"))
    if journal_name:
        return journal_name, "journal.name"

    if isinstance(venue_value, str):
        venue_name = _clean_text(venue_value)
    else:
        venue_name = _clean_text(raw.get("venue"))
    if venue_name:
        return venue_name, "venue"

    return None, "missing"


def _extract_open_access(raw: dict[str, Any], platform: str) -> bool | None:
    for key in ("is_oa", "is_open_access", "openaccess_flag", "openaccessFlag"):
        value = _coerce_bool(raw.get(key))
        if value is not None:
            return value

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _openalex_payload(raw)
        oa = _coerce_mapping(payload.get("open_access"))
        value = _coerce_bool(oa.get("is_oa"))
        if value is not None:
            return value
    if platform == "semanticscholar":
        payload = _coerce_mapping(extra.get("semanticscholar"))
        value = _coerce_bool(payload.get("isOpenAccess"))
        if value is not None:
            return value
    return None


def _extract_citations(raw: dict[str, Any], platform: str) -> tuple[int | None, bool]:
    def from_mapping(payload: dict[str, Any], keys: Iterable[str]) -> tuple[int | None, bool]:
        for key in keys:
            if key in payload:
                return _coerce_int(payload.get(key)), True
        return None, False

    if platform == "scopus":
        keys = ("citedby-count", "cited_by_count", "citedby_count")
        if raw.get("citations_field_present") is False:
            raw_value, raw_present = None, False
        else:
            raw_value, raw_present = from_mapping(raw, keys)
        if raw_present and raw_value is not None:
            return raw_value, True
        extra = _coerce_mapping(raw.get("extra"))
        scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
        value, present = from_mapping(scopus_raw, keys)
        if present:
            return value, True
        abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
        value, present = from_mapping(abstract, keys)
        if present:
            return value, True
        if raw_present:
            return None, True
        return None, False

    keys = ("citations", "cited_by_count", "citationCount", "citedby-count")
    raw_value, raw_present = from_mapping(raw, keys)
    if raw_present and raw_value is not None:
        return raw_value, True

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _openalex_payload(raw)
        value, present = from_mapping(payload, ("cited_by_count",))
        if present:
            return value, True
    elif platform == "semanticscholar":
        payload = _coerce_mapping(extra.get("semanticscholar"))
        value, present = from_mapping(payload, ("citationCount",))
        if present:
            return value, True
    elif platform == "core":
        payload = _coerce_mapping(extra.get("core"))
        value, present = from_mapping(payload, ("citationCount", "citations"))
        if present:
            return value, True

    if raw_present:
        return None, True
    return None, False


def _extract_platform(raw: dict[str, Any], platform: str | None) -> str:
    candidate = _clean_text(platform) or _clean_text(raw.get("platform"))
    return candidate or "unknown"


def _extract_id(raw: dict[str, Any]) -> str:
    for key in ("id", "raw_id", "scopus_id", "eid"):
        value = _clean_text(raw.get(key))
        if value:
            return value
    doi = normalise_doi(raw.get("doi"))  # type: ignore[arg-type]
    if doi:
        return doi
    url = _clean_text(raw.get("url"))
    if url:
        return url
    title = _clean_text(raw.get("title")) or "record"
    rank = _clean_text(raw.get("rank")) or "na"
    return f"{title}#{rank}"


def _extract_doi(raw: dict[str, Any]) -> str | None:
    doi = normalise_doi(raw.get("doi"))  # type: ignore[arg-type]
    if doi:
        return doi
    return doi_from_url(_clean_text(raw.get("url")))


def _extract_core_accepted_year(raw: dict[str, Any]) -> int | None:
    for key in ("acceptedDate", "accepted_date"):
        year = _coerce_year(raw.get(key))
        if year is not None:
            return year
    payload = _coerce_mapping(_coerce_mapping(raw.get("extra")).get("core"))
    for key in ("acceptedDate", "accepted_date"):
        year = _coerce_year(payload.get(key))
        if year is not None:
            return year
    return None


def _is_older_record_for_citation_signal(record: dict[str, Any], *, current_year: int) -> bool:
    year = _coerce_year(record.get("year"))
    if year is not None:
        return year <= current_year - 3
    source = _coerce_mapping(record.get("source"))
    accepted_year = _coerce_year(source.get("accepted_year"))
    if accepted_year is not None:
        return accepted_year <= current_year - 3
    return False


def _extract_source_metadata(raw: dict[str, Any], platform: str) -> dict[str, Any]:
    source = {
        "platform": platform,
        "rank": _coerce_int(raw.get("rank")),
        "query_id": _clean_text(raw.get("query_id")),
        "raw_id": _clean_text(raw.get("raw_id")),
    }
    if platform == "core":
        accepted_year = _extract_core_accepted_year(raw)
        if accepted_year is not None:
            source["accepted_year"] = accepted_year
    return source


def normalize_record(raw: dict[str, Any], platform: str) -> dict[str, Any]:
    """Normalize a single record into canonical metadata fields."""

    if not isinstance(raw, dict):
        raw = {}
    platform_name = _extract_platform(raw, platform)
    record_id = _extract_id(raw)
    doi = _extract_doi(raw)
    year_resolution = _resolve_year_with_provenance(raw=raw, platform=platform_name)
    year = year_resolution["year"]
    language = _clean_text(raw.get("language"))
    is_oa = _extract_open_access(raw, platform_name)
    citations, citations_present = _extract_citations(raw, platform_name)
    doc_type = _clean_text(raw.get("doc_type")) or _clean_text(raw.get("subtype"))
    publisher = _extract_publisher(raw, platform_name)
    publisher_provenance = "missing"
    if platform_name == "openalex":
        if _clean_publisher(raw.get("publisher")):
            publisher_provenance = "record_field"
        else:
            payload = _openalex_payload(raw)
            publisher, publisher_provenance = extract_openalex_publisher(payload)
    journal_title = _extract_journal_title(raw, platform_name)
    journal_title_provenance = "missing"

    if platform_name == "semanticscholar":
        journal_title, journal_title_provenance = _resolve_semanticscholar_journal_title(raw)
        if journal_title is None:
            fallback = _extract_journal_title(raw, platform_name)
            if fallback:
                journal_title = fallback

    issn_info = canonical_issn_selection(raw, platform=platform_name)
    issn_values = issn_info["issn_list"]
    issn_method = "direct" if issn_values else None
    country_info = _resolve_country_fields(raw, platform=platform_name)
    source_meta = _extract_source_metadata(raw, platform_name)

    journal_match = _coerce_mapping(raw.get("journal_match"))
    if not journal_match:
        journal_match = {
            "method": issn_method,
            "confidence": (1.0 if issn_values else None),
            "matched_venue_id": None,
        }

    if not journal_match.get("matched_venue_id"):
        payload = _openalex_payload(raw)
        source = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(payload).get("primary_location")).get(
                "source"
            )
        )
        source_id = _clean_text(source.get("id"))
        if source_id:
            journal_match["matched_venue_id"] = source_id
            if journal_match.get("method") is None:
                journal_match["method"] = "openalex_venue_lookup"
                journal_match["confidence"] = 0.95

    metrics_quality = _coerce_mapping(raw.get("metrics_quality"))
    if not metrics_quality:
        metrics_quality = {
            "citations": "ok" if citations is not None else ("missing" if not citations_present else "missing")
        }

    canonical_extra: dict[str, Any] = {}
    provenance: dict[str, Any] = {
        "year": {
            "year_raw": year_resolution["year_raw"],
            "year_enriched": year_resolution["year_enriched"],
            "year": year_resolution["year"],
            "year_provenance": year_resolution["year_provenance"],
            "year_source": year_resolution["year_source"],
            "year_candidates": year_resolution["year_candidates"],
        },
        "issn": {
            "preferred_source": issn_info["issn_source"],
            "sources_used": issn_info["sources_used"],
            "count": len(issn_values),
            "provenance": issn_info["issn_provenance"],
            "source_rows": issn_info["source_rows"],
        },
        "country": {
            "countries": country_info["countries"],
            "country_primary": country_info["country_primary"],
            "affiliation_countries": country_info["affiliation_countries"],
            "country_dominant": country_info["country_dominant"],
            "country_count": country_info["country_count"],
            "country_provenance": country_info["country_provenance"],
            "country_is_fractional": country_info["country_is_fractional"],
            "country_counts": country_info["country_counts"],
        },
        "journal_title": journal_title_provenance,
    }
    if platform_name == "openalex":
        canonical_extra["openalex"] = {"publisher_provenance": publisher_provenance}
    canonical_extra["provenance"] = provenance

    out: dict[str, Any] = {
        "id": record_id,
        "platform": platform_name,
        "rank": _coerce_int(raw.get("rank")),
        "title": _clean_text(raw.get("title")),
        "raw_id": _clean_text(raw.get("raw_id")),
        "doi": doi,
        "year": year,
        "language": language,
        "is_oa": is_oa,
        "citations": citations,
        "doc_type": doc_type,
        "publisher": publisher,
        "journal_title": journal_title,
        "issn": issn_info["issn"],
        "eissn": issn_info["eissn"],
        "issn_list": issn_values,
        "issn_source": issn_info["issn_source"],
        "issn_provenance": issn_info["issn_provenance"],
        "year_raw": year_resolution["year_raw"],
        "year_enriched": year_resolution["year_enriched"],
        "year_provenance": year_resolution["year_provenance"],
        "countries": country_info["countries"],
        "country_primary": country_info["country_primary"],
        "affiliation_countries": country_info["affiliation_countries"],
        "country_dominant": country_info["country_dominant"],
        "country_count": country_info["country_count"],
        "country_provenance": country_info["country_provenance"],
        "country_is_fractional": country_info["country_is_fractional"],
        "source": source_meta,
        "journal_match": journal_match,
        "metrics_quality": metrics_quality,
    }
    if canonical_extra:
        out["extra"] = canonical_extra
    return out


def _group_by_platform(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        platform = _clean_text(item.get("platform")) or "unknown"
        grouped.setdefault(platform, []).append(item)
    return grouped


def _mark_suspicious_citations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_year = datetime.utcnow().year
    for platform, items in _group_by_platform(records).items():
        known = [item for item in items if isinstance(item.get("citations"), int)]
        if not known:
            for item in items:
                quality = _coerce_mapping(item.get("metrics_quality"))
                quality["citations"] = "missing"
                item["metrics_quality"] = quality
            continue

        zeros = [item for item in known if int(item["citations"]) == 0]
        zero_rate = len(zeros) / len(known) if known else 0.0
        older_known = [
            item for item in known if _is_older_record_for_citation_signal(item, current_year=current_year)
        ]
        older_zero_count = len([item for item in older_known if item.get("citations") == 0])
        older_zero_rate = (
            (older_zero_count / len(older_known))
            if older_known
            else 0.0
        )
        older_zero_share = (older_zero_count / len(items)) if items else 0.0

        core_structural_unavailable = (
            platform == "core"
            and zero_rate >= 0.98
            and len(older_known) >= 5
            and older_zero_rate >= 0.98
        )
        if core_structural_unavailable:
            for item in items:
                item["citations"] = None
                quality = _coerce_mapping(item.get("metrics_quality"))
                quality["citations"] = "structurally_unavailable"
                item["metrics_quality"] = quality
            LOGGER.info(
                "CORE citationCount unreliable (structural limitation) "
                "(zero_rate=%.3f older_zero_rate=%.3f older_zero_count=%d known_count=%d)",
                zero_rate,
                older_zero_rate,
                older_zero_count,
                len(known),
            )
            continue

        suspicious = zero_rate >= 0.85 and older_zero_share > 0.30

        if suspicious:
            for item in items:
                quality = _coerce_mapping(item.get("metrics_quality"))
                quality["citations"] = "suspicious"
                item["metrics_quality"] = quality
            LOGGER.warning(
                "Suspicious citations detected for platform=%s zero_rate=%.3f older_zero_share=%.3f "
                "known_count=%d",
                platform,
                zero_rate,
                older_zero_share,
                len(known),
            )
            continue

        for item in items:
            quality = _coerce_mapping(item.get("metrics_quality"))
            quality["citations"] = "ok" if item.get("citations") is not None else "missing"
            item["metrics_quality"] = quality

    return records


def normalize_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a sequence of records and apply citations quality checks."""

    normalized = [
        normalize_record(record, platform=_clean_text(record.get("platform")) or "unknown")
        for record in records
    ]
    return _mark_suspicious_citations(normalized)


def normalize_records_frame(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Normalize a DataFrame into canonical record fields."""

    import pandas as pd

    if frame.empty:
        return pd.DataFrame()
    records = frame.to_dict(orient="records")
    normalized = normalize_records(records)
    return pd.DataFrame(normalized)
