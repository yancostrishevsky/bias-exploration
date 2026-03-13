"""Sanity checks and diagnostics for metadata quality."""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from ai_bias_search.normalize.records import normalize_country_code, normalize_records
from ai_bias_search.utils.config import DiagnosticsConfig

_DEFAULT_REDACT_FIELDS = {
    "apikey",
    "api_key",
    "api-key",
    "insttoken",
    "authorization",
    "token",
    "x-els-apikey",
    "x-els-insttoken",
}


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _coerce_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


_LIST_LIKE_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")


def _parse_list_like_text(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    quoted: list[str] = []
    for first, second in _LIST_LIKE_QUOTED_RE.findall(text):
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
            token = _clean_text(item)
            if token:
                out.append(token)
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


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, set):
        items = sorted(value)
    elif isinstance(value, str):
        parsed = _parse_list_like_text(value)
        if parsed:
            items = parsed
        elif "," in value or ";" in value:
            items = [part.strip() for part in value.replace(";", ",").split(",")]
        else:
            items = [value]
    elif hasattr(value, "tolist"):
        try:
            converted = value.tolist()
        except Exception:
            converted = []
        if isinstance(converted, list):
            items = converted
        elif isinstance(converted, tuple):
            items = list(converted)
        else:
            return []
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = _clean_text(item)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _canonical_country_list(record: Mapping[str, Any]) -> list[str]:
    raw_values = record.get("countries")
    if raw_values is None:
        raw_values = record.get("affiliation_countries")
    values = _coerce_str_list(raw_values)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = normalize_country_code(value)
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _canonical_country_primary(record: Mapping[str, Any]) -> str | None:
    primary = _clean_text(record.get("country_primary"))
    if primary is None:
        primary = _clean_text(record.get("country_dominant"))
    if not primary:
        return None
    if primary.upper() == "MULTI":
        return "MULTI"
    return normalize_country_code(primary)


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _platform_summary(
    records: list[dict[str, Any]],
    platform: str,
    *,
    capabilities: Mapping[str, Any] | None = None,
    publisher_parser_bug_rate: float = 0.0,
    publisher_structural_missing_rate: float | None = None,
    result_status: str | None = None,
) -> dict[str, Any]:
    total = len(records)
    issn_missing = sum(1 for item in records if not item.get("issn"))
    publisher_missing = sum(1 for item in records if not item.get("publisher"))
    citations_known = [item for item in records if item.get("citations") is not None]
    citations_zero = [item for item in citations_known if item.get("citations") == 0]
    quality = Counter(
        str((item.get("metrics_quality") or {}).get("citations") or "missing") for item in records
    )
    citations_quality_label = "ok"
    if int(quality.get("structurally_unavailable", 0)) > 0:
        citations_quality_label = "structurally_unavailable"
    elif int(quality.get("suspicious", 0)) > 0:
        citations_quality_label = "suspicious"
    if total > 0 and len(citations_known) == 0 and citations_quality_label == "ok":
        citations_quality_label = "missing"

    caps = dict(capabilities or {})
    return {
        "platform": platform,
        "total": total,
        "issn_missing_rate": (issn_missing / total) if total else None,
        "publisher_missing_rate": (publisher_missing / total) if total else None,
        "citations_missing_rate": ((total - len(citations_known)) / total if total else None),
        "citations_zero_rate": (len(citations_zero) / len(citations_known) if citations_known else None),
        "citations_quality": dict(quality),
        "citations_quality_label": citations_quality_label,
        "citations_structurally_unavailable": (
            bool(citations_quality_label == "structurally_unavailable")
        ),
        "publisher_parser_bug_rate": (publisher_parser_bug_rate if total else None),
        "publisher_structural_missing_rate": publisher_structural_missing_rate,
        "publisher_structurally_unavailable": (
            bool(caps.get("publisher_available") is False) if caps else None
        ),
        "result_status": result_status,
    }


def _platform_health_from_requests(
    request_logs: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    health: dict[str, dict[str, Any]] = {}
    for platform, entries in request_logs.items():
        statuses = [
            _coerce_int(entry.get("status_code"))
            for entry in entries
            if isinstance(entry, dict)
        ]
        status_values = [status for status in statuses if status is not None]
        request_count = len(status_values)
        error_count = len([status for status in status_values if status >= 400])
        error_rate = (error_count / request_count) if request_count > 0 else 0.0

        enabled = True
        reason: str | None = None
        if platform == "core":
            permanent_errors = [status for status in status_values if 400 <= status < 500]
            if permanent_errors:
                enabled = False
                reason = f"CORE request failed with HTTP {permanent_errors[-1]}"
            elif request_count > 0 and error_count == request_count:
                enabled = False
                reason = "CORE requests failed during this run"

        health[str(platform)] = {
            "enabled": enabled,
            "reason": reason,
            "error_rate": error_rate,
            "request_count": request_count,
            "error_count": error_count,
        }
    return health


def _result_status(
    *,
    platform: str,
    records: Sequence[dict[str, Any]],
    request_logs: Mapping[str, list[dict[str, Any]]],
) -> str:
    if records:
        return "ok"
    statuses = [
        _coerce_int(entry.get("status_code"))
        for entry in request_logs.get(platform, [])
        if isinstance(entry, dict)
    ]
    if any(status is not None and status >= 400 for status in statuses):
        return "connector_error"
    return "no_results"


def _year_discrepancies_by_platform(
    normalized_grouped: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for platform, items in sorted(normalized_grouped.items()):
        total = len(items)
        examples: list[dict[str, Any]] = []
        count = 0
        for item in items:
            year_raw = _coerce_int(item.get("year_raw"))
            year_enriched = _coerce_int(item.get("year_enriched"))
            if year_raw is None or year_enriched is None:
                continue
            if year_raw == year_enriched:
                continue
            count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "year_raw": year_raw,
                        "year_enriched": year_enriched,
                        "year_provenance": item.get("year_provenance"),
                    }
                )
        out[platform] = {
            "count": count,
            "share": (count / total) if total > 0 else 0.0,
            "examples": examples,
        }
    return out


def _failing_field_samples(
    *,
    raw_grouped: Mapping[str, list[dict[str, Any]]],
    normalized_grouped: Mapping[str, list[dict[str, Any]]],
    redact_fields: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    fields = ("publisher", "year", "issn", "citations")
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for platform, canonical_items in normalized_grouped.items():
        raw_items = raw_grouped.get(platform, [])
        platform_payload: dict[str, dict[str, Any]] = {}
        for raw, canonical in zip(raw_items, canonical_items, strict=False):
            missing_map = {
                "publisher": not _clean_text(canonical.get("publisher")),
                "year": _coerce_int(canonical.get("year")) is None,
                "issn": not _clean_text(canonical.get("issn")) and not _safe_list(canonical.get("issn_list")),
                "citations": canonical.get("citations") is None,
            }
            for field in fields:
                if not missing_map[field]:
                    continue
                if field in platform_payload:
                    continue
                platform_payload[field] = {
                    "record_id": canonical.get("id") or raw.get("id") or raw.get("raw_id"),
                    "raw_snippet": _redact(_raw_sample_for_platform(raw, platform), redact_fields=redact_fields),
                    "canonical_value": canonical.get(field),
                    "field_paths": _field_paths(platform, field),
                }
        if platform_payload:
            out[platform] = platform_payload
    return out


def _has_semanticscholar_publisher_field(payload: Mapping[str, Any]) -> tuple[bool, bool]:
    present = False
    value_present = False
    venue = _coerce_mapping(payload.get("venue"))
    journal = _coerce_mapping(payload.get("journal"))
    publication_venue = _coerce_mapping(payload.get("publicationVenue"))
    candidates: list[tuple[bool, object]] = [
        ("publisher" in venue, venue.get("publisher")),
        ("publisher" in journal, journal.get("publisher")),
        ("publisher" in publication_venue, publication_venue.get("publisher")),
        ("publisher" in payload, payload.get("publisher")),
    ]
    for field_present, value in candidates:
        if field_present:
            present = True
        if _clean_text(value):
            value_present = True
    return present, value_present


def _openalex_raw_publisher_values(payload: Mapping[str, Any]) -> list[str]:
    primary_location = _coerce_mapping(payload.get("primary_location"))
    source = _coerce_mapping(primary_location.get("source"))
    host = _coerce_mapping(payload.get("host_venue"))

    lineage = source.get("host_organization_lineage_names")
    lineage_last = None
    if isinstance(lineage, list):
        for candidate in reversed(lineage):
            cleaned = _clean_text(candidate)
            if cleaned:
                lineage_last = cleaned
                break

    values = [
        _clean_text(source.get("host_organization_name")),
        lineage_last,
        _clean_text(host.get("publisher")),
        _clean_text(source.get("display_name")),
    ]
    return [value for value in values if value]


def _semanticscholar_raw_publisher_values(payload: Mapping[str, Any]) -> list[str]:
    venue = _coerce_mapping(payload.get("venue"))
    journal = _coerce_mapping(payload.get("journal"))
    publication_venue = _coerce_mapping(payload.get("publicationVenue"))
    values = [
        _clean_text(venue.get("publisher")),
        _clean_text(journal.get("publisher")),
        _clean_text(publication_venue.get("publisher")),
        _clean_text(payload.get("publisher")),
    ]
    return [value for value in values if value]


def _scopus_raw_publisher_values(payload: Mapping[str, Any]) -> list[str]:
    values = [
        _clean_text(payload.get("publishername")),
        _clean_text(payload.get("dc:publisher")),
        _clean_text(payload.get("publisher")),
        _clean_text(payload.get("prism:publisher")),
    ]
    return [value for value in values if value]


def _publisher_source_values(record: Mapping[str, Any], platform: str) -> list[str]:
    payload = _coerce_mapping(_coerce_mapping(record.get("extra")).get(platform))
    if platform == "openalex":
        payload = _coerce_mapping(
            _coerce_mapping(record.get("extra")).get("openalex_enrich")
        ) or _coerce_mapping(_coerce_mapping(record.get("extra")).get("openalex"))
        return _openalex_raw_publisher_values(payload)
    if platform == "semanticscholar":
        return _semanticscholar_raw_publisher_values(payload)
    if platform == "scopus":
        scopus_payload = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(record.get("extra")).get("scopus")).get("raw")
        )
        return _scopus_raw_publisher_values(scopus_payload)
    return []


def _publisher_parse_stats(
    *,
    raw_records: Sequence[dict[str, Any]],
    canonical_records: Sequence[dict[str, Any]],
    platform: str,
) -> dict[str, Any]:
    if not raw_records:
        return {
            "parser_bug_rate": 0.0,
            "structural_missing_rate": None,
            "records_with_source_values": 0,
            "records_without_source_values": 0,
        }

    total_with_source_value = 0
    total_without_source_value = 0
    missing_after_parse = 0
    for raw, canonical in zip(raw_records, canonical_records, strict=False):
        values = _publisher_source_values(raw, platform)
        if not values:
            total_without_source_value += 1
            continue
        total_with_source_value += 1
        if not _clean_text(canonical.get("publisher")):
            missing_after_parse += 1

    total_records = len(canonical_records)
    parser_bug_rate = (
        (missing_after_parse / total_with_source_value)
        if total_with_source_value > 0
        else 0.0
    )
    structural_missing_rate = (
        (total_without_source_value / total_records) if total_records > 0 else None
    )
    return {
        "parser_bug_rate": parser_bug_rate,
        "structural_missing_rate": structural_missing_rate,
        "records_with_source_values": total_with_source_value,
        "records_without_source_values": total_without_source_value,
    }


def _infer_platform_capabilities(
    raw_grouped: Mapping[str, list[dict[str, Any]]],
    normalized_grouped: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    caps: dict[str, dict[str, Any]] = {}
    for platform in sorted(set(raw_grouped.keys()) | set(normalized_grouped.keys())):
        records = raw_grouped.get(platform, [])
        publisher_available: bool | None = None
        publisher_reason: str | None = None
        if platform == "semanticscholar":
            present_any = False
            value_any = False
            seen_payload = False
            for record in records:
                payload = _coerce_mapping(_coerce_mapping(record.get("extra")).get("semanticscholar"))
                if payload:
                    seen_payload = True
                present, value_present = _has_semanticscholar_publisher_field(payload)
                present_any = present_any or present
                value_any = value_any or value_present
            if not seen_payload:
                publisher_available = None
            elif not present_any:
                publisher_available = False
                publisher_reason = "publisher not exposed by Semantic Scholar payload schema"
            else:
                publisher_available = True
                if not value_any:
                    publisher_reason = "publisher field present but values absent in sampled payloads"
        citations_available: bool | None = True
        citations_reason: str | None = None
        normalized_records = normalized_grouped.get(platform, [])
        quality = Counter(
            str((item.get("metrics_quality") or {}).get("citations") or "missing")
            for item in normalized_records
        )
        if int(quality.get("structurally_unavailable", 0)) > 0:
            citations_available = False
            if platform == "core":
                citations_reason = "CORE citationCount unreliable (structural limitation)"
            else:
                citations_reason = "citation signal treated as structurally unavailable"

        has_geo_values = any(bool(_canonical_country_list(item)) for item in normalized_records)
        geo_available: bool | None = None
        geo_reason: str | None = None
        if platform in {"semanticscholar", "core"}:
            if has_geo_values:
                geo_available = True
            else:
                geo_available = False
                geo_reason = "geo metadata structurally unavailable from source payload"
        elif normalized_records:
            geo_available = has_geo_values
        caps[platform] = {
            "publisher_available": publisher_available,
            "publisher_reason": publisher_reason,
            "citations_available": citations_available,
            "citations_reason": citations_reason,
            "geo_available": geo_available,
            "geo_reason": geo_reason,
        }
    return caps


def _redact(value: Any, *, redact_fields: set[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in redact_fields:
                out[key_text] = "****"
            else:
                out[key_text] = _redact(item, redact_fields=redact_fields)
        return out
    if isinstance(value, list):
        return [_redact(item, redact_fields=redact_fields) for item in value]
    return value


def _sanitize_endpoint(endpoint: object) -> object:
    if not isinstance(endpoint, str):
        return endpoint
    try:
        parts = urlsplit(endpoint)
    except Exception:
        return endpoint.split("?", 1)[0]
    if not parts.scheme and not parts.netloc:
        return endpoint.split("?", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _sanitize_request_logs(request_logs: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    sanitized: dict[str, list[dict[str, Any]]] = {}
    for platform, entries in request_logs.items():
        platform_entries: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            copied = dict(entry)
            copied["endpoint"] = _sanitize_endpoint(copied.get("endpoint"))
            platform_entries.append(copied)
        sanitized[str(platform)] = platform_entries
    return sanitized


def _openalex_sample(payload: Mapping[str, Any]) -> dict[str, Any]:
    locations = _safe_list(payload.get("locations"))
    first_location = locations[0] if locations else None
    is_oa = None
    open_access = _coerce_mapping(payload.get("open_access"))
    if "is_oa" in open_access:
        is_oa = open_access.get("is_oa")
    elif "is_oa" in payload:
        is_oa = payload.get("is_oa")
    return {
        "id": payload.get("id"),
        "host_venue": payload.get("host_venue"),
        "primary_location": payload.get("primary_location"),
        "locations_0": first_location,
        "cited_by_count": payload.get("cited_by_count"),
        "is_oa": is_oa,
        "type": payload.get("type"),
        "publication_year": payload.get("publication_year"),
    }


def _scopus_sample(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dc:identifier": payload.get("dc:identifier"),
        "eid": payload.get("eid"),
        "prism:doi": payload.get("prism:doi"),
        "prism:publicationName": payload.get("prism:publicationName"),
        "prism:issn": payload.get("prism:issn"),
        "prism:eIssn": payload.get("prism:eIssn") or payload.get("prism:eissn"),
        "prism:coverDate": payload.get("prism:coverDate"),
        "citedby-count": payload.get("citedby-count"),
        "openaccessFlag": payload.get("openaccessFlag"),
        "subtype": payload.get("subtype"),
        "subtypeDescription": payload.get("subtypeDescription"),
    }


def _semanticscholar_sample(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paperId": payload.get("paperId"),
        "externalIds": payload.get("externalIds"),
        "venue": payload.get("venue"),
        "journal": payload.get("journal"),
        "publicationVenue": payload.get("publicationVenue"),
        "year": payload.get("year"),
        "publicationDate": payload.get("publicationDate"),
        "citationCount": payload.get("citationCount"),
        "isOpenAccess": payload.get("isOpenAccess"),
        "publicationTypes": payload.get("publicationTypes"),
    }


def _raw_sample_for_platform(record: Mapping[str, Any], platform: str) -> dict[str, Any]:
    extra = _coerce_mapping(record.get("extra"))
    if platform == "openalex":
        payload = _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))
        return _openalex_sample(payload)
    if platform == "scopus":
        scopus_block = _coerce_mapping(extra.get("scopus"))
        payload = _coerce_mapping(scopus_block.get("raw")) or _coerce_mapping(scopus_block.get("abstract"))
        return _scopus_sample(payload)
    if platform == "semanticscholar":
        payload = _coerce_mapping(extra.get("semanticscholar"))
        return _semanticscholar_sample(payload)
    if platform == "core":
        payload = _coerce_mapping(extra.get("core"))
        return {
            "id": payload.get("id") or payload.get("coreId"),
            "title": payload.get("title"),
            "doi": payload.get("doi"),
            "publisher": payload.get("publisher"),
            "citationCount": payload.get("citationCount"),
            "year": payload.get("year") or payload.get("yearPublished"),
        }
    return {}


def _field_paths(platform: str, field: str) -> list[str]:
    defaults: dict[str, dict[str, list[str]]] = {
        "openalex": {
            "publisher": [
                "publisher",
                "extra.openalex.primary_location.source.host_organization_name",
                "extra.openalex.primary_location.source.host_organization_lineage_names[-1]",
                "extra.openalex.host_venue.publisher",
                "extra.openalex.primary_location.source.display_name (fallback)",
                "extra.openalex.publisher_provenance",
            ],
            "issn": [
                "issn_list",
                "extra.openalex.primary_location.source.issn",
                "extra.openalex.host_venue.issn",
            ],
            "citations": ["citations", "cited_by_count", "extra.openalex.cited_by_count"],
            "doi": ["doi", "url", "extra.openalex.doi", "extra.openalex.ids.doi"],
            "year": [
                "year",
                "publication_year",
                "extra.openalex_enrich.publication_year",
                "extra.openalex.publication_year",
            ],
        },
        "scopus": {
            "publisher": [
                "publisher",
                "extra.scopus.raw.publishername",
                "extra.scopus.raw.dc:publisher",
                "extra.scopus.raw.prism:publisher",
            ],
            "issn": ["issn_list", "extra.scopus.raw.prism:issn", "extra.scopus.raw.prism:eIssn"],
            "citations": ["citedby-count", "cited_by_count", "extra.scopus.raw.citedby-count"],
            "doi": ["doi", "url", "extra.scopus.raw.prism:doi", "extra.scopus.abstract.prism:doi"],
            "year": [
                "year",
                "publication_year",
                "scopus.abstract.publication_year",
                "extra.scopus_enrich.cover_date",
            ],
        },
        "semanticscholar": {
            "publisher": [
                "publisher",
                "extra.semanticscholar.venue.publisher",
                "extra.semanticscholar.journal.publisher",
                "extra.semanticscholar.publicationVenue.publisher",
            ],
            "issn": [
                "issn_list",
                "extra.semanticscholar.journal.issn",
                "extra.semanticscholar.publicationVenue.issn",
                "extra.semanticscholar.externalIds.ISSN",
            ],
            "citations": ["citationCount", "citations", "extra.semanticscholar.citationCount"],
            "doi": ["doi", "url", "extra.semanticscholar.externalIds.DOI"],
            "year": [
                "year",
                "publication_year",
                "extra.semanticscholar.year",
                "extra.semanticscholar.publicationDate",
                "extra.openalex_enrich.publication_year",
                "scopus.abstract.publication_year",
            ],
        },
        "core": {
            "publisher": ["publisher", "extra.core.publisher"],
            "issn": ["issn_list", "extra.core.issn", "extra.core.identifiers.issn"],
            "citations": ["citations", "cited_by_count", "extra.core.citationCount"],
            "doi": ["doi", "url", "extra.core.doi", "extra.core.identifiers.doi"],
            "year": ["year", "publication_year", "extra.core.year", "extra.core.yearPublished"],
        },
    }
    return defaults.get(platform, {}).get(field, [field])


def _mapping_attempts(platform: str) -> dict[str, list[str]]:
    return {
        "publisher": _field_paths(platform, "publisher"),
        "issn": _field_paths(platform, "issn"),
        "citations": _field_paths(platform, "citations"),
        "doi": _field_paths(platform, "doi"),
        "year": _field_paths(platform, "year"),
    }


def _extract_enrich_trace(record: Mapping[str, Any], *, max_entries: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entries: object) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = repr(sorted(entry.items()))
            if key in seen:
                continue
            seen.add(key)
            collected.append(dict(entry))
            if max_entries >= 0 and len(collected) >= max_entries:
                return

    extra = _coerce_mapping(record.get("extra"))
    add(extra.get("enrich_trace"))
    scopus_meta = _coerce_mapping(record.get("scopus_meta"))
    add(scopus_meta.get("enrich_trace"))
    if max_entries <= 0:
        return []
    return collected[:max_entries]


def _sample_notes(
    *,
    canonical: Mapping[str, Any],
    capabilities: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    if capabilities.get("publisher_available") is False and not canonical.get("publisher"):
        notes.append("publisher missing due to API structural unavailability")
    quality = _coerce_mapping(canonical.get("metrics_quality"))
    if quality.get("citations") == "suspicious":
        notes.append("citations marked suspicious due to all-zero pattern in older records")
    if quality.get("citations") == "structurally_unavailable":
        notes.append("citations treated as structurally unavailable for this platform")
    year_raw = _coerce_int(canonical.get("year_raw"))
    year_enriched = _coerce_int(canonical.get("year_enriched"))
    if year_raw is not None and year_enriched is not None and year_raw != year_enriched:
        notes.append("year discrepancy between raw and enriched candidates")
    return notes


def _samples_by_platform(
    *,
    raw_grouped: Mapping[str, list[dict[str, Any]]],
    normalized_grouped: Mapping[str, list[dict[str, Any]]],
    capabilities: Mapping[str, Mapping[str, Any]],
    max_sample_records: int,
    max_enrich_trace_entries: int,
    redact_fields: set[str],
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    for platform, raw_items in sorted(raw_grouped.items()):
        canonical_items = normalized_grouped.get(platform, [])
        capped: list[dict[str, Any]] = []
        for raw, canonical in list(zip(raw_items, canonical_items, strict=False))[:max_sample_records]:
            caps = dict(capabilities.get(platform) or {})
            raw_sample = _raw_sample_for_platform(raw, platform)
            record_id = canonical.get("id") or raw.get("raw_id") or raw.get("id")
            trace_entries = _extract_enrich_trace(raw, max_entries=max_enrich_trace_entries)
            capped.append(
                {
                    "record_id": str(record_id) if record_id is not None else None,
                    "raw_snippet": _redact(raw_sample, redact_fields=redact_fields),
                    "canonical": _redact(dict(canonical), redact_fields=redact_fields),
                    "mapping_attempts": _mapping_attempts(platform),
                    "enrich_trace": _redact(trace_entries, redact_fields=redact_fields),
                    "notes": _sample_notes(canonical=canonical, capabilities=caps),
                }
            )
        samples[platform] = capped
    return samples


def _raw_geo_sample_for_platform(record: Mapping[str, Any], platform: str) -> dict[str, Any]:
    extra = _coerce_mapping(record.get("extra"))
    if platform == "openalex":
        payload = _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))
        authorships = _safe_list(payload.get("authorships"))
        compact_authorships: list[dict[str, Any]] = []
        for authorship in authorships[:3]:
            if not isinstance(authorship, dict):
                continue
            institutions = _safe_list(authorship.get("institutions"))
            compact_inst: list[dict[str, Any]] = []
            for institution in institutions[:5]:
                if not isinstance(institution, dict):
                    continue
                compact_inst.append(
                    {
                        "id": institution.get("id"),
                        "country_code": institution.get("country_code"),
                    }
                )
            compact_authorships.append({"institutions": compact_inst})
        return {"authorships": compact_authorships}
    if platform == "scopus":
        scopus = _coerce_mapping(record.get("scopus"))
        abstract = _coerce_mapping(scopus.get("abstract"))
        response = _coerce_mapping(abstract.get("response"))
        affiliations = _safe_list(response.get("affiliation"))
        return {
            "abstract.affiliation_countries": abstract.get("affiliation_countries"),
            "abstract.countries": abstract.get("countries"),
            "response.affiliation": [
                {
                    "id": entry.get("@id") or entry.get("affiliation-id") or entry.get("id"),
                    "affiliation-country": entry.get("affiliation-country"),
                    "country": entry.get("country"),
                }
                for entry in affiliations[:5]
                if isinstance(entry, dict)
            ],
        }
    return {
        "record.countries": record.get("countries"),
        "record.affiliation_countries": record.get("affiliation_countries"),
    }


def _geo_coverage_summary(
    records: Sequence[dict[str, Any]],
    *,
    capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    total = len(records)
    country_lists = [_canonical_country_list(record) for record in records]
    available_count = sum(1 for values in country_lists if values)
    available_share = (available_count / total) if total > 0 else None
    multi_count = sum(1 for values in country_lists if len(values) > 1)
    multi_share = (multi_count / available_count) if available_count > 0 else None

    structural_unavailable = bool(
        isinstance(capability, Mapping) and capability.get("geo_available") is False
    )
    reason: str | None = None
    if structural_unavailable:
        reason = "structural_unavailability"
    elif total == 0:
        reason = "no_records"
    elif available_count == 0:
        reason = "no_country_data"

    return {
        "available": bool(available_count > 0 and not structural_unavailable),
        "reason": reason,
        "total_records": total,
        "available_count": available_count,
        "available_share": available_share,
        "multi_count": multi_count,
        "multi_country_share": multi_share,
    }


def _geo_samples_by_platform(
    *,
    raw_grouped: Mapping[str, list[dict[str, Any]]],
    normalized_grouped: Mapping[str, list[dict[str, Any]]],
    redact_fields: set[str],
    limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for platform in sorted(set(raw_grouped.keys()) | set(normalized_grouped.keys())):
        raw_items = raw_grouped.get(platform, [])
        canonical_items = normalized_grouped.get(platform, [])
        samples: list[dict[str, Any]] = []
        for raw, canonical in zip(raw_items, canonical_items, strict=False):
            countries = _canonical_country_list(canonical)
            primary = _canonical_country_primary(canonical)
            provenance = _clean_text(canonical.get("country_provenance")) or "missing"
            sample = {
                "record_id": canonical.get("id") or raw.get("id") or raw.get("raw_id"),
                "raw_country_snippet": _raw_geo_sample_for_platform(raw, platform),
                "canonical": {
                    "countries": countries or None,
                    "country_primary": primary,
                    "country_count": canonical.get("country_count"),
                    "country_provenance": provenance,
                },
            }
            samples.append(_redact(sample, redact_fields=redact_fields))
            if len(samples) >= limit:
                break
        out[platform] = samples
    return out


def run_sanity_checks(
    records: Sequence[dict[str, Any]],
    *,
    diagnostics: DiagnosticsConfig | None = None,
    request_logs: Mapping[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run platform-level sanity checks, capabilities analysis, and optional samples."""

    cfg = diagnostics or DiagnosticsConfig()
    raw_records = [item if isinstance(item, dict) else {} for item in records]
    normalized = normalize_records(raw_records)

    raw_grouped: dict[str, list[dict[str, Any]]] = {}
    normalized_grouped: dict[str, list[dict[str, Any]]] = {}
    for raw, canonical in zip(raw_records, normalized, strict=False):
        platform = str(canonical.get("platform") or raw.get("platform") or "unknown")
        raw_grouped.setdefault(platform, []).append(raw)
        normalized_grouped.setdefault(platform, []).append(canonical)

    platform_capabilities = _infer_platform_capabilities(raw_grouped, normalized_grouped)
    platform_health = _platform_health_from_requests(dict(request_logs or {}))
    by_platform: dict[str, dict[str, Any]] = {}
    geo_capabilities: dict[str, dict[str, Any]] = {}
    geo_coverage: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    platforms = sorted(set(normalized_grouped.keys()) | set((request_logs or {}).keys()))
    for platform in platforms:
        items = normalized_grouped.get(platform, [])
        raw_items = raw_grouped.get(platform, [])
        capabilities = platform_capabilities.get(platform, {})
        publisher_stats = _publisher_parse_stats(
            raw_records=raw_items,
            canonical_records=items,
            platform=platform,
        )
        bug_rate = float(publisher_stats.get("parser_bug_rate") or 0.0)
        structural_missing_rate = publisher_stats.get("structural_missing_rate")
        summary = _platform_summary(
            items,
            platform,
            capabilities=capabilities,
            publisher_parser_bug_rate=bug_rate,
            publisher_structural_missing_rate=(
                float(structural_missing_rate)
                if isinstance(structural_missing_rate, (int, float))
                else None
            ),
            result_status=_result_status(
                platform=platform,
                records=items,
                request_logs=dict(request_logs or {}),
            ),
        )
        health_entry = platform_health.get(platform)
        if health_entry:
            summary["platform_health"] = {
                "enabled": health_entry.get("enabled"),
                "reason": health_entry.get("reason"),
                "error_rate": health_entry.get("error_rate"),
            }
        coverage = _geo_coverage_summary(items, capability=capabilities)
        summary["geo_coverage"] = coverage
        geo_available = capabilities.get("geo_available")
        geo_capabilities[platform] = {
            "available": (
                bool(geo_available)
                if geo_available is not None
                else bool(coverage.get("available"))
            ),
            "reason": _clean_text(capabilities.get("geo_reason")) or coverage.get("reason"),
        }
        geo_coverage[platform] = coverage
        by_platform[platform] = summary

        if summary.get("result_status") == "connector_error":
            reason = _clean_text((health_entry or {}).get("reason"))
            if reason:
                warnings.append(f"{platform}: connector_error ({reason})")
            else:
                warnings.append(f"{platform}: connector_error (HTTP failures during collection)")
            continue

        if (summary.get("issn_missing_rate") or 0.0) > 0.7:
            warnings.append(f"{platform}: high ISSN missing rate ({summary['issn_missing_rate']:.1%})")
        publisher_missing_rate = float(summary.get("publisher_missing_rate") or 0.0)
        if publisher_missing_rate > 0.7:
            if capabilities.get("publisher_available") is False:
                reason = capabilities.get("publisher_reason") or "publisher not provided by API"
                warnings.append(
                    f"{platform}: high publisher missing rate ({publisher_missing_rate:.1%}) due to structural API limitation ({reason})"
                )
            elif bug_rate > 0.1:
                warnings.append(
                    f"{platform}: high publisher missing rate ({publisher_missing_rate:.1%}); publisher values are present in raw payloads (likely parser/mapping bug)"
                )
            elif (summary.get("publisher_structural_missing_rate") or 0.0) > 0.7:
                warnings.append(
                    f"{platform}: high publisher missing rate ({publisher_missing_rate:.1%}); publisher fields are mostly absent in sampled raw payloads"
                )
            else:
                warnings.append(
                    f"{platform}: high publisher missing rate ({publisher_missing_rate:.1%})"
                )
        quality = summary.get("citations_quality") or {}
        suspicious = int(quality.get("suspicious", 0))
        citations_available = capabilities.get("citations_available")
        citations_reason = _clean_text(capabilities.get("citations_reason"))
        citations_label = summary.get("citations_quality_label")
        if platform == "core" and citations_available is False and citations_label == "structurally_unavailable":
            warnings.append(
                "core: citations unavailable (structural limitation), citation-based metrics disabled"
            )
        elif suspicious > 0:
            warnings.append(
                f"{platform}: suspicious citations detected ({suspicious} records flagged)"
            )
        elif citations_available is False and citations_reason:
            warnings.append(f"{platform}: citations unavailable ({citations_reason})")

    redact_fields = {field.lower() for field in _DEFAULT_REDACT_FIELDS}
    redact_fields.update(str(field).strip().lower() for field in cfg.redact_fields if str(field).strip())
    geo_samples = _geo_samples_by_platform(
        raw_grouped=raw_grouped,
        normalized_grouped=normalized_grouped,
        redact_fields=redact_fields,
        limit=3,
    )
    payload: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "total_records": len(normalized),
        "by_platform": by_platform,
        "platform_capabilities": platform_capabilities,
        "platform_health": platform_health,
        "year_discrepancies": _year_discrepancies_by_platform(normalized_grouped),
        "failing_field_samples": _failing_field_samples(
            raw_grouped=raw_grouped,
            normalized_grouped=normalized_grouped,
            redact_fields=redact_fields,
        ),
        "geo": {
            "capabilities": geo_capabilities,
            "coverage": geo_coverage,
            "samples": geo_samples,
        },
        "warnings": warnings,
    }
    if cfg.capture_samples:
        payload["samples"] = _samples_by_platform(
            raw_grouped=raw_grouped,
            normalized_grouped=normalized_grouped,
            capabilities=platform_capabilities,
            max_sample_records=cfg.max_sample_records,
            max_enrich_trace_entries=cfg.max_enrich_trace_entries,
            redact_fields=redact_fields,
        )
    if cfg.capture_requests:
        payload["requests"] = _redact(
            _sanitize_request_logs(dict(request_logs or {})),
            redact_fields=redact_fields,
        )
    return payload
