"""Canonical metadata normalization for cross-platform bias analyses."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Sequence

import pandas as pd

from ai_bias_search.rankings.base import normalize_issn
from ai_bias_search.utils.ids import doi_from_url, normalise_doi
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _coerce_year(value: object) -> int | None:
    year = _coerce_int(value)
    if year is None:
        return None
    if 1800 <= year <= 2100:
        return year
    return None


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


def _extract_issn(raw: dict[str, Any], platform: str) -> tuple[list[str], str | None]:
    out: list[str] = []
    seen: set[str] = set()

    _add_issn(raw.get("issn"), out, seen)
    _add_issn(raw.get("eissn"), out, seen)
    _add_issn(raw.get("issn_list"), out, seen)

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        source = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(extra.get("openalex")).get("primary_location")).get(
                "source"
            )
        )
        host = _coerce_mapping(_coerce_mapping(extra.get("openalex")).get("host_venue"))
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
        payload = _coerce_mapping(extra.get("semanticscholar"))
        journal = _coerce_mapping(payload.get("journal"))
        venue = _coerce_mapping(payload.get("publicationVenue"))
        ext = _coerce_mapping(payload.get("externalIds"))
        _add_issn(journal.get("issn"), out, seen)
        _add_issn(venue.get("issn"), out, seen)
        _add_issn(ext.get("ISSN"), out, seen)
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
    publisher = _clean_text(raw.get("publisher"))
    if publisher:
        return publisher

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))
        source = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(payload.get("primary_location")).get("source"))
        )
        host = _coerce_mapping(payload.get("host_venue"))
        return _clean_text(source.get("publisher")) or _clean_text(host.get("publisher"))
    if platform == "scopus":
        if _clean_text(raw.get("jcr_publisher")):
            return _clean_text(raw.get("jcr_publisher"))
        scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
        for key in ("publishername", "dc:publisher", "publisher", "prism:publisher"):
            value = _clean_text(scopus_raw.get(key))
            if value:
                return value
        scopus_abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
        for key in ("publisher", "publishername", "dc:publisher"):
            value = _clean_text(scopus_abstract.get(key))
            if value:
                return value
    if platform == "semanticscholar":
        payload = _coerce_mapping(extra.get("semanticscholar"))
        journal = _coerce_mapping(payload.get("journal"))
        venue = _coerce_mapping(payload.get("publicationVenue"))
        return (
            _clean_text(journal.get("publisher"))
            or _clean_text(venue.get("publisher"))
            or _clean_text(payload.get("publisher"))
        )
    if platform == "core":
        payload = _coerce_mapping(extra.get("core"))
        return _clean_text(payload.get("publisher"))

    return None


def _extract_journal_title(raw: dict[str, Any], platform: str) -> str | None:
    for key in ("journal_title", "publication_name", "publicationName", "host_venue", "source"):
        value = _clean_text(raw.get(key))
        if value:
            return value

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))
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
        payload = _coerce_mapping(extra.get("semanticscholar"))
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


def _extract_open_access(raw: dict[str, Any], platform: str) -> bool | None:
    for key in ("is_oa", "is_open_access", "openaccess_flag", "openaccessFlag"):
        value = _coerce_bool(raw.get(key))
        if value is not None:
            return value

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))
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
        value, present = from_mapping(raw, keys)
        if present:
            return value, True
        extra = _coerce_mapping(raw.get("extra"))
        scopus_raw = _coerce_mapping(_coerce_mapping(extra.get("scopus")).get("raw"))
        value, present = from_mapping(scopus_raw, keys)
        if present:
            return value, True
        abstract = _coerce_mapping(_coerce_mapping(raw.get("scopus")).get("abstract"))
        value, present = from_mapping(abstract, keys)
        if present:
            return value, True
        return None, False

    keys = ("citations", "cited_by_count", "citationCount", "citedby-count")
    value, present = from_mapping(raw, keys)
    if present:
        return value, True

    extra = _coerce_mapping(raw.get("extra"))
    if platform == "openalex":
        payload = _coerce_mapping(extra.get("openalex_enrich")) or _coerce_mapping(extra.get("openalex"))
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


def _extract_source_metadata(raw: dict[str, Any], platform: str) -> dict[str, Any]:
    source = {
        "platform": platform,
        "rank": _coerce_int(raw.get("rank")),
        "query_id": _clean_text(raw.get("query_id")),
        "raw_id": _clean_text(raw.get("raw_id")),
    }
    return source


def normalize_record(raw: dict[str, Any], platform: str) -> dict[str, Any]:
    """Normalize a single record into canonical metadata fields."""

    if not isinstance(raw, dict):
        raw = {}
    platform_name = _extract_platform(raw, platform)
    record_id = _extract_id(raw)
    doi = _extract_doi(raw)
    year = _extract_year(raw)
    language = _clean_text(raw.get("language"))
    is_oa = _extract_open_access(raw, platform_name)
    citations, citations_present = _extract_citations(raw, platform_name)
    doc_type = _clean_text(raw.get("doc_type")) or _clean_text(raw.get("subtype"))
    publisher = _extract_publisher(raw, platform_name)
    journal_title = _extract_journal_title(raw, platform_name)
    issn_values, issn_method = _extract_issn(raw, platform_name)

    journal_match = _coerce_mapping(raw.get("journal_match"))
    if not journal_match:
        journal_match = {
            "method": issn_method,
            "confidence": (1.0 if issn_values else None),
            "matched_venue_id": None,
        }

    if not journal_match.get("matched_venue_id"):
        extra = _coerce_mapping(raw.get("extra"))
        source = _coerce_mapping(
            _coerce_mapping(_coerce_mapping(extra.get("openalex")).get("primary_location")).get(
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

    return {
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
        "issn": issn_values,
        "issn_list": issn_values,
        "affiliation_countries": _coerce_str_list(raw.get("affiliation_countries")) or None,
        "source": _extract_source_metadata(raw, platform_name),
        "journal_match": journal_match,
        "metrics_quality": metrics_quality,
    }


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
        years = [int(item["year"]) for item in known if isinstance(item.get("year"), int)]
        older_share = (
            len([year for year in years if year <= current_year - 2]) / len(years) if years else 0.0
        )
        suspicious = len(known) >= 20 and zero_rate > 0.7 and older_share > 0.4

        if suspicious:
            changed_ids: list[str] = []
            for item in items:
                quality = _coerce_mapping(item.get("metrics_quality"))
                if item.get("citations") == 0:
                    item["citations"] = None
                    changed_ids.append(str(item.get("id")))
                quality["citations"] = "suspicious"
                item["metrics_quality"] = quality
            sample = ", ".join(changed_ids[:5])
            LOGGER.warning(
                "Suspicious citations detected for platform=%s zero_rate=%.3f older_share=%.3f "
                "records_adjusted=%d sample_ids=%s",
                platform,
                zero_rate,
                older_share,
                len(changed_ids),
                sample or "-",
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


def normalize_records_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame into canonical record fields."""

    if frame.empty:
        return pd.DataFrame()
    records = frame.to_dict(orient="records")
    normalized = normalize_records(records)
    return pd.DataFrame(normalized)
