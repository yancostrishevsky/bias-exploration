"""Bias-oriented metrics for retrieved records."""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime
import math
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ai_bias_search.normalize.records import normalize_country_code, normalize_records

CORE_RANK_LABELS = {"A*": "a_star", "A": "a", "B": "b", "C": "c"}
TOP_K_VALUES = (10, 20, 50)
DEFAULT_GEO_MIN_COVERAGE = 0.4


def compute_bias_metrics(
    frame: pd.DataFrame,
    *,
    geo_min_coverage: float = DEFAULT_GEO_MIN_COVERAGE,
) -> Dict[str, Any]:
    """Compute a suite of bias metrics from an enriched dataframe."""

    canonical = _canonicalize_frame(frame)
    platform_capabilities = _infer_platform_capabilities(frame, canonical)
    overall = _compute_bias_set(canonical, geo_min_coverage=geo_min_coverage)
    by_platform: Dict[str, Dict[str, Any]] = {}
    if "platform" in canonical.columns:
        platforms = sorted(
            str(platform) for platform in canonical["platform"].dropna().unique().tolist()
        )
        for platform in platforms:
            subset = canonical[canonical["platform"].astype(str) == platform]
            by_platform[platform] = _compute_bias_set(
                subset,
                capabilities=platform_capabilities.get(platform),
                geo_min_coverage=geo_min_coverage,
            )

    if len(by_platform) > 1:
        overall["citations_not_cross_platform_comparable"] = True
        overall["rank_vs_citations"] = {
            "citations_not_cross_platform_comparable": True,
            "spearman": None,
            "per_platform": {
                platform: values.get("rank_vs_citations") for platform, values in by_platform.items()
            },
            "platform_medians": {
                platform: (
                    values.get("top_k_bias", {})
                    .get("citations", {})
                    .get("overall_median")
                )
                for platform, values in by_platform.items()
            },
        }
        overall["top_k_bias"]["citations"] = {
            "citations_not_cross_platform_comparable": True,
            "per_platform": {
                platform: values.get("top_k_bias", {}).get("citations")
                for platform, values in by_platform.items()
            },
        }

    overall["by_platform"] = by_platform
    overall["platform_capabilities"] = platform_capabilities
    overall["open_access_by_platform"] = _open_access_by_platform(canonical)
    return overall


def _canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalized = normalize_records(frame.to_dict(orient="records"))
    canonical = pd.DataFrame(normalized)
    if canonical.empty:
        return canonical
    canonical["publication_year"] = canonical.get("year")
    canonical["cited_by_count"] = canonical.get("citations")
    canonical["is_open_access"] = canonical.get("is_oa")
    canonical["issn_list"] = (
        canonical.get("issn_list")
        if "issn_list" in canonical.columns
        else canonical.get("issn")
    )
    return canonical


def _country_lists(frame: pd.DataFrame) -> Optional[pd.Series]:
    values = _list_feature_series(frame, "countries")
    if values is not None:
        return values
    return _list_feature_series(frame, "affiliation_countries")


def _country_primary_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    primary = _text_series(frame.get("country_primary"))
    if primary is not None and primary.notna().any():
        return primary.str.upper()
    dominant = _text_series(frame.get("country_dominant"))
    if dominant is None:
        return None
    return dominant.str.upper()


def _infer_platform_capabilities(
    raw_frame: pd.DataFrame,
    canonical_frame: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    if canonical_frame.empty or "platform" not in canonical_frame.columns:
        return {}
    capabilities: Dict[str, Dict[str, Any]] = {}
    for platform in sorted(str(value) for value in canonical_frame["platform"].dropna().unique().tolist()):
        subset = canonical_frame[canonical_frame["platform"].astype(str) == platform]
        raw_subset = (
            raw_frame[raw_frame["platform"].astype(str) == platform]
            if "platform" in raw_frame.columns
            else pd.DataFrame()
        )
        publisher_available: bool | None = None
        publisher_reason: str | None = None
        if platform == "semanticscholar":
            extras = raw_subset.get("extra")
            has_field = False
            has_value = False
            seen_payload = False
            if isinstance(extras, pd.Series):
                for value in extras.tolist():
                    extra = value if isinstance(value, dict) else {}
                    payload = extra.get("semanticscholar") if isinstance(extra, dict) else {}
                    payload = payload if isinstance(payload, dict) else {}
                    if payload:
                        seen_payload = True
                    venue = payload.get("venue") if isinstance(payload.get("venue"), dict) else {}
                    venue = venue if isinstance(venue, dict) else {}
                    journal = payload.get("journal") if isinstance(payload.get("journal"), dict) else {}
                    journal = journal if isinstance(journal, dict) else {}
                    publication_venue = (
                        payload.get("publicationVenue")
                        if isinstance(payload.get("publicationVenue"), dict)
                        else {}
                    )
                    publication_venue = (
                        publication_venue if isinstance(publication_venue, dict) else {}
                    )
                    candidates = [
                        ("publisher" in venue, venue.get("publisher")),
                        ("publisher" in journal, journal.get("publisher")),
                        ("publisher" in publication_venue, publication_venue.get("publisher")),
                        ("publisher" in payload, payload.get("publisher")),
                    ]
                    for present, item in candidates:
                        if present:
                            has_field = True
                        if isinstance(item, str) and item.strip():
                            has_value = True
            if not seen_payload:
                publisher_available = None
            elif not has_field:
                publisher_available = False
                publisher_reason = "publisher not exposed by Semantic Scholar payload schema"
            else:
                publisher_available = True
                if not has_value:
                    publisher_reason = "publisher field present but values absent in sampled payloads"

        quality_counts = _citations_quality_counts(subset)
        citations_quality = _citations_quality_label(quality_counts)
        citations_available = citations_quality != "structurally_unavailable"
        citations_reason: str | None = None
        if citations_quality == "structurally_unavailable":
            citations_reason = (
                "CORE citationCount unreliable (structural limitation)"
                if platform == "core"
                else "citation signal treated as structurally unavailable"
            )

        geo_available: bool | None = None
        geo_reason: str | None = None
        country_lists = _country_lists(subset)
        has_geo_values = bool(
            country_lists is not None and (country_lists.apply(len) > 0).any()
        )
        if platform in {"semanticscholar", "core"}:
            if has_geo_values:
                geo_available = True
            else:
                geo_available = False
                geo_reason = "geo metadata structurally unavailable from source payload"
        elif country_lists is not None:
            geo_available = has_geo_values

        capabilities[platform] = {
            "publisher_available": publisher_available,
            "publisher_reason": publisher_reason,
            "citations_available": citations_available,
            "citations_reason": citations_reason,
            "citations_quality_label": citations_quality,
            "geo_available": geo_available,
            "geo_reason": geo_reason,
        }
    return capabilities


def _compute_bias_set(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
    geo_min_coverage: float = DEFAULT_GEO_MIN_COVERAGE,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["recency"] = _recency_metrics(frame)
    metrics["completeness"] = _metadata_completeness(frame, capabilities=capabilities)
    metrics["language"] = _language_bias(frame)
    metrics["open_access"] = _open_access_bias(frame)
    metrics["geo_bias"] = _geo_coverage_metrics(
        frame,
        capabilities=capabilities,
        min_coverage=geo_min_coverage,
    )
    metrics["core_ranking"] = _core_ranking_metrics(frame)
    metrics["publisher_hhi"] = _publisher_hhi(frame, capabilities=capabilities)
    metrics["rank_vs_citations"] = _rank_correlation(frame, capabilities=capabilities)
    metrics["feature_availability"] = _feature_availability(frame, capabilities=capabilities)
    metrics["top_k_bias"] = _top_k_bias_metrics(
        frame,
        ks=TOP_K_VALUES,
        capabilities=capabilities,
        geo_min_coverage=geo_min_coverage,
    )
    return metrics


def _open_access_by_platform(frame: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    if "platform" not in frame.columns:
        return {}
    result: Dict[str, Dict[str, Optional[float]]] = {}
    platforms = sorted(str(platform) for platform in frame["platform"].dropna().unique().tolist())
    for platform in platforms:
        subset = frame[frame["platform"].astype(str) == platform]
        result[platform] = _open_access_bias(subset)
    return result


def _recency_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    empty_result: Dict[str, Optional[float]] = {
        "median_year": None,
        "median_age_years": None,
        "share_age_le_2": None,
        "share_age_le_5": None,
        "share_age_le_10": None,
        "share_age_gt_10": None,
    }
    years = _select_year_column(frame)
    if years is None:
        return empty_result

    numeric_years = pd.to_numeric(years, errors="coerce").dropna().astype(int)
    current_year = datetime.utcnow().year
    plausible_years = numeric_years[(numeric_years >= 1800) & (numeric_years <= current_year)]
    if plausible_years.empty:
        return empty_result

    ages = current_year - plausible_years
    return {
        "median_year": float(plausible_years.median()),
        "median_age_years": float(ages.median()),
        "share_age_le_2": float((ages <= 2).mean()),
        "share_age_le_5": float((ages <= 5).mean()),
        "share_age_le_10": float((ages <= 10).mean()),
        "share_age_gt_10": float((ages > 10).mean()),
    }


def _metadata_completeness(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
) -> Dict[str, Optional[float]]:
    total = len(frame)
    citations = _numeric_series(_citation_series(frame))
    issn = _list_feature_series(frame, "issn")
    issn_coverage = (
        float((issn.apply(len) > 0).sum() / total) if issn is not None and total > 0 else None
    )
    citation_block = _citation_metric_block(
        frame,
        capabilities=capabilities,
    )
    citations_coverage = None if citation_block is not None else _coverage_ratio(citations, total)
    return {
        "doi": _coverage_ratio(frame.get("doi"), total),
        "year_coverage": _coverage_ratio(_select_year_column(frame), total),
        "language": _coverage_ratio(frame.get("language"), total),
        "publisher": _coverage_ratio(frame.get("publisher"), total),
        "is_oa": _coverage_ratio(frame.get("is_oa"), total),
        "citations": citations_coverage,
        "issn_coverage": issn_coverage,
    }


def _select_year_column(frame: pd.DataFrame) -> Optional[pd.Series]:
    years = frame.get("publication_year")
    if years is not None:
        return years
    return frame.get("year")


def _coverage_ratio(series: Optional[pd.Series], total: int) -> Optional[float]:
    if series is None or total == 0:
        return None
    non_missing = series.dropna()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        non_missing = non_missing[non_missing.astype(str).str.strip() != ""]
    count = len(non_missing)
    return float(count / total) if total > 0 else None


def _language_bias(frame: pd.DataFrame) -> Dict[str, Any]:
    languages = frame.get("language")
    if languages is None or languages.dropna().empty:
        return {}
    counts = Counter(str(lang).lower() for lang in languages.dropna())
    total = sum(counts.values())
    return {language: count / total for language, count in counts.items()}


def _open_access_bias(frame: pd.DataFrame) -> Dict[str, Any]:
    is_oa = _open_access_series(frame)
    if is_oa is None:
        return {"share_open_access": None}
    numeric = is_oa.dropna().astype(int)
    if numeric.empty:
        return {"share_open_access": None}
    return {"share_open_access": float(numeric.mean())}


def _core_ranking_metrics(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    stats = _core_ranking_stats(frame)
    return {
        "share_core_a_star": stats["a_star_share"],
        "share_core_a_or_higher": _safe_sum(stats["a_star_share"], stats["a_share"]),
        "share_core_ranked": _safe_sum(
            stats["a_star_share"],
            stats["a_share"],
            stats["b_share"],
            stats["c_share"],
        ),
        "share_core_unranked_or_missing": stats["missing_share"],
        "core_rank_coverage": stats["core_rank_coverage"],
    }


def core_ranking_table(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    """Return CORE ranking counts/shares overall and per platform."""

    rows = [{"platform": "overall", **_core_ranking_stats(frame)}]
    if "platform" in frame.columns:
        for platform, subset in frame.groupby("platform"):
            rows.append({"platform": str(platform), **_core_ranking_stats(subset)})
    return rows


def _core_ranking_stats(frame: pd.DataFrame) -> Dict[str, Any]:
    eligible = _select_core_ranking_frame(frame)
    total = len(eligible)
    stats: Dict[str, Any] = {"eligible_count": total}
    if total == 0 or "core_rank" not in eligible.columns:
        for label in CORE_RANK_LABELS.values():
            stats[f"{label}_count"] = None
            stats[f"{label}_share"] = None
        stats["missing_count"] = None
        stats["missing_share"] = None
        stats["core_rank_coverage"] = None
        return stats

    ranks = eligible.get("core_rank")
    if ranks is None:
        for label in CORE_RANK_LABELS.values():
            stats[f"{label}_count"] = None
            stats[f"{label}_share"] = None
        stats["missing_count"] = None
        stats["missing_share"] = None
        stats["core_rank_coverage"] = None
        return stats

    normalized = ranks.fillna("").astype(str).str.strip().str.upper()
    counts = {rank: int((normalized == rank).sum()) for rank in CORE_RANK_LABELS}
    ranked_count = sum(counts.values())
    missing_count = int(total - ranked_count)

    for rank, label in CORE_RANK_LABELS.items():
        stats[f"{label}_count"] = counts[rank]
        stats[f"{label}_share"] = float(counts[rank] / total) if total > 0 else None

    stats["missing_count"] = missing_count
    stats["missing_share"] = float(missing_count / total) if total > 0 else None
    stats["core_rank_coverage"] = _coverage_ratio(ranks, total)
    return stats


def _select_core_ranking_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "venue_type" not in frame.columns:
        return frame
    venue_type = frame.get("venue_type")
    if venue_type is None:
        return frame
    normalized = venue_type.astype(str).str.strip().str.lower()
    missing = venue_type.isna() | (normalized == "") | (normalized == "nan")
    conference = normalized == "conference"
    return frame[conference | missing]


def _publisher_hhi(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    publisher_available = (
        capabilities.get("publisher_available") if isinstance(capabilities, dict) else None
    )
    if publisher_available is False:
        return {
            "hhi": None,
            "available": False,
            "reason": capabilities.get("publisher_reason")
            or "publisher structurally unavailable for this platform",
        }
    publishers = frame.get("publisher")
    if publishers is None:
        return {"hhi": None, "available": False}
    publishers = publishers.dropna()
    if publishers.empty:
        return {"hhi": None, "available": False}
    counts = Counter(str(pub).lower() for pub in publishers)
    total = sum(counts.values())
    hhi = sum((count / total) ** 2 for count in counts.values())
    return {"hhi": float(hhi), "available": True}


def _rank_correlation(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    citation_block = _citation_metric_block(frame, capabilities=capabilities)
    if citation_block is not None:
        return {
            "spearman": None,
            "reason": citation_block["reason"],
            "note": citation_block["note"],
        }
    citation_series = _citation_series(frame)
    if "rank" not in frame.columns or citation_series is None:
        return {"spearman": None}
    subset = pd.DataFrame({"rank": frame["rank"], "citations": citation_series}).dropna()
    if subset.empty:
        return {"spearman": None}
    rho, _ = spearmanr(subset["rank"], subset["citations"])
    if np.isnan(rho):
        return {"spearman": None}
    return {"spearman": float(rho)}


def _citations_quality_counts(frame: pd.DataFrame) -> dict[str, int]:
    quality = frame.get("metrics_quality")
    if not isinstance(quality, pd.Series):
        return {}
    counter = Counter(
        str((item or {}).get("citations") if isinstance(item, dict) else "missing")
        for item in quality.tolist()
    )
    return {str(key): int(value) for key, value in counter.items()}


def _citations_quality_label(counts: Mapping[str, int]) -> str:
    if int(counts.get("structurally_unavailable", 0)) > 0:
        return "structurally_unavailable"
    if int(counts.get("suspicious", 0)) > 0:
        return "suspicious"
    if sum(int(value) for value in counts.values()) == 0:
        return "missing"
    if int(counts.get("missing", 0)) == sum(int(value) for value in counts.values()):
        return "missing"
    return "ok"


def _citation_metric_block(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
) -> Dict[str, str] | None:
    if isinstance(capabilities, Mapping) and capabilities.get("citations_available") is False:
        reason = "citations_unavailable"
        note = str(capabilities.get("citations_reason") or "citation signal unavailable")
        return {"reason": reason, "note": note}

    quality_label = _citations_quality_label(_citations_quality_counts(frame))
    if quality_label == "structurally_unavailable":
        return {
            "reason": "citations_quality_structurally_unavailable",
            "note": "citation signal treated as structurally unavailable",
        }
    if quality_label == "suspicious":
        return {
            "reason": "citations_quality_suspicious",
            "note": "citation distribution flagged as suspicious",
        }
    return None


def _citations_suspicious(frame: pd.DataFrame) -> bool:
    return _citations_quality_label(_citations_quality_counts(frame)) == "suspicious"


def _open_access_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    series = frame.get("is_oa")
    if series is None:
        series = frame.get("is_open_access")
    if series is None:
        return None
    return series.apply(_coerce_bool)


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


def _citation_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    if "citations" in frame.columns:
        return frame["citations"]
    return frame.get("cited_by_count")


def _minimum_reliable_count(k: int, effective_k: int) -> int:
    if effective_k <= 0:
        return 0
    return min(effective_k, max(10, int(math.ceil(0.3 * max(k, 1)))))


def _reliability_label(*, available_count: int, effective_k: int, missing_pct: float | None) -> str:
    if effective_k <= 0 or available_count <= 0:
        return "low"
    missing = missing_pct if missing_pct is not None else float(1.0 - (available_count / effective_k))
    if available_count >= max(30, int(math.ceil(0.6 * effective_k))) and missing <= 0.2:
        return "high"
    if available_count >= max(10, int(math.ceil(0.3 * effective_k))) and missing <= 0.5:
        return "medium"
    return "low"


def _feature_availability(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    total = int(len(frame))
    publisher_available = (
        capabilities.get("publisher_available") if isinstance(capabilities, dict) else None
    )
    publisher_reason = (
        capabilities.get("publisher_reason") if isinstance(capabilities, dict) else None
    )
    if total == 0:
        return {
            "total_records": 0,
            "oa": {"available_count": 0, "available_share": None},
            "country": {"available_count": 0, "available_share": None},
            "citations": {"available_count": 0, "available_share": None},
            "doc_type": {"available_count": 0, "available_share": None},
            "publisher": {"available_count": 0, "available_share": None},
            "issn": {"available_count": 0, "available_share": None},
            "journal_title": {"available_count": 0, "available_share": None},
        }

    oa = _open_access_series(frame)
    countries = _country_lists(frame)
    citations = _numeric_series(_citation_series(frame))
    quality_counts = _citations_quality_counts(frame)
    citations_quality = _citations_quality_label(quality_counts)
    citation_block = _citation_metric_block(frame, capabilities=capabilities)
    citations_available_count = int(citations.notna().sum()) if citations is not None else 0
    doc_type = _text_series(frame.get("doc_type"))
    publishers = _text_series(frame.get("publisher"))
    issn = _list_feature_series(frame, "issn")
    if issn is None:
        issn = _list_feature_series(frame, "issn_list")
    journals = _text_series(frame.get("journal_title"))

    return {
        "total_records": total,
        "oa": _availability_block(int(oa.notna().sum()) if oa is not None else 0, total),
        "country": _availability_block(
            int((countries.apply(len) > 0).sum()) if countries is not None else 0,
            total,
        ),
        "citations": {
            **_availability_block(citations_available_count, total),
            "available_for_bias_metrics": bool(
                (citations_available_count > 0) and citation_block is None
            ),
            "quality": citations_quality,
            "note": (citation_block.get("note") if citation_block is not None else None),
        },
        "doc_type": _availability_block(int(doc_type.notna().sum()) if doc_type is not None else 0, total),
        "publisher": {
            **_availability_block(int(publishers.notna().sum()) if publishers is not None else 0, total),
            "structurally_unavailable": bool(publisher_available is False),
            "note": (
                publisher_reason
                if publisher_available is False
                else None
            ),
        },
        "issn": _availability_block(
            int((issn.apply(len) > 0).sum()) if issn is not None else 0,
            total,
        ),
        "journal_title": _availability_block(
            int(journals.notna().sum()) if journals is not None else 0,
            total,
        ),
    }


def _availability_block(count: int, total: int) -> Dict[str, Any]:
    return {
        "available_count": int(count),
        "available_share": (float(count / total) if total > 0 else None),
    }


def _geo_coverage_metrics(
    frame: pd.DataFrame,
    *,
    capabilities: Mapping[str, Any] | None = None,
    min_coverage: float = DEFAULT_GEO_MIN_COVERAGE,
) -> Dict[str, Any]:
    total = len(frame)
    country_lists = _country_lists(frame)
    dominant_series = _country_primary_series(frame)

    available_count = (
        int((country_lists.apply(len) > 0).sum()) if country_lists is not None else 0
    )
    available_share = (float(available_count / total) if total > 0 else None)
    missing_pct = (float(1.0 - available_share) if available_share is not None else None)

    if dominant_series is not None:
        multi_count = int((dominant_series == "MULTI").sum())
    elif country_lists is not None:
        multi_count = int((country_lists.apply(len) > 1).sum())
    else:
        multi_count = 0
    multi_share = (
        float(multi_count / available_count) if available_count > 0 else None
    )

    structural_unavailable = bool(
        isinstance(capabilities, Mapping) and capabilities.get("geo_available") is False
    )
    reason: str | None = None
    if structural_unavailable:
        reason = str(capabilities.get("geo_reason") or "structural_unavailability")
    elif total == 0:
        reason = "no_records"
    elif available_share is None or available_share < min_coverage:
        reason = f"insufficient_coverage (< {min_coverage:.2f})"

    enabled = bool(
        not structural_unavailable
        and total > 0
        and available_share is not None
        and available_share >= min_coverage
    )
    return {
        "available_count": available_count,
        "available_share": available_share,
        "missing_pct": missing_pct,
        "multi_count": multi_count,
        "multi_country_share": multi_share,
        "multi_share": multi_share,
        "min_coverage_threshold": float(min_coverage),
        "enabled_for_bias_metrics": enabled,
        "structurally_unavailable": structural_unavailable,
        "reason": reason,
    }


def _top_k_bias_metrics(
    frame: pd.DataFrame,
    *,
    ks: tuple[int, ...],
    capabilities: Mapping[str, Any] | None = None,
    geo_min_coverage: float = DEFAULT_GEO_MIN_COVERAGE,
) -> Dict[str, Any]:
    ranked = _ranked_frame(frame)
    return {
        "ks": [int(k) for k in ks],
        "oa": _oa_top_k_bias(frame, ranked, ks=ks),
        "country": _country_top_k_bias(
            frame,
            ranked,
            ks=ks,
            capabilities=capabilities,
            min_coverage=geo_min_coverage,
        ),
        "citations": _citation_top_k_bias(frame, ranked, ks=ks, capabilities=capabilities),
        "journal_issn": _journal_issn_top_k_bias(frame, ranked, ks=ks),
        "doc_type": _doc_type_top_k_bias(frame, ranked, ks=ks),
    }


def _ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "rank" not in frame.columns:
        return pd.DataFrame()
    ranked = frame.copy()
    ranked["_rank_numeric"] = pd.to_numeric(ranked["rank"], errors="coerce")
    ranked = ranked.dropna(subset=["_rank_numeric"])
    if ranked.empty:
        return ranked
    ranked["_doi_sort"] = ranked.get("doi", pd.Series(index=ranked.index)).fillna("").astype(str)
    ranked["_title_sort"] = (
        ranked.get("title", pd.Series(index=ranked.index)).fillna("").astype(str)
    )
    ranked["_raw_id_sort"] = (
        ranked.get("raw_id", pd.Series(index=ranked.index)).fillna("").astype(str)
    )
    return ranked.sort_values(
        by=["_rank_numeric", "_doi_sort", "_title_sort", "_raw_id_sort"],
        kind="mergesort",
        na_position="last",
    )


def _oa_top_k_bias(frame: pd.DataFrame, ranked: pd.DataFrame, *, ks: tuple[int, ...]) -> Dict[str, Any]:
    oa = _open_access_series(frame)
    total = len(frame)
    missing_pct = 1.0 if total > 0 else None
    overall_share: float | None = None
    overall_available_count = 0
    if oa is not None and total > 0:
        missing_pct = float(1.0 - (oa.notna().sum() / total))
        available = oa.dropna().astype(int)
        overall_available_count = int(len(available))
        if not available.empty:
            overall_share = float(available.mean())
    overall_minimum = _minimum_reliable_count(total, total)

    per_k: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        effective_k = int(min(max(k, 0), len(ranked)))
        top = ranked.head(k)
        top_oa = _open_access_series(top)
        if top_oa is None:
            per_k[str(k)] = {
                "available": False,
                "effective_k": effective_k,
                "available_count": 0,
                "minimum_required": _minimum_reliable_count(k, effective_k),
                "top_k_share": None,
                "overall_share": overall_share,
                "delta_share": None,
                "missing_pct": 1.0 if effective_k > 0 else None,
                "reliability": "low",
            }
            continue
        top_available = top_oa.dropna().astype(int)
        available_count = int(len(top_available))
        minimum_required = _minimum_reliable_count(k, effective_k)
        missing_top = float(1.0 - (len(top_available) / effective_k)) if effective_k else None
        if top_available.empty:
            per_k[str(k)] = {
                "available": False,
                "effective_k": effective_k,
                "available_count": available_count,
                "minimum_required": minimum_required,
                "top_k_share": None,
                "overall_share": overall_share,
                "delta_share": None,
                "missing_pct": missing_top,
                "reliability": _reliability_label(
                    available_count=available_count,
                    effective_k=effective_k,
                    missing_pct=missing_top,
                ),
            }
            continue
        top_share = float(top_available.mean())
        stable = (
            overall_share is not None
            and available_count >= minimum_required
            and overall_available_count >= overall_minimum
        )
        per_k[str(k)] = {
            "available": stable,
            "effective_k": effective_k,
            "available_count": available_count,
            "minimum_required": minimum_required,
            "top_k_share": top_share,
            "overall_share": overall_share,
            "delta_share": ((top_share - overall_share) if stable and overall_share is not None else None),
            "missing_pct": missing_top,
            "reliability": _reliability_label(
                available_count=available_count,
                effective_k=effective_k,
                missing_pct=missing_top,
            ),
        }

    return {
        "available": overall_share is not None,
        "missing_pct": missing_pct,
        "overall_share": overall_share,
        "overall_available_count": overall_available_count,
        "minimum_required": overall_minimum,
        "reliability": _reliability_label(
            available_count=overall_available_count,
            effective_k=total,
            missing_pct=missing_pct,
        ),
        "per_k": per_k,
    }


def _country_top_k_bias(
    frame: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    ks: tuple[int, ...],
    capabilities: Mapping[str, Any] | None = None,
    min_coverage: float = DEFAULT_GEO_MIN_COVERAGE,
) -> Dict[str, Any]:
    country_lists = _country_lists(frame)
    dominant_series = _country_primary_series(frame)
    total = len(frame)
    overall_minimum = _minimum_reliable_count(total, total)
    coverage = _geo_coverage_metrics(
        frame,
        capabilities=capabilities,
        min_coverage=min_coverage,
    )

    def empty_row(k: int) -> dict[str, Any]:
        effective_k = int(min(max(k, 0), len(ranked)))
        return {
            "available": False,
            "effective_k": effective_k,
            "available_count": 0,
            "minimum_required": _minimum_reliable_count(k, effective_k),
            "js_divergence": None,
            "overrepresentation_ratio": {},
            "top_distribution": {},
            "missing_pct_top_k": (1.0 if effective_k > 0 else None),
            "reliability": "low",
            "variants": {
                "countries": {
                    "fractional": {
                        "available": False,
                        "top_distribution": {},
                        "js_divergence": None,
                        "overrepresentation_ratio": {},
                    },
                    "dominant": {
                        "available": False,
                        "top_distribution": {},
                        "js_divergence": None,
                        "overrepresentation_ratio": {},
                    },
                },
                "regions": {
                    "fractional": {
                        "available": False,
                        "top_distribution": {},
                        "js_divergence": None,
                        "overrepresentation_ratio": {},
                    },
                    "dominant": {
                        "available": False,
                        "top_distribution": {},
                        "js_divergence": None,
                        "overrepresentation_ratio": {},
                    },
                },
            },
        }

    if country_lists is None or total == 0:
        return {
            "available": False,
            "missing_pct": coverage.get("missing_pct"),
            "available_share": coverage.get("available_share"),
            "multi_country_share": coverage.get("multi_country_share"),
            "multi_share": coverage.get("multi_share"),
            "enabled_for_bias_metrics": coverage.get("enabled_for_bias_metrics"),
            "structurally_unavailable": coverage.get("structurally_unavailable"),
            "reason": coverage.get("reason"),
            "overall_distribution": {},
            "overall_distribution_variants": {
                "countries": {"fractional": {}, "dominant": {}},
                "regions": {"fractional": {}, "dominant": {}},
            },
            "overall_available_count": int(coverage.get("available_count") or 0),
            "minimum_required": overall_minimum,
            "min_coverage_threshold": float(coverage.get("min_coverage_threshold") or min_coverage),
            "reliability": "low",
            "per_k": {str(k): empty_row(k) for k in ks},
        }

    overall_country_fractional = _country_distribution_fractional(country_lists)
    overall_country_dominant = _country_distribution_dominant(
        country_lists=country_lists,
        dominant_series=dominant_series,
    )
    overall_region_fractional = _distribution_to_regions(overall_country_fractional)
    overall_region_dominant = _distribution_to_regions(overall_country_dominant)
    gating_enabled = bool(coverage.get("enabled_for_bias_metrics"))

    per_k: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        top_frame = ranked.head(k)
        top_lists = _country_lists(top_frame)
        top_dominant = _country_primary_series(top_frame)
        if top_lists is None:
            per_k[str(k)] = empty_row(k)
            continue

        effective_k = int(min(max(k, 0), len(ranked)))
        minimum_required = _minimum_reliable_count(k, effective_k)
        available_count = int((top_lists.apply(len) > 0).sum())
        missing_top = (
            float((top_lists.apply(len) == 0).sum() / effective_k) if effective_k > 0 else None
        )
        top_country_fractional = _country_distribution_fractional(top_lists)
        top_country_dominant = _country_distribution_dominant(
            country_lists=top_lists,
            dominant_series=top_dominant,
        )
        top_region_fractional = _distribution_to_regions(top_country_fractional)
        top_region_dominant = _distribution_to_regions(top_country_dominant)

        reliable_sample = (
            gating_enabled
            and available_count >= minimum_required
            and int(coverage.get("available_count") or 0) >= overall_minimum
        )

        def variant_payload(
            top_distribution: Dict[str, float],
            overall_distribution: Dict[str, float],
        ) -> Dict[str, Any]:
            stable = reliable_sample and bool(top_distribution) and bool(overall_distribution)
            return {
                "available": stable,
                "top_distribution": top_distribution,
                "js_divergence": (
                    _js_divergence(overall_distribution, top_distribution) if stable else None
                ),
                "overrepresentation_ratio": (
                    _overrepresentation_ratio(top_distribution, overall_distribution, top_n=10)
                    if stable
                    else {}
                ),
            }

        country_fractional_variant = variant_payload(
            top_country_fractional, overall_country_fractional
        )
        per_k[str(k)] = {
            "available": country_fractional_variant["available"],
            "effective_k": effective_k,
            "available_count": available_count,
            "minimum_required": minimum_required,
            "js_divergence": country_fractional_variant["js_divergence"],
            "top_distribution": country_fractional_variant["top_distribution"],
            "overrepresentation_ratio": country_fractional_variant["overrepresentation_ratio"],
            "missing_pct_top_k": missing_top,
            "reliability": _reliability_label(
                available_count=available_count,
                effective_k=effective_k,
                missing_pct=missing_top,
            ),
            "variants": {
                "countries": {
                    "fractional": country_fractional_variant,
                    "dominant": variant_payload(top_country_dominant, overall_country_dominant),
                },
                "regions": {
                    "fractional": variant_payload(top_region_fractional, overall_region_fractional),
                    "dominant": variant_payload(top_region_dominant, overall_region_dominant),
                },
            },
        }

    return {
        "available": bool(overall_country_fractional) and gating_enabled,
        "missing_pct": coverage.get("missing_pct"),
        "available_share": coverage.get("available_share"),
        "multi_country_share": coverage.get("multi_country_share"),
        "multi_share": coverage.get("multi_share"),
        "enabled_for_bias_metrics": gating_enabled,
        "structurally_unavailable": bool(coverage.get("structurally_unavailable")),
        "reason": coverage.get("reason"),
        "overall_available_count": int(coverage.get("available_count") or 0),
        "minimum_required": overall_minimum,
        "min_coverage_threshold": float(coverage.get("min_coverage_threshold") or min_coverage),
        "reliability": _reliability_label(
            available_count=int(coverage.get("available_count") or 0),
            effective_k=total,
            missing_pct=coverage.get("missing_pct"),
        ),
        # Backward-compatible defaults used by existing report code.
        "overall_distribution": overall_country_fractional,
        "overall_distribution_variants": {
            "countries": {
                "fractional": overall_country_fractional,
                "dominant": overall_country_dominant,
            },
            "regions": {
                "fractional": overall_region_fractional,
                "dominant": overall_region_dominant,
            },
        },
        "per_k": per_k,
    }


def _citation_top_k_bias(
    frame: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    ks: tuple[int, ...],
    capabilities: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    citations = _numeric_series(_citation_series(frame))
    total = len(frame)
    quality_counts = _citations_quality_counts(frame)
    citation_block = _citation_metric_block(frame, capabilities=capabilities)
    citations_blocked = citation_block is not None
    if total == 0:
        return {
            "available": False,
            "missing_pct": None,
            "spearman_rank_vs_citations": None,
            "overall_median": None,
            "reliability": "low",
            "reason": citation_block.get("reason") if citation_block else None,
            "note": citation_block.get("note") if citation_block else None,
            "quality": quality_counts or None,
            "per_k": {str(k): {"available": False, "effective_k": 0, "reliability": "low"} for k in ks},
        }

    if citations is None:
        missing_pct = 1.0
        available = pd.Series(dtype=float)
    else:
        missing_pct = float(citations.isna().mean())
        available = citations.dropna()
    overall_median = float(available.median()) if not available.empty else None

    spearman: float | None = None
    ranked_citations = _numeric_series(_citation_series(ranked))
    if not citations_blocked and not ranked.empty and ranked_citations is not None:
        corr_frame = pd.DataFrame(
            {"_rank_numeric": ranked["_rank_numeric"], "cited_by_count": ranked_citations}
        )
        corr_frame = corr_frame.dropna()
        if not corr_frame.empty:
            rho, _ = spearmanr(corr_frame["_rank_numeric"], corr_frame["cited_by_count"])
            if not np.isnan(rho):
                spearman = float(rho)

    per_k: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        effective_k = int(min(max(k, 0), len(ranked)))
        top = ranked.head(k).copy()
        rest = ranked.iloc[k:].copy()
        top_series = _numeric_series(_citation_series(top))
        rest_series = _numeric_series(_citation_series(rest))
        top_vals = top_series.dropna() if top_series is not None else pd.Series(dtype=float)
        rest_vals = rest_series.dropna() if rest_series is not None else pd.Series(dtype=float)
        available_count = int(len(top_vals))
        minimum_required = _minimum_reliable_count(k, effective_k)
        missing_top = float(1.0 - (len(top_vals) / effective_k)) if effective_k > 0 else None
        delta = None
        if (
            not citations_blocked
            and available_count >= minimum_required
            and int(len(rest_vals)) >= minimum_required
        ):
            delta = float(top_vals.median() - rest_vals.median())
        per_k[str(k)] = {
            "available": (
                not citations_blocked
                and available_count >= minimum_required
                and overall_median is not None
            ),
            "effective_k": effective_k,
            "available_count": available_count,
            "minimum_required": minimum_required,
            "median_top_k": (float(top_vals.median()) if not top_vals.empty else None),
            "median_rest": (float(rest_vals.median()) if not rest_vals.empty else None),
            "delta_median_top_vs_rest": delta,
            "missing_pct_top_k": missing_top,
            "reliability": (
                "low"
                if citations_blocked
                else _reliability_label(
                    available_count=available_count,
                    effective_k=effective_k,
                    missing_pct=missing_top,
                )
            ),
        }

    return {
        "available": bool((not available.empty) and not citations_blocked),
        "missing_pct": missing_pct,
        "spearman_rank_vs_citations": spearman,
        "overall_median": overall_median,
        "reliability": (
            "low"
            if citations_blocked
            else _reliability_label(
                available_count=int(len(available)),
                effective_k=total,
                missing_pct=missing_pct,
            )
        ),
        "reason": citation_block.get("reason") if citation_block else None,
        "note": citation_block.get("note") if citation_block else None,
        "quality": quality_counts or None,
        "per_k": per_k,
    }


def _journal_issn_top_k_bias(
    frame: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    ks: tuple[int, ...],
) -> Dict[str, Any]:
    journals = _text_series(frame.get("journal_title"))
    issn_lists = _list_feature_series(frame, "issn_list")
    total = len(frame)

    journal_missing = None
    issn_missing = None
    if total > 0:
        journal_missing = (
            float(journals.isna().mean()) if journals is not None else 1.0
        )
        issn_missing = (
            float((issn_lists.apply(len) == 0).mean()) if issn_lists is not None else 1.0
        )

    per_k: Dict[str, Dict[str, Any]] = {}
    overall_journal_available = int(journals.notna().sum()) if journals is not None else 0
    overall_issn_available = int((issn_lists.apply(len) > 0).sum()) if issn_lists is not None else 0
    for k in ks:
        top = ranked.head(k)
        effective_k = int(min(max(k, 0), len(ranked)))
        minimum_required = _minimum_reliable_count(k, effective_k)
        top_journals = _text_series(top.get("journal_title"))
        top_issn_lists = _list_feature_series(top, "issn_list")
        unique_journals = (
            sorted(set(top_journals.dropna().tolist())) if top_journals is not None else []
        )
        unique_issn: set[str] = set()
        journal_available_count = int(top_journals.notna().sum()) if top_journals is not None else 0
        issn_available_count = (
            int((top_issn_lists.apply(len) > 0).sum()) if top_issn_lists is not None else 0
        )
        journal_missing_top = (
            float(1.0 - (journal_available_count / effective_k)) if effective_k > 0 else None
        )
        issn_missing_top = (
            float(1.0 - (issn_available_count / effective_k)) if effective_k > 0 else None
        )
        if top_issn_lists is not None:
            for values in top_issn_lists:
                unique_issn.update(values)
        per_k[str(k)] = {
            "available": effective_k > 0,
            "effective_k": effective_k,
            "minimum_required": minimum_required,
            "available_journal_count": journal_available_count,
            "available_issn_count": issn_available_count,
            "unique_journal_title_count": len(unique_journals),
            "unique_issn_count": len(unique_issn),
            "journal_diversity": (
                float(len(unique_journals) / effective_k) if effective_k > 0 else None
            ),
            "issn_diversity": (float(len(unique_issn) / effective_k) if effective_k > 0 else None),
            "missing_pct_journal_title_top_k": journal_missing_top,
            "missing_pct_issn_top_k": issn_missing_top,
            "reliability_journal_title": _reliability_label(
                available_count=journal_available_count,
                effective_k=effective_k,
                missing_pct=journal_missing_top,
            ),
            "reliability_issn": _reliability_label(
                available_count=issn_available_count,
                effective_k=effective_k,
                missing_pct=issn_missing_top,
            ),
        }

    return {
        "available": (journals is not None and journals.notna().any())
        or (issn_lists is not None and (issn_lists.apply(len) > 0).any()),
        "missing_pct_journal_title": journal_missing,
        "missing_pct_issn": issn_missing,
        "overall_available_journal_count": overall_journal_available,
        "overall_available_issn_count": overall_issn_available,
        "reliability_journal_title": _reliability_label(
            available_count=overall_journal_available,
            effective_k=total,
            missing_pct=journal_missing,
        ),
        "reliability_issn": _reliability_label(
            available_count=overall_issn_available,
            effective_k=total,
            missing_pct=issn_missing,
        ),
        "per_k": per_k,
    }


def _doc_type_top_k_bias(
    frame: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    ks: tuple[int, ...],
) -> Dict[str, Any]:
    doc_type = _text_series(frame.get("doc_type"))
    total = len(frame)
    if doc_type is None or total == 0:
        return {
            "available": False,
            "missing_pct": (1.0 if total > 0 else None),
            "overall_distribution": {},
            "overall_available_count": 0,
            "minimum_required": _minimum_reliable_count(total, total),
            "reliability": "low",
            "per_k": {
                str(k): {
                    "available": False,
                    "effective_k": int(min(max(k, 0), len(ranked))),
                    "available_count": 0,
                    "minimum_required": _minimum_reliable_count(k, int(min(max(k, 0), len(ranked)))),
                    "js_divergence": None,
                    "top_distribution": {},
                    "missing_pct_top_k": (1.0 if int(min(max(k, 0), len(ranked))) > 0 else None),
                    "reliability": "low",
                }
                for k in ks
            },
        }

    cleaned = doc_type.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    missing_pct = float(1.0 - (len(cleaned) / total)) if total else None
    overall_available_count = int(len(cleaned))
    overall_minimum = _minimum_reliable_count(total, total)
    overall_distribution = _normalized_distribution(Counter(cleaned.tolist()))

    per_k: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        top = ranked.head(k)
        top_doc_type = _text_series(top.get("doc_type"))
        effective_k = int(min(max(k, 0), len(ranked)))
        minimum_required = _minimum_reliable_count(k, effective_k)
        if top_doc_type is None:
            per_k[str(k)] = {
                "available": False,
                "effective_k": effective_k,
                "available_count": 0,
                "minimum_required": minimum_required,
                "js_divergence": None,
                "top_distribution": {},
                "missing_pct_top_k": 1.0 if effective_k > 0 else None,
                "reliability": "low",
            }
            continue
        top_cleaned = top_doc_type.fillna("").astype(str).str.strip()
        top_cleaned = top_cleaned[top_cleaned != ""]
        available_count = int(len(top_cleaned))
        missing_top = (
            float(1.0 - (available_count / effective_k)) if effective_k > 0 else None
        )
        top_distribution = _normalized_distribution(Counter(top_cleaned.tolist()))
        stable = (
            available_count >= minimum_required
            and overall_available_count >= overall_minimum
            and bool(overall_distribution)
            and bool(top_distribution)
        )
        per_k[str(k)] = {
            "available": stable,
            "effective_k": effective_k,
            "available_count": available_count,
            "minimum_required": minimum_required,
            "js_divergence": (
                _js_divergence(overall_distribution, top_distribution) if stable else None
            ),
            "top_distribution": top_distribution,
            "missing_pct_top_k": missing_top,
            "reliability": _reliability_label(
                available_count=available_count,
                effective_k=effective_k,
                missing_pct=missing_top,
            ),
        }

    return {
        "available": bool(overall_distribution),
        "missing_pct": missing_pct,
        "overall_available_count": overall_available_count,
        "minimum_required": overall_minimum,
        "reliability": _reliability_label(
            available_count=overall_available_count,
            effective_k=total,
            missing_pct=missing_pct,
        ),
        "overall_distribution": overall_distribution,
        "per_k": per_k,
    }


def _list_feature_series(frame: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in frame.columns:
        return None
    return frame[column].apply(_coerce_list_of_strings)


def _text_series(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    cleaned = series.apply(
        lambda value: str(value).strip() if value is not None and not pd.isna(value) else None
    )
    return cleaned.where(cleaned.notna() & (cleaned != ""), None)


def _numeric_series(series: object) -> Optional[pd.Series]:
    if not isinstance(series, pd.Series):
        return None
    return pd.to_numeric(series, errors="coerce")


_LIST_LIKE_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")


def _parse_list_like_text(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    quoted: list[str] = []
    for first, second in _LIST_LIKE_QUOTED_RE.findall(text):
        token = str(first or second).strip()
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
            token = str(item).strip()
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
            return [part.strip().strip("'\"") for part in middle.replace(";", ",").split(",") if part.strip()]
    return []


def _coerce_list_of_strings(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _parse_list_like_text(text)
        if parsed:
            items = parsed
        elif "," in text or ";" in text:
            items = [part.strip() for part in text.replace(";", ",").split(",")]
        else:
            items = [text]
    elif hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
        except Exception:
            converted = []
        if isinstance(converted, (list, tuple, set)):
            items = converted
        else:
            return []
    else:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


_ISO2_REGION_MAP: dict[str, str] = {
    "AU": "Oceania",
    "AT": "Europe",
    "BE": "Europe",
    "BR": "South America",
    "CA": "North America",
    "CH": "Europe",
    "CN": "Asia",
    "CZ": "Europe",
    "DE": "Europe",
    "DK": "Europe",
    "ES": "Europe",
    "FI": "Europe",
    "FR": "Europe",
    "GB": "Europe",
    "GR": "Europe",
    "HU": "Europe",
    "IE": "Europe",
    "IN": "Asia",
    "IT": "Europe",
    "JP": "Asia",
    "KR": "Asia",
    "LU": "Europe",
    "NL": "Europe",
    "NO": "Europe",
    "NZ": "Oceania",
    "PL": "Europe",
    "PT": "Europe",
    "RO": "Europe",
    "RU": "Europe",
    "SE": "Europe",
    "SG": "Asia",
    "TR": "Asia",
    "TW": "Asia",
    "US": "North America",
    "ZA": "Africa",
}


def _country_distribution_fractional(country_lists: pd.Series) -> Dict[str, float]:
    counter: Counter[str] = Counter()
    for values in country_lists.tolist():
        cleaned = _coerce_list_of_strings(values)
        normalized: list[str] = []
        seen: set[str] = set()
        for country in cleaned:
            code = normalize_country_code(country)
            if not code or code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        if not normalized:
            continue
        weight = 1.0 / float(len(normalized))
        for country in normalized:
            counter[country] += weight
    return _normalized_distribution(counter)


def _country_distribution_dominant(
    *,
    country_lists: pd.Series,
    dominant_series: Optional[pd.Series],
) -> Dict[str, float]:
    counter: Counter[str] = Counter()
    dominant_values = (
        dominant_series.tolist()
        if dominant_series is not None
        else [None] * len(country_lists)
    )
    for values, dominant in zip(country_lists.tolist(), dominant_values, strict=False):
        dominant_text = str(dominant).strip().upper() if isinstance(dominant, str) else ""
        if dominant_text and dominant_text != "MULTI":
            code = normalize_country_code(dominant_text)
            if code:
                counter[code] += 1.0
            continue
        cleaned = _coerce_list_of_strings(values)
        if len(cleaned) == 1:
            code = normalize_country_code(cleaned[0])
            if code:
                counter[code] += 1.0
        elif dominant_text == "MULTI":
            counter["MULTI"] += 1.0
    return _normalized_distribution(counter)


def _distribution_to_regions(distribution: Dict[str, float]) -> Dict[str, float]:
    counter: Counter[str] = Counter()
    for country, share in distribution.items():
        code = str(country).upper()
        if code == "MULTI":
            region = "Multiple countries"
        else:
            region = _ISO2_REGION_MAP.get(code, "Other")
        counter[region] += float(share)
    return _normalized_distribution(counter)


def _normalized_distribution(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    pairs = sorted(
        ((key, float(value / total)) for key, value in counter.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return {key: share for key, share in pairs}


def _js_divergence(left: Dict[str, float], right: Dict[str, float]) -> float | None:
    if not left or not right:
        return None
    labels = sorted(set(left.keys()) | set(right.keys()))
    if not labels:
        return None
    p = np.array([left.get(label, 0.0) for label in labels], dtype=float)
    q = np.array([right.get(label, 0.0) for label in labels], dtype=float)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _overrepresentation_ratio(
    top_distribution: Dict[str, float],
    overall_distribution: Dict[str, float],
    *,
    top_n: int,
) -> Dict[str, float | None]:
    if not top_distribution:
        return {}
    ordered = sorted(top_distribution.items(), key=lambda pair: (-pair[1], pair[0]))[:top_n]
    ratios: Dict[str, float | None] = {}
    for country, top_share in ordered:
        overall_share = overall_distribution.get(country, 0.0)
        ratios[country] = (top_share / overall_share) if overall_share > 0 else None
    return ratios


def _safe_sum(*values: Optional[float]) -> Optional[float]:
    if any(value is None for value in values):
        return None
    cleaned = [value for value in values if value is not None]
    return float(sum(cleaned))
