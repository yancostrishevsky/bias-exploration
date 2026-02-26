"""Bias-oriented metrics for retrieved records."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ai_bias_search.normalize.records import normalize_records

CORE_RANK_LABELS = {"A*": "a_star", "A": "a", "B": "b", "C": "c"}
TOP_K_VALUES = (10, 20, 50)


def compute_bias_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute a suite of bias metrics from an enriched dataframe."""

    canonical = _canonicalize_frame(frame)
    overall = _compute_bias_set(canonical)
    by_platform: Dict[str, Dict[str, Any]] = {}
    if "platform" in canonical.columns:
        platforms = sorted(
            str(platform) for platform in canonical["platform"].dropna().unique().tolist()
        )
        for platform in platforms:
            subset = canonical[canonical["platform"].astype(str) == platform]
            by_platform[platform] = _compute_bias_set(subset)

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
    canonical["issn_list"] = canonical.get("issn")
    return canonical


def _compute_bias_set(frame: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["recency"] = _recency_metrics(frame)
    metrics["completeness"] = _metadata_completeness(frame)
    metrics["language"] = _language_bias(frame)
    metrics["open_access"] = _open_access_bias(frame)
    metrics["core_ranking"] = _core_ranking_metrics(frame)
    metrics["publisher_hhi"] = _publisher_hhi(frame)
    metrics["rank_vs_citations"] = _rank_correlation(frame)
    metrics["feature_availability"] = _feature_availability(frame)
    metrics["top_k_bias"] = _top_k_bias_metrics(frame, ks=TOP_K_VALUES)
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


def _metadata_completeness(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    total = len(frame)
    citations = _numeric_series(_citation_series(frame))
    issn = _list_feature_series(frame, "issn")
    issn_coverage = (
        float((issn.apply(len) > 0).sum() / total) if issn is not None and total > 0 else None
    )
    return {
        "doi": _coverage_ratio(frame.get("doi"), total),
        "year_coverage": _coverage_ratio(_select_year_column(frame), total),
        "language": _coverage_ratio(frame.get("language"), total),
        "publisher": _coverage_ratio(frame.get("publisher"), total),
        "is_oa": _coverage_ratio(frame.get("is_oa"), total),
        "citations": _coverage_ratio(citations, total),
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


def _publisher_hhi(frame: pd.DataFrame) -> Dict[str, Any]:
    publishers = frame.get("publisher")
    if publishers is None:
        return {"hhi": None}
    publishers = publishers.dropna()
    if publishers.empty:
        return {"hhi": None}
    counts = Counter(str(pub).lower() for pub in publishers)
    total = sum(counts.values())
    hhi = sum((count / total) ** 2 for count in counts.values())
    return {"hhi": float(hhi)}


def _rank_correlation(frame: pd.DataFrame) -> Dict[str, Any]:
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


def _feature_availability(frame: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(frame))
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
    countries = _list_feature_series(frame, "affiliation_countries")
    citations = _numeric_series(_citation_series(frame))
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
        "citations": _availability_block(int(citations.notna().sum()) if citations is not None else 0, total),
        "doc_type": _availability_block(int(doc_type.notna().sum()) if doc_type is not None else 0, total),
        "publisher": _availability_block(
            int(publishers.notna().sum()) if publishers is not None else 0,
            total,
        ),
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


def _top_k_bias_metrics(frame: pd.DataFrame, *, ks: tuple[int, ...]) -> Dict[str, Any]:
    ranked = _ranked_frame(frame)
    return {
        "ks": [int(k) for k in ks],
        "oa": _oa_top_k_bias(frame, ranked, ks=ks),
        "country": _country_top_k_bias(frame, ranked, ks=ks),
        "citations": _citation_top_k_bias(frame, ranked, ks=ks),
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
) -> Dict[str, Any]:
    country_lists = _list_feature_series(frame, "affiliation_countries")
    total = len(frame)
    if country_lists is None or total == 0:
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
                    "overrepresentation_ratio": {},
                    "missing_pct_top_k": (1.0 if int(min(max(k, 0), len(ranked))) > 0 else None),
                    "reliability": "low",
                }
                for k in ks
            },
        }

    missing_pct = float((country_lists.apply(len) == 0).sum() / total) if total else None
    overall_available_count = int((country_lists.apply(len) > 0).sum())
    overall_minimum = _minimum_reliable_count(total, total)
    overall_counter = Counter()
    for values in country_lists:
        for country in values:
            overall_counter[country] += 1
    overall_distribution = _normalized_distribution(overall_counter)

    per_k: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        top_lists = _list_feature_series(ranked.head(k), "affiliation_countries")
        effective_k = int(min(max(k, 0), len(ranked)))
        minimum_required = _minimum_reliable_count(k, effective_k)
        if top_lists is None:
            per_k[str(k)] = {
                "available": False,
                "effective_k": effective_k,
                "available_count": 0,
                "minimum_required": minimum_required,
                "js_divergence": None,
                "overrepresentation_ratio": {},
                "missing_pct_top_k": 1.0 if effective_k > 0 else None,
                "reliability": "low",
            }
            continue
        available_count = int((top_lists.apply(len) > 0).sum())
        missing_top = float((top_lists.apply(len) == 0).sum() / effective_k) if effective_k > 0 else None
        top_counter = Counter()
        for values in top_lists:
            for country in values:
                top_counter[country] += 1
        top_distribution = _normalized_distribution(top_counter)
        stable = (
            available_count >= minimum_required
            and overall_available_count >= overall_minimum
            and bool(overall_distribution)
            and bool(top_distribution)
        )
        jsd = _js_divergence(overall_distribution, top_distribution) if stable else None
        per_k[str(k)] = {
            "available": stable,
            "effective_k": effective_k,
            "available_count": available_count,
            "minimum_required": minimum_required,
            "js_divergence": jsd,
            "top_distribution": top_distribution,
            "overrepresentation_ratio": (
                _overrepresentation_ratio(top_distribution, overall_distribution, top_n=10)
                if stable
                else {}
            ),
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


def _citation_top_k_bias(
    frame: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    ks: tuple[int, ...],
) -> Dict[str, Any]:
    citations = _numeric_series(_citation_series(frame))
    total = len(frame)
    if total == 0:
        return {
            "available": False,
            "missing_pct": None,
            "spearman_rank_vs_citations": None,
            "overall_median": None,
            "reliability": "low",
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
    if not ranked.empty and ranked_citations is not None:
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
        if available_count >= minimum_required and int(len(rest_vals)) >= minimum_required:
            delta = float(top_vals.median() - rest_vals.median())
        per_k[str(k)] = {
            "available": available_count >= minimum_required and overall_median is not None,
            "effective_k": effective_k,
            "available_count": available_count,
            "minimum_required": minimum_required,
            "median_top_k": (float(top_vals.median()) if not top_vals.empty else None),
            "median_rest": (float(rest_vals.median()) if not rest_vals.empty else None),
            "delta_median_top_vs_rest": delta,
            "missing_pct_top_k": missing_top,
            "reliability": _reliability_label(
                available_count=available_count,
                effective_k=effective_k,
                missing_pct=missing_top,
            ),
        }

    quality = frame.get("metrics_quality")
    quality_counts: dict[str, int] = {}
    if isinstance(quality, pd.Series):
        counter = Counter(
            str((item or {}).get("citations") if isinstance(item, dict) else "missing")
            for item in quality.tolist()
        )
        quality_counts = {str(key): int(value) for key, value in counter.items()}

    return {
        "available": not available.empty,
        "missing_pct": missing_pct,
        "spearman_rank_vs_citations": spearman,
        "overall_median": overall_median,
        "reliability": _reliability_label(
            available_count=int(len(available)),
            effective_k=total,
            missing_pct=missing_pct,
        ),
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


def _coerce_list_of_strings(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text or ";" in text:
            items = [part.strip() for part in text.replace(";", ",").split(",")]
        else:
            items = [text]
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
