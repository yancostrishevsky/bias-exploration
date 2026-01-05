"""Bias-oriented metrics for retrieved records."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_bias_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute a suite of bias metrics from an enriched dataframe."""

    overall = _compute_bias_set(frame)
    by_platform: Dict[str, Dict[str, Any]] = {}
    if "platform" in frame.columns:
        for platform, subset in frame.groupby("platform"):
            by_platform[str(platform)] = _compute_bias_set(subset)

    overall["by_platform"] = by_platform
    overall["open_access_by_platform"] = _open_access_by_platform(frame)
    return overall


def _compute_bias_set(frame: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["recency"] = _recency_metrics(frame)
    metrics["completeness"] = _metadata_completeness(frame)
    metrics["language"] = _language_bias(frame)
    metrics["open_access"] = _open_access_bias(frame)
    metrics["publisher_hhi"] = _publisher_hhi(frame)
    metrics["rank_vs_citations"] = _rank_correlation(frame)
    return metrics


def _open_access_by_platform(frame: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    if "platform" not in frame.columns:
        return {}
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for platform, subset in frame.groupby("platform"):
        result[str(platform)] = _open_access_bias(subset)
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
    return {
        "doi": _coverage_ratio(frame.get("doi"), total),
        "year_coverage": _coverage_ratio(_select_year_column(frame), total),
        "language": _coverage_ratio(frame.get("language"), total),
        "publisher": _coverage_ratio(frame.get("publisher"), total),
        "is_oa": _coverage_ratio(frame.get("is_oa"), total),
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
    is_oa = frame.get("is_oa")
    if is_oa is None or is_oa.dropna().empty:
        return {"share_open_access": None}
    numeric = is_oa.dropna().astype(int)
    return {"share_open_access": float(numeric.mean())}


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
    if {"rank", "cited_by_count"} - set(frame.columns):
        return {"spearman": None}
    subset = frame[["rank", "cited_by_count"]].dropna()
    if subset.empty:
        return {"spearman": None}
    rho, _ = spearmanr(subset["rank"], subset["cited_by_count"])
    if np.isnan(rho):
        return {"spearman": None}
    return {"spearman": float(rho)}
