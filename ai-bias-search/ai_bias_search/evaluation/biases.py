"""Bias-oriented metrics for retrieved records."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_bias_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute a suite of bias metrics from an enriched dataframe."""

    metrics: Dict[str, Any] = {}
    metrics["recency"] = _recency_metrics(frame)
    metrics["language"] = _language_bias(frame)
    metrics["open_access"] = _open_access_bias(frame)
    metrics["publisher_hhi"] = _publisher_hhi(frame)
    metrics["rank_vs_citations"] = _rank_correlation(frame)
    return metrics


def _recency_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    years = frame.get("publication_year")
    if years is None:
        years = frame.get("year")
    if years is None:
        return {"median_year": None, "share_last_12_months": None}
    years = years.dropna().astype(int)
    if years.empty:
        return {"median_year": None, "share_last_12_months": None}
    current_year = datetime.utcnow().year
    share = float((years >= current_year - 1).mean())
    return {"median_year": float(years.median()), "share_last_12_months": share}


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
