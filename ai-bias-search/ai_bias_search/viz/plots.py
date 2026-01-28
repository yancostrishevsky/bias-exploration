"""Matplotlib-based plotting utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def plot_year_distribution(frame: pd.DataFrame, output: Path) -> None:
    if "publication_year" not in frame.columns:
        LOGGER.warning("publication_year column missing; skipping year plot")
        return
    data = frame["publication_year"].dropna()
    if data.empty:
        LOGGER.warning("No publication_year data to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    data.astype(int).plot(kind="hist", bins=min(20, data.nunique()))
    plt.title("Publication year distribution")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_publisher_hhi(frame: pd.DataFrame, output: Path) -> None:
    if "publisher" not in frame.columns:
        LOGGER.warning("publisher column missing; skipping HHI plot")
        return
    counts = frame["publisher"].dropna().value_counts().head(10)
    if counts.empty:
        LOGGER.warning("No publisher data to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Top publishers")
    plt.xlabel("Publisher")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_year_distribution_by_platform(frame: pd.DataFrame, output: Path) -> None:
    if {"platform", "publication_year"} - set(frame.columns):
        LOGGER.warning("platform/publication_year missing; skipping per-platform year distribution")
        return
    subset = frame[["platform", "publication_year"]].dropna()
    if subset.empty:
        LOGGER.warning("No publication_year data to plot per platform")
        return
    subset = subset.assign(publication_year=subset["publication_year"].astype(int))
    counts = (
        subset.groupby(["publication_year", "platform"]).size().unstack(fill_value=0).sort_index()
    )
    if counts.empty:
        LOGGER.warning("No per-platform counts for publication_year")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(counts.columns) * 1.6), 4))
    counts.plot(kind="area", stacked=True, alpha=0.8, ax=plt.gca())
    plt.title("Publication year distribution by platform")
    plt.xlabel("Publication year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_rank_vs_citations(frame: pd.DataFrame, output: Path) -> None:
    if {"rank", "cited_by_count"} - set(frame.columns):
        LOGGER.warning("rank/cited_by_count missing; skipping scatter plot")
        return
    cols = ["rank", "cited_by_count"]
    include_platform = "platform" in frame.columns
    if include_platform:
        cols.append("platform")
    data = frame[cols].dropna()
    if data.empty:
        LOGGER.warning("No citation data to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, 4 + (len(data["platform"].unique()) if include_platform else 0)), 4))
    if include_platform:
        for platform, subset in data.groupby("platform"):
            plt.scatter(subset["rank"], subset["cited_by_count"], alpha=0.6, label=str(platform))
        plt.legend(title="Platform", bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        plt.scatter(data["rank"], data["cited_by_count"], alpha=0.6)
    plt.title("Rank vs cited_by_count")
    plt.xlabel("Rank (lower is better)")
    plt.ylabel("Citations")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_platform_counts(frame: pd.DataFrame, output: Path) -> None:
    if "platform" not in frame.columns:
        LOGGER.warning("platform column missing; skipping platform mix plot")
        return
    counts = frame["platform"].dropna().value_counts()
    if counts.empty:
        LOGGER.warning("No platform data to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Records per platform")
    plt.xlabel("Platform")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_recency_by_platform(frame: pd.DataFrame, output: Path) -> None:
    if {"platform", "publication_year"} - set(frame.columns):
        LOGGER.warning("platform/publication_year missing; skipping recency boxplot")
        return
    subset = frame[["platform", "publication_year"]].dropna()
    if subset.empty:
        LOGGER.warning("No recency data to plot")
        return
    grouped: Dict[str, List[int]] = (
        subset.assign(publication_year=subset["publication_year"].astype(int))
        .groupby("platform")["publication_year"]
        .apply(list)
        .to_dict()
    )
    if not grouped:
        LOGGER.warning("No grouped recency data to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(grouped) * 1.4), 4))
    plt.boxplot(grouped.values(), labels=grouped.keys(), vert=True)
    plt.title("Publication year by platform")
    plt.xlabel("Platform")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_language_distribution(frame: pd.DataFrame, output: Path, top_n: int = 5) -> None:
    if {"platform", "language"} - set(frame.columns):
        LOGGER.warning("platform/language missing; skipping language plot")
        return
    subset = frame[["platform", "language"]].dropna()
    if subset.empty:
        LOGGER.warning("No language data to plot")
        return
    top_langs = subset["language"].str.lower().value_counts().head(top_n).index.tolist()
    filtered = subset[subset["language"].str.lower().isin(top_langs)]
    if filtered.empty:
        LOGGER.warning("No language data after filtering to top languages")
        return
    ctab = pd.crosstab(
        filtered["platform"], filtered["language"].str.lower(), normalize="index"
    ).reindex(columns=top_langs, fill_value=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(ctab) * 1.4), 4))
    ctab.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title("Language distribution (top languages)")
    plt.xlabel("Platform")
    plt.ylabel("Share")
    plt.legend(title="Language", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_open_access_share(frame: pd.DataFrame, output: Path) -> None:
    if {"platform", "is_oa"} - set(frame.columns):
        LOGGER.warning("platform/is_oa missing; skipping OA plot")
        return
    subset = frame[["platform", "is_oa"]].dropna()
    if subset.empty:
        LOGGER.warning("No OA data to plot")
        return
    shares = subset.assign(is_oa=subset["is_oa"].astype(int)).groupby("platform")["is_oa"].mean()
    if shares.empty:
        LOGGER.warning("No OA shares to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    shares.plot(kind="bar")
    plt.title("Open access share by platform")
    plt.xlabel("Platform")
    plt.ylabel("Share OA")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_publisher_hhi_per_platform(frame: pd.DataFrame, output: Path) -> None:
    if {"platform", "publisher"} - set(frame.columns):
        LOGGER.warning("platform/publisher missing; skipping HHI per platform")
        return
    subset = frame[["platform", "publisher"]].dropna()
    if subset.empty:
        LOGGER.warning("No publisher data to plot")
        return
    hhis: Dict[str, float] = {}
    for platform, pubs in subset.groupby("platform")["publisher"]:
        counts = Counter(str(pub).lower() for pub in pubs)
        total = sum(counts.values())
        if not total:
            continue
        hhis[platform] = sum((count / total) ** 2 for count in counts.values())
    if not hhis:
        LOGGER.warning("No HHI values to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    pd.Series(hhis).sort_values(ascending=False).plot(kind="bar")
    plt.title("Publisher concentration (HHI) by platform")
    plt.xlabel("Platform")
    plt.ylabel("HHI (0-1)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_pairwise_jaccard(metrics: Dict[str, Any], output: Path) -> None:
    pairwise = metrics.get("pairwise") if isinstance(metrics, dict) else None
    if not pairwise:
        LOGGER.warning("Pairwise metrics missing; skipping overlap heatmap")
        return
    platform_set: set[str] = set()
    for key in pairwise.keys():
        if "_vs_" in key:
            left, right = key.split("_vs_", 1)
            platform_set.update([left, right])
    if not platform_set:
        LOGGER.warning("No platforms found in pairwise metrics; skipping heatmap")
        return
    platforms = sorted(platform_set)
    index = {name: idx for idx, name in enumerate(platforms)}
    matrix = [
        [1.0 if i == j else None for j in range(len(platforms))] for i in range(len(platforms))
    ]
    for key, vals in pairwise.items():
        if "_vs_" not in key:
            continue
        left, right = key.split("_vs_", 1)
        if left not in index or right not in index:
            continue
        score = vals.get("jaccard")
        if score is None:
            continue
        i, j = index[left], index[right]
        matrix[i][j] = matrix[j][i] = float(score)
    # Replace Nones with 0 for plotting
    filled = pd.DataFrame(matrix, index=platforms, columns=platforms).fillna(0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(filled, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Jaccard overlap")
    plt.xticks(range(len(platforms)), platforms, rotation=45, ha="right")
    plt.yticks(range(len(platforms)), platforms)
    # annotate
    for i in range(len(platforms)):
        for j in range(len(platforms)):
            plt.text(
                j,
                i,
                f"{filled.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    plt.title("Platform overlap (Jaccard)")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
