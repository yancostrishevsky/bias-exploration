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
    plt.boxplot(list(grouped.values()), tick_labels=list(grouped.keys()), vert=True)
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


def plot_oa_share_top_k_by_platform(
    frame: pd.DataFrame,
    output: Path,
    *,
    top_k: int = 10,
) -> None:
    if {"platform", "rank"} - set(frame.columns):
        LOGGER.warning("platform/rank missing; skipping OA top-k plot")
        return
    oa_col = "is_oa" if "is_oa" in frame.columns else ("is_open_access" if "is_open_access" in frame.columns else None)
    if oa_col is None:
        LOGGER.warning("is_oa/is_open_access missing; skipping OA top-k plot")
        return

    subset = frame[["platform", "rank", oa_col]].copy()
    subset["_rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    subset[oa_col] = subset[oa_col].map(
        lambda value: value
        if isinstance(value, bool)
        else (bool(int(value)) if str(value).strip() in {"0", "1"} else None)
    )
    subset = subset.dropna(subset=["platform", "_rank", oa_col])
    if subset.empty:
        LOGGER.warning("No OA data for top-k plot")
        return

    shares: dict[str, float] = {}
    for platform, platform_rows in subset.groupby("platform"):
        sorted_rows = platform_rows.sort_values(by="_rank", kind="mergesort").head(top_k)
        if sorted_rows.empty:
            continue
        shares[str(platform)] = float(sorted_rows[oa_col].astype(int).mean())
    if not shares:
        LOGGER.warning("No OA top-k shares to plot")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(shares) * 1.4), 4))
    pd.Series(shares).sort_index().plot(kind="bar")
    plt.title(f"Open access share in Top-{top_k}")
    plt.xlabel("Platform")
    plt.ylabel("Share OA")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_country_top_overall_vs_top_k(
    frame: pd.DataFrame,
    output: Path,
    *,
    top_k: int = 10,
    top_n_countries: int = 10,
    platform: str | None = None,
) -> None:
    if "affiliation_countries" not in frame.columns:
        LOGGER.warning("affiliation_countries missing; skipping country top-k plot")
        return
    if "rank" not in frame.columns:
        LOGGER.warning("rank missing; skipping country top-k plot")
        return

    subset = frame.copy()
    if platform is None and "platform" in subset.columns:
        platforms = sorted(str(value) for value in subset["platform"].dropna().unique())
        if platforms:
            platform = platforms[0]
    if platform is not None and "platform" in subset.columns:
        subset = subset[subset["platform"].astype(str) == platform]

    if subset.empty:
        LOGGER.warning("No data for country top-k plot")
        return

    subset["_rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    subset = subset.dropna(subset=["_rank"])
    if subset.empty:
        LOGGER.warning("No ranked data for country top-k plot")
        return

    def to_country_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            chunks = text.replace(";", ",").split(",") if "," in text or ";" in text else [text]
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        return []

    overall = Counter()
    for value in subset["affiliation_countries"]:
        for country in to_country_list(value):
            overall[country] += 1
    if not overall:
        LOGGER.warning("No country values for country top-k plot")
        return

    top_subset = subset.sort_values(by="_rank", kind="mergesort").head(top_k)
    top_counter = Counter()
    for value in top_subset["affiliation_countries"]:
        for country in to_country_list(value):
            top_counter[country] += 1
    if not top_counter:
        LOGGER.warning("No country values in top-k for country top-k plot")
        return

    country_order = [name for name, _ in overall.most_common(top_n_countries)]
    overall_total = sum(overall.values())
    top_total = sum(top_counter.values())
    overall_share = [overall.get(name, 0) / overall_total for name in country_order]
    top_share = [top_counter.get(name, 0) / top_total for name in country_order]

    output.parent.mkdir(parents=True, exist_ok=True)
    width = 0.4
    x = range(len(country_order))
    plt.figure(figsize=(max(7, len(country_order) * 0.8), 4))
    plt.bar([idx - width / 2 for idx in x], overall_share, width=width, label="Overall")
    plt.bar([idx + width / 2 for idx in x], top_share, width=width, label=f"Top-{top_k}")
    title_platform = f" ({platform})" if platform else ""
    plt.title(f"Country share: overall vs Top-{top_k}{title_platform}")
    plt.xlabel("Country")
    plt.ylabel("Share")
    plt.xticks(list(x), country_order, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_citedby_top_k_vs_rest(
    frame: pd.DataFrame,
    output: Path,
    *,
    top_k: int = 10,
) -> None:
    if {"platform", "rank", "cited_by_count"} - set(frame.columns):
        LOGGER.warning("platform/rank/cited_by_count missing; skipping citation top-k plot")
        return
    subset = frame[["platform", "rank", "cited_by_count"]].copy()
    subset["_rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    subset["_cited"] = pd.to_numeric(subset["cited_by_count"], errors="coerce")
    subset = subset.dropna(subset=["platform", "_rank", "_cited"])
    if subset.empty:
        LOGGER.warning("No citation data for top-k plot")
        return

    labels: list[str] = []
    values: list[list[float]] = []
    for platform, rows in subset.groupby("platform"):
        ordered = rows.sort_values(by="_rank", kind="mergesort")
        top_vals = ordered.head(top_k)["_cited"].tolist()
        rest_vals = ordered.iloc[top_k:]["_cited"].tolist()
        if top_vals:
            labels.append(f"{platform} top-{top_k}")
            values.append(top_vals)
        if rest_vals:
            labels.append(f"{platform} rest")
            values.append(rest_vals)
    if not values:
        LOGGER.warning("No values for citation top-k plot")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(labels) * 1.0), 4))
    plt.boxplot(values, tick_labels=labels, vert=True)
    plt.title(f"Citations in Top-{top_k} vs rest")
    plt.xlabel("Group")
    plt.ylabel("cited_by_count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_doc_type_top_k_per_platform(
    frame: pd.DataFrame,
    output: Path,
    *,
    top_k: int = 10,
) -> None:
    if {"platform", "rank", "doc_type"} - set(frame.columns):
        LOGGER.warning("platform/rank/doc_type missing; skipping doc-type top-k plot")
        return
    subset = frame[["platform", "rank", "doc_type"]].copy()
    subset["_rank"] = pd.to_numeric(subset["rank"], errors="coerce")
    subset["doc_type"] = subset["doc_type"].astype(str).str.strip()
    subset = subset.dropna(subset=["platform", "_rank"])
    subset = subset[subset["doc_type"] != ""]
    subset = subset[subset["doc_type"].str.lower() != "nan"]
    if subset.empty:
        LOGGER.warning("No doc_type data for top-k plot")
        return

    rows: list[dict[str, object]] = []
    for platform, data in subset.groupby("platform"):
        top = data.sort_values(by="_rank", kind="mergesort").head(top_k)
        for doc_type, count in top["doc_type"].value_counts().items():
            rows.append({"platform": str(platform), "doc_type": str(doc_type), "count": int(count)})
    if not rows:
        LOGGER.warning("No doc_type values in top-k")
        return
    chart = pd.DataFrame(rows)
    pivot = chart.pivot_table(
        index="platform",
        columns="doc_type",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    if pivot.empty:
        LOGGER.warning("No doc_type pivot for top-k plot")
        return

    top_types = pivot.sum(axis=0).sort_values(ascending=False).head(6).index.tolist()
    pivot = pivot.reindex(columns=top_types, fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, len(pivot.index) * 1.4), 4))
    pivot.sort_index().plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title(f"Doc type distribution in Top-{top_k}")
    plt.xlabel("Platform")
    plt.ylabel("Share")
    plt.legend(title="Doc type", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
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
