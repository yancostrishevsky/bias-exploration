"""Matplotlib-based plotting utilities."""

from __future__ import annotations

from pathlib import Path

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


def plot_rank_vs_citations(frame: pd.DataFrame, output: Path) -> None:
    if {"rank", "cited_by_count"} - set(frame.columns):
        LOGGER.warning("rank/cited_by_count missing; skipping scatter plot")
        return
    data = frame[["rank", "cited_by_count"]].dropna()
    if data.empty:
        LOGGER.warning("No citation data to plot")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(data["rank"], data["cited_by_count"], alpha=0.6)
    plt.title("Rank vs cited_by_count")
    plt.xlabel("Rank (lower is better)")
    plt.ylabel("Citations")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
