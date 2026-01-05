"""HTML report generation for evaluation results."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai_bias_search.utils.io import read_parquet
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.viz.plots import (
    plot_language_distribution,
    plot_open_access_share,
    plot_pairwise_jaccard,
    plot_platform_counts,
    plot_publisher_hhi,
    plot_publisher_hhi_per_platform,
    plot_rank_vs_citations,
    plot_recency_by_platform,
    plot_year_distribution,
    plot_year_distribution_by_platform,
)


LOGGER = configure_logging()


def generate_report(enriched_path: Path, metrics_dir: Path, output_path: Path) -> Path:
    """Render an HTML report combining enriched data and metrics."""

    LOGGER.info("Generating report from %s", enriched_path)
    data = read_parquet(enriched_path)
    latest_metrics, metrics_timestamp = _load_latest_metrics(metrics_dir)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")

    summary = {
        "total_records": len(data),
        "platforms": data.get("platform").dropna().unique().tolist() if "platform" in data.columns else [],
    }

    plots = _generate_plots(data, latest_metrics)

    preview = data.drop(columns=["extra"], errors="ignore")
    rendered = template.render(
        summary=summary,
        latest_metrics=latest_metrics,
        metrics_timestamp=metrics_timestamp,
        table=preview.to_dict(orient="records"),
        plots=plots,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)
    return output_path


def _load_latest_metrics(metrics_dir: Path) -> tuple[Dict[str, dict], str | None]:
    if not metrics_dir.exists():
        LOGGER.warning("Metrics directory %s missing", metrics_dir)
        return {}, None
    candidates = sorted(metrics_dir.glob("*.json"))
    if not candidates:
        LOGGER.warning("No metrics files found in %s", metrics_dir)
        return {}, None
    latest_path = candidates[-1]
    try:
        metrics = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse metrics file %s: %s", latest_path, exc)
        return {}, latest_path.stem
    return metrics, latest_path.stem


def _generate_plots(frame: pd.DataFrame, metrics: Dict[str, dict]) -> List[Dict[str, str]]:
    """Render available plots to base64-encoded PNG strings for embedding."""

    plot_specs = [
        ("platform_mix", "Records per platform", lambda path: plot_platform_counts(frame, path)),
        ("year_distribution", "Publication year distribution", lambda path: plot_year_distribution(frame, path)),
        (
            "year_distribution_platform",
            "Publication year distribution by platform",
            lambda path: plot_year_distribution_by_platform(frame, path),
        ),
        ("recency_by_platform", "Publication year by platform", lambda path: plot_recency_by_platform(frame, path)),
        ("language_distribution", "Language distribution (top)", lambda path: plot_language_distribution(frame, path)),
        ("open_access_share", "Open access share by platform", lambda path: plot_open_access_share(frame, path)),
        ("publisher_top", "Top publishers (count)", lambda path: plot_publisher_hhi(frame, path)),
        ("publisher_hhi_platform", "Publisher concentration (HHI) by platform", lambda path: plot_publisher_hhi_per_platform(frame, path)),
        ("rank_vs_citations", "Rank vs cited_by_count", lambda path: plot_rank_vs_citations(frame, path)),
        ("pairwise_overlap", "Platform overlap (Jaccard)", lambda path: plot_pairwise_jaccard(metrics, path)),
    ]

    rendered: List[Dict[str, str]] = []
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for slug, title, func in plot_specs:
            output_path = tmpdir_path / f"{slug}.png"
            try:
                func(output_path)
            except Exception as exc:
                LOGGER.warning("Failed to render %s plot: %s", slug, exc)
                continue
            if not output_path.exists():
                continue
            rendered.append({"slug": slug, "title": title, "data_url": _encode_png(output_path)})
    return rendered


def _encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
