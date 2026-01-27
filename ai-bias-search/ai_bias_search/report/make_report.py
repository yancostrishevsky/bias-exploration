"""HTML report generation for evaluation results."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai_bias_search.evaluation.biases import core_ranking_table
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
        "platforms": (
            data.get("platform").dropna().unique().tolist() if "platform" in data.columns else []
        ),
    }

    plots = _generate_plots(data, latest_metrics)
    jcr_summary = _jcr_summary(data)

    preview = _prepare_sample_table(data.drop(columns=["extra"], errors="ignore")).head(50)
    core_ranking_rows = core_ranking_table(data)
    rendered = template.render(
        summary=summary,
        latest_metrics=latest_metrics,
        metrics_timestamp=metrics_timestamp,
        table=preview.to_dict(orient="records"),
        core_ranking_table=core_ranking_rows,
        jcr_summary=jcr_summary,
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
    latest_path = _select_latest_metrics_file(candidates)
    try:
        metrics = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse metrics file %s: %s", latest_path, exc)
        return {}, latest_path.stem
    return metrics, latest_path.stem


def _select_latest_metrics_file(candidates: List[Path]) -> Path:
    parsed: list[tuple[datetime, Path]] = []
    for path in candidates:
        timestamp = _parse_metrics_timestamp(path.stem)
        if timestamp is not None:
            parsed.append((timestamp, path))
    if parsed:
        return max(parsed, key=lambda item: item[0])[1]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_metrics_timestamp(stem: str) -> datetime | None:
    try:
        return datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _generate_plots(frame: pd.DataFrame, metrics: Dict[str, dict]) -> List[Dict[str, str]]:
    """Render available plots to base64-encoded PNG strings for embedding."""

    plot_specs = [
        ("platform_mix", "Records per platform", lambda path: plot_platform_counts(frame, path)),
        (
            "year_distribution",
            "Publication year distribution",
            lambda path: plot_year_distribution(frame, path),
        ),
        (
            "year_distribution_platform",
            "Publication year distribution by platform",
            lambda path: plot_year_distribution_by_platform(frame, path),
        ),
        (
            "recency_by_platform",
            "Publication year by platform",
            lambda path: plot_recency_by_platform(frame, path),
        ),
        (
            "language_distribution",
            "Language distribution (top)",
            lambda path: plot_language_distribution(frame, path),
        ),
        (
            "open_access_share",
            "Open access share by platform",
            lambda path: plot_open_access_share(frame, path),
        ),
        ("publisher_top", "Top publishers (count)", lambda path: plot_publisher_hhi(frame, path)),
        (
            "publisher_hhi_platform",
            "Publisher concentration (HHI) by platform",
            lambda path: plot_publisher_hhi_per_platform(frame, path),
        ),
        (
            "rank_vs_citations",
            "Rank vs cited_by_count",
            lambda path: plot_rank_vs_citations(frame, path),
        ),
        (
            "pairwise_overlap",
            "Platform overlap (Jaccard)",
            lambda path: plot_pairwise_jaccard(metrics, path),
        ),
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


def _prepare_sample_table(frame: pd.DataFrame) -> pd.DataFrame:
    preview = frame.copy()
    impact_values = _column_or_default(preview, "impact_factor", _format_number)
    quartile_values = _column_or_default(preview, "jcr_quartile", _format_text)
    jci_values = _column_or_default(preview, "jcr_jci", _format_number)
    jif_5y_values = _column_or_default(preview, "jcr_jif_5y", _format_number)
    publisher_values = _column_or_default(preview, "jcr_publisher", _format_text)
    preview = preview.drop(
        columns=["impact_factor", "jcr_quartile", "jcr_jci", "jcr_jif_5y", "jcr_publisher"],
        errors="ignore",
    )
    preview["Impact Factor"] = impact_values
    preview["JIF Quartile"] = quartile_values
    preview["JCI"] = jci_values
    preview["5-Year JIF"] = jif_5y_values
    preview["JCR Publisher"] = publisher_values
    return preview


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"
        return f"{float(str(value)):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _column_or_default(
    frame: pd.DataFrame,
    column: str,
    formatter: Callable[[object], str],
) -> pd.Series:
    if column in frame.columns:
        return frame[column].apply(formatter)
    return pd.Series(["-"] * len(frame), index=frame.index)


def _jcr_summary(frame: pd.DataFrame) -> Optional[List[Dict[str, object]]]:
    if "impact_factor" not in frame.columns:
        LOGGER.warning("No JCR data to plot")
        return None
    values = pd.to_numeric(frame["impact_factor"], errors="coerce")
    if values.notna().sum() == 0:
        LOGGER.warning("No JCR data to plot")
        return None
    platforms = frame["platform"].dropna().unique().tolist() if "platform" in frame.columns else []
    summary_rows: List[Dict[str, object]] = []
    for platform in platforms:
        subset = frame[frame["platform"] == platform]
        total = len(subset)
        if total == 0:
            continue
        impact_values = pd.to_numeric(subset["impact_factor"], errors="coerce").dropna()
        eligible = int(impact_values.count())
        missing = total - eligible

        jci_median = None
        if "jcr_jci" in subset.columns:
            jci_values = pd.to_numeric(subset["jcr_jci"], errors="coerce").dropna()
            if not jci_values.empty:
                jci_median = float(jci_values.median())

        q1_share = None
        if "jcr_quartile" in subset.columns:
            quartiles = subset["jcr_quartile"].dropna().astype(str).str.strip().str.upper()
            q1_count = int((quartiles == "Q1").sum())
            q1_share = (q1_count / total) if total else None

        top_publishers = None
        if "jcr_publisher" in subset.columns:
            publishers = subset["jcr_publisher"].dropna().astype(str).str.strip()
            publishers = publishers[publishers != ""]
            if not publishers.empty:
                counts = publishers.value_counts().head(3)
                top_publishers = ", ".join(f"{name} ({count})" for name, count in counts.items())

        summary_rows.append(
            {
                "platform": platform,
                "eligible_count": eligible,
                "mean": float(impact_values.mean()) if eligible else None,
                "median": float(impact_values.median()) if eligible else None,
                "median_jci": jci_median,
                "q1_share": q1_share,
                "top_publishers": top_publishers,
                "missing_count": missing,
                "missing_share": (missing / total) if total else None,
            }
        )
    return summary_rows


def _encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
