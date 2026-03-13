"""HTML report generation for evaluation results."""

from __future__ import annotations

import ast
import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai_bias_search.evaluation.biases import core_ranking_table
from ai_bias_search.normalize.records import normalize_records
from ai_bias_search.utils.io import read_parquet
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def _json_compatible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_compatible(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def generate_report(enriched_path: Path, metrics_dir: Path, output_path: Path) -> Path:
    """Render an HTML report combining enriched data and metrics."""

    LOGGER.info("Generating report from %s", enriched_path)
    data = read_parquet(enriched_path)
    latest_metrics, metrics_timestamp = _load_latest_metrics(metrics_dir)
    diagnostics = _load_diagnostics(metrics_dir, latest_metrics)
    metrics_path = (metrics_dir / f"{metrics_timestamp}.json") if metrics_timestamp else None
    context = build_report_context(
        frame=data,
        latest_metrics=latest_metrics,
        metrics_timestamp=metrics_timestamp,
        diagnostics=diagnostics,
        output_path=output_path,
        metrics_path=metrics_path if metrics_path and metrics_path.exists() else None,
        enriched_download_path=enriched_path,
    )
    return render_report_context(output_path=output_path, context=context)


def render_report_context(*, output_path: Path, context: Mapping[str, Any]) -> Path:
    """Render the shared report template with a prepared context."""

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")
    rendered = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)
    return output_path


def build_report_context(
    *,
    frame: pd.DataFrame,
    latest_metrics: Mapping[str, Any],
    metrics_timestamp: str | None,
    diagnostics: Mapping[str, Any],
    output_path: Path,
    metrics_path: Path | None,
    enriched_download_path: Path | None,
    report_title: str = "AI Bias exploration Report",
    entity_label_singular: str = "Platform",
    entity_label_plural: str = "Platforms",
    enriched_download_label: str = "Enriched Parquet",
    summary_override: Mapping[str, Any] | None = None,
    llm_specific: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the context consumed by the shared HTML report template."""

    summary = dict(summary_override or {})
    if "total_records" not in summary:
        summary["total_records"] = len(frame)
    if "platforms" not in summary:
        summary["platforms"] = (
            sorted(str(value) for value in frame.get("platform").dropna().unique().tolist())
            if "platform" in frame.columns
            else []
        )
    summary.setdefault("entity_label_singular", entity_label_singular)
    summary.setdefault("entity_label_plural", entity_label_plural)

    interactive_context = _build_interactive_context(
        frame=frame,
        metrics=latest_metrics,
        diagnostics=diagnostics,
        metrics_path=metrics_path,
        enriched_download_path=enriched_download_path,
        output_path=output_path,
    )
    jcr_summary = _jcr_summary(frame)
    jif_context = _generate_jif_context(frame)
    rankings_coverage_table = _rankings_coverage_table(frame)
    core_match_source_breakdown = _core_match_source_rows(frame)
    scopus_summary = _scopus_summary(frame)
    bias_features_availability = _bias_features_availability_rows(frame)
    citations_quality_rows = _citations_quality_rows(frame)
    top_k_bias_summary = _top_k_bias_summary_rows(dict(latest_metrics))

    preview = _prepare_sample_table(frame.drop(columns=["extra"], errors="ignore")).head(50)
    core_ranking_rows = core_ranking_table(frame)
    return {
        "report_title": report_title,
        "entity_label_singular": entity_label_singular,
        "entity_label_plural": entity_label_plural,
        "enriched_download_label": enriched_download_label,
        "summary": summary,
        "latest_metrics": latest_metrics,
        "metrics_timestamp": metrics_timestamp,
        "table": preview.to_dict(orient="records"),
        "core_ranking_table": core_ranking_rows,
        "rankings_coverage_table": rankings_coverage_table,
        "core_match_source_breakdown": core_match_source_breakdown,
        "scopus_summary": scopus_summary,
        "bias_features_availability": bias_features_availability,
        "citations_quality_rows": citations_quality_rows,
        "diagnostics": diagnostics,
        "top_k_bias_summary": top_k_bias_summary,
        "jcr_summary": jcr_summary,
        "interactive": interactive_context,
        "llm_specific": llm_specific,
        **jif_context,
    }


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


def _load_diagnostics(metrics_dir: Path, latest_metrics: Dict[str, dict]) -> Dict[str, Any]:
    if not isinstance(latest_metrics, dict):
        latest_metrics = {}
    diagnostics_name = latest_metrics.get("diagnostics_path")
    candidates: list[Path] = []
    if isinstance(diagnostics_name, str) and diagnostics_name.strip():
        candidates.append(metrics_dir.parent / diagnostics_name.strip())
    candidates.append(metrics_dir.parent / "diagnostics.json")

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse diagnostics file %s: %s", path, exc)
            continue
        if isinstance(payload, dict):
            return payload
    return {}


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

    # Import lazily to avoid importing matplotlib during `collect` / `enrich`.
    from ai_bias_search.viz.plots import (
        plot_citedby_top_k_vs_rest,
        plot_country_top_overall_vs_top_k,
        plot_doc_type_top_k_per_platform,
        plot_language_distribution,
        plot_oa_share_top_k_by_platform,
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
        (
            "open_access_top_k",
            "Open access share in Top-10 by platform",
            lambda path: plot_oa_share_top_k_by_platform(frame, path, top_k=10),
        ),
        (
            "country_overall_vs_top_k",
            "Country share: overall vs Top-10",
            lambda path: plot_country_top_overall_vs_top_k(frame, path, top_k=10),
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
            "citations_top_k_vs_rest",
            "Citations in Top-10 vs rest",
            lambda path: plot_citedby_top_k_vs_rest(frame, path, top_k=10),
        ),
        (
            "doc_type_top_k",
            "Doc type distribution in Top-10",
            lambda path: plot_doc_type_top_k_per_platform(frame, path, top_k=10),
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


def _extract_bias_block(metrics: Mapping[str, Any], platform: str) -> dict[str, Any]:
    biases = metrics.get("biases")
    if not isinstance(biases, dict):
        return {}
    if platform == "all":
        return biases
    by_platform = biases.get("by_platform")
    if not isinstance(by_platform, dict):
        return {}
    candidate = by_platform.get(platform)
    return candidate if isinstance(candidate, dict) else {}


def _extract_top_ks(metrics: Mapping[str, Any]) -> list[int]:
    candidate = _extract_bias_block(metrics, "all")
    top_k = candidate.get("top_k_bias") if isinstance(candidate.get("top_k_bias"), dict) else {}
    ks = top_k.get("ks") if isinstance(top_k, dict) else None
    values: list[int] = []
    if isinstance(ks, list):
        for item in ks:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                continue
    if not values:
        values = [10, 20, 50]
    return sorted({value for value in values if value > 0})


def _ranked_subset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "rank" not in frame.columns:
        return pd.DataFrame()
    ranked = frame.copy()
    ranked["_rank_numeric"] = pd.to_numeric(ranked["rank"], errors="coerce")
    ranked = ranked.dropna(subset=["_rank_numeric"]).sort_values("_rank_numeric", kind="mergesort")
    return ranked


def _to_clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _year_distribution(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or "year" not in frame.columns:
        return []
    years = pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int)
    if years.empty:
        return []
    counts = years.value_counts().sort_index()
    return [{"year": int(year), "count": int(count)} for year, count in counts.items()]


def _rank_vs_citations_points(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ranked = _ranked_subset(frame)
    if ranked.empty:
        return []
    citation_column = "citations" if "citations" in ranked.columns else "cited_by_count"
    if citation_column not in ranked.columns:
        return []
    citations = pd.to_numeric(ranked[citation_column], errors="coerce")
    if citations.isna().all() and citation_column != "cited_by_count" and "cited_by_count" in ranked.columns:
        citations = pd.to_numeric(ranked["cited_by_count"], errors="coerce")
    ranked = ranked.assign(_citations_numeric=citations).dropna(
        subset=["_rank_numeric", "_citations_numeric"]
    )
    if ranked.empty:
        return []
    points: list[dict[str, Any]] = []
    for row in ranked.to_dict(orient="records"):
        points.append(
            {
                "rank": float(row.get("_rank_numeric")),
                "citations": float(row.get("_citations_numeric")),
                "title": _to_clean_text(row.get("title")),
                "doi": _to_clean_text(row.get("doi")),
                "year": row.get("year"),
            }
        )
    return points


def _oa_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    if "is_oa" in frame.columns and "is_open_access" in frame.columns:
        return frame["is_oa"].combine_first(frame["is_open_access"])
    if "is_oa" in frame.columns:
        return frame["is_oa"]
    if "is_open_access" in frame.columns:
        return frame["is_open_access"]
    return None


def _oa_share(series: Optional[pd.Series]) -> float | None:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    numeric = values.astype(int)
    return float(numeric.mean())


def _doc_type_distribution(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "doc_type" not in frame.columns:
        return {}
    series = frame["doc_type"].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return {}
    counts = series.value_counts(normalize=True).sort_values(ascending=False)
    top = counts.head(8)
    return {str(label): float(value) for label, value in top.items()}


def _per_platform_hhi(metrics: Mapping[str, Any], platforms: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for platform in platforms:
        bias = _extract_bias_block(metrics, platform)
        hhi_block = bias.get("publisher_hhi") if isinstance(bias.get("publisher_hhi"), dict) else {}
        out.append(
            {
                "platform": platform,
                "hhi": hhi_block.get("hhi"),
                "available": bool(hhi_block.get("available")),
                "reason": hhi_block.get("reason"),
            }
        )
    return out


def _series_value(per_k: Mapping[str, Any], k: int, key: str) -> Any:
    item = per_k.get(str(k))
    if not isinstance(item, dict):
        return None
    return item.get(key)


def _build_platform_chart_payload(
    frame: pd.DataFrame,
    *,
    platform: str,
    ks: Sequence[int],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    subset = frame if platform == "all" else frame[frame["platform"].astype(str) == platform]
    ranked = _ranked_subset(subset)
    top_k_bias = _extract_bias_block(metrics, platform).get("top_k_bias")
    top_k_bias = top_k_bias if isinstance(top_k_bias, dict) else {}
    oa_bias = top_k_bias.get("oa") if isinstance(top_k_bias.get("oa"), dict) else {}
    oa_per_k = oa_bias.get("per_k") if isinstance(oa_bias.get("per_k"), dict) else {}
    doc_bias = top_k_bias.get("doc_type") if isinstance(top_k_bias.get("doc_type"), dict) else {}
    doc_per_k = doc_bias.get("per_k") if isinstance(doc_bias.get("per_k"), dict) else {}
    country_bias = top_k_bias.get("country") if isinstance(top_k_bias.get("country"), dict) else {}
    citations_bias = (
        top_k_bias.get("citations") if isinstance(top_k_bias.get("citations"), dict) else {}
    )
    rank_vs_citations = _extract_bias_block(metrics, platform).get("rank_vs_citations")
    rank_vs_citations = rank_vs_citations if isinstance(rank_vs_citations, dict) else {}

    oa = _oa_series(subset)
    oa_overall = _oa_share(oa)
    oa_top_k: dict[str, dict[str, Any]] = {}
    doc_top_k: dict[str, dict[str, Any]] = {}
    for k in ks:
        top = ranked.head(k)
        oa_top_k[str(k)] = {
            "top_k_share": _oa_share(_oa_series(top)),
            "overall_share": oa_overall,
            "delta_share": (
                (_oa_share(_oa_series(top)) - oa_overall)
                if (_oa_share(_oa_series(top)) is not None and oa_overall is not None)
                else None
            ),
            "reliability": _series_value(oa_per_k, k, "reliability"),
            "missing_pct": _series_value(oa_per_k, k, "missing_pct"),
        }
        doc_top_k[str(k)] = {
            "top_distribution": _doc_type_distribution(top),
            "overall_distribution": _doc_type_distribution(subset),
            "reliability": _series_value(doc_per_k, k, "reliability"),
            "missing_pct_top_k": _series_value(doc_per_k, k, "missing_pct_top_k"),
            "js_divergence": _series_value(doc_per_k, k, "js_divergence"),
        }

    return {
        "record_count": int(len(subset)),
        "year_distribution": _year_distribution(subset),
        "rank_vs_citations": {
            "points": _rank_vs_citations_points(subset),
            "reason": rank_vs_citations.get("reason"),
            "note": rank_vs_citations.get("note"),
            "spearman": rank_vs_citations.get("spearman"),
            "reliability": citations_bias.get("reliability"),
        },
        "oa_top_k": oa_top_k,
        "doc_type_top_k": doc_top_k,
        "geo_bias": {
            "available_share": country_bias.get("available_share"),
            "multi_country_share": country_bias.get("multi_country_share"),
            "multi_share": country_bias.get("multi_share"),
            "enabled_for_bias_metrics": country_bias.get("enabled_for_bias_metrics"),
            "reason": country_bias.get("reason"),
            "min_coverage_threshold": country_bias.get("min_coverage_threshold"),
            "overall_distribution_variants": (
                country_bias.get("overall_distribution_variants")
                if isinstance(country_bias.get("overall_distribution_variants"), dict)
                else {}
            ),
            "per_k": (
                country_bias.get("per_k")
                if isinstance(country_bias.get("per_k"), dict)
                else {}
            ),
        },
        "citations": {
            "reason": citations_bias.get("reason"),
            "note": citations_bias.get("note"),
            "reliability": citations_bias.get("reliability"),
        },
    }


def _build_chart_rows(
    platform_payload: Mapping[str, Mapping[str, Any]],
    *,
    ks: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for platform, payload in platform_payload.items():
        year_dist = payload.get("year_distribution")
        if isinstance(year_dist, list):
            for item in year_dist:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "chart": "year_distribution",
                        "platform": platform,
                        "k": None,
                        "label": item.get("year"),
                        "value": item.get("count"),
                    }
                )
        oa_top_k = payload.get("oa_top_k")
        if isinstance(oa_top_k, dict):
            for k in ks:
                item = oa_top_k.get(str(k))
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "chart": "oa_top_k",
                        "platform": platform,
                        "k": int(k),
                        "label": "top_k_share",
                        "value": item.get("top_k_share"),
                    }
                )
                rows.append(
                    {
                        "chart": "oa_top_k",
                        "platform": platform,
                        "k": int(k),
                        "label": "overall_share",
                        "value": item.get("overall_share"),
                    }
                )
    return rows


def _field_present(row: Mapping[str, Any], field: str) -> bool:
    if field == "doi":
        return _to_clean_text(row.get("doi")) is not None
    if field == "year":
        year = pd.to_numeric(pd.Series([row.get("year")]), errors="coerce").iloc[0]
        return bool(pd.notna(year))
    if field == "journal_title":
        return _to_clean_text(row.get("journal_title") or row.get("source")) is not None
    if field == "issn":
        values = row.get("issn_list")
        if isinstance(values, list) and values:
            return True
        return _to_clean_text(row.get("issn")) is not None or _to_clean_text(row.get("eissn")) is not None
    if field == "citations":
        value = row.get("citations")
        if value is None:
            value = row.get("cited_by_count")
        return bool(pd.notna(value))
    if field == "publisher":
        return _to_clean_text(row.get("publisher")) is not None
    if field == "oa":
        return row.get("is_oa") is not None or row.get("is_open_access") is not None
    return False


def _coverage_blocks(
    raw_frame: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    fields: Sequence[str],
) -> dict[str, Any]:
    def coverage_for_frame(frame: pd.DataFrame) -> dict[str, float | None]:
        total = len(frame)
        out: dict[str, float | None] = {}
        for field in fields:
            if total == 0:
                out[field] = None
                continue
            present = 0
            for row in frame.to_dict(orient="records"):
                if _field_present(row, field):
                    present += 1
            out[field] = float(present / total)
        return out

    overall_before = coverage_for_frame(raw_frame)
    overall_after = coverage_for_frame(normalized)
    per_platform: dict[str, dict[str, dict[str, float | None]]] = {}
    platforms = sorted(
        set(
            str(value)
            for value in raw_frame.get("platform", pd.Series(dtype=str)).dropna().tolist()
        )
        | set(
            str(value)
            for value in normalized.get("platform", pd.Series(dtype=str)).dropna().tolist()
        )
    )
    for platform in platforms:
        raw_subset = (
            raw_frame[raw_frame["platform"].astype(str) == platform]
            if "platform" in raw_frame.columns
            else pd.DataFrame()
        )
        norm_subset = (
            normalized[normalized["platform"].astype(str) == platform]
            if "platform" in normalized.columns
            else pd.DataFrame()
        )
        per_platform[platform] = {
            "before": coverage_for_frame(raw_subset),
            "after": coverage_for_frame(norm_subset),
        }

    return {
        "fields": list(fields),
        "overall": {"before": overall_before, "after": overall_after},
        "per_platform": per_platform,
    }


def _missingness_heatmap(
    normalized: pd.DataFrame,
    *,
    fields: Sequence[str],
    platforms: Sequence[str],
) -> dict[str, Any]:
    z: list[list[float]] = []
    for platform in platforms:
        subset = normalized[normalized["platform"].astype(str) == platform]
        total = len(subset)
        row_values: list[float] = []
        rows = subset.to_dict(orient="records")
        for field in fields:
            if total == 0:
                row_values.append(1.0)
                continue
            present = sum(1 for row in rows if _field_present(row, field))
            row_values.append(float(1.0 - (present / total)))
        z.append(row_values)
    return {"platforms": list(platforms), "fields": list(fields), "z": z}


def _pairwise_overlap_heatmap(metrics: Mapping[str, Any], platforms: Sequence[str]) -> dict[str, Any]:
    matrix: dict[str, dict[str, float]] = {
        platform: {other: (1.0 if platform == other else 0.0) for other in platforms}
        for platform in platforms
    }
    pairwise = metrics.get("pairwise")
    if isinstance(pairwise, dict):
        for key, value in pairwise.items():
            if not isinstance(key, str) or "_vs_" not in key or not isinstance(value, dict):
                continue
            left, right = key.split("_vs_", 1)
            score = value.get("jaccard")
            if isinstance(score, (int, float)):
                matrix.setdefault(left, {})[right] = float(score)
                matrix.setdefault(right, {})[left] = float(score)
    z = [[float(matrix.get(p, {}).get(o, 0.0)) for o in platforms] for p in platforms]
    return {"platforms": list(platforms), "z": z}


def _recency_overlay(normalized: pd.DataFrame, platforms: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for platform in platforms:
        subset = normalized[normalized["platform"].astype(str) == platform]
        years = pd.to_numeric(subset.get("year"), errors="coerce").dropna().astype(int)
        if years.empty:
            payload[platform] = []
            continue
        counts = years.value_counts().sort_index()
        payload[platform] = [
            {"year": int(year), "count": int(count)}
            for year, count in counts.items()
        ]
    return payload


def _oa_vs_recency(normalized: pd.DataFrame, platforms: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for platform in platforms:
        subset = normalized[normalized["platform"].astype(str) == platform]
        if subset.empty:
            payload[platform] = []
            continue
        years = pd.to_numeric(subset.get("year"), errors="coerce")
        oa = _oa_series(subset)
        if oa is None:
            payload[platform] = []
            continue
        frame = pd.DataFrame({"year": years, "oa": oa}).dropna()
        if frame.empty:
            payload[platform] = []
            continue
        frame["year"] = frame["year"].astype(int)
        grouped = frame.groupby("year", as_index=False)["oa"].agg(["mean", "count"]).reset_index()
        payload[platform] = [
            {"year": int(row["year"]), "oa_share": float(row["mean"]), "count": int(row["count"])}
            for _, row in grouped.iterrows()
        ]
    return payload


def _topk_vs_rest_share(
    frame: pd.DataFrame,
    *,
    field: str,
    positive_values: set[str],
    ks: Sequence[int],
) -> dict[str, dict[str, float | int | None]]:
    ranked = _ranked_subset(frame)
    payload: dict[str, dict[str, float | int | None]] = {}
    for k in ks:
        top = ranked.head(k)
        rest = ranked.iloc[k:]
        top_values = top.get(field)
        rest_values = rest.get(field)
        if not isinstance(top_values, pd.Series) or not isinstance(rest_values, pd.Series):
            payload[str(k)] = {"top_k_share": None, "rest_share": None, "effective_k": int(len(top))}
            continue
        top_clean = top_values.dropna().astype(str).str.strip().str.upper()
        rest_clean = rest_values.dropna().astype(str).str.strip().str.upper()
        top_share = (
            float(top_clean.isin(positive_values).mean())
            if not top_clean.empty
            else None
        )
        rest_share = (
            float(rest_clean.isin(positive_values).mean())
            if not rest_clean.empty
            else None
        )
        payload[str(k)] = {
            "top_k_share": top_share,
            "rest_share": rest_share,
            "effective_k": int(len(top)),
        }
    return payload


def _jif_quartile_topk_vs_rest(
    frame: pd.DataFrame,
    *,
    ks: Sequence[int],
) -> dict[str, dict[str, dict[str, float | int]]]:
    ranked = _ranked_subset(frame)
    payload: dict[str, dict[str, dict[str, float | int]]] = {}
    for k in ks:
        top = ranked.head(k)
        rest = ranked.iloc[k:]
        entry: dict[str, dict[str, float | int]] = {}
        for label, subset in (("top", top), ("rest", rest)):
            quartile = subset.get("jcr_quartile")
            if not isinstance(quartile, pd.Series):
                entry[label] = {}
                continue
            values = quartile.dropna().astype(str).str.strip().str.upper()
            if values.empty:
                entry[label] = {}
                continue
            counts = values.value_counts(normalize=True)
            entry[label] = {str(idx): float(val) for idx, val in counts.items()}
        payload[str(k)] = entry
    return payload


def _venue_diversity(
    normalized: pd.DataFrame,
    platforms: Sequence[str],
    *,
    ks: Sequence[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for platform in platforms:
        subset = normalized[normalized["platform"].astype(str) == platform]
        ranked = _ranked_subset(subset)
        journals = subset.get("journal_title")
        overall_unique = (
            int(journals.dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique())
            if isinstance(journals, pd.Series)
            else 0
        )
        per_k: dict[str, dict[str, int | float]] = {}
        for k in ks:
            top = ranked.head(k)
            top_j = top.get("journal_title")
            unique_top = (
                int(top_j.dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique())
                if isinstance(top_j, pd.Series)
                else 0
            )
            eff_k = int(len(top))
            per_k[str(k)] = {
                "unique_top_k": unique_top,
                "unique_overall": overall_unique,
                "effective_k": eff_k,
                "top_k_diversity": (float(unique_top / eff_k) if eff_k > 0 else 0.0),
            }
        out[platform] = {"overall_unique": overall_unique, "per_k": per_k}
    return out


def _platform_compare(metrics: Mapping[str, Any], platforms: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for platform in platforms:
        bias = _extract_bias_block(metrics, platform)
        recency = bias.get("recency") if isinstance(bias.get("recency"), dict) else {}
        open_access = bias.get("open_access") if isinstance(bias.get("open_access"), dict) else {}
        publisher_hhi = bias.get("publisher_hhi") if isinstance(bias.get("publisher_hhi"), dict) else {}
        rows.append(
            {
                "platform": platform,
                "median_year": recency.get("median_year"),
                "oa_share": open_access.get("share_open_access"),
                "publisher_hhi": publisher_hhi.get("hhi"),
                "publisher_available": publisher_hhi.get("available"),
            }
        )
    return rows


def _to_relative_link(target: Path, *, base: Path) -> str:
    try:
        return os.path.relpath(str(target.resolve()), start=str(base.resolve()))
    except Exception:
        return str(target)


def _load_plotly_js() -> tuple[str | None, bool]:
    try:
        from plotly.offline import get_plotlyjs  # type: ignore[import-not-found]

        return get_plotlyjs(), True
    except Exception as exc:
        LOGGER.warning("Plotly offline bundle unavailable; interactive charts disabled: %s", exc)
        return None, False


def _plotly_cdn_fallback_url() -> str | None:
    disabled = os.getenv("REPORT_PLOTLY_CDN_DISABLED", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None
    url = os.getenv("REPORT_PLOTLY_CDN_URL", "https://cdn.plot.ly/plotly-2.32.0.min.js").strip()
    return url or None


def _build_interactive_context(
    *,
    frame: pd.DataFrame,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    metrics_path: Path | None,
    enriched_download_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    normalized = (
        pd.DataFrame(normalize_records(frame.to_dict(orient="records")))
        if not frame.empty
        else pd.DataFrame()
    )
    platforms = (
        sorted(str(value) for value in normalized.get("platform", pd.Series(dtype=str)).dropna().unique())
        if not normalized.empty and "platform" in normalized.columns
        else []
    )
    ks = _extract_top_ks(metrics)
    platform_payload: dict[str, dict[str, Any]] = {
        "all": _build_platform_chart_payload(
            normalized,
            platform="all",
            ks=ks,
            metrics=metrics,
        )
    }
    for platform in platforms:
        platform_payload[platform] = _build_platform_chart_payload(
            normalized,
            platform=platform,
            ks=ks,
            metrics=metrics,
        )

    hhi_payload = _per_platform_hhi(metrics, platforms)
    field_names = ("doi", "year", "journal_title", "issn", "citations", "publisher", "oa")
    coverage_payload = _coverage_blocks(frame, normalized, fields=field_names)
    missingness_payload = _missingness_heatmap(normalized, fields=field_names, platforms=platforms)
    overlap_payload = _pairwise_overlap_heatmap(metrics, platforms)
    recency_payload = _recency_overlay(normalized, platforms)
    oa_recency_payload = _oa_vs_recency(normalized, platforms)
    core_rank_payload: dict[str, Any] = {}
    jif_quartile_payload: dict[str, Any] = {}
    for platform in platforms:
        subset = frame[frame["platform"].astype(str) == platform] if "platform" in frame.columns else pd.DataFrame()
        core_rank_payload[platform] = _topk_vs_rest_share(
            subset,
            field="core_rank",
            positive_values={"A*", "A"},
            ks=ks,
        )
        jif_quartile_payload[platform] = _jif_quartile_topk_vs_rest(subset, ks=ks)
    venue_diversity_payload = _venue_diversity(normalized, platforms, ks=ks)
    platform_compare_payload = _platform_compare(metrics, platforms)

    chart_rows = _build_chart_rows(platform_payload, ks=ks)
    chart_data_path = output_path.with_name(f"{output_path.stem}_chart_data.csv")
    if chart_rows:
        chart_data_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(chart_rows).to_csv(chart_data_path, index=False)

    warnings = diagnostics.get("warnings") if isinstance(diagnostics.get("warnings"), list) else []
    plotly_js, plotly_available = _load_plotly_js()
    plotly_cdn_url: str | None = None
    if not plotly_available:
        plotly_cdn_url = _plotly_cdn_fallback_url()
        if plotly_cdn_url:
            LOGGER.warning(
                "Falling back to Plotly CDN for interactive charts because local offline bundle is unavailable"
            )

    downloads: dict[str, str | None] = {
        "metrics_json": (
            _to_relative_link(metrics_path, base=output_path.parent)
            if metrics_path is not None and metrics_path.exists()
            else None
        ),
        "enriched_dataset": (
            _to_relative_link(enriched_download_path, base=output_path.parent)
            if enriched_download_path is not None and enriched_download_path.exists()
            else None
        ),
        "enriched_parquet": (
            _to_relative_link(enriched_download_path, base=output_path.parent)
            if enriched_download_path is not None and enriched_download_path.exists()
            else None
        ),
        "chart_data_csv": (
            _to_relative_link(chart_data_path, base=output_path.parent)
            if chart_data_path.exists()
            else None
        ),
    }

    payload = {
        "platforms": ["all", *platforms],
        "top_ks": ks,
        "metric_groups": [
            {"id": "all", "label": "all"},
            {"id": "completeness", "label": "completeness"},
            {"id": "recency", "label": "recency"},
            {"id": "citations", "label": "citations"},
            {"id": "doc_type", "label": "doc_type"},
            {"id": "oa", "label": "oa"},
            {"id": "venue_concentration", "label": "venue_concentration"},
            {"id": "geo", "label": "geo"},
            {"id": "missingness", "label": "missingness"},
            {"id": "enrichment", "label": "enrichment"},
            {"id": "overlap", "label": "overlap"},
        ],
        "compare_modes": ["overlay", "small_multiples"],
        "platform_data": platform_payload,
        "publisher_hhi": hhi_payload,
        "missingness": missingness_payload,
        "enrichment_gain": coverage_payload,
        "overlap": overlap_payload,
        "recency_overlay": recency_payload,
        "oa_vs_recency": oa_recency_payload,
        "core_rank_topk_vs_rest": core_rank_payload,
        "jif_quartile_topk_vs_rest": jif_quartile_payload,
        "venue_diversity": venue_diversity_payload,
        "platform_compare": platform_compare_payload,
        "warnings": [str(item) for item in warnings],
    }
    return {
        "payload": _json_compatible(payload),
        "plotly_js": plotly_js,
        "plotly_available": plotly_available,
        "plotly_cdn_url": plotly_cdn_url,
        "downloads": downloads,
    }


def _generate_jif_context(frame: pd.DataFrame) -> Dict[str, object]:
    """Prepare plot payloads for the JIF/JCR section."""

    try:
        from ai_bias_search.report.jif_plots import build_jif_context
    except Exception as exc:
        LOGGER.warning("JIF/JCR plotting helpers unavailable: %s", exc)
        return {
            "jif_enabled": False,
            "jif_message": "JIF/JCR plots unavailable.",
            "jif_coverage_pct": 0.0,
            "jif_match_plot_png": None,
            "jif_distribution_plot_png": None,
            "jif_quartile_plot_png": None,
            "jif_publisher_plot_png": None,
            "jif_publisher_hhi_text": None,
            "jif_rank_plot_png": None,
            "jif_rank_corr_text": None,
        }

    try:
        return build_jif_context(frame)
    except Exception as exc:
        LOGGER.warning("Failed to build JIF/JCR section: %s", exc)
        return {
            "jif_enabled": False,
            "jif_message": "Failed to render JIF/JCR section.",
            "jif_coverage_pct": 0.0,
            "jif_match_plot_png": None,
            "jif_distribution_plot_png": None,
            "jif_quartile_plot_png": None,
            "jif_publisher_plot_png": None,
            "jif_publisher_hhi_text": None,
            "jif_rank_plot_png": None,
            "jif_rank_corr_text": None,
        }


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
    scopus_metrics = pd.DataFrame(
        [extract_scopus_metrics_for_table(record) for record in preview.to_dict(orient="records")],
        index=preview.index,
    )
    preview = pd.concat([preview, scopus_metrics], axis=1)
    return preview


def extract_scopus_metrics_for_table(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten Scopus ranking metrics into table-friendly display columns."""

    def _coerce_year(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None

    def _metric_text(scopus_payload: dict[str, Any], key: str) -> str:
        metric = scopus_payload.get(key)
        if not isinstance(metric, dict):
            return "-"
        value_text = _format_number(metric.get("value"))
        if value_text == "-":
            return "-"
        year = _coerce_year(metric.get("year"))
        if year is None:
            return value_text
        return f"{value_text} ({year})"

    default = {
        "Scopus CiteScore (year)": "-",
        "Scopus SJR (year)": "-",
        "Scopus SNIP (year)": "-",
    }
    rankings = record.get("rankings")
    if not isinstance(rankings, dict):
        return default
    scopus = rankings.get("scopus")
    if not isinstance(scopus, dict):
        return default
    return {
        "Scopus CiteScore (year)": _metric_text(scopus, "citescore"),
        "Scopus SJR (year)": _metric_text(scopus, "sjr"),
        "Scopus SNIP (year)": _metric_text(scopus, "snip"),
    }


def _rankings_coverage_table(frame: pd.DataFrame) -> Optional[List[Dict[str, object]]]:
    if "rankings" not in frame.columns:
        return None
    total = int(len(frame))
    if total == 0:
        return []

    values = frame["rankings"]
    ranking_ids: set[str] = set()
    for item in values:
        if isinstance(item, dict):
            ranking_ids.update(str(key) for key in item.keys())

    if not ranking_ids:
        return []

    methods = ("issn_exact", "title_exact", "title_fuzzy", "unmatched")
    rows: List[Dict[str, object]] = []
    for ranking_id in sorted(ranking_ids):
        method_counts = {method: 0 for method in methods}
        matched_count = 0

        for item in values:
            result = item.get(ranking_id) if isinstance(item, dict) else None
            value = None
            method = "unmatched"
            if isinstance(result, dict):
                value = result.get("value")
                method = str(result.get("method") or "unmatched")

            if value is None:
                method = "unmatched"
            if method not in method_counts:
                method = "unmatched"

            if value is not None:
                matched_count += 1
            method_counts[method] += 1

        breakdown = ", ".join(
            f"{method}:{method_counts[method]}" for method in methods if method_counts[method] > 0
        )
        matched_pct = float(matched_count / total * 100.0) if total else 0.0
        rows.append(
            {
                "ranking_id": ranking_id,
                "matched_count": matched_count,
                "total": total,
                "matched_pct": matched_pct,
                "method_breakdown": breakdown,
            }
        )

    return rows


def _scopus_summary(frame: pd.DataFrame) -> Dict[str, object]:
    total = int(len(frame))
    if total == 0 or "scopus" not in frame.columns:
        return {
            "total": total,
            "enriched_count": 0,
            "enriched_pct": 0.0,
            "with_citedby_count": 0,
            "with_serial_metrics": 0,
            "with_citation_overview": 0,
        }

    enriched_count = 0
    with_citedby = 0
    with_serial = 0
    with_citov = 0

    rankings_values = (
        frame["rankings"]
        if "rankings" in frame.columns
        else pd.Series([None] * total, index=frame.index)
    )
    for item, rankings_item in zip(frame["scopus"], rankings_values, strict=False):
        if not isinstance(item, dict):
            continue
        abstract = item.get("abstract")
        if isinstance(abstract, dict) and abstract.get("scopus_id"):
            enriched_count += 1
            if abstract.get("citedby_count") is not None:
                with_citedby += 1

        has_serial_metrics = False
        serial = item.get("serial_metrics")
        if isinstance(serial, dict) and serial:
            has_serial_metrics = True
        elif _has_scopus_rankings_metrics(rankings_item):
            has_serial_metrics = True
        if has_serial_metrics:
            with_serial += 1

        citov = item.get("citation_overview")
        if isinstance(citov, dict) and citov:
            with_citov += 1

    enriched_pct = float(enriched_count / total * 100.0) if total else 0.0
    return {
        "total": total,
        "enriched_count": enriched_count,
        "enriched_pct": enriched_pct,
        "with_citedby_count": with_citedby,
        "with_serial_metrics": with_serial,
        "with_citation_overview": with_citov,
    }


def _has_scopus_rankings_metrics(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    scopus = value.get("scopus")
    if not isinstance(scopus, dict):
        return False
    for metric in ("citescore", "citescore_tracker", "sjr", "snip"):
        payload = scopus.get(metric)
        if not isinstance(payload, dict):
            continue
        if payload.get("value") is not None:
            return True
    series = scopus.get("series")
    if not isinstance(series, dict):
        return False
    for key in ("citescore", "sjr", "snip"):
        values = series.get(key)
        if isinstance(values, list) and values:
            return True
    return False


def _bias_features_availability_rows(frame: pd.DataFrame) -> List[Dict[str, object]]:
    if frame.empty:
        return []

    normalized = pd.DataFrame(normalize_records(frame.to_dict(orient="records")))
    if normalized.empty:
        return []

    rows: List[Dict[str, object]] = []
    platform_order = (
        sorted(str(value) for value in normalized["platform"].dropna().unique())
        if "platform" in normalized.columns
        else []
    )
    datasets: list[tuple[str, pd.DataFrame]] = [("overall", normalized)]
    datasets.extend(
        (platform, normalized[normalized["platform"].astype(str) == platform])
        for platform in platform_order
    )

    for platform, subset in datasets:
        total = int(len(subset))
        if total == 0:
            continue
        oa_series = subset["is_oa"] if "is_oa" in subset.columns else None
        country_column = (
            "countries"
            if "countries" in subset.columns
            else ("affiliation_countries" if "affiliation_countries" in subset.columns else None)
        )
        country_series = (
            subset[country_column].apply(_coerce_str_list)
            if country_column is not None
            else None
        )
        citations = pd.to_numeric(subset.get("citations"), errors="coerce")
        doc_type = subset["doc_type"].apply(_coerce_text) if "doc_type" in subset.columns else None
        publisher = subset["publisher"].apply(_coerce_text) if "publisher" in subset.columns else None
        issn_series = (
            subset["issn"].apply(_coerce_str_list)
            if "issn" in subset.columns
            else (
                subset["issn_list"].apply(_coerce_str_list)
                if "issn_list" in subset.columns
                else None
            )
        )

        oa_count = int(oa_series.notna().sum()) if oa_series is not None else 0
        country_count = int((country_series.apply(len) > 0).sum()) if country_series is not None else 0
        citations_count = int(citations.notna().sum()) if hasattr(citations, "notna") else 0
        doc_type_count = int(doc_type.notna().sum()) if doc_type is not None else 0
        publisher_count = int(publisher.notna().sum()) if publisher is not None else 0
        issn_count = int((issn_series.apply(len) > 0).sum()) if issn_series is not None else 0

        rows.append(
            {
                "platform": platform,
                "total": total,
                "oa_count": oa_count,
                "oa_pct": float(oa_count / total * 100.0),
                "country_count": country_count,
                "country_pct": float(country_count / total * 100.0),
                "citations_count": citations_count,
                "citations_pct": float(citations_count / total * 100.0),
                "doc_type_count": doc_type_count,
                "doc_type_pct": float(doc_type_count / total * 100.0),
                "publisher_count": publisher_count,
                "publisher_pct": float(publisher_count / total * 100.0),
                "issn_count": issn_count,
                "issn_pct": float(issn_count / total * 100.0),
            }
        )

    return rows


def _core_match_source_rows(frame: pd.DataFrame) -> List[Dict[str, object]]:
    if frame.empty or "rankings" not in frame.columns:
        return []

    rows: list[dict[str, object]] = []
    platform_order = (
        sorted(str(value) for value in frame["platform"].dropna().unique())
        if "platform" in frame.columns
        else []
    )
    datasets: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    datasets.extend((platform, frame[frame["platform"].astype(str) == platform]) for platform in platform_order)

    for platform, subset in datasets:
        total = int(len(subset))
        if total == 0:
            continue
        source_counts = {"issn": 0, "title": 0, "fuzzy": 0, "unmatched": 0}
        for payload in subset["rankings"]:
            core = payload.get("core") if isinstance(payload, dict) else None
            if not isinstance(core, dict):
                source_counts["unmatched"] += 1
                continue
            value = core.get("rank")
            if value is None:
                value = core.get("value")
            if value is None:
                source_counts["unmatched"] += 1
                continue
            source = str(core.get("source") or "").strip().lower()
            if not source:
                method = str(core.get("method") or "").strip().lower()
                if method == "issn_exact":
                    source = "issn"
                elif method == "title_exact":
                    source = "title"
                elif method == "title_fuzzy":
                    source = "fuzzy"
            if source not in source_counts:
                source = "unmatched"
            source_counts[source] += 1

        matched = source_counts["issn"] + source_counts["title"] + source_counts["fuzzy"]
        rows.append(
            {
                "platform": platform,
                "total": total,
                "matched_count": matched,
                "matched_pct": float(matched / total * 100.0),
                "issn_count": source_counts["issn"],
                "title_count": source_counts["title"],
                "fuzzy_count": source_counts["fuzzy"],
                "unmatched_count": source_counts["unmatched"],
            }
        )
    return rows


def _citations_quality_rows(frame: pd.DataFrame) -> List[Dict[str, object]]:
    if frame.empty:
        return []

    normalized = pd.DataFrame(normalize_records(frame.to_dict(orient="records")))
    if normalized.empty:
        return []

    rows: list[dict[str, object]] = []
    platform_order = (
        sorted(str(value) for value in normalized["platform"].dropna().unique())
        if "platform" in normalized.columns
        else []
    )
    datasets: list[tuple[str, pd.DataFrame]] = [("overall", normalized)]
    datasets.extend(
        (platform, normalized[normalized["platform"].astype(str) == platform])
        for platform in platform_order
    )

    for platform, subset in datasets:
        total = int(len(subset))
        if total == 0:
            continue

        known_count = int(pd.to_numeric(subset.get("citations"), errors="coerce").notna().sum())
        quality_counts = {"ok": 0, "missing": 0, "suspicious": 0}
        quality_series = subset.get("metrics_quality")
        if isinstance(quality_series, pd.Series):
            for item in quality_series.tolist():
                label = (
                    str(item.get("citations"))
                    if isinstance(item, dict) and item.get("citations") is not None
                    else "missing"
                )
                if label not in quality_counts:
                    label = "missing"
                quality_counts[label] += 1

        rows.append(
            {
                "platform": platform,
                "total": total,
                "known_count": known_count,
                "known_pct": float(known_count / total * 100.0),
                "ok_count": quality_counts["ok"],
                "missing_count": quality_counts["missing"],
                "suspicious_count": quality_counts["suspicious"],
            }
        )
    return rows


def _top_k_bias_summary_rows(metrics: Dict[str, dict]) -> List[Dict[str, object]]:
    if not isinstance(metrics, dict):
        return []
    biases = metrics.get("biases")
    if not isinstance(biases, dict):
        return []
    by_platform = biases.get("by_platform")
    if not isinstance(by_platform, dict):
        return []

    rows: List[Dict[str, object]] = []
    for platform in sorted(by_platform.keys()):
        platform_metrics = by_platform.get(platform)
        if not isinstance(platform_metrics, dict):
            continue
        topk = platform_metrics.get("top_k_bias")
        if not isinstance(topk, dict):
            continue
        ks = topk.get("ks")
        if not isinstance(ks, list):
            continue
        oa = topk.get("oa") if isinstance(topk.get("oa"), dict) else {}
        country = topk.get("country") if isinstance(topk.get("country"), dict) else {}
        doc_type = topk.get("doc_type") if isinstance(topk.get("doc_type"), dict) else {}
        citations = topk.get("citations") if isinstance(topk.get("citations"), dict) else {}
        oa_per_k = oa.get("per_k") if isinstance(oa.get("per_k"), dict) else {}
        country_per_k = country.get("per_k") if isinstance(country.get("per_k"), dict) else {}
        doc_type_per_k = doc_type.get("per_k") if isinstance(doc_type.get("per_k"), dict) else {}
        citations_per_k = citations.get("per_k") if isinstance(citations.get("per_k"), dict) else {}

        for k in sorted(int(value) for value in ks):
            key = str(k)
            oa_k = oa_per_k.get(key) if isinstance(oa_per_k.get(key), dict) else {}
            country_k = (
                country_per_k.get(key) if isinstance(country_per_k.get(key), dict) else {}
            )
            doc_type_k = (
                doc_type_per_k.get(key) if isinstance(doc_type_per_k.get(key), dict) else {}
            )
            citations_k = (
                citations_per_k.get(key) if isinstance(citations_per_k.get(key), dict) else {}
            )
            rows.append(
                {
                    "platform": platform,
                    "k": k,
                    "oa_top_k_share": oa_k.get("top_k_share"),
                    "oa_overall_share": oa_k.get("overall_share"),
                    "oa_delta_share": oa_k.get("delta_share"),
                    "oa_reliability": oa_k.get("reliability"),
                    "country_js_divergence": country_k.get("js_divergence"),
                    "country_reliability": country_k.get("reliability"),
                    "doc_type_js_divergence": doc_type_k.get("js_divergence"),
                    "doc_type_reliability": doc_type_k.get("reliability"),
                    "citation_spearman": citations.get("spearman_rank_vs_citations"),
                    "citation_median_top_k": citations_k.get("median_top_k"),
                    "citation_median_rest": citations_k.get("median_rest"),
                    "citation_reliability": citations_k.get("reliability"),
                }
            )
    return rows


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text if text else None


def _coerce_str_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    elif isinstance(value, set):
        values = sorted(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        values = _parse_list_like_text(text)
        if not values:
            values = text.replace(";", ",").split(",") if "," in text or ";" in text else [text]
    elif hasattr(value, "tolist"):
        try:
            converted = value.tolist()
        except Exception:
            converted = []
        if isinstance(converted, list):
            values = converted
        elif isinstance(converted, tuple):
            values = list(converted)
        else:
            return []
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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
