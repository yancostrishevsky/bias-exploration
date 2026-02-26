"""HTML report generation for evaluation results."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai_bias_search.evaluation.biases import core_ranking_table
from ai_bias_search.normalize.records import normalize_records
from ai_bias_search.utils.io import read_parquet
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()


def generate_report(enriched_path: Path, metrics_dir: Path, output_path: Path) -> Path:
    """Render an HTML report combining enriched data and metrics."""

    LOGGER.info("Generating report from %s", enriched_path)
    data = read_parquet(enriched_path)
    latest_metrics, metrics_timestamp = _load_latest_metrics(metrics_dir)
    diagnostics = _load_diagnostics(metrics_dir, latest_metrics)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")

    summary = {
        "total_records": len(data),
        "platforms": (
            sorted(str(value) for value in data.get("platform").dropna().unique().tolist())
            if "platform" in data.columns
            else []
        ),
    }

    plots = _generate_plots(data, latest_metrics)
    jcr_summary = _jcr_summary(data)
    jif_context = _generate_jif_context(data)
    rankings_coverage_table = _rankings_coverage_table(data)
    core_match_source_breakdown = _core_match_source_rows(data)
    scopus_summary = _scopus_summary(data)
    bias_features_availability = _bias_features_availability_rows(data)
    citations_quality_rows = _citations_quality_rows(data)
    top_k_bias_summary = _top_k_bias_summary_rows(latest_metrics)

    preview = _prepare_sample_table(data.drop(columns=["extra"], errors="ignore")).head(50)
    core_ranking_rows = core_ranking_table(data)
    rendered = template.render(
        summary=summary,
        latest_metrics=latest_metrics,
        metrics_timestamp=metrics_timestamp,
        table=preview.to_dict(orient="records"),
        core_ranking_table=core_ranking_rows,
        rankings_coverage_table=rankings_coverage_table,
        core_match_source_breakdown=core_match_source_breakdown,
        scopus_summary=scopus_summary,
        bias_features_availability=bias_features_availability,
        citations_quality_rows=citations_quality_rows,
        diagnostics=diagnostics,
        top_k_bias_summary=top_k_bias_summary,
        jcr_summary=jcr_summary,
        plots=plots,
        **jif_context,
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
        country_series = (
            subset["affiliation_countries"].apply(_coerce_str_list)
            if "affiliation_countries" in subset.columns
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
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        values = text.replace(";", ",").split(",") if "," in text or ";" in text else [text]
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


def _oa_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    if "is_oa" in frame.columns and "is_open_access" in frame.columns:
        return frame["is_oa"].combine_first(frame["is_open_access"])
    if "is_oa" in frame.columns:
        return frame["is_oa"]
    if "is_open_access" in frame.columns:
        return frame["is_open_access"]
    return None


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
