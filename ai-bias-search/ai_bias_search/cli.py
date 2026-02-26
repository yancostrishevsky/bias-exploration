"""Command line interface for the AI Bias Search project."""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

from ai_bias_search.connectors import get_connector
from ai_bias_search.connectors.base import ConnectorError
from ai_bias_search.diagnostics.sanity import run_sanity_checks
from ai_bias_search.enrichment.scopus_rankings import enrich_with_scopus_rankings
from ai_bias_search.evaluation.biases import compute_bias_metrics
from ai_bias_search.normalize.records import normalize_records
from ai_bias_search.evaluation.overlap import jaccard, overlap_at_k
from ai_bias_search.evaluation.ranking_similarity import rbo
from ai_bias_search.normalization.openalex_enrich import enrich_with_openalex
from ai_bias_search.normalization.scopus_enrich import enrich_with_scopus
from ai_bias_search.report.make_report import generate_report
from ai_bias_search.utils.config import AppConfig, RateLimitConfig, load_config
from ai_bias_search.utils.io import (
    ensure_directory,
    load_queries,
    read_jsonl,
    read_parquet,
    utc_timestamp,
    write_jsonl,
    write_parquet,
)
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.rate_limit import RateLimiter

LOGGER = configure_logging()
app = typer.Typer(add_completion=False)

CONFIG_OPTION = typer.Option(..., help="Path to YAML configuration file.")
COLLECT_TIMESTAMP_OPTION = typer.Option(None, help="Specific collection timestamp to process.")
ENRICHED_TIMESTAMP_OPTION = typer.Option(None, help="Specific enrichment timestamp to evaluate.")
METRICS_TIMESTAMP_OPTION = typer.Option(None, help="Specific metrics timestamp to include.")
REPORT_ENRICHED_TIMESTAMP_OPTION = typer.Option(None, help="Specific enrichment timestamp to use.")
ENRICH_SCOPUS_RANKINGS_OPTION = typer.Option(
    False,
    "--enrich-scopus-rankings/--no-enrich-scopus-rankings",
    help="Enrich records with Scopus Serial Title journal metrics (CiteScore/SJR/SNIP).",
)


def _load_env() -> None:
    load_dotenv()
    configure_logging()


def _load_app_config(config_path: Path) -> AppConfig:
    config = load_config(config_path)
    return config


def _instantiate_connector(name: str, config: AppConfig) -> Any:
    connector_cls = get_connector(name)
    rate_limit_cfg: Optional[RateLimitConfig] = (
        config.rate_limit.get(name) if config.rate_limit else None
    )
    limiter = None
    if rate_limit_cfg:
        limiter = RateLimiter(rate=rate_limit_cfg.rps, burst=rate_limit_cfg.burst)
    kwargs: Dict[str, object] = {"rate_limiter": limiter, "retries": config.retries}
    if name == "openalex":
        kwargs["mailto"] = config.openalex_mailto
    if name == "scopus":
        kwargs["config"] = config.scopus
    try:
        connector = connector_cls(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        connector = connector_cls()  # type: ignore[call-arg]
    return connector


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _json_compatible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
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


def _merge_canonical_metadata(
    record: Dict[str, Any],
    canonical: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(record)

    citations = canonical.get("citations")
    updated["citations"] = citations
    updated["cited_by_count"] = citations

    issn_values = canonical.get("issn")
    if isinstance(issn_values, list):
        updated["issn_list"] = issn_values or None
        if issn_values and not updated.get("issn"):
            updated["issn"] = issn_values[0]
        if len(issn_values) > 1 and not updated.get("eissn"):
            updated["eissn"] = issn_values[1]

    for key in ("publisher", "journal_title", "doc_type", "language"):
        value = canonical.get(key)
        current = updated.get(key)
        if value is None:
            continue
        if current is None or (isinstance(current, str) and not current.strip()):
            updated[key] = value

    if canonical.get("is_oa") is not None and updated.get("is_oa") is None:
        updated["is_oa"] = canonical["is_oa"]

    updated["journal_match"] = canonical.get("journal_match")
    updated["metrics_quality"] = canonical.get("metrics_quality")
    return updated


@app.command()
def collect(
    *,
    config: Path = CONFIG_OPTION,
) -> None:
    """Collect search results from configured platforms."""

    _load_env()
    app_config = _load_app_config(config)
    base_dir = config.parent
    queries_path = app_config.resolve_queries_path(base_dir)
    queries = load_queries(queries_path)
    timestamp = utc_timestamp()

    for platform in app_config.platforms:
        connector = _instantiate_connector(platform, app_config)
        results: List[dict] = []
        for query in queries:
            query_text = query.get("text") or ""
            try:
                platform_results = connector.search(
                    query=query_text,
                    k=app_config.top_k,
                    prompt_template=app_config.prompt_template,
                    params=None,
                )
            except ConnectorError as exc:
                LOGGER.error("Connector %s failed for query '%s': %s", platform, query_text, exc)
                continue
            for item in platform_results:
                item["platform"] = platform
                item["query_id"] = query.get("query_id")
                item["query_text"] = query_text
                results.append(item)

        output_path = Path("data/raw") / platform / f"{timestamp}.jsonl"
        write_jsonl(output_path, results)
        LOGGER.info("Saved %s records for %s to %s", len(results), platform, output_path)


@app.command()
def enrich(
    *,
    config: Path = CONFIG_OPTION,
    run_timestamp: Optional[str] = COLLECT_TIMESTAMP_OPTION,
    enrich_scopus_rankings: bool = ENRICH_SCOPUS_RANKINGS_OPTION,
) -> None:
    """Enrich collected records with OpenAlex metadata and store as Parquet."""

    _load_env()
    app_config = _load_app_config(config)
    records: List[dict] = []
    for platform in app_config.platforms:
        raw_dir = Path("data/raw") / platform
        raw_path: Path | None
        raw_path = (
            raw_dir / f"{run_timestamp}.jsonl"
            if run_timestamp
            else _latest_file(raw_dir, "*.jsonl")
        )
        if raw_path is None or not raw_path.exists():
            LOGGER.warning("No raw data found for %s", platform)
            continue
        records.extend(read_jsonl(raw_path))

    if not records:
        LOGGER.error("No records available for enrichment")
        raise typer.Exit(code=1)

    openalex_rate = app_config.rate_limit.get("openalex") if app_config.rate_limit else None
    enrich_limiter = (
        RateLimiter(rate=openalex_rate.rps, burst=openalex_rate.burst) if openalex_rate else None
    )
    enriched = enrich_with_openalex(
        records,
        app_config.openalex_mailto,
        app_config.impact_factor,
        rate_limiter=enrich_limiter,
        retries=app_config.retries,
    )
    scopus_cfg = app_config.scopus if app_config.scopus.enabled else app_config.scopus_enrich
    if scopus_cfg.enabled:
        enriched = enrich_with_scopus(
            enriched,
            scopus_cfg,
            retries=app_config.retries,
        )
    run_rankings = bool(enrich_scopus_rankings or scopus_cfg.rankings.enabled)
    if run_rankings:
        rankings_cfg = scopus_cfg.rankings.model_copy(update={"enabled": True})
        enriched = enrich_with_scopus_rankings(
            enriched,
            cfg=rankings_cfg,
            retries=app_config.retries,
        )
    canonical = normalize_records(enriched)
    enriched = [
        _merge_canonical_metadata(record, normalized)
        for record, normalized in zip(enriched, canonical, strict=False)
    ]
    timestamp = utc_timestamp()
    output_path = Path("data/enriched") / f"{timestamp}.parquet"
    write_parquet(output_path, enriched)
    LOGGER.info("Enriched dataset stored at %s", output_path)


def _pairwise_metrics(frame: pd.DataFrame, platforms: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for left, right in itertools.combinations(platforms, 2):
        l_records = frame[frame["platform"] == left].sort_values("rank").to_dict(orient="records")
        r_records = frame[frame["platform"] == right].sort_values("rank").to_dict(orient="records")
        key = f"{left}_vs_{right}"
        metrics[key] = {
            "jaccard": jaccard(
                [record.get("doi") for record in l_records],
                [record.get("doi") for record in r_records],
            ),
            "overlap_at_k": overlap_at_k(
                l_records, r_records, k=min(len(l_records), len(r_records))
            ),
            "rbo": rbo(l_records, r_records),
        }
    return metrics


@app.command()
def eval(
    *,
    config: Path = CONFIG_OPTION,
    run_timestamp: Optional[str] = ENRICHED_TIMESTAMP_OPTION,
) -> None:
    """Compute evaluation metrics and store them as JSON."""

    _load_env()
    _ = _load_app_config(config)
    enriched_dir = Path("data/enriched")
    enriched_path: Path | None
    enriched_path = (
        enriched_dir / f"{run_timestamp}.parquet"
        if run_timestamp
        else _latest_file(enriched_dir, "*.parquet")
    )
    if enriched_path is None or not enriched_path.exists():
        LOGGER.error("No enriched dataset available")
        raise typer.Exit(code=1)

    frame = read_parquet(enriched_path)
    diagnostics = run_sanity_checks(frame.to_dict(orient="records"))
    diagnostics_path = Path("results/diagnostics.json")
    ensure_directory(diagnostics_path)
    diagnostics_path.write_text(
        json.dumps(_json_compatible(diagnostics), indent=2),
        encoding="utf-8",
    )
    for warning in diagnostics.get("warnings", []):
        LOGGER.warning("Sanity check warning: %s", warning)

    platforms = (
        sorted(frame["platform"].dropna().unique().tolist()) if "platform" in frame.columns else []
    )

    pairwise = _pairwise_metrics(frame, platforms) if platforms else {}
    bias_metrics = compute_bias_metrics(frame)

    timestamp = utc_timestamp()
    output_path = Path("results/metrics") / f"{timestamp}.json"
    ensure_directory(output_path)
    payload = {
        "pairwise": pairwise,
        "biases": bias_metrics,
        "source": enriched_path.name,
        "diagnostics_path": diagnostics_path.name,
    }
    output_path.write_text(json.dumps(_json_compatible(payload), indent=2), encoding="utf-8")
    LOGGER.info("Metrics saved to %s", output_path)


@app.command()
def report(
    *,
    config: Path = CONFIG_OPTION,
    metrics_timestamp: Optional[str] = METRICS_TIMESTAMP_OPTION,
    enriched_timestamp: Optional[str] = REPORT_ENRICHED_TIMESTAMP_OPTION,
) -> None:
    """Generate an HTML report for the latest metrics."""

    _load_env()
    _ = _load_app_config(config)

    metrics_dir = Path("results/metrics")
    chosen_metrics_dir = metrics_dir

    enriched_dir = Path("data/enriched")
    enriched_path: Path | None
    enriched_path = (
        enriched_dir / f"{enriched_timestamp}.parquet"
        if enriched_timestamp
        else _latest_file(enriched_dir, "*.parquet")
    )

    if enriched_path is None or not enriched_path.exists():
        LOGGER.error("No enriched dataset available for reporting")
        raise typer.Exit(code=1)

    timestamp = utc_timestamp()
    output_path = Path("results/reports") / f"{timestamp}.html"
    generate_report(enriched_path, chosen_metrics_dir, output_path)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
