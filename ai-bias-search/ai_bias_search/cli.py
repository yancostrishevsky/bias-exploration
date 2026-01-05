"""Command line interface for the AI Bias Search project."""

import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer
from dotenv import load_dotenv

from ai_bias_search.connectors import get_connector
from ai_bias_search.connectors.base import ConnectorError
from ai_bias_search.evaluation.biases import compute_bias_metrics
from ai_bias_search.evaluation.overlap import jaccard, overlap_at_k
from ai_bias_search.evaluation.ranking_similarity import rbo
from ai_bias_search.normalization.openalex_enrich import enrich_with_openalex
from ai_bias_search.report.make_report import generate_report
from ai_bias_search.utils.config import AppConfig, RateLimitConfig, load_config
from ai_bias_search.utils.io import (
    ensure_directory,
    load_queries,
    read_jsonl,
    read_parquet,
    write_jsonl,
    write_parquet,
    utc_timestamp,
)
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.rate_limit import RateLimiter


LOGGER = configure_logging()
app = typer.Typer(add_completion=False)


def _load_env() -> None:
    load_dotenv()


def _load_app_config(config_path: Path) -> AppConfig:
    config = load_config(config_path)
    return config


def _instantiate_connector(name: str, config: AppConfig) -> object:
    connector_cls = get_connector(name)
    rate_limit_cfg: Optional[RateLimitConfig] = config.rate_limit.get(name) if config.rate_limit else None
    limiter = None
    if rate_limit_cfg:
        limiter = RateLimiter(rate=rate_limit_cfg.rps, burst=rate_limit_cfg.burst)
    kwargs: Dict[str, object] = {"rate_limiter": limiter, "retries": config.retries}
    if name == "openalex":
        kwargs["mailto"] = config.openalex_mailto
    try:
        connector = connector_cls(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        connector = connector_cls()  # type: ignore[call-arg]
    return connector


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


@app.command()
def collect(
    *,
    config: Path = typer.Option(..., help="Path to YAML configuration file."),
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
    config: Path = typer.Option(..., help="Path to YAML configuration file."),
    run_timestamp: Optional[str] = typer.Option(
        None, help="Specific collection timestamp to process."
    ),
) -> None:
    """Enrich collected records with OpenAlex metadata and store as Parquet."""

    _load_env()
    app_config = _load_app_config(config)
    records: List[dict] = []
    for platform in app_config.platforms:
        raw_dir = Path("data/raw") / platform
        if run_timestamp:
            raw_path = raw_dir / f"{run_timestamp}.jsonl"
        else:
            raw_path = _latest_file(raw_dir, "*.jsonl")
        if raw_path is None or not raw_path.exists():
            LOGGER.warning("No raw data found for %s", platform)
            continue
        records.extend(read_jsonl(raw_path))

    if not records:
        LOGGER.error("No records available for enrichment")
        raise typer.Exit(code=1)

    enriched = enrich_with_openalex(records, app_config.openalex_mailto)
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
            "overlap_at_k": overlap_at_k(l_records, r_records, k=min(len(l_records), len(r_records))),
            "rbo": rbo(l_records, r_records),
        }
    return metrics


@app.command()
def eval(
    *,
    config: Path = typer.Option(..., help="Path to YAML configuration file."),
    run_timestamp: Optional[str] = typer.Option(
        None, help="Specific enrichment timestamp to evaluate."
    ),
) -> None:
    """Compute evaluation metrics and store them as JSON."""

    _load_env()
    _ = _load_app_config(config)
    enriched_dir = Path("data/enriched")
    if run_timestamp:
        enriched_path = enriched_dir / f"{run_timestamp}.parquet"
    else:
        enriched_path = _latest_file(enriched_dir, "*.parquet")
    if enriched_path is None or not enriched_path.exists():
        LOGGER.error("No enriched dataset available")
        raise typer.Exit(code=1)

    frame = read_parquet(enriched_path)
    platforms = sorted(frame["platform"].dropna().unique().tolist()) if "platform" in frame.columns else []

    pairwise = _pairwise_metrics(frame, platforms) if platforms else {}
    bias_metrics = compute_bias_metrics(frame)

    timestamp = utc_timestamp()
    output_path = Path("results/metrics") / f"{timestamp}.json"
    ensure_directory(output_path)
    payload = {"pairwise": pairwise, "biases": bias_metrics, "source": enriched_path.name}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Metrics saved to %s", output_path)


@app.command()
def report(
    *,
    config: Path = typer.Option(..., help="Path to YAML configuration file."),
    metrics_timestamp: Optional[str] = typer.Option(
        None, help="Specific metrics timestamp to include."
    ),
    enriched_timestamp: Optional[str] = typer.Option(
        None, help="Specific enrichment timestamp to use."
    ),
) -> None:
    """Generate an HTML report for the latest metrics."""

    _load_env()
    _ = _load_app_config(config)

    metrics_dir = Path("results/metrics")
    if metrics_timestamp:
        chosen_metrics_dir = metrics_dir
    else:
        chosen_metrics_dir = metrics_dir

    enriched_dir = Path("data/enriched")
    if enriched_timestamp:
        enriched_path = enriched_dir / f"{enriched_timestamp}.parquet"
    else:
        enriched_path = _latest_file(enriched_dir, "*.parquet")

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
