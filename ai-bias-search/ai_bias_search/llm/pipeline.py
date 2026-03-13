"""Pipeline orchestration for the OpenRouter-based LLM audit workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

import yaml  # type: ignore[import-untyped]

from ai_bias_search.llm.enrichment import enrich_recommendations
from ai_bias_search.llm.evaluator import evaluate_run
from ai_bias_search.llm.normalizer import normalize_responses
from ai_bias_search.llm.prompts import load_query_prompt_tasks, load_scenario_prompt_tasks
from ai_bias_search.llm.report import render_llm_report
from ai_bias_search.llm.schemas import (
    ChatMessage,
    EnrichedRecommendationRecord,
    NormalizedResponseRecord,
    PromptTask,
    ProviderRequest,
    RawResponseRecord,
    RunManifest,
)
from ai_bias_search.providers import OpenRouterClient, OpenRouterError
from ai_bias_search.utils.config import AppConfig
from ai_bias_search.utils.io import ensure_directory, read_jsonl, utc_timestamp, write_jsonl
from ai_bias_search.utils.logging import configure_logging
from ai_bias_search.utils.rate_limit import RateLimiter

LOGGER = configure_logging()
T = TypeVar("T")


@dataclass(frozen=True)
class RunPaths:
    """Resolved artifact paths for one LLM run."""

    output_root: Path
    run_dir: Path
    raw_file: Path
    normalized_file: Path
    enriched_file: Path
    metrics_file: Path
    report_file: Path
    manifest_file: Path
    config_snapshot_file: Path
    payload_dir: Path
    pairwise_csv: Path
    cross_model_csv: Path


@dataclass(frozen=True)
class PromptTaskBundle:
    """Prompt tasks plus the source files used to build them."""

    mode: str
    tasks: list[PromptTask]
    input_file: Path
    prompt_template_file: Path | None = None


def run_llm_collect(config: AppConfig, *, base_dir: Path, run_id: str | None = None) -> Path:
    """Collect raw LLM responses for all configured queries/models."""

    llm_cfg = _require_llm_config(config)
    paths = _resolve_run_paths(base_dir=base_dir, output_dir=llm_cfg.output_dir, run_id=run_id)
    prompt_bundle = _load_prompt_tasks(config=config, base_dir=base_dir)
    manifest = _initialize_manifest(config=config, paths=paths, prompt_bundle=prompt_bundle)
    _write_manifest(paths.manifest_file, manifest)
    _write_config_snapshot(paths.config_snapshot_file, config)

    client = _build_provider_client(config)
    records: list[RawResponseRecord] = []
    try:
        for task in prompt_bundle.tasks:
            for model in llm_cfg.models:
                for repeat_index in range(llm_cfg.generation.repeats_per_query):
                    request_id = _request_id(task, model=model, repeat_index=repeat_index)
                    request = _provider_request(config=config, task=task, model=model)
                    timestamp = utc_timestamp()
                    try:
                        response = client.complete(request)
                        record = RawResponseRecord(
                            run_id=paths.run_dir.name,
                            timestamp=timestamp,
                            request_id=request_id,
                            source_mode=task.source_mode,
                            query_id=task.query_id,
                            query_text=task.query_text,
                            query_category=task.query_category,
                            query_language=task.query_language,
                            mode=task.mode,
                            model=model,
                            provider=llm_cfg.provider,
                            repeat_index=repeat_index,
                            prompt_text=task.prompt_text,
                            input_metadata=task.metadata,
                            expected_format=task.expected_format,
                            topic=task.topic,
                            pair_id=task.pair_id,
                            variant=task.variant,
                            control_or_treatment=task.control_or_treatment,
                            request_payload=client.build_payload(request),
                            raw_response_text=response.output_text,
                            raw_response_json=response.raw_response,
                            latency_ms=response.latency_ms,
                            token_usage=response.token_usage,
                            success=True,
                        )
                    except OpenRouterError as exc:
                        LOGGER.warning(
                            "LLM collection failed mode=%s query_id=%s model=%s repeat=%s error=%s",
                            task.source_mode,
                            task.query_id,
                            model,
                            repeat_index,
                            exc,
                        )
                        record = RawResponseRecord(
                            run_id=paths.run_dir.name,
                            timestamp=timestamp,
                            request_id=request_id,
                            source_mode=task.source_mode,
                            query_id=task.query_id,
                            query_text=task.query_text,
                            query_category=task.query_category,
                            query_language=task.query_language,
                            mode=task.mode,
                            model=model,
                            provider=llm_cfg.provider,
                            repeat_index=repeat_index,
                            prompt_text=task.prompt_text,
                            input_metadata=task.metadata,
                            expected_format=task.expected_format,
                            topic=task.topic,
                            pair_id=task.pair_id,
                            variant=task.variant,
                            control_or_treatment=task.control_or_treatment,
                            request_payload=client.build_payload(request),
                            success=False,
                            error_message=str(exc),
                        )
                    records.append(record)
                    if llm_cfg.save_payloads:
                        _write_payload_snapshot(paths.payload_dir / f"{request_id}.json", record)
    finally:
        client.close()

    write_jsonl(paths.raw_file, [record.model_dump(mode="json") for record in records])
    manifest.stages["collect"] = {
        "completed_at": utc_timestamp(),
        "raw_record_count": len(records),
        "query_count": len({record.query_id for record in records}),
        "success_count": sum(1 for record in records if record.success),
        "failure_count": sum(1 for record in records if not record.success),
    }
    manifest.artifacts["raw_responses"] = paths.raw_file.name
    _write_manifest(paths.manifest_file, manifest)
    return paths.run_dir


def run_llm_normalize(config: AppConfig, *, base_dir: Path, run_id: str | None = None) -> Path:
    """Normalize raw LLM responses for one run."""

    paths = _existing_run_paths(config=config, base_dir=base_dir, run_id=run_id)
    raw_records = _load_jsonl_models(paths.raw_file, RawResponseRecord)
    normalized = normalize_responses(raw_records)
    write_jsonl(paths.normalized_file, [record.model_dump(mode="json") for record in normalized])
    manifest = _load_manifest(paths.manifest_file)
    manifest.stages["normalize"] = {
        "completed_at": utc_timestamp(),
        "normalized_record_count": len(normalized),
        "parse_success_count": sum(1 for record in normalized if record.parse_success),
        "parse_failure_count": sum(1 for record in normalized if not record.parse_success),
    }
    manifest.artifacts["normalized_responses"] = paths.normalized_file.name
    _write_manifest(paths.manifest_file, manifest)
    return paths.normalized_file


def run_llm_eval(config: AppConfig, *, base_dir: Path, run_id: str | None = None) -> Path:
    """Enrich normalized recommendations and compute metrics for one run."""

    llm_cfg = _require_llm_config(config)
    paths = _existing_run_paths(config=config, base_dir=base_dir, run_id=run_id)
    raw_records = _load_jsonl_models(paths.raw_file, RawResponseRecord)
    normalized = _load_jsonl_models(paths.normalized_file, NormalizedResponseRecord)
    openalex_rate = config.rate_limit.get("openalex") if config.rate_limit else None
    rate_limiter = (
        RateLimiter(rate=openalex_rate.rps, burst=openalex_rate.burst) if openalex_rate else None
    )
    enriched = enrich_recommendations(
        normalized,
        enabled=llm_cfg.enrichment.enabled,
        openalex_mailto=config.openalex_mailto,
        impact_factor=config.impact_factor,
        retries=config.retries,
        rate_limiter=rate_limiter,
    )
    write_jsonl(paths.enriched_file, [record.model_dump(mode="json") for record in enriched])
    metrics = evaluate_run(raw_records, normalized, enriched)
    paths.metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_pairwise_csv(paths.pairwise_csv, metrics.get("pairwise_comparisons", []))
    _write_pairwise_csv(
        paths.cross_model_csv,
        metrics.get("cross_model_divergence", {}).get("pairwise", []),
    )

    manifest = _load_manifest(paths.manifest_file)
    manifest.stages["eval"] = {
        "completed_at": utc_timestamp(),
        "enriched_record_count": len(enriched),
        "metrics_file": paths.metrics_file.name,
    }
    manifest.artifacts["enriched_recommendations"] = paths.enriched_file.name
    manifest.artifacts["metrics"] = paths.metrics_file.name
    manifest.artifacts["pairwise_csv"] = paths.pairwise_csv.name
    manifest.artifacts["cross_model_csv"] = paths.cross_model_csv.name
    _write_manifest(paths.manifest_file, manifest)
    return paths.metrics_file


def run_llm_report(config: AppConfig, *, base_dir: Path, run_id: str | None = None) -> Path:
    """Generate the HTML report for one run."""

    paths = _existing_run_paths(config=config, base_dir=base_dir, run_id=run_id)
    raw_records = _load_jsonl_models(paths.raw_file, RawResponseRecord)
    normalized = _load_jsonl_models(paths.normalized_file, NormalizedResponseRecord)
    enriched = _load_jsonl_models(paths.enriched_file, EnrichedRecommendationRecord)
    metrics = json.loads(paths.metrics_file.read_text(encoding="utf-8"))
    output_path = render_llm_report(
        run_dir=paths.run_dir,
        raw_records=raw_records,
        normalized_records=normalized,
        enriched_records=enriched,
        metrics=metrics,
    )
    manifest = _load_manifest(paths.manifest_file)
    manifest.stages["report"] = {"completed_at": utc_timestamp(), "report_file": output_path.name}
    manifest.artifacts["report"] = output_path.name
    _write_manifest(paths.manifest_file, manifest)
    return output_path


def run_llm_pipeline(config: AppConfig, *, base_dir: Path) -> Path:
    """Run collect -> normalize -> eval -> report using a single run id."""

    run_dir = run_llm_collect(config, base_dir=base_dir)
    run_id = run_dir.name
    run_llm_normalize(config, base_dir=base_dir, run_id=run_id)
    run_llm_eval(config, base_dir=base_dir, run_id=run_id)
    run_llm_report(config, base_dir=base_dir, run_id=run_id)
    return run_dir


def latest_llm_run_dir(config: AppConfig, *, base_dir: Path) -> Path:
    """Return the most recent LLM run directory."""

    llm_cfg = _require_llm_config(config)
    output_root = _resolve_path(base_dir, llm_cfg.output_dir)
    if not output_root.exists():
        raise FileNotFoundError(f"LLM output directory does not exist: {output_root}")
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No LLM run directories found in {output_root}")
    return sorted(candidates)[-1]


def _require_llm_config(config: AppConfig):
    llm_cfg = config.llm
    if not llm_cfg.enabled:
        raise ValueError("LLM pipeline is disabled in config")
    if not llm_cfg.models:
        raise ValueError("llm.models must not be empty")
    return llm_cfg


def _load_prompt_tasks(config: AppConfig, *, base_dir: Path) -> PromptTaskBundle:
    llm_cfg = _require_llm_config(config)
    if llm_cfg.mode == "query_csv":
        query_csv_path = _resolve_path(base_dir, llm_cfg.queries.input_csv)
        prompt_template_path = _resolve_path(base_dir, llm_cfg.queries.prompt_template_file)
        tasks = load_query_prompt_tasks(
            query_csv_path,
            prompt_template_path,
            top_k_articles=llm_cfg.generation.top_k_articles,
        )
        return PromptTaskBundle(
            mode="query_csv",
            tasks=tasks,
            input_file=query_csv_path,
            prompt_template_file=prompt_template_path,
        )

    scenario_path = _resolve_path(base_dir, llm_cfg.controlled_bias_probes.input_file)
    tasks = load_scenario_prompt_tasks(scenario_path)
    return PromptTaskBundle(mode="scenarios", tasks=tasks, input_file=scenario_path)


def _resolve_run_paths(*, base_dir: Path, output_dir: Path, run_id: str | None) -> RunPaths:
    output_root = _resolve_path(base_dir, output_dir)
    chosen_run_id = run_id or utc_timestamp()
    run_dir = output_root / chosen_run_id
    return RunPaths(
        output_root=output_root,
        run_dir=run_dir,
        raw_file=run_dir / "raw_responses.jsonl",
        normalized_file=run_dir / "normalized_responses.jsonl",
        enriched_file=run_dir / "enriched_recommendations.jsonl",
        metrics_file=run_dir / "metrics.json",
        report_file=run_dir / "report.html",
        manifest_file=run_dir / "manifest.json",
        config_snapshot_file=run_dir / "effective_config.yaml",
        payload_dir=run_dir / "payloads",
        pairwise_csv=run_dir / "pairwise_comparisons.csv",
        cross_model_csv=run_dir / "cross_model_divergence.csv",
    )


def _existing_run_paths(config: AppConfig, *, base_dir: Path, run_id: str | None) -> RunPaths:
    llm_cfg = _require_llm_config(config)
    if run_id:
        paths = _resolve_run_paths(base_dir=base_dir, output_dir=llm_cfg.output_dir, run_id=run_id)
    else:
        latest = latest_llm_run_dir(config, base_dir=base_dir)
        paths = _resolve_run_paths(
            base_dir=base_dir, output_dir=llm_cfg.output_dir, run_id=latest.name
        )
    if not paths.run_dir.exists():
        raise FileNotFoundError(f"LLM run directory does not exist: {paths.run_dir}")
    return paths


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _initialize_manifest(
    config: AppConfig,
    *,
    paths: RunPaths,
    prompt_bundle: PromptTaskBundle,
) -> RunManifest:
    return RunManifest(
        run_id=paths.run_dir.name,
        created_at=utc_timestamp(),
        provider=config.llm.provider,
        models=list(config.llm.models),
        mode=config.llm.mode,
        input_file=str(prompt_bundle.input_file),
        prompt_template_file=(
            str(prompt_bundle.prompt_template_file) if prompt_bundle.prompt_template_file else None
        ),
        output_dir=str(paths.run_dir),
        config_snapshot=paths.config_snapshot_file.name,
        stages={},
        artifacts={},
    )


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    ensure_directory(path)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _write_config_snapshot(path: Path, config: AppConfig) -> None:
    ensure_directory(path)
    payload = _yaml_ready(config.model_dump(mode="python"))
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _yaml_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_ready(item) for item in value]
    return value


def _build_provider_client(config: AppConfig) -> OpenRouterClient:
    if config.llm.provider != "openrouter":
        raise ValueError(f"Unsupported llm.provider: {config.llm.provider}")
    return OpenRouterClient(retries=config.retries)


def _provider_request(config: AppConfig, task: PromptTask, *, model: str) -> ProviderRequest:
    system_prompt = _system_prompt(config=config, task=task)
    return ProviderRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=task.prompt_text),
        ],
        temperature=config.llm.generation.temperature,
        max_tokens=config.llm.generation.max_tokens,
        top_p=config.llm.generation.top_p,
        timeout_seconds=config.llm.generation.timeout_seconds,
        require_json=config.llm.parsing.require_json,
    )


def _system_prompt(config: AppConfig, task: PromptTask) -> str:
    parts = [
        (
            "You are participating in a reproducible audit of LLM behavior for "
            "scientific literature retrieval."
        ),
        "Follow the requested response format exactly.",
    ]
    if config.llm.parsing.require_json:
        parts.append("Return only valid JSON. Do not include markdown fences or commentary.")
    if task.expected_format:
        parts.append(f"Expected format: {task.expected_format}.")
    if task.mode == "article_recommendation":
        parts.append(
            "Recommend real scholarly literature when possible. Include DOI only when "
            "known and do not invent bibliographic metadata."
        )
    return " ".join(parts)


def _request_id(task: PromptTask, *, model: str, repeat_index: int) -> str:
    safe_model = model.replace("/", "__").replace(":", "_")
    safe_query_id = task.query_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    return f"{safe_query_id}__{safe_model}__r{repeat_index + 1}"


def _write_payload_snapshot(path: Path, record: RawResponseRecord) -> None:
    ensure_directory(path)
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=2), encoding="utf-8")


def _load_jsonl_models(path: Path, model_type: type[T]) -> list[T]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [model_type.model_validate(item) for item in read_jsonl(path)]


def _write_pairwise_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_directory(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        values: list[str] = []
        for key in header:
            value = row.get(key)
            text = "" if value is None else str(value)
            if any(char in text for char in [",", '"', "\n"]):
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
