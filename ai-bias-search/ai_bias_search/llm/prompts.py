"""Prompt loading and rendering helpers for the LLM audit pipeline."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ai_bias_search.llm.schemas import PromptDataset, PromptScenario, PromptTask, QueryInputRow

_DEFAULT_EXPECTED_FORMAT = "json object with an `articles` array"
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def load_query_rows(path: Path) -> list[QueryInputRow]:
    """Load literature-retrieval queries from CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [QueryInputRow.model_validate(row) for row in reader if _has_content(row)]
    if not rows:
        raise ValueError(f"Query file {path} did not contain any rows")
    return rows


def load_query_prompt_tasks(
    query_csv_path: Path,
    prompt_template_path: Path,
    *,
    top_k_articles: int,
) -> list[PromptTask]:
    """Render one prompt task per CSV query row."""

    template = prompt_template_path.read_text(encoding="utf-8").strip()
    if not template:
        raise ValueError(f"Prompt template file {prompt_template_path} is empty")

    tasks: list[PromptTask] = []
    for row in load_query_rows(query_csv_path):
        prompt_text = render_query_prompt(template, row=row, top_k_articles=top_k_articles)
        tasks.append(
            PromptTask(
                source_mode="query_csv",
                query_id=row.query_id,
                query_text=row.query_text,
                query_category=row.category,
                query_language=row.language,
                mode="article_recommendation",
                prompt_text=prompt_text,
                expected_format=_DEFAULT_EXPECTED_FORMAT,
                metadata=row.metadata,
            )
        )
    return tasks


def render_query_prompt(template: str, *, row: QueryInputRow, top_k_articles: int) -> str:
    """Render the reusable article-retrieval prompt for one query row."""

    context = {
        "query_id": row.query_id,
        "query_text": row.query_text,
        "category": row.category or "",
        "language": row.language or "",
        "top_k": top_k_articles,
        "top_k_articles": top_k_articles,
        **row.metadata,
    }
    prompt = template
    for key, value in context.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    unresolved = sorted(
        {
            match.group(1)
            for match in _PLACEHOLDER_RE.finditer(prompt)
            if match.group(1) not in context
        }
    )
    if unresolved:
        raise ValueError(
            f"Prompt template missing variable for query {row.query_id!r}: {unresolved[0]}"
        )
    rendered = prompt.strip()
    if not rendered:
        raise ValueError(f"Rendered prompt is empty for query {row.query_id!r}")
    return rendered


def load_scenario_prompt_tasks(path: Path) -> list[PromptTask]:
    """Load optional controlled-bias scenarios and convert them to prompt tasks."""

    return [scenario_to_prompt_task(scenario) for scenario in load_prompt_scenarios(path)]


def scenario_to_prompt_task(scenario: PromptScenario) -> PromptTask:
    """Convert a structured scenario definition into a prompt task."""

    return PromptTask(
        source_mode="scenarios",
        query_id=scenario.id,
        query_text=(scenario.topic or scenario.prompt or scenario.id),
        query_category=scenario.category,
        query_language=scenario.language,
        mode=scenario.mode,
        prompt_text=scenario.prompt or "",
        expected_format=scenario.expected_format,
        topic=scenario.topic,
        pair_id=scenario.pair_id,
        variant=scenario.variant,
        control_or_treatment=scenario.control_or_treatment,
        metadata=scenario.metadata,
    )


def load_prompt_scenarios(path: Path) -> list[PromptScenario]:
    """Load prompt scenarios from YAML or JSON."""

    raw = _load_structured_file(path)
    if isinstance(raw, list):
        dataset = PromptDataset(scenarios=raw)
    elif isinstance(raw, dict) and isinstance(raw.get("scenarios"), list):
        dataset = PromptDataset.model_validate(raw)
    else:
        raise ValueError(f"Prompt file {path} must contain a list or a 'scenarios' array")
    if not dataset.scenarios:
        raise ValueError(f"Prompt file {path} did not contain any scenarios")
    return dataset.scenarios


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported prompt file format: {path.suffix}")


def _has_content(row: dict[str, Any]) -> bool:
    return any(str(value).strip() for value in row.values() if value is not None)
