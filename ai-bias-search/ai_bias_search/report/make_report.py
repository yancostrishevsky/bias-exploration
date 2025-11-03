"""HTML report generation for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai_bias_search.utils.io import read_parquet
from ai_bias_search.utils.logging import configure_logging


LOGGER = configure_logging()


def generate_report(enriched_path: Path, metrics_dir: Path, output_path: Path) -> Path:
    """Render an HTML report combining enriched data and metrics."""

    LOGGER.info("Generating report from %s", enriched_path)
    data = read_parquet(enriched_path)
    metrics = _load_metrics(metrics_dir)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")

    summary = {
        "total_records": len(data),
        "platforms": data.get("platform").dropna().unique().tolist() if "platform" in data.columns else [],
    }

    rendered = template.render(summary=summary, metrics=metrics, table=data.head(20).to_dict(orient="records"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)
    return output_path


def _load_metrics(metrics_dir: Path) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    if not metrics_dir.exists():
        LOGGER.warning("Metrics directory %s missing", metrics_dir)
        return results
    for path in metrics_dir.glob("*.json"):
        try:
            results[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse metrics file %s: %s", path, exc)
    return results
