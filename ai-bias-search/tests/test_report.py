from __future__ import annotations

from pathlib import Path

import pytest

from ai_bias_search.report import make_report
from ai_bias_search.utils.io import write_parquet


def _write_metrics(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_generate_report_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(make_report, "_generate_plots", lambda *_: [])

    enriched_path = tmp_path / "enriched.parquet"
    write_parquet(
        enriched_path,
        [
            {
                "platform": "openalex",
                "rank": 1,
                "title": "Paper",
                "doi": "10.0/example",
                "impact_factor": 1.23,
                "rankings": {
                    "scopus": {
                        "citescore": {"value": 3.14, "year": 2024},
                        "sjr": {"value": 0.77, "year": 2023},
                        "snip": {"value": 1.02, "year": 2023},
                    }
                },
            }
        ],
    )

    metrics_dir = tmp_path / "metrics"
    _write_metrics(metrics_dir / "20231231T235959Z.json")
    _write_metrics(metrics_dir / "20240101T000000Z.json")

    output_path = tmp_path / "report.html"
    make_report.generate_report(enriched_path, metrics_dir, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "AI Bias exploration Report" in rendered
    assert "Metrics (20240101T000000Z)" in rendered
    assert "Bias Features Availability by Platform" in rendered
    assert "Top-K Bias Summary" in rendered
    assert "CORE match source breakdown" in rendered
    assert "Citations quality diagnostics" in rendered
    assert "Journal Impact Factor (JIF/JCR)" in rendered
    assert "Scopus CiteScore (year)" in rendered
    assert "Scopus SJR (year)" in rendered
    assert "Scopus SNIP (year)" in rendered
    assert "Sample Records" in rendered


def test_generate_report_without_metrics_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_report, "_generate_plots", lambda *_: [])

    enriched_path = tmp_path / "enriched.parquet"
    write_parquet(
        enriched_path,
        [
            {
                "platform": "openalex",
                "rank": 1,
                "title": "Paper",
                "doi": "10.0/example",
            }
        ],
    )

    metrics_dir = tmp_path / "missing-metrics"
    output_path = tmp_path / "report.html"
    make_report.generate_report(enriched_path, metrics_dir, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Journal Impact Factor (JIF/JCR)" in rendered
    assert "Bias Features Availability by Platform" in rendered
    assert "Top-K Bias Summary" in rendered
    assert "CORE match source breakdown" in rendered
    assert "Citations quality diagnostics" in rendered
    assert "No metrics available." in rendered
