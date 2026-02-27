from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ai_bias_search.report import make_report
from ai_bias_search.utils.io import write_parquet


def _write_metrics(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_generate_report_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(make_report, "_generate_plots", lambda *_: [])
    monkeypatch.setattr(
        make_report,
        "_load_plotly_js",
        lambda: ("window.Plotly={newPlot:function(){return null;}};", True),
    )

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
    assert "Interactive visualizations" in rendered
    assert "id=\"platform-filter\"" in rendered
    assert "id=\"metric-group-filter\"" in rendered
    assert "id=\"top-k-filter\"" in rendered
    assert "id=\"geo-variant-filter\"" in rendered
    assert "id=\"geo-granularity-filter\"" in rendered
    assert "id=\"chart-geo-top\"" in rendered
    assert "id=\"chart-geo-overunder\"" in rendered
    assert "window.__REPORT_DATA__" in rendered
    assert "Download data" in rendered
    assert "Metrics JSON" in rendered
    assert "Enriched Parquet" in rendered
    assert "Chart aggregates CSV" in rendered
    assert "Bias Features Availability by Platform" in rendered
    assert "Top-K Bias Summary" in rendered
    assert "CORE match source breakdown" in rendered
    assert "Citations quality diagnostics" in rendered
    assert "Journal Impact Factor (JIF/JCR)" in rendered
    assert "Scopus CiteScore (year)" in rendered
    assert "Scopus SJR (year)" in rendered
    assert "Scopus SNIP (year)" in rendered
    assert "Sample Records" in rendered
    chart_csv = output_path.with_name(f"{output_path.stem}_chart_data.csv")
    assert chart_csv.exists()


def test_generate_report_without_metrics_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_report, "_generate_plots", lambda *_: [])
    monkeypatch.setattr(
        make_report,
        "_load_plotly_js",
        lambda: ("window.Plotly={newPlot:function(){return null;}};", True),
    )

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
    assert "Interactive visualizations" in rendered
    assert "id=\"platform-filter\"" in rendered
    assert "Geo Bias" in rendered
    assert "Journal Impact Factor (JIF/JCR)" in rendered
    assert "Bias Features Availability by Platform" in rendered
    assert "Top-K Bias Summary" in rendered
    assert "CORE match source breakdown" in rendered
    assert "Citations quality diagnostics" in rendered
    assert "No metrics available." in rendered


def test_generate_report_uses_plotly_cdn_fallback_when_package_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_report, "_load_plotly_js", lambda: (None, False))
    monkeypatch.setenv("REPORT_PLOTLY_CDN_URL", "https://example.test/plotly.min.js")

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

    metrics_dir = tmp_path / "metrics"
    _write_metrics(metrics_dir / "20240101T000000Z.json")

    output_path = tmp_path / "report.html"
    make_report.generate_report(enriched_path, metrics_dir, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert 'src="https://example.test/plotly.min.js"' in rendered


def test_generate_report_payload_is_valid_json_when_nan_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        make_report,
        "_load_plotly_js",
        lambda: ("window.Plotly={newPlot:function(){return null;}};", True),
    )

    enriched_path = tmp_path / "enriched.parquet"
    write_parquet(
        enriched_path,
        [
            {
                "platform": "openalex",
                "rank": 1,
                "title": "Paper with NaN year",
                "doi": "10.0/example",
                "year": float("nan"),
                "citations": 1,
            }
        ],
    )

    metrics_dir = tmp_path / "metrics"
    _write_metrics(metrics_dir / "20240101T000000Z.json")

    output_path = tmp_path / "report.html"
    make_report.generate_report(enriched_path, metrics_dir, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="interactive-report-data" type="application/json">(.*?)</script>',
        rendered,
        flags=re.S,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict)
