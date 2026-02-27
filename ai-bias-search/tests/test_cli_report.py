from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_bias_search import cli
from ai_bias_search.utils.io import write_parquet


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_report_sets_mplconfigdir_to_tmp(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
queries_file: queries/queries.csv
platforms: [openalex]
top_k: 5
prompt_template: null
openalex_mailto: null
rate_limit:
  openalex: { rps: 5, burst: 5 }
retries: { max: 1, backoff: 1.0 }
""",
        encoding="utf-8",
    )

    enriched_path = tmp_path / "data" / "enriched" / "20260101T000000Z.parquet"
    write_parquet(
        enriched_path,
        [{"platform": "openalex", "rank": 1, "title": "Paper", "doi": "10.1/x"}],
    )

    metrics_path = tmp_path / "results" / "metrics" / "20260101T000000Z.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    captured: dict[str, str | None] = {}

    def fake_generate_report(enriched: Path, metrics_dir: Path, output: Path) -> Path:
        captured["mplconfigdir"] = os.getenv("MPLCONFIGDIR")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<html></html>", encoding="utf-8")
        return output

    monkeypatch.setattr(cli, "generate_report", fake_generate_report)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli.app, ["report", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert captured["mplconfigdir"] == "/tmp"
