import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ai_bias_search import cli
from ai_bias_search.utils.io import write_parquet


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_eval_serializes_numpy_scalars(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enriched_ts = "20240201T000000Z"
    enriched_path = tmp_path / "data" / "enriched" / f"{enriched_ts}.parquet"
    write_parquet(
        enriched_path,
        [
            {
                "platform": "scopus",
                "rank": 1,
                "doi": "10.1234/example",
                "title": "Paper",
            }
        ],
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
queries_file: queries/queries.csv
platforms: [scopus]
top_k: 10
prompt_template: null
openalex_mailto: null
rate_limit:
  scopus: { rps: 10, burst: 10 }
retries: { max: 1, backoff: 1.0 }
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "_pairwise_metrics", lambda *_: {})
    monkeypatch.setattr(
        cli,
        "compute_bias_metrics",
        lambda *_: {
            "flag": np.bool_(True),
            "count": np.int64(7),
            "score": np.float64(0.5),
            "nested": {"missing": np.nan},
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sanity_checks",
        lambda *_args, **_kwargs: {
            "generated_at": "2026-02-26T00:00:00Z",
            "total_records": np.int64(1),
            "samples": {
                "openalex": [
                    {
                        "raw_record": {
                            "issn": np.array(["1234-5678", "8765-4321"], dtype=object),
                        }
                    }
                ]
            },
            "warnings": [],
        },
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        cli.app,
        ["eval", "--config", str(config_path), "--run-timestamp", enriched_ts],
    )
    assert result.exit_code == 0, result.stdout

    metrics_files = sorted((tmp_path / "results" / "metrics").glob("*.json"))
    assert len(metrics_files) == 1
    payload = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    assert payload["diagnostics_path"] == "diagnostics.json"
    assert payload["biases"]["flag"] is True
    assert payload["biases"]["count"] == 7
    assert payload["biases"]["score"] == 0.5
    assert payload["biases"]["nested"]["missing"] is None

    diagnostics_path = tmp_path / "results" / "diagnostics.json"
    assert diagnostics_path.exists()
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics_payload["samples"]["openalex"][0]["raw_record"]["issn"] == [
        "1234-5678",
        "8765-4321",
    ]
