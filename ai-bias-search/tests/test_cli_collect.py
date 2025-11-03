import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ai_bias_search.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@respx.mock
def test_cli_collect_creates_jsonl(tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    queries_path = tmp_path / "queries.csv"
    queries_path.write_text("query_id,language,domain,text\nq1,en,test,example query\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
queries_file: queries.csv
platforms: [openalex, semanticscholar]
top_k: 2
prompt_template: null
openalex_mailto: null
rate_limit:
  openalex: { rps: 10, burst: 10 }
  semanticscholar: { rps: 10, burst: 10 }
retries: { max: 1, backoff: 1.0 }
""",
        encoding="utf-8",
    )

    openalex_route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "Paper One",
                        "doi": "10.0/one",
                        "id": "https://openalex.org/W1",
                        "host_venue": {"display_name": "Journal"},
                        "publication_year": 2023,
                        "authorships": [{"author": {"display_name": "Author"}}],
                    },
                    {
                        "display_name": "Paper Two",
                        "doi": "10.0/two",
                        "id": "https://openalex.org/W2",
                        "host_venue": {"display_name": "Journal"},
                        "publication_year": 2022,
                        "authorships": [],
                    },
                ]
            },
        )
    )

    semanticscholar_route = respx.post("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "title": "Paper A",
                        "paperId": "S1",
                        "url": "https://example.com/a",
                        "year": 2023,
                        "venue": "Venue",
                        "externalIds": {"DOI": "10.0/a"},
                        "authors": [{"name": "Author"}],
                    },
                    {
                        "title": "Paper B",
                        "paperId": "S2",
                        "url": "https://example.com/b",
                        "year": 2022,
                        "venue": "Venue",
                        "externalIds": {"DOI": "10.0/b"},
                        "authors": [{"name": "Author"}],
                    },
                ]
            },
        )
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["collect", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert openalex_route.called
    assert semanticscholar_route.called

    openalex_dir = tmp_path / "data" / "raw" / "openalex"
    semanticscholar_dir = tmp_path / "data" / "raw" / "semanticscholar"
    openalex_files = list(openalex_dir.glob("*.jsonl"))
    sem_files = list(semanticscholar_dir.glob("*.jsonl"))
    assert len(openalex_files) == 1
    assert len(sem_files) == 1

    with openalex_files[0].open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == 2
    assert [record["rank"] for record in lines] == [1, 2]
