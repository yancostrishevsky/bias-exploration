from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from ai_bias_search.cli import app
from ai_bias_search.llm.schemas import EnrichedRecommendationRecord, ProviderResponse, TokenUsage


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class FakeProviderClient:
    def build_payload(self, request: object) -> dict[str, object]:
        return {"model": getattr(request, "model", None)}

    def complete(self, request: object) -> ProviderResponse:
        model = str(request.model)
        body = {
            "articles": [
                {
                    "rank": 1,
                    "title": f"{model} paper",
                    "doi": f"10.1000/{model.split('/')[-1]}",
                    "year": 2024,
                    "journal": "Journal of Tests",
                    "authors": ["Alice"],
                    "rationale": "Relevant",
                }
            ]
        }
        return ProviderResponse(
            provider="openrouter",
            model=model,
            output_text=json.dumps(body),
            raw_response={"choices": [{"message": {"content": json.dumps(body)}}]},
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
            latency_ms=25,
        )

    def close(self) -> None:
        return None


def _write_config(tmp_path: Path, *, llm_enabled: bool) -> Path:
    scholarly_queries_path = tmp_path / "queries.csv"
    scholarly_queries_path.write_text(
        "query_id,language,domain,text\nq1,en,test,example query\n", encoding="utf-8"
    )
    llm_queries_path = tmp_path / "llm_queries.csv"
    llm_queries_path.write_text(
        "query_id,query_text,category,language\nq1,heart failure treatment,cardiology,en\n",
        encoding="utf-8",
    )
    template_path = tmp_path / "article_retrieval.txt"
    template_path.write_text(
        """You are helping with scientific literature retrieval.

Task:
Return the top {top_k_articles} most relevant scholarly articles for the query: \"{query_text}\".

Requirements:
- Prefer real peer-reviewed scientific literature
- Prioritize relevance to the query
- Include DOI if known
- If DOI is uncertain, return null instead of inventing one
- Do not invent bibliographic metadata
- Return only JSON
- No markdown

Return JSON in this format:
{
  \"articles\": [
    {
      \"rank\": 1,
      \"title\": \"...\",
      \"doi\": \"10....\" or null,
      \"year\": 2024 or null,
      \"journal\": \"...\",
      \"authors\": [\"...\"],
      \"rationale\": \"short reason\"
    }
  ]
}
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
queries_file: {scholarly_queries_path.name}
platforms: [openalex]
top_k: 2
prompt_template: null
openalex_mailto: null
rate_limit:
  openalex: {{ rps: 10, burst: 10 }}
retries: {{ max: 1, backoff: 1.0 }}
llm:
  enabled: {str(llm_enabled).lower()}
  provider: openrouter
  mode: query_csv
  models: [openai/gpt-4o-mini]
  generation:
    temperature: 0.2
    max_tokens: 500
    top_p: 1.0
    timeout_seconds: 30
    repeats_per_query: 1
    top_k_articles: 5
  queries:
    input_csv: {llm_queries_path.name}
    prompt_template_file: {template_path.name}
  output_dir: runs/llm
  parsing:
    require_json: true
  enrichment:
    enabled: true
  save_payloads: true
""",
        encoding="utf-8",
    )
    return config_path


def test_collect_does_not_create_llm_runs_when_using_regular_collect(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, llm_enabled=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
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
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.Client

    def client_factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs = dict(kwargs)
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["collect", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "data" / "raw" / "openalex").exists()
    assert not (tmp_path / "runs" / "llm").exists()


def test_grouped_llm_run_writes_query_centric_artifacts(
    tmp_path: Path,
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, llm_enabled=True)

    def fake_enrich(*args: object, **kwargs: object) -> list[EnrichedRecommendationRecord]:
        normalized = args[0]
        out: list[EnrichedRecommendationRecord] = []
        for record in normalized:
            item = record.article_recommendations[0]
            out.append(
                EnrichedRecommendationRecord(
                    run_id=record.run_id,
                    request_id=record.request_id,
                    source_mode=record.source_mode,
                    query_id=record.query_id,
                    query_text=record.query_text,
                    query_category=record.query_category,
                    query_language=record.query_language,
                    model=record.model,
                    provider=record.provider,
                    repeat_index=record.repeat_index,
                    recommended_rank=item.rank,
                    llm_claimed_title=item.title,
                    llm_claimed_doi=item.doi,
                    llm_claimed_year=item.year,
                    llm_claimed_journal=item.journal,
                    llm_claimed_authors=item.authors,
                    rationale=item.rationale,
                    parse_status=record.parse_status,
                    enriched_title=item.title,
                    enriched_doi=item.doi,
                    enriched_year=item.year,
                    enriched_journal=item.journal,
                    openalex_id="W1",
                    openalex_match_found=True,
                    valid_doi=True,
                    is_oa=True,
                    cited_by_count=42,
                    publisher="Publisher A",
                    country_primary="US",
                )
            )
        return out

    monkeypatch.setattr(
        "ai_bias_search.llm.pipeline._build_provider_client", lambda config: FakeProviderClient()
    )
    monkeypatch.setattr("ai_bias_search.llm.pipeline.enrich_recommendations", fake_enrich)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["llm", "run", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout

    run_root = tmp_path / "runs" / "llm"
    run_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "raw_responses.jsonl").exists()
    assert (run_dir / "normalized_responses.jsonl").exists()
    assert (run_dir / "enriched_recommendations.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "manifest.json").exists()

    raw_rows = [
        json.loads(line)
        for line in (run_dir / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert raw_rows[0]["provider"] == "openrouter"
    assert raw_rows[0]["query_id"] == "q1"
    assert raw_rows[0]["query_text"] == "heart failure treatment"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overview"]["call_count"] == 1
    assert metrics["overview"]["query_count"] == 1
    assert metrics["overview"]["parse_success_rate"] == 1.0
    report_html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Interactive visualizations" in report_html
    assert "LLM Hallucination / Verifiability" in report_html
    assert "LLM Parse Robustness" in report_html
    assert "LLM Query Details" in report_html
