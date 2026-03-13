from __future__ import annotations

from pathlib import Path

from ai_bias_search.utils.config import load_config


def test_load_config_with_query_first_llm_block(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    template_path = tmp_path / "article_retrieval.txt"
    template_path.write_text("Query: {query_text} / top {top_k}\n", encoding="utf-8")
    query_csv_path = tmp_path / "queries.csv"
    query_csv_path.write_text(
        "query_id,query_text,category,language\nq1,heart failure treatment,cardiology,en\n",
        encoding="utf-8",
    )

    config_path.write_text(
        f"""
queries_file: {query_csv_path.name}
platforms: []
rate_limit:
  openalex: {{ rps: 2, burst: 5 }}
llm:
  enabled: true
  provider: openrouter
  mode: query_csv
  models: [openai/gpt-4o-mini, openai/gpt-4o-mini, anthropic/claude-3.5-sonnet]
  generation:
    temperature: 0.3
    max_tokens: 900
    top_p: 0.8
    timeout_seconds: 45
    repeats_per_query: 2
    top_k_articles: 12
  queries:
    input_csv: {query_csv_path.name}
    prompt_template_file: {template_path.name}
  output_dir: runs/llm
  parsing:
    require_json: true
  enrichment:
    enabled: false
  save_payloads: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.llm.enabled is True
    assert config.llm.provider == "openrouter"
    assert config.llm.mode == "query_csv"
    assert config.llm.models == ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
    assert config.llm.generation.max_tokens == 900
    assert config.llm.generation.repeats_per_query == 2
    assert config.llm.generation.top_k_articles == 12
    assert config.llm.queries.input_csv == Path(query_csv_path.name)
    assert config.llm.queries.prompt_template_file == Path(template_path.name)
    assert config.llm.output_dir == Path("runs/llm")
    assert config.llm.enrichment.enabled is False


def test_load_config_maps_legacy_llm_prompt_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text("scenarios: []\n", encoding="utf-8")
    query_csv_path = tmp_path / "queries.csv"
    query_csv_path.write_text(
        "query_id,query_text,category,language\nq1,example,test,en\n",
        encoding="utf-8",
    )

    config_path.write_text(
        f"""
queries_file: {query_csv_path.name}
platforms: []
rate_limit:
  openalex: {{ rps: 2, burst: 5 }}
llm:
  enabled: true
  provider: openrouter
  models: [openai/gpt-4o-mini]
  generation:
    max_output_tokens: 777
    repeats_per_prompt: 3
  prompts:
    input_file: {prompts_path.name}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.llm.generation.max_tokens == 777
    assert config.llm.generation.repeats_per_query == 3
    assert config.llm.mode == "scenarios"
    assert config.llm.controlled_bias_probes.input_file == Path(prompts_path.name)
