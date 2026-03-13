from __future__ import annotations

from pathlib import Path

from ai_bias_search.llm.prompts import load_query_prompt_tasks, load_scenario_prompt_tasks


def test_load_query_prompt_tasks_from_csv_and_template(tmp_path: Path) -> None:
    query_csv = tmp_path / "queries.csv"
    query_csv.write_text(
        "query_id,query_text,category,language\nq1,heart failure treatment,cardiology,en\n",
        encoding="utf-8",
    )
    template = tmp_path / "article_retrieval.txt"
    template.write_text("Query: {query_text}\nTop K: {top_k_articles}\n", encoding="utf-8")

    tasks = load_query_prompt_tasks(query_csv, template, top_k_articles=7)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.source_mode == "query_csv"
    assert task.query_id == "q1"
    assert task.query_text == "heart failure treatment"
    assert task.query_category == "cardiology"
    assert task.mode == "article_recommendation"
    assert "Top K: 7" in task.prompt_text


def test_query_prompt_template_allows_literal_json_braces(tmp_path: Path) -> None:
    query_csv = tmp_path / "queries.csv"
    query_csv.write_text(
        "query_id,query_text,category,language\nq1,heart failure treatment,cardiology,en\n",
        encoding="utf-8",
    )
    template = tmp_path / "article_retrieval.txt"
    template.write_text(
        'Task: "{query_text}"\nReturn only JSON:\n{"articles": [{"title": "..."}]}\n',
        encoding="utf-8",
    )

    tasks = load_query_prompt_tasks(query_csv, template, top_k_articles=5)

    assert '{"articles": [{"title": "..."}]}' in tasks[0].prompt_text


def test_load_scenario_prompt_tasks_is_secondary_mode(tmp_path: Path) -> None:
    scenario_file = tmp_path / "scenarios.yaml"
    scenario_file.write_text(
        """
scenarios:
  - id: s1
    category: geography_bias
    mode: article_recommendation
    topic: climate adaptation policy
    prompt: Return JSON.
""",
        encoding="utf-8",
    )

    tasks = load_scenario_prompt_tasks(scenario_file)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.source_mode == "scenarios"
    assert task.query_id == "s1"
    assert task.query_text == "climate adaptation policy"
