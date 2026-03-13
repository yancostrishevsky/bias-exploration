from __future__ import annotations

from ai_bias_search.llm.parser import parse_json_response


def test_parse_json_response_accepts_strict_json() -> None:
    parsed = parse_json_response('{"articles": [{"title": "Paper"}]}')
    assert parsed.success is True
    assert parsed.parse_method == "strict_json"
    assert parsed.parsed_json["articles"][0]["title"] == "Paper"


def test_parse_json_response_accepts_fenced_json() -> None:
    parsed = parse_json_response('```json\n{"articles": [{"title": "Paper"}]}\n```')
    assert parsed.success is True
    assert parsed.parse_method == "fenced_json"


def test_parse_json_response_reports_failure_for_malformed_json() -> None:
    parsed = parse_json_response('{"articles": [}')
    assert parsed.success is False
    assert parsed.parse_error == "json_parse_failed"
