# Development and testing

This repository uses Poetry (`pyproject.toml`) for packaging and a `justfile` to standardize common developer workflows.

## Setup (recommended)

The `justfile` uses `uv` to create and populate a virtual environment:
```bash
just setup
source .venv/bin/activate
```

## Quality gates

```bash
just fmt
just lint
just typecheck
just test
```

## Testing strategy

The test suite is built around `pytest` and avoids real network calls by:
- mocking `httpx.Client` or providing an `httpx.MockTransport`
- setting API-key environment variables with `pytest`’s `monkeypatch`

Key tests to use as references:
- CLI and connector integration: `tests/test_cli_collect.py`
- CORE connector paging/fallback/error-handling: `tests/test_core_connector.py`
- OpenAlex enrichment and retry policy: `tests/test_openalex_enrich.py`, `tests/test_openalex_retry_policy.py`
- Metrics: `tests/test_overlap.py`, `tests/test_ranking.py`, `tests/test_biases.py`

