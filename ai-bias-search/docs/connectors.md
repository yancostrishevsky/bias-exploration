# Connectors

This document describes the connector API, the built-in connectors, and the repository’s conventions for adding a new platform connector.

## Connector interface

The connector contract is expressed as a Python `Protocol`:
- `ai_bias_search/connectors/base.py:SearchConnector`

Requirements:
- `name: str` — connector identifier used in YAML config (`platforms: [...]`) and file paths (`data/raw/<name>/...`).
- `search(query: str, k: int, prompt_template: str | None = None, params: dict | None = None) -> list[dict]`

Returned records should be normalized into `ai_bias_search.utils.models.Record` and then emitted via `Record.model_dump()` so downstream stages operate on consistent keys (`title`, `doi`, `rank`, `source`, ...).

## Connector registry

Connectors are registered in a static mapping:
- `ai_bias_search/connectors/__init__.py:CONNECTOR_REGISTRY`

The CLI instantiates connectors via:
- `ai_bias_search/cli.py:_instantiate_connector`

Instantiation injects:
- `rate_limiter` (when configured under `rate_limit.<connector_name>`)
- `retries` (from top-level config)
- connector-specific parameters (currently only `openalex_mailto` → OpenAlex connector)

## Built-in connectors

### OpenAlex (`openalex`)
- File: `ai_bias_search/connectors/openalex.py`
- Purpose: baseline scholarly search via OpenAlex “works” endpoint.
- Notes:
  - uses `mailto` when configured (recommended by OpenAlex).
  - extracts DOI from multiple fields; normalizes to lower-case.

### Semantic Scholar (`semanticscholar`)
- File: `ai_bias_search/connectors/semanticscholar.py`
- Purpose: scholarly search via Semantic Scholar Graph API.
- Notes:
  - attempts the bulk search API first; falls back to ranked search.
  - includes optional authentication via `SEMANTIC_SCHOLAR_API_KEY`.

### CORE (`core`)
- File: `ai_bias_search/connectors/core.py`
- Purpose: scholarly search via CORE v3.
- Notes:
  - requires `CORE_API_KEY`.
  - is explicitly configurable via environment variables (base URL, auth header, paging params, GET vs POST).
  - uses retries for transient failures and returns partial results when later pages fail.

### Placeholders (`perplexity`, `consensus`, `scite`)
- Files:
  - `ai_bias_search/connectors/perplexity.py`
  - `ai_bias_search/connectors/consensus.py`
  - `ai_bias_search/connectors/scite.py`
- Status: not implemented; raise `ConnectorError` (with guidance on required API keys).

## Adding a new connector (recommended workflow)

1. Implement the connector in `ai_bias_search/connectors/<name>.py`.
2. Use `httpx` for HTTP requests and accept optional injected dependencies:
   - `rate_limiter: RateLimiter | None`
   - `retries: RetryConfig | None`
   - `client: httpx.Client | None` (useful for testing)
3. Normalize results into `Record`:
   - `Record(title=..., doi=..., rank=..., extra={"<connector>": raw_payload}).model_dump()`
4. Register in `CONNECTOR_REGISTRY`.
5. Add a unit test using `httpx.MockTransport` to avoid network calls (see `tests/test_core_connector.py` and `tests/test_cli_collect.py`).

