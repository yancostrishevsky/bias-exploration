# AI Bias Search (`ai_bias_search`)

`ai_bias_search` is a research codebase for **controlled comparisons of scholarly literature search platforms**. It provides an end-to-end pipeline to (1) collect ranked results from multiple “connectors”, (2) enrich records with OpenAlex metadata (with caching), (3) compute overlap/ranking/bias-oriented metrics, and (4) generate a self-contained HTML report suitable for reproducible experiments.

This repository is intended as a master’s-thesis-quality artifact: readable, testable, and designed around the practical constraints of working with external search APIs (rate limits, retries, partial failures, and metadata incompleteness).

Optional extended documentation:
- `docs/architecture.md`
- `docs/connectors.md`
- `docs/enrichment.md`
- `docs/evaluation.md`
- `docs/development.md`

---

## 1. Project Overview

### What problem the project solves
Scientific search platforms (traditional indices and AI-assisted search tools) can return materially different sets of papers for the same query. These differences affect downstream evidence synthesis and can introduce systematic distortions (e.g., language, recency, open-access visibility, publisher concentration).

### Research motivation (bias, reproducibility, AI search)
The project supports **reproducible retrieval experiments** by:
- recording raw connector outputs as timestamped snapshots,
- enriching records into a single normalized dataset,
- computing interpretable, platform-comparable metrics, and
- producing a single HTML artifact that can be archived with a thesis or paper.

### High-level system design
The pipeline is modular:
- **Connectors** isolate platform-specific API logic behind a shared interface.
- **Enrichment** normalizes identifiers and attaches OpenAlex metadata, optional CORE conference ranks, and optional journal impact-factor data.
- **Evaluation** computes overlap/ranking similarity and bias-oriented summaries from the enriched dataset.
- **Reporting** renders a static HTML report embedding plots as base64 PNGs.

---

## 2. Architecture

### Pipeline stages (collect → enrich → eval → report)

```
queries.csv
  │
  ▼
collect  ──▶  data/raw/<platform>/<timestamp>.jsonl
  │
  ▼
enrich   ──▶  data/enriched/<timestamp>.parquet      (OpenAlex + caching + optional enrichments)
  │
  ▼
eval     ──▶  results/metrics/<timestamp>.json       (pairwise + bias metrics)
  │
  ▼
report   ──▶  results/reports/<timestamp>.html       (self-contained report)
```

### Data flow between stages
- `collect` reads `queries_file` from the YAML config and calls `connector.search(...)` per platform per query.
- Raw results are stored as JSON Lines, preserving connector-provided fields and adding:
  - `platform`, `query_id`, `query_text`.
- `enrich` reads the latest (or specified) raw snapshots across all configured platforms, merges them into a single record list, and enriches each record using OpenAlex metadata (cached in `data/cache/openalex`).
- `eval` reads one enriched Parquet file and computes:
  - pairwise platform overlap/ranking similarity metrics, and
  - bias-oriented metrics (recency, language, open access, publisher concentration, etc.).
- `report` reads one enriched Parquet file and the **latest metrics JSON** in `results/metrics/`, then renders `ai_bias_search/report/templates/report.html.j2`.

### Key abstractions and interfaces
- **Connector interface**: `ai_bias_search.connectors.base.SearchConnector` (Protocol)
  - required: `name: str`, `search(query: str, k: int, ...) -> list[dict]`
- **Canonical record models** (used by connectors/enrichment):
  - `ai_bias_search.utils.models.Record`
  - `ai_bias_search.utils.models.EnrichedRecord`

---

## 3. Connectors

### What a connector is
A connector is a thin adapter from a platform-specific search API to a common record schema (`Record.model_dump()`), enabling consistent downstream enrichment and evaluation.

### How connectors are discovered/registered
Connectors are registered in a static mapping:
- `ai_bias_search/connectors/__init__.py` (`CONNECTOR_REGISTRY`)

`ai_bias_search.cli` uses `get_connector(name)` and instantiates the connector with optional:
- `rate_limiter` (token bucket; `ai_bias_search.utils.rate_limit.RateLimiter`)
- `retries` (`ai_bias_search.utils.config.RetryConfig`)

### Implemented connectors

#### `openalex` — OpenAlex works search
- Implementation: `ai_bias_search/connectors/openalex.py`
- Endpoint: `GET https://api.openalex.org/works?search=<query>&per_page=<k>`
- Normalization:
  - DOI is normalized via `ai_bias_search.utils.ids.normalise_doi` (lower-cased).
  - Falls back to extracting a DOI from DOI-resolver URLs when possible.
- Retries:
  - tenacity retries on `httpx.HTTPError` up to `retries.max` with exponential backoff (`retries.backoff`).
- Rate limiting:
  - default `2 rps` with burst `5` (overridable via YAML `rate_limit.openalex`).
- Configuration:
  - `openalex_mailto` (passed as `mailto` query parameter when provided).

#### `semanticscholar` — Semantic Scholar Graph API search
- Implementation: `ai_bias_search/connectors/semanticscholar.py`
- Endpoints:
  - primary: `/graph/v1/paper/search/bulk` (token-pagination)
  - fallback: `/graph/v1/paper/search` (ranked, single-page)
- Authentication:
  - uses `SEMANTIC_SCHOLAR_API_KEY` if present (sent as `x-api-key`).
- Rate limiting:
  - default `1 rps` with burst `2` (overridable via YAML `rate_limit.semanticscholar`).
- Retries and 429 handling:
  - sleeps on `Retry-After` when present, then retries via tenacity.
- User-Agent:
  - configurable via `AI_BIAS_USER_AGENT` (default `ai-bias-search/0.1 (+contact@example.com)`).

#### `core` — CORE v3 works search
- Implementation: `ai_bias_search/connectors/core.py`
- Status: implemented; designed to be resilient to API variants and partial failures.
- Requires:
  - `CORE_API_KEY` (connector fails fast if missing).
- Key environment variables (all optional unless noted):
  - `CORE_API_BASE_URL` (default `https://api.core.ac.uk/v3`)
  - `CORE_SEARCH_PATH` (default `/search/works`)
  - `CORE_SEARCH_METHOD` (`AUTO`, `GET`, or `POST`; default `AUTO`)
  - `CORE_QUERY_PARAM`, `CORE_LIMIT_PARAM`, `CORE_OFFSET_PARAM`
  - `CORE_MAX_PAGE_SIZE` (default `25`)
  - `CORE_AUTH_HEADER` (default `Authorization`)
  - `CORE_AUTH_PREFIX` (default `Bearer`)
  - `AI_BIAS_USER_AGENT` (default `ai-bias-search/0.1 (+contact@example.com)`)
- Pagination:
  - uses `limit`/`offset` style paging and de-duplicates within a query run.
- Error handling:
  - retries transient HTTP failures (timeouts, 429/5xx, etc.) and certain 200-OK payloads that still indicate backend overload.
  - returns partial results when later pages fail, rather than failing the whole run.

### Placeholder connectors (scaffolding)
These connectors exist as explicit placeholders and raise `ConnectorError`:
- `perplexity` (`PERPLEXITY_API_KEY` gate; not implemented)
- `consensus` (`CONSENSUS_API_KEY` gate; not implemented)
- `scite` (`SCITE_API_KEY` gate; not implemented)

---

## 4. Enrichment Layer

### Purpose of enrichment
The enrichment layer normalizes records across platforms and attaches metadata needed for bias-oriented analysis (e.g., language, open access, publication year, citations, venue, publisher).

### OpenAlex enrichment (with persistent caching)
- Implementation: `ai_bias_search/normalization/openalex_enrich.py`
- Cache backend: `diskcache.Cache` stored in `data/cache/openalex/`
  - both hits and misses are cached for 7 days to avoid repeated failed lookups.
- Identifier resolution:
  - uses `ai_bias_search.utils.ids.best_identifier` (DOI → DOI-from-URL → `raw_id`)
  - for DOIs, prefers direct `GET /works/doi:<doi>`; falls back to `GET /works?filter=doi:<doi>` and finally to `GET /works?search=<identifier>`.
- Retry policy:
  - 404/410 are treated as non-retryable in DOI-direct resolution.
  - 429/5xx and network timeouts are treated as retryable (tenacity exponential backoff).

### Venue normalization and CORE conference rankings
- CORE ranking lookup: `ai_bias_search/utils/core_rankings.py`
  - default file: `CORE.csv` at repo root (gitignored; you must provide it)
  - override: `CORE_RANKINGS_PATH=/path/to/CORE.csv`
  - output field: `core_rank` in `{A*, A, B, C}` or `None`
- Enrichment uses OpenAlex venue fields plus heuristics to extract a conference acronym where possible (e.g., from Semantic Scholar DBLP hints) and filters out obvious false positives (e.g., Roman numerals).

### Journal impact factor enrichment (optional)
- Implementation: `ai_bias_search/utils/impact_factor.py` + integration in `openalex_enrich.py`
- Controlled by YAML `impact_factor.enabled`.
- Input is an XLSX file (default `data/vendor/jif.xlsx`; gitignored), parsed with `openpyxl`.
- Matching strategy (in order):
  1) ISSN/eISSN exact match (if present)
  2) normalized title exact match
  3) optional fuzzy title match (requires `impact_factor.allow_fuzzy: true`)
- Enriched fields include:
  - `impact_factor`, `impact_factor_year`, `impact_factor_match`
  - and additional `jcr_*` fields (publisher, quartile, JCI, etc.) when present in the XLSX.

---

## 5. Evaluation Layer

### Implemented evaluation metrics
Computed in `ai_bias_search/cli.py` using:
- Jaccard overlap on identifier sets: `ai_bias_search.evaluation.overlap.jaccard`
- Overlap@k on ranked lists: `ai_bias_search.evaluation.overlap.overlap_at_k`
- Rank-Biased Overlap (RBO): `ai_bias_search.evaluation.ranking_similarity.rbo`

Pairwise metrics are computed for every platform pair and stored under:
- `pairwise.<platformA>_vs_<platformB>`

### Bias-related measurements (implemented in code)
Computed in `ai_bias_search.evaluation.biases.compute_bias_metrics`:
- **Recency**: median year and age distribution.
- **Metadata completeness**: coverage of DOI/year/language/publisher/open-access.
- **Language distribution**: normalized counts by language.
- **Open-access share**: mean of `is_oa`.
- **CORE ranking**: shares of A*/A/B/C for eligible records (conference or missing venue type).
- **Publisher concentration**: Herfindahl–Hirschman Index (HHI) over publishers.
- **Rank vs citations**: Spearman correlation between `rank` and `cited_by_count`.

Metrics are computed “overall” and additionally “by_platform” when the `platform` column is present.

---

## 6. Configuration

### YAML config file (`config.yaml`)
Configuration is loaded via Pydantic (`ai_bias_search.utils.config.load_config`). Unknown keys are allowed (they are parsed but not used unless referenced in code).

Required keys (minimal):
```yaml
queries_file: queries/queries.csv
platforms: [openalex]
top_k: 25
prompt_template: null
openalex_mailto: "you@example.com"
rate_limit:
  openalex: { rps: 2, burst: 5 }
retries:
  max: 3
  backoff: 1.5
```

See `configs/config.yaml.example` for the full template (including `impact_factor`).

### Query file format
`queries_file` is read with `csv.DictReader` and must include a header row. The pipeline uses:
- `text` (required for meaningful runs)
- `query_id` (optional but strongly recommended)

Example:
```csv
query_id,text
q1,"information retrieval conference paper"
```

### Environment variables
Environment variables are loaded from `.env` at runtime (`python-dotenv`). See `.env.example`.

Common variables:
- `LOG_LEVEL` (default `INFO`)
- `SEMANTIC_SCHOLAR_API_KEY`
- `CORE_API_KEY`
- `CORE_RANKINGS_PATH` (default `CORE.csv`)
- `CORE_SEARCH_METHOD` (`AUTO`/`GET`/`POST`, default `AUTO`)
- `AI_BIAS_USER_AGENT` (used by `core` and `semanticscholar`)

---

## 7. Command Line Interface (CLI)

The CLI is implemented with Typer in `ai_bias_search/cli.py`. The package also installs an entrypoint:
- `ai-bias-search` → `ai_bias_search.cli:main`

### Available commands
- `collect --config <path>`: fetch raw results and write JSONL snapshots under `data/raw/`
- `enrich --config <path> [--run-timestamp <collect_ts>]`: enrich latest/specified raw snapshots and write Parquet under `data/enriched/`
- `eval --config <path> [--run-timestamp <enriched_ts>]`: compute metrics from latest/specified Parquet and write JSON under `results/metrics/`
- `report --config <path> [--enriched-timestamp <enriched_ts>]`: generate HTML under `results/reports/`

Notes:
- `report` always loads the **latest metrics JSON** present in `results/metrics/`. The CLI option `--metrics-timestamp` is currently accepted but not used by the implementation.

### Example workflows

Local, explicit stage-by-stage execution:
```bash
cp configs/config.yaml.example config.yaml
cp .env.example .env

python -m ai_bias_search.cli collect --config config.yaml
python -m ai_bias_search.cli enrich  --config config.yaml
python -m ai_bias_search.cli eval    --config config.yaml
python -m ai_bias_search.cli report  --config config.yaml
```

Task runner shortcuts:
```bash
just setup
just pipeline
```

Docker:
```bash
just docker-build
docker compose run --rm ai-bias-search just pipeline CONFIG=/app/config.yaml
```

---

## 8. Outputs and Reproducibility

### Output directory structure
- Raw snapshots (per platform):
  - `data/raw/<platform>/<timestamp>.jsonl`
- Enriched dataset:
  - `data/enriched/<timestamp>.parquet`
- Metrics JSON:
  - `results/metrics/<timestamp>.json`
- HTML report:
  - `results/reports/<timestamp>.html`
- OpenAlex enrichment cache:
  - `data/cache/openalex/`

Timestamps are UTC in the format `%Y%m%dT%H%M%SZ` (see `ai_bias_search.utils.io.utc_timestamp`).

### Determinism, caching, and repeatability
This project interacts with external APIs; results may vary over time due to index updates and platform-side changes. The codebase supports reproducibility through:
- timestamped raw snapshots,
- explicit rate limiting and retry policies, and
- cached OpenAlex enrichment (including cached misses).

For experiment archiving, store at minimum:
- the exact `config.yaml` and `queries.csv`,
- the generated `data/raw/...` snapshot files,
- the enriched Parquet file referenced by the metrics JSON (`metrics["source"]`),
- and the HTML report.

---

## 9. Extending the Project

### How to add a new connector
1. Create a new module in `ai_bias_search/connectors/` implementing:
   - `name: str`
   - `search(query: str, k: int, ...) -> list[dict]`
2. Normalize outputs into `ai_bias_search.utils.models.Record` and return `record.model_dump()`.
3. Register the connector in `ai_bias_search/connectors/__init__.py` (`CONNECTOR_REGISTRY`).
4. Add config entries under `rate_limit.<name>` and document required API keys in `.env.example`.
5. Add tests using `httpx.MockTransport` (see `tests/test_cli_collect.py`).

### How to add a new enrichment step
Enrichment is currently performed in `ai_bias_search.normalization.openalex_enrich.enrich_with_openalex`. To add additional enrichment:
- extend `EnrichedRecord` in `ai_bias_search/utils/models.py` with new fields, and/or
- augment records within `enrich_with_openalex(...)`, or add a new enrichment module and call it from `ai_bias_search/cli.py:enrich`.

### How to add new evaluation metrics
1. Implement metric functions in `ai_bias_search/evaluation/`.
2. Integrate them into `ai_bias_search/cli.py:eval` and/or `ai_bias_search.evaluation.biases.compute_bias_metrics`.
3. If the metric should appear in the HTML report, update `ai_bias_search/report/make_report.py` and/or the Jinja template.

---

## 10. Testing and Development

### Local setup
The repo supports Poetry (`pyproject.toml`) and includes a `justfile` using `uv`:
```bash
just setup
source .venv/bin/activate
```

### How to run tests
```bash
just test
```

### Mocking external APIs
Tests avoid network calls and use:
- `httpx.MockTransport` to intercept HTTP requests
- `pytest` `monkeypatch` to set environment variables and replace clients

See:
- `tests/test_cli_collect.py`
- `tests/test_openalex_enrich.py`
- `tests/test_core_connector.py`

---

## 11. Limitations and Design Decisions

- **Per-query vs aggregated evaluation**: current evaluation computes pairwise metrics and bias metrics over the *entire enriched dataset*, not per query. If `queries.csv` contains multiple queries, the `rank` field (which is per-query) is reused across queries, and the evaluation logic does not group by `query_id`.
- **Identifier dependence**: overlap/ranking metrics use `doi` by default. Missing/invalid DOIs reduce the effective overlap signal.
- **Connector coverage**: `perplexity`, `consensus`, and `scite` are placeholders and are not implemented.
- **Report metrics selection**: `report` selects the latest `results/metrics/*.json` and ignores `--metrics-timestamp` in the current implementation.
- **External-service dependence**: OpenAlex/CORE/Semantic Scholar availability and policy changes can affect runs; rate limits and retries mitigate but do not eliminate this.

---

## License

MIT — see `LICENSE`.
