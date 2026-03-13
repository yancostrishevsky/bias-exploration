# Architecture

This document expands on the pipeline structure described in `README.md` and maps the high-level “collect → enrich → eval → report” stages to concrete modules and file artifacts in the repository.

## Pipeline stages

### 1) Collect

Entry point:
- `ai_bias_search/cli.py` → `collect`

Inputs:
- YAML config (`--config`)
- query CSV (`queries_file` from config)

Processing:
- Reads `queries_file` using `csv.DictReader` (see `ai_bias_search/utils/io.py:load_queries`).
- For each configured platform:
  - instantiates a connector from `ai_bias_search/connectors/__init__.py`
  - calls `connector.search(query=<text>, k=<top_k>, prompt_template=<...>, params=None)`
  - attaches `platform`, `query_id`, `query_text` to each returned record

Outputs:
- `data/raw/<platform>/<timestamp>.jsonl`

### Separate LLM pipeline

Entry points:
- `ai_bias_search/cli.py` → `llm collect`, `llm normalize`, `llm eval`, `llm report`, `llm run`
- Flat aliases remain: `llm-collect`, `llm-normalize`, `llm-eval`, `llm-report`, `llm-run`

Processing:
- `llm-collect` defaults to query-driven collection from `llm.queries.input_csv`, renders
  prompts from `llm.queries.prompt_template_file`, sends them through
  `ai_bias_search/providers/openrouter.py`, and writes raw JSONL outputs.
- Optional `llm.mode: scenarios` uses `llm.controlled_bias_probes.input_file` for paired,
  hand-authored bias probes.
- `llm-normalize` parses raw outputs into typed recommendation/ranking records.
- `llm-eval` optionally enriches recommended articles via OpenAlex and computes
  LLM-specific bias, divergence, hallucination, and stability metrics.
- `llm-report` renders the shared HTML report from one run directory, with LLM-specific sections when present.

Outputs:
- `runs/llm/<timestamp>/raw_responses.jsonl`
- `runs/llm/<timestamp>/normalized_responses.jsonl`
- `runs/llm/<timestamp>/enriched_recommendations.jsonl`
- `runs/llm/<timestamp>/metrics.json`
- `runs/llm/<timestamp>/report.html`
- `runs/llm/<timestamp>/manifest.json`

### 2) Enrich

Entry point:
- `ai_bias_search/cli.py` → `enrich`

Inputs:
- one snapshot per platform (`data/raw/<platform>/<timestamp>.jsonl`) or a specific timestamp via `--run-timestamp`

Processing:
- Concatenates records across platforms.
- Enriches each record with OpenAlex metadata via `ai_bias_search/normalization/openalex_enrich.py`.
- Optionally enriches missing metadata from Elsevier Scopus if `scopus.enabled` (or legacy `scopus_enrich.enabled`) is true in the YAML config.
- Optionally enriches journal impact-factor fields if `impact_factor.enabled` is true in the YAML config.

Outputs:
- `data/enriched/<timestamp>.parquet`
- cache side effects:
  - `data/cache/openalex/` (DiskCache; 7-day TTL on entries)
  - `data/cache/scopus/` (DiskCache; TTL controlled by `scopus.cache_ttl_days`)

### 3) Eval

Entry point:
- `ai_bias_search/cli.py` → `eval`

Inputs:
- one enriched Parquet file (`data/enriched/<timestamp>.parquet`) or a specific timestamp via `--run-timestamp`

Processing:
- Pairwise platform metrics:
  - Jaccard overlap on DOI sets
  - Overlap@k on ranked lists (by `rank`)
  - Rank-Biased Overlap (RBO)
- Bias metrics:
  - recency, language, open access, publisher concentration, CORE ranks, etc.

Outputs:
- `results/metrics/<timestamp>.json`
  - JSON structure: `{ "pairwise": {...}, "biases": {...}, "source": "<enriched_filename>" }`

### 4) Report

Entry point:
- `ai_bias_search/cli.py` → `report`

Inputs:
- one enriched Parquet file (latest or `--enriched-timestamp`)
- metrics directory `results/metrics/` (the report selects the latest metrics file)

Processing:
- Loads the latest `results/metrics/*.json` by timestamp/mtime.
- Renders `ai_bias_search/report/templates/report.html.j2` with:
  - summary counts
  - metrics JSON (when present)
  - a sample table (first 50 rows, excluding `extra`)
  - plots rendered to base64 PNGs (matplotlib; see `ai_bias_search/viz/plots.py`)

Outputs:
- `results/reports/<timestamp>.html`

## Data model overview

The system uses dictionary records rather than a strict schema at the storage boundary, but records are constructed via Pydantic models:
- `ai_bias_search/utils/models.py:Record`
- `ai_bias_search/utils/models.py:EnrichedRecord`

The Parquet dataset is the primary “analysis-ready” artifact for downstream evaluation and reporting.
