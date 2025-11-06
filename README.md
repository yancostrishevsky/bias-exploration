# AI Bias Search

AI Bias Search is a production-ready scaffold for benchmarking how different literature retrieval platforms (classical and AI-powered) respond to the same set of research questions. The project focuses on reproducibility, bias analysis, and extensibility so that you can plug in new data sources or evaluation logic with minimal work.

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────────┐
│ queries  │ →  │ connectors   │ →  │ enrichment│ →  │ evaluation  │
└──────────┘    └──────────────┘    └───────────┘    └────┬────────┘
                                                         │
                                                         ↓
                                                   ┌──────────┐
                                                   │ reporting│
                                                   └──────────┘
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Walkthrough](#pipeline-walkthrough)
3. [Configuration Reference](#configuration-reference)
4. [Data Artifacts & Schemas](#data-artifacts--schemas)
5. [Report Structure & Column Guide](#report-structure--column-guide)
6. [Extending the Project](#extending-the-project)
7. [Testing & Quality Gates](#testing--quality-gates)
8. [Troubleshooting & Tips](#troubleshooting--tips)

---

## Quick Start

### Prerequisites

- Python 3.11+
- Recommended package manager: [uv](https://github.com/astral-sh/uv) or Poetry

### Bootstrap the environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .          # or: poetry install

cp .env.example .env         # configure API keys and secrets
cp config.yaml.example config.yaml
```

Fill `queries/queries.csv` with your search topics and update `config.yaml` with the platforms you want to benchmark.

### Run the full pipeline

```bash
python -m ai_bias_search.cli collect --config config.yaml
python -m ai_bias_search.cli enrich  --config config.yaml
python -m ai_bias_search.cli eval    --config config.yaml
python -m ai_bias_search.cli report  --config config.yaml
```

Each command accepts `--run-timestamp` to re-use historical artifacts (see [Pipeline Walkthrough](#pipeline-walkthrough)).

---

## Pipeline Walkthrough

### 1. `collect`

- Loads queries from the CSV configured in `config.yaml`.
- For each platform (OpenAlex, Semantic Scholar, etc.) the CLI instantiates a connector (`ai_bias_search/connectors/`) with rate limiting and retry policies (`ai_bias_search/cli.py:47`).
- Calls `connector.search(...)` and writes results to `data/raw/{platform}/{UTC_TIMESTAMP}.jsonl`.
- Raw records include the original payload plus contextual fields (`platform`, `query_id`, `query_text`).

### 2. `enrich`

- Reads the latest (or specified) JSONL snapshots from each platform.
- Uses `normalization/openalex_enrich.py` to fetch additional metadata from OpenAlex:
  - Preferred language, open-access status, referenced venue/publisher, citation counts, etc.
  - Requests are cached via DiskCache to avoid hitting API limits on repeated runs.
- Produces a single Parquet dataset at `data/enriched/{UTC_TIMESTAMP}.parquet` that merges results from all platforms.

### 3. `eval`

- Loads the Parquet dataset and splits data by platform.
- Computes two groups of metrics:
  1. **Pairwise overlap and ranking correlations** (`evaluation/overlap.py`, `evaluation/ranking.py`).
  2. **Bias-oriented metrics** (`evaluation/biases.py`), such as language distribution or recency.
- Writes `results/metrics/{UTC_TIMESTAMP}.json` with the metric payload and a pointer to the source enrichment file.

### 4. `report`

- Renders an HTML report using Jinja2 templates (`report/templates/report.html.j2`).
- Combines:
  - Summary counts (total records, platforms included).
  - Metrics JSON (all files under `results/metrics/`).
  - A preview table with the first 20 enriched records (configurable in the template or `make_report.py:37`).
- Saves `results/reports/{UTC_TIMESTAMP}.html` for convenient sharing.

---

## Configuration Reference

### `config.yaml`

| Field              | Description                                                                                  | Required |
|--------------------|----------------------------------------------------------------------------------------------|----------|
| `queries_file`     | Path to the CSV with query definitions (can be relative to the config file).                 | yes       |
| `platforms`        | List of connectors to execute (must exist in `CONNECTOR_REGISTRY`).                          | yes       |
| `top_k`            | Number of records to request per platform and query.                                         | yes       |
| `prompt_template`  | Optional prompt text passed to AI-based connectors (if supported).                           | no       |
| `openalex_mailto`  | Email appended to OpenAlex requests to improve rate limits.                                  | no       |
| `rate_limit`       | Per-platform rate limiting configuration (`rps`, `burst`) used when instantiating connectors.| yes       |
| `retries`          | Backoff configuration for HTTP retries (`max`, `backoff`).                                   | yes       |

Tip: Keep a dedicated `config.yaml` per experiment so you can re-run with `--run-timestamp`.

### `.env`

Environment variables enable external APIs. Examples:

- `PERPLEXITY_API_KEY`, `CONSENSUS_API_KEY`, `SCITE_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`
- Custom keys for your own connectors

Use `dotenv` to load them automatically (`ai_bias_search/cli.py:38`).

### `queries/queries.csv`

Minimum columns:

| Column     | Purpose                                   |
|------------|-------------------------------------------|
| `query_id` | Stable identifier used in outputs.        |
| `text`     | The actual search query passed to APIs.   |

Additional columns are preserved, allowing you to store language hints, domains, etc.

---

## Data Artifacts & Schemas

### Raw results (`data/raw/{platform}/{timestamp}.jsonl`)

Each line is a JSON object that follows the `Record` schema (`ai_bias_search/utils/models.py:14`):

| Field          | Description                                                                          |
|----------------|--------------------------------------------------------------------------------------|
| `title`        | Title returned by the platform.                                                      |
| `doi`          | DOI normalised to lowercase if provided.                                             |
| `url`          | Landing page or platform-specific URL.                                               |
| `rank`         | Position in the platform result list (1-based).                                      |
| `raw_id`       | Native identifier (e.g. OpenAlex ID, Semantic Scholar paperId).                      |
| `source`       | Venue or collection label from the platform.                                         |
| `year`         | Publication year reported by the platform (if any).                                  |
| `authors`      | List of author strings.                                                              |
| `extra`        | Connector-specific payload retained for reference.                                   |
| `platform`     | Added during collection: name of the connector that produced the record.             |
| `query_id`     | Identifier from `queries.csv`.                                                        |
| `query_text`   | Original query text.                                                                 |

### Enriched dataset (`data/enriched/{timestamp}.parquet`)

Produced by OpenAlex enrichment. Extends the raw schema with:

| Field                  | Description                                                                       |
|------------------------|-----------------------------------------------------------------------------------|
| `language`             | Detected language code from OpenAlex metadata.                                    |
| `is_oa`                | Boolean indicating open-access availability.                                      |
| `publication_year`     | Publication year resolved by OpenAlex (may differ from raw `year`).               |
| `host_venue`           | Venue name from OpenAlex.                                                          |
| `publisher`            | Publisher name.                                                                   |
| `cited_by_count`       | Total citation count from OpenAlex.                                               |
| `extra.openalex_enrich`| Full OpenAlex API response cached for reproducibility.                            |

Access the data with:

```python
import pandas as pd
frame = pd.read_parquet("data/enriched/<timestamp>.parquet")
```

### Metrics (`results/metrics/{timestamp}.json`)

Structure:

```json
{
  "pairwise": {
    "openalex_vs_semanticscholar": {
      "jaccard": 0.12,
      "overlap_at_k": 0.28,
      "spearman": 0.41,
      "kendall": 0.35
    }
  },
  "biases": {
    "recency": {"median_year": 2022.5, "share_last_12_months": 0.2},
    "language": {"en": 0.8, "es": 0.2},
    "open_access": {"share_open_access": 0.6},
    "publisher_hhi": {"hhi": 0.23},
    "rank_vs_citations": {"spearman": -0.12}
  },
  "source": "20240201T120000Z.parquet"
}
```

### Reports (`results/reports/{timestamp}.html`)

Self-contained HTML with embedded table and metrics JSON.

---

## Report Structure & Column Guide

`ai_bias_search/report/make_report.py` loads the enriched dataset and metrics directory, then renders `report/templates/report.html.j2`. The default report contains three sections:

1. **Summary** – total record count and list of platforms present.
2. **Metrics** – each JSON file in `results/metrics/` appears as a pretty-printed block.
3. **Sample Records** – the first 20 rows of the enriched dataset (edit `.head(20)` in `make_report.py` to change this).

The sample table includes the following columns (depending on availability):

| Column                   | Source                          | Meaning                                                             |
|--------------------------|----------------------------------|---------------------------------------------------------------------|
| `platform`               | Collect step                     | Connector that returned the record.                                 |
| `query_id`               | Collect step                     | ID of the originating query.                                        |
| `query_text`             | Collect step                     | Text shown to the platform.                                         |
| `rank`                   | Collect step                     | Ranking returned by the platform (1 = highest).                     |
| `title`                  | Collect step                     | Returned document title.                                            |
| `doi`                    | Collect & enrichment             | DOI string; normalised if possible.                                 |
| `url`                    | Collect step                     | Landing page.                                                       |
| `source` / `host_venue`  | Collect & enrichment             | Venue names from platform / OpenAlex respectively.                  |
| `year` / `publication_year` | Collect & enrichment         | Reported vs OpenAlex-resolved publication year.                     |
| `language`               | Enrichment                       | ISO language code from OpenAlex.                                    |
| `is_oa`                  | Enrichment                       | Boolean flag for open access.                                       |
| `publisher`              | Enrichment                       | Publisher from OpenAlex.                                            |
| `cited_by_count`         | Enrichment                       | Total citations in OpenAlex.                                        |
| `authors`                | Collect step                     | List of author names.                                               |

To expose the full dataset in the report, remove or adjust `.head(20)` in `make_report.py`.

---

## Extending the Project

### Add a New Search Platform

1. **Implement a connector class** in `ai_bias_search/connectors/` that exposes:

   ```python
   class MyConnector:
       name = "myplatform"

       def __init__(self, *, rate_limiter=None, retries=None, **kwargs):
           ...

       def search(self, query: str, k: int, prompt_template: str | None = None, params: dict | None = None) -> list[dict]:
           ...
   ```

   Reuse `RateLimiter` and `RetryConfig` to match existing behaviour (see `connectors/openalex.py:28` for reference).

2. **Register the connector** inside `CONNECTOR_REGISTRY` (`connectors/__init__.py`) so the CLI can instantiate it.

3. **Update `config.yaml`** to include the new platform and rate limit values.

4. **Write tests** (e.g. with [Respx](https://lundberg.github.io/respx/)) under `tests/` to mock API responses and verify JSONL output structure (`tests/test_cli_collect.py` is a good blueprint).

5. **Document required environment variables** in `.env.example` if your connector needs API keys.

### Add or Modify Bias Metrics

1. Extend `evaluation/biases.py` with your metric (e.g. new helper function plus an entry in `compute_bias_metrics`).
2. If the metric requires additional columns, ensure the enrichment step populates them (modify `enrich_with_openalex` or build a dedicated enrichment function).
3. Update or add tests to cover the metric logic.
4. Optional: expose the metric in the HTML report template (e.g. custom formatting or charts).

### Extend Pairwise Metrics

Add new functions to `evaluation/overlap.py` or `evaluation/ranking.py`, then update `_pairwise_metrics` in `cli.py` to include them in the metrics JSON payload.

### Customise Reporting & Visualisation

- Modify `report/templates/report.html.j2` to change layout, add charts, or embed external assets.
- Generate additional plots via `viz/plots.py` or `viz/upset.py`. Call these utilities from the CLI or a custom script and store the results under `results/`.
- If you want to expose the full dataset in HTML, remove or adjust `.head(20)` in `make_report.py`.

---

## Testing & Quality Gates

Recommended commands:

```bash
pytest                # unit tests covering connectors and metric logic
ruff check .          # linting
black --check .       # formatting
mypy .                # optional static type checks
```

Tests rely on mocked HTTP calls so they do not require live API keys.

---

## Troubleshooting & Tips

- **API throttling**: set `openalex_mailto` and appropriate rate limits to stay within platform quotas. DiskCache ensures we do not re-fetch metadata unnecessarily.
- **Re-running experiments**: supply `--run-timestamp` to `collect`, `enrich`, `eval`, or `report` to regenerate downstream artifacts for a specific snapshot.
- **Inspecting cache**: enrichment metadata is stored in `data/cache/openalex`. Clear it if you need to refresh stale entries.
- **Data validation**: inspect `data/enriched/*.parquet` directly with pandas when debugging metrics.

---

## Project Structure

```
ai_bias_search/
  cli.py              # Typer-powered CLI pipeline
  connectors/         # API clients and stubs
  normalization/      # OpenAlex enrichment logic
  evaluation/         # Overlap, ranking, and bias metrics
  report/             # HTML report generation & templates
  utils/              # Configuration, IO helpers, rate limiting, models
  viz/                # Optional plotting utilities
data/                 # (gitignored) raw/enriched datasets, caches
results/              # (gitignored) metrics, figures, HTML reports
queries/              # CSV definitions of search topics
tests/                # pytest suite
```

---

## License

MIT – see [LICENSE](LICENSE).

