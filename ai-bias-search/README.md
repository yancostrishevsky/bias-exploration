# AI Bias Search

AI Bias Search is a production-ready scaffold for comparing scientific literature retrieval platforms (baseline search engines such as OpenAlex alongside AI-powered systems). The project emphasises reproducibility, bias assessment, and extensibility for new connectors.

## Features

- Pluggable connector architecture with rate limiting, retries, and logging.
- Metadata enrichment via OpenAlex with persistent caching.
- Evaluation layer covering overlap, ranking correlations, and bias metrics.
- CLI pipeline: collect → enrich → eval → report.
- Built-in tests, linters (ruff, black), and typing support.

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

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or Poetry for dependency management.

## Installation

1. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

   (Alternatively: `poetry install`.)

2. Copy environment defaults and update secrets:

   ```bash
   cp .env.example .env
   ```

3. Review `config.yaml.example` and adjust settings (API keys, rate limits, `top_k`, etc.).

4. Populate `queries/queries.csv` with your search topics (sample provided with 10 multilingual entries).

## CLI Usage

From the project root run:

```bash
python -m ai_bias_search.cli collect --config config.yaml.example
python -m ai_bias_search.cli enrich --config config.yaml.example
python -m ai_bias_search.cli eval --config config.yaml.example
python -m ai_bias_search.cli report --config config.yaml.example
```

Command overview:

- `collect`: Calls configured connectors for each query and persists raw JSONL snapshots per platform under `data/raw/{platform}/`.
- `enrich`: Resolves OpenAlex metadata for collected records and saves `data/enriched/{timestamp}.parquet`.
- `eval`: Computes overlap, ranking, and bias metrics → `results/metrics/{timestamp}.json`.
- `report`: Builds a self-contained HTML report mixing metrics and sample records → `results/reports/{timestamp}.html`.

Each command supports optional `--run-timestamp` switches to reprocess previous snapshots.

## Testing & Quality

Run the test suite and optional tooling:

```bash
pytest
ruff check .
black --check .
```

## Project Structure

```
ai_bias_search/
  connectors/        # API clients and stubs
  normalization/     # OpenAlex enrichment
  evaluation/        # Metrics for overlap, ranking, biases
  utils/             # IO, config, logging, rate limiting
  viz/               # Plot utilities
  report/            # HTML report generator
data/                # (gitignored) raw/enriched caches
results/             # (gitignored) metrics, figs, reports
queries/             # CSV definitions of search topics
tests/               # pytest suite
```

## Example End-to-End Run

```
uv venv && uv pip install -e .
cp .env.example .env
python -m ai_bias_search.cli collect --config config.yaml.example
python -m ai_bias_search.cli enrich --config config.yaml.example
python -m ai_bias_search.cli eval --config config.yaml.example
python -m ai_bias_search.cli report --config config.yaml.example
```

The workflow will produce JSONL snapshots in `data/raw/`, enriched Parquet files in `data/enriched/`, metrics JSON in `results/metrics/`, and a final HTML report in `results/reports/`.

## License

MIT – see [LICENSE](LICENSE).
