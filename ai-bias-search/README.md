# AI Bias Search

AI Bias Search is a production-ready scaffold for comparing scientific literature retrieval platforms (baseline search engines such as OpenAlex alongside AI-powered systems). The project emphasises reproducibility, bias assessment, and extensibility for new connectors.

## Features

- Pluggable connector architecture with rate limiting, retries, and logging.
- Metadata enrichment via OpenAlex with persistent caching.
- Evaluation layer covering overlap, ranking correlations, and bias metrics.
- CLI pipeline: collect → enrich → eval → report.
- Reproducible local and container workflows via `just` and Docker.

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
- `uv` for local dependency management (or Poetry)
- `just` for task execution
- Docker + Docker Compose (optional; for container runs)

## Quick Start (Local)

1. Install dependencies with `uv`:

   ```bash
   just setup
   source .venv/bin/activate
   ```

2. Create local configuration and environment files:

   ```bash
   cp configs/config.yaml.example config.yaml
   cp .env.example .env
   ```

3. Update `config.yaml` and `queries/queries.csv` for your experiment.

4. Run the full pipeline:

   ```bash
   just pipeline
   ```

## Quick Start (Docker)

1. Build the image:

   ```bash
   just docker-build
   ```

2. Run the pipeline inside the container:

   ```bash
   docker compose run --rm ai-bias-search just pipeline CONFIG=/app/config.yaml
   ```

The container runs as a non-root user and writes to `data/` and `results/` mounted from the project directory.

## CLI Usage (Backwards Compatible)

All existing CLI entrypoints still work:

```bash
python -m ai_bias_search.cli collect --config config.yaml
python -m ai_bias_search.cli enrich --config config.yaml
python -m ai_bias_search.cli eval --config config.yaml
python -m ai_bias_search.cli report --config config.yaml
```

## Configuration & Environment

- Example configs live in `configs/config.yaml.example`.
- Local runs default to `config.yaml` at the repo root.
- Queries live in `queries/queries.csv`.
- Environment variables are loaded from `.env` (see `.env.example`).

Environment variables currently supported:

- `SEMANTIC_SCHOLAR_API_KEY`
- `PERPLEXITY_API_KEY`
- `CONSENSUS_API_KEY`
- `SCITE_API_KEY`
- `LOG_LEVEL` (optional; defaults to `INFO`)
- `CORE_RANKINGS_PATH` (optional; defaults to `CORE.csv`)

## Artifacts & Outputs

Generated data stays under `data/` and `results/` (both gitignored):

- `data/raw/<platform>/<timestamp>.jsonl` — raw platform snapshots
- `data/enriched/<timestamp>.parquet` — merged enrichment output
- `results/metrics/<timestamp>.json` — computed metrics
- `results/reports/<timestamp>.html` — HTML report

## Operational Notes

- Rate limits and retries are configured per platform in `config.yaml`.
- OpenAlex enrichment uses DiskCache under `data/cache/openalex`.
- Clear artifacts and caches with `just clean`.
- Re-run historical snapshots with `--run-timestamp` or `--metrics-timestamp`.

## CORE Rankings Enrichment

Place the CORE conference rankings CSV at `CORE.csv` (default) or set `CORE_RANKINGS_PATH` in `.env` to a custom location. During enrichment, OpenAlex venue metadata is matched against this file to populate `core_rank` (A*, A, B, C), plus `venue_type` and `is_core_listed`. Non-ranked CORE labels are treated as missing. This is intended for CS conference venues; journals and non-conference records may remain empty or be excluded from CORE ranking metrics.

## Quality & Tooling

Run quality gates locally:

```bash
just fmt
just lint
just test
just typecheck
```

Optional pre-commit hooks:

```bash
pre-commit install
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
configs/             # Example configurations
scripts/             # Docker entrypoint wrapper
data/                # (gitignored) raw/enriched caches
results/             # (gitignored) metrics, figs, reports
queries/             # CSV definitions of search topics
tests/               # pytest suite
```

## License

MIT – see [LICENSE](LICENSE).
