# Enrichment

Enrichment transforms raw connector outputs into an analysis-ready dataset and attaches metadata needed for bias-oriented evaluation.

## OpenAlex metadata enrichment

Implementation:
- `ai_bias_search/normalization/openalex_enrich.py`

### Inputs
- A list of record dictionaries (typically from `data/raw/.../*.jsonl`).

### Identifier resolution strategy
The enrichment uses `ai_bias_search.utils.ids.best_identifier` which prioritizes:
1) `doi` (normalized)
2) DOI extracted from `url` when it is a DOI resolver URL
3) `raw_id`

For DOIs, OpenAlex lookup attempts:
- `GET /works/doi:<doi>` (preferred)
- fallback: `GET /works?filter=doi:<doi>`
- last resort: `GET /works?search=<identifier>`

### Caching
- Cache backend: `diskcache.Cache`
- Location: `data/cache/openalex/`
- TTL: 7 days
- Both successful lookups and misses (`None`) are cached.

### Output fields
Enrichment fills or adds:
- `language`, `publication_year`, `cited_by_count`
- `is_oa` (from OpenAlex `open_access.is_oa`)
- `venue_type`, `publisher`, `is_core_listed`
- `host_venue`, `core_rank` (when a valid venue can be resolved and matched)
- `rankings` (per-ranking match results; see below)
- `extra.openalex_enrich` (raw OpenAlex payload)
- `extra.core_lookup` (venue/acronym selection diagnostics)

## Rankings enrichment (CORE/JIF + future lists)

Rankings are handled via a unified, config-driven registry and matching engine.

Implementation:
- `ai_bias_search/rankings/` (configs, loaders, matching engine, registry)
- integrated via `ai_bias_search/normalization/openalex_enrich.py`

Configs and datasets:
- YAML configs live in `ai_bias_search/rankings/sources/*.yaml`
- datasets can live in-repo (e.g., `ai_bias_search/rankings/datasets/core.csv`) or be referenced externally
- CORE dataset path can be overridden with `CORE_RANKINGS_PATH`

Matching precedence (deterministic):
1) ISSN exact match (when ISSNs are available)
2) exact match on normalized title
3) optional fuzzy title match (RapidFuzz when installed)

Output fields:
- `rankings`: `{ranking_id: {matched, value, method, score, evidence}}`
- compatibility fields preserved:
  - CORE: `core_rank` in `{A*, A, B, C}` or `None`
  - JIF/JCR (when enabled): `impact_factor` + `impact_factor_*` + `jcr_*`

Adding a new ranking list:
- add a dataset file + a YAML config in `ai_bias_search/rankings/sources/` (no code changes)

Notes:
- `impact_factor.enabled: true` gates whether the `jif` ranking provider is invoked during enrichment.

## Scopus metadata enrichment (optional)

Implementation:
- `ai_bias_search/enrichment/scopus.py` (also re-exported from `ai_bias_search/normalization/scopus_enrich.py`)

Enabled by:
- `scopus.enabled: true` in YAML config (`scopus_enrich` legacy alias is still supported).

Environment variables:
- `SCOPUS_API_KEY` (or `ELSEVIER_API_KEY`)
- optional `SCOPUS_INSTTOKEN` (or legacy `SCOPUS_INST_TOKEN`; header: `X-ELS-Insttoken`)

Lookup strategy (best-effort):
1) DOI → Abstract Retrieval by DOI (when supported)
2) DOI → Search (`DOI(<doi>)`) → Abstract Retrieval by `scopus_id`
3) Optional title search (only when enabled and title length ≥ `title_search_min_len`)

Optional endpoints (may require additional entitlements):
- Citation Overview API (`enable_citation_overview`)
- Serial Title API (`enable_serial_title_metrics`)
- PlumX Metrics API (`enable_plumx`)

Caching:
- Cache backend: `diskcache.Cache`
- Location: `data/cache/scopus/`
- TTL: controlled by `scopus_enrich.cache_ttl_days` (default 7)
- Both successful lookups and misses (`None`) are cached.

Merge policy:
- Fills only missing fields by default.
- To allow overwriting non-null fields, set `scopus.overwrite_existing: true`.

Failure mode:
- `scopus.fail_open: true` (default): best-effort mode; enrichment errors do not abort pipeline.
- `scopus.fail_open: false`: fail-fast mode; 401/403 and other errors raise immediately.

Fields filled (when present in Scopus responses):
- `scopus_id`, `issn`, `eissn`, `publication_year`, `cited_by_count`, `authors`
- `extra.scopus_enrich` (compact payload: cover date, subtype, keywords, abstract, affiliation countries)
- structured payloads under `scopus` and merge diagnostics under `scopus_meta`

## Scopus journal rankings enrichment (optional)

Implementation:
- `ai_bias_search/enrichment/scopus_rankings.py`

Run path:
- CLI enrich flag: `--enrich-scopus-rankings` (runs after OpenAlex/Scopus enrichment).

Configuration:
- YAML block: `scopus.rankings` (or `scopus_enrich.rankings` legacy alias path).
- Key options:
  - `enabled`
  - `api_key` (usually read from env)
  - `insttoken` (optional)
  - `view_preference` (default `["ENHANCED", "STANDARD"]`)
  - `timeout_s`, `cache_ttl_days`, `rate_limit`

Environment variables:
- `SCOPUS_API_KEY` or `ELSEVIER_API_KEY` (required when enabled)
- optional `SCOPUS_INSTTOKEN`/`SCOPUS_INST_TOKEN`
- optional overrides:
  - `SCOPUS_RANKINGS_API_KEY`
  - `SCOPUS_RANKINGS_INSTTOKEN`
  - `SCOPUS_RANKINGS_VIEW_PREFERENCE`
  - `SCOPUS_RANKINGS_TIMEOUT_S`
  - `SCOPUS_RANKINGS_CACHE_TTL_DAYS`
  - `SCOPUS_RANKINGS_RPS`

API usage:
- Serial Title endpoint by ISSN:
  - `GET /content/serial/title/issn/{issn}`
- View fallback:
  - tries `ENHANCED` first and falls back to `STANDARD`
- ISSN fallback:
  - tries print ISSN first, then other ISSN candidates (including eISSN)

Output fields:
- `record["rankings"]["scopus"]`:
  - `citescore` (`value`, `year`)
  - `citescore_tracker` (`value`, `year`)
  - `sjr` (`value`, `year`)
  - `snip` (`value`, `year`)
  - `series` (`citescore`, `sjr`, `snip` yearly lists)
  - `source_issn_used`
  - `retrieved_at` (UTC ISO8601)

Limitations:
- Availability of `ENHANCED` fields depends on Elsevier entitlement/service-level access.
- Some journals expose partial metrics only; missing values are stored as `None`.
