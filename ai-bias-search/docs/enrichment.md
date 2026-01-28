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
- `extra.openalex_enrich` (raw OpenAlex payload)
- `extra.core_lookup` (venue/acronym selection diagnostics)

## CORE conference ranking enrichment

Implementation:
- `ai_bias_search/utils/core_rankings.py` (lookup)
- integrated via `openalex_enrich.py`

Input:
- `CORE.csv` at repo root by default; override with `CORE_RANKINGS_PATH`.

Output:
- `core_rank` in `{A*, A, B, C}` or `None` when missing/unranked.

Notes:
- Enrichment uses heuristics to extract conference acronyms and filters out common false positives (e.g., Roman numerals).
- CORE ranking metrics treat conference and unknown venue types as “eligible” (see `ai_bias_search/evaluation/biases.py`).

## Journal Impact Factor enrichment (optional)

Implementation:
- `ai_bias_search/utils/impact_factor.py`

Enabled by:
- `impact_factor.enabled: true` in YAML config.

Input:
- An XLSX file (default `data/vendor/jif.xlsx`).

Matching precedence:
1) ISSN/eISSN exact match (if present)
2) exact match on normalized title
3) optional fuzzy match on normalized title (`impact_factor.allow_fuzzy`)

The enrichment attaches `impact_factor` and several `jcr_*` fields when present in the source XLSX.

