# Evaluation

This document describes the metrics computed by `ai_bias_search` and the assumptions implied by the current implementation.

## Pairwise platform metrics

Computed in:
- `ai_bias_search/cli.py:_pairwise_metrics`

For every platform pair `(A, B)`, the pipeline computes:

### Jaccard overlap
Implementation:
- `ai_bias_search/evaluation/overlap.py:jaccard`

Definition:
- Let `S_A` and `S_B` be the sets of non-empty identifiers (default: DOIs).  
  `J(A,B) = |S_A ∩ S_B| / |S_A ∪ S_B|`

### Overlap@k
Implementation:
- `ai_bias_search/evaluation/overlap.py:overlap_at_k`

Definition:
- Let `TopA(k)` and `TopB(k)` be the sets of identifiers in the first `k` records of each ranked list.  
  `Overlap@k = |TopA(k) ∩ TopB(k)| / min(|TopA(k)|, |TopB(k)|)`

### Rank-Biased Overlap (RBO)
Implementation:
- `ai_bias_search/evaluation/ranking_similarity.py:rbo`

Notes:
- RBO is top-weighted and remains well-defined for different list lengths and small intersections.
- The implementation uses a default `p = 0.9` and compares at depth `k` (default: min list length).

## Bias-oriented metrics

Computed in:
- `ai_bias_search/evaluation/biases.py:compute_bias_metrics`

The bias metric suite includes:
- Recency metrics over `publication_year` (preferred) or `year`
- Metadata completeness (coverage ratios)
- Language distribution over `language`
- Open-access share over `is_oa`
- CORE ranking shares over `core_rank` (for eligible records)
- Publisher concentration (HHI) over `publisher`
- Spearman correlation between `rank` and `cited_by_count`

## Important implementation assumptions

- Metrics are computed over the entire enriched dataset (and per-platform subsets), not per query.
- The `rank` field is produced by connectors as a per-query rank (1..k). When multiple queries are present, the current evaluation logic does not group by `query_id`.
- Overlap/ranking similarity defaults to DOI matching; missing DOIs reduce measured overlap.

