"""Ranking correlation metrics."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _aligned_rank_lists(
    records_a: Sequence[Mapping[str, object]],
    records_b: Sequence[Mapping[str, object]],
    *,
    key: str = "doi",
) -> Tuple[np.ndarray, np.ndarray]:
    lookup_a = {}
    for idx, record in enumerate(records_a, start=1):
        identifier = record.get(key)
        if not identifier:
            continue
        lookup_a[str(identifier).lower()] = float(record.get("rank") or idx)

    lookup_b = {}
    for idx, record in enumerate(records_b, start=1):
        identifier = record.get(key)
        if not identifier:
            continue
        lookup_b[str(identifier).lower()] = float(record.get("rank") or idx)

    shared_ids = sorted(set(lookup_a) & set(lookup_b))
    ranks_a = np.array([lookup_a[identifier] for identifier in shared_ids], dtype=float)
    ranks_b = np.array([lookup_b[identifier] for identifier in shared_ids], dtype=float)
    return ranks_a, ranks_b


def spearman_rho(
    records_a: Sequence[Mapping[str, object]],
    records_b: Sequence[Mapping[str, object]],
    *,
    key: str = "doi",
) -> float:
    """Compute Spearman's rank correlation for the shared identifiers."""

    ranks_a, ranks_b = _aligned_rank_lists(records_a, records_b, key=key)
    if len(ranks_a) == 0:
        return float("nan")
    value, _ = spearmanr(ranks_a, ranks_b)
    return float(value)


def kendall_tau(
    records_a: Sequence[Mapping[str, object]],
    records_b: Sequence[Mapping[str, object]],
    *,
    key: str = "doi",
) -> float:
    """Compute Kendall's tau for the shared identifiers."""

    ranks_a, ranks_b = _aligned_rank_lists(records_a, records_b, key=key)
    if len(ranks_a) == 0:
        return float("nan")
    value, _ = kendalltau(ranks_a, ranks_b)
    return float(value)
