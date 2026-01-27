"""Rank similarity metrics based on Rank-Biased Overlap (RBO).

RBO emphasises agreement near the top of ranked lists and stays well-defined
even when the lists are of different lengths or share few items, unlike
Spearman or Kendall correlations that only consider the intersection.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Set


def _normalize_identifiers(records: Sequence[Mapping[str, object]], key: str) -> Sequence[str]:
    identifiers = []
    for record in records:
        identifier = record.get(key)
        if identifier is None:
            continue
        if isinstance(identifier, str):
            identifier = identifier.strip()
            if not identifier:
                continue
            normalized = identifier.lower()
        else:
            normalized = str(identifier)
            if not normalized:
                continue
        identifiers.append(normalized)
    return identifiers


def rbo(
    records_a: Sequence[Mapping[str, object]],
    records_b: Sequence[Mapping[str, object]],
    *,
    k: Optional[int] = None,
    p: float = 0.9,
    key: str = "doi",
) -> float:
    """Compute extrapolated RBO for two ranked lists.

    Args:
        records_a: Ranked records for list A.
        records_b: Ranked records for list B.
        k: Evaluation depth. If None, compare to min(len(a), len(b)).
        p: Top-weightedness parameter (0 < p < 1); higher values discount deep ranks less.
        key: Mapping key used to extract identifiers (lower-cased for strings).
    """

    if not 0 < p < 1:
        raise ValueError("p must be in the open interval (0, 1)")

    ids_a = _normalize_identifiers(records_a, key)
    ids_b = _normalize_identifiers(records_b, key)

    depth_limit = min(len(records_a), len(records_b)) if k is None else k
    if depth_limit == 0:
        return float("nan")

    evaluation_depth = min(depth_limit, len(ids_a), len(ids_b))
    if evaluation_depth == 0:
        return float("nan")

    seen_a: Set[str] = set()
    seen_b: Set[str] = set()
    overlap = 0
    cumulative = 0.0

    for depth in range(1, evaluation_depth + 1):
        id_a = ids_a[depth - 1]
        id_b = ids_b[depth - 1]

        if id_a not in seen_a:
            seen_a.add(id_a)
            if id_a in seen_b:
                overlap += 1
        if id_b not in seen_b:
            seen_b.add(id_b)
            if id_b in seen_a:
                overlap += 1

        agreement = overlap / depth
        cumulative += (1 - p) * agreement * (p ** (depth - 1))

    agreement_at_k = overlap / evaluation_depth
    return cumulative + agreement_at_k * (p**evaluation_depth)
