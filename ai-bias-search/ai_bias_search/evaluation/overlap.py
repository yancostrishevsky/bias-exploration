"""Overlap metrics for retrieved document sets."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence


def jaccard(set_a: Iterable[str], set_b: Iterable[str]) -> float:
    """Compute the Jaccard index between two iterables."""

    a = set(item for item in set_a if item)
    b = set(item for item in set_b if item)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_at_k(
    records_a: Sequence[Mapping[str, object]],
    records_b: Sequence[Mapping[str, object]],
    *,
    k: Optional[int] = None,
    key: str = "doi",
) -> float:
    """Compute Overlap@k between two ranked lists."""

    if k is None:
        k = min(len(records_a), len(records_b))
    top_a = _collect_identifiers(records_a, key, k)
    top_b = _collect_identifiers(records_b, key, k)
    if not top_a and not top_b:
        return 1.0
    if not top_a or not top_b:
        return 0.0
    return len(top_a & top_b) / min(len(top_a), len(top_b))


def _collect_identifiers(records: Sequence[Mapping[str, object]], key: str, k: int) -> set[str]:
    identifiers: set[str] = set()
    for record in records[:k]:
        value = record.get(key)
        if isinstance(value, str) and value:
            identifiers.add(value.lower())
    return identifiers
