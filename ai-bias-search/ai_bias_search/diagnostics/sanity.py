"""Sanity checks for metadata quality diagnostics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

from ai_bias_search.normalize.records import normalize_records


def _platform_summary(records: list[dict[str, Any]], platform: str) -> dict[str, Any]:
    total = len(records)
    issn_missing = sum(1 for item in records if not item.get("issn"))
    publisher_missing = sum(1 for item in records if not item.get("publisher"))
    citations_known = [item for item in records if item.get("citations") is not None]
    citations_zero = [item for item in citations_known if item.get("citations") == 0]
    quality = Counter(
        str((item.get("metrics_quality") or {}).get("citations") or "missing") for item in records
    )

    return {
        "platform": platform,
        "total": total,
        "issn_missing_rate": (issn_missing / total) if total else None,
        "publisher_missing_rate": (publisher_missing / total) if total else None,
        "citations_missing_rate": (
            (total - len(citations_known)) / total if total else None
        ),
        "citations_zero_rate": (
            len(citations_zero) / len(citations_known) if citations_known else None
        ),
        "citations_quality": dict(quality),
    }


def run_sanity_checks(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Run platform-level sanity checks for key metadata fields."""

    normalized = normalize_records(list(records))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        platform = str(item.get("platform") or "unknown")
        grouped.setdefault(platform, []).append(item)

    by_platform: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for platform, items in sorted(grouped.items()):
        summary = _platform_summary(items, platform)
        by_platform[platform] = summary

        if (summary.get("issn_missing_rate") or 0.0) > 0.7:
            warnings.append(
                f"{platform}: high ISSN missing rate ({summary['issn_missing_rate']:.1%})"
            )
        if (summary.get("publisher_missing_rate") or 0.0) > 0.7:
            warnings.append(
                f"{platform}: high publisher missing rate ({summary['publisher_missing_rate']:.1%})"
            )
        quality = summary.get("citations_quality") or {}
        suspicious = int(quality.get("suspicious", 0))
        if suspicious > 0:
            warnings.append(
                f"{platform}: suspicious citations detected ({suspicious} records adjusted)"
            )

    return {
        "generated_at": datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "total_records": len(normalized),
        "by_platform": by_platform,
        "warnings": warnings,
    }
