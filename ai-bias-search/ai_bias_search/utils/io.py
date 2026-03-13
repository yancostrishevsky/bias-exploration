"""Input/output helpers for the AI Bias Search project."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from ai_bias_search.utils.logging import configure_logging

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

LOGGER = configure_logging()


def ensure_directory(path: Path) -> None:
    """Ensure the parent directory of *path* exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to a JSON Lines file."""

    ensure_directory(path)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_parquet(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Persist *records* to Parquet using pandas."""

    import pandas as pd

    ensure_directory(path)
    frame = pd.DataFrame(list(records))
    frame = _sanitize_empty_dict_columns(frame)
    try:
        frame.to_parquet(path, index=False)
    except Exception as exc:
        LOGGER.warning(
            "Parquet engine unavailable (%s). Falling back to pickle for %s",
            exc,
            path,
        )
        frame.to_pickle(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file as a pandas DataFrame."""

    import pandas as pd

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning(
            "Parquet engine unavailable while reading %s (%s). Falling back to pickle.",
            path,
            exc,
        )
        return pd.read_pickle(path)


def _sanitize_empty_dict_columns(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Replace columns composed only of empty dict values with nulls.

    PyArrow cannot persist an object column inferred as `struct<>` (struct with
    no children). This happens when a column contains only `{}` values.
    """

    if frame.empty:
        return frame

    sanitized = frame.copy()
    for column in sanitized.columns:
        series = sanitized[column]
        non_null = series[series.notna()]
        if non_null.empty:
            continue
        if not non_null.map(lambda value: isinstance(value, dict)).all():
            continue
        if non_null.map(lambda value: len(value) == 0).all():
            sanitized[column] = None
    return sanitized


def load_queries(csv_path: Path) -> List[Dict[str, str]]:
    """Load search queries from a CSV file."""

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def utc_timestamp() -> str:
    """Return a compact UTC timestamp string suitable for filenames."""

    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
