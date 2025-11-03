"""Input/output helpers for the AI Bias Search project."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pandas as pd


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

    ensure_directory(path)
    frame = pd.DataFrame(list(records))
    frame.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file as a pandas DataFrame."""

    return pd.read_parquet(path)


def load_queries(csv_path: Path) -> List[Dict[str, str]]:
    """Load search queries from a CSV file."""

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def utc_timestamp() -> str:
    """Return a compact UTC timestamp string suitable for filenames."""

    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
