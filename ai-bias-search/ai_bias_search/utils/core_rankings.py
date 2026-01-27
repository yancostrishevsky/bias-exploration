"""CORE conference ranking lookup helpers."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()

DEFAULT_CORE_PATH = Path(__file__).resolve().parents[2] / "CORE.csv"

ALLOWED_RANKS = {"A*", "A", "B", "C"}
RANK_ORDER = {"A*": 4, "A": 3, "B": 2, "C": 1, None: 0}
HEADER_TOKENS = {"title", "acronym", "abbr", "shortname", "rank", "name"}

PARENS_RE = re.compile(r"\([^)]*\)")
PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")

try:  # optional fuzzy matching
    from rapidfuzz import fuzz, process  # type: ignore

    RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None  # type: ignore[assignment]
    process = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CoreRankings:
    """Lookup tables for CORE rankings."""

    acronym_map: Dict[str, Optional[str]]
    title_map: Dict[str, Optional[str]]
    acronym_keys: Tuple[str, ...]
    title_keys: Tuple[str, ...]


_CORE_CACHE: CoreRankings | None = None
_CORE_PATH_OVERRIDE: Path | None = None


def set_core_rankings_path(path: Path | None) -> None:
    """Override the default CORE rankings path and reset the cache."""

    global _CORE_PATH_OVERRIDE, _CORE_CACHE
    _CORE_PATH_OVERRIDE = path
    _CORE_CACHE = None


def clear_core_rankings_cache() -> None:
    """Clear the in-memory CORE rankings cache."""

    global _CORE_CACHE
    _CORE_CACHE = None


def lookup_core_rank(venue_name: str | None, venue_acronym: str | None = None) -> str | None:
    """Lookup the CORE rank for a venue name/acronym (A*, A, B, C)."""
    LOGGER.debug("RAPIDFUZZ_AVAILABLE=%r", RAPIDFUZZ_AVAILABLE)

    rankings = _load_core_rankings()
    if not rankings.title_map and not rankings.acronym_map:
        return None

    if venue_acronym:
        acronym_key = normalize_text(venue_acronym)
        if acronym_key and acronym_key in rankings.acronym_map:
            return rankings.acronym_map[acronym_key]

    if venue_name:
        title_key = normalize_text(venue_name)
        if title_key and title_key in rankings.title_map:
            return rankings.title_map[title_key]

    query, choices = _select_fuzzy_inputs(venue_name, venue_acronym, rankings)
    if not query or not choices:
        return None

    matched = _fuzzy_match(query, choices)
    if matched is None:
        return None
    match_key, score = matched
    LOGGER.debug(
        "CORE fuzzy match for %r -> %r (score=%.1f)",
        venue_name or venue_acronym,
        match_key,
        score,
    )
    if venue_name:
        return rankings.title_map.get(match_key)
    return rankings.acronym_map.get(match_key)


def is_known_core_acronym(value: str | None) -> bool:
    """Return True when *value* appears as an acronym in the CORE index."""

    if not value:
        return False
    rankings = _load_core_rankings()
    if not rankings.acronym_map:
        return False
    key = normalize_text(value)
    return bool(key and key in rankings.acronym_map)


def normalize_text(value: str | None, *, strip_parens: bool = True) -> str:
    """Normalize text for matching CORE titles/acronyms."""

    if not value:
        return ""
    text = value.strip().lower()
    if strip_parens:
        text = PARENS_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def _select_fuzzy_inputs(
    venue_name: str | None,
    venue_acronym: str | None,
    rankings: CoreRankings,
) -> tuple[str | None, Iterable[str]]:
    if venue_name:
        query = normalize_text(venue_name)
        return query, rankings.title_keys
    if venue_acronym:
        query = normalize_text(venue_acronym)
        return query, rankings.acronym_keys
    return None, ()


def _fuzzy_match(query: str, choices: Iterable[str]) -> tuple[str, float] | None:
    if not RAPIDFUZZ_AVAILABLE:
        return None
    choices_list = list(choices)
    if not query or not choices_list:
        return None
    matches = process.extract(query, choices_list, scorer=fuzz.ratio, limit=2)
    if not matches:
        return None
    best, score, _ = matches[0]
    if score < 95:
        return None
    if len(matches) > 1 and matches[1][1] == score:
        return None
    return str(best), float(score)


def _load_core_rankings() -> CoreRankings:
    global _CORE_CACHE
    if _CORE_CACHE is not None:
        return _CORE_CACHE

    path = _resolve_core_path()
    if not path.exists():
        LOGGER.warning("CORE rankings file not found at %s; core ranks disabled", path)
        _CORE_CACHE = CoreRankings({}, {}, (), ())
        return _CORE_CACHE

    acronym_map: Dict[str, Optional[str]] = {}
    title_map: Dict[str, Optional[str]] = {}
    total_rows = 0
    ranked_rows = 0

    def process_row(row: list[str]) -> None:
        nonlocal total_rows, ranked_rows
        if not row or len(row) < 5:
            return
        if not any(cell.strip() for cell in row):
            return
        total_rows += 1
        title = row[1].strip() if len(row) > 1 else ""
        acronym = row[2].strip() if len(row) > 2 else ""
        rank = _normalize_rank(row[4] if len(row) > 4 else "")
        if rank is not None:
            ranked_rows += 1
        title_key = normalize_text(title)
        acronym_key = normalize_text(acronym)
        _update_map(title_map, title_key, rank)
        _update_map(acronym_map, acronym_key, rank)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is not None:
            if not _is_header_row(first):
                process_row(first)
        for row in reader:
            process_row(row)

    _CORE_CACHE = CoreRankings(
        acronym_map=acronym_map,
        title_map=title_map,
        acronym_keys=tuple(acronym_map.keys()),
        title_keys=tuple(title_map.keys()),
    )
    LOGGER.info(
        "Loaded CORE rankings from %s (rows=%d, titles=%d, acronyms=%d, ranked_rows=%d)",
        path,
        total_rows,
        len(title_map),
        len(acronym_map),
        ranked_rows,
    )

    LOGGER.debug("Sample CORE acronym keys: %r", list(acronym_map.keys())[:20])
    LOGGER.debug("Sample CORE title keys: %r", list(title_map.keys())[:5])

    return _CORE_CACHE


def _resolve_core_path() -> Path:
    if _CORE_PATH_OVERRIDE is not None:
        return _CORE_PATH_OVERRIDE
    env_path = os.getenv("CORE_RANKINGS_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_CORE_PATH


def _normalize_rank(value: str | None) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().upper().replace(" ", "")
    if cleaned in ALLOWED_RANKS:
        return cleaned
    return None


def _update_map(target: Dict[str, Optional[str]], key: str, rank: Optional[str]) -> None:
    if not key:
        return
    current = target.get(key)
    if key not in target or _rank_value(rank) > _rank_value(current):
        target[key] = rank


def _rank_value(rank: Optional[str]) -> int:
    return RANK_ORDER.get(rank, 0)


def _is_header_row(row: list[str]) -> bool:
    hits = 0
    for cell in row:
        cell_value = cell.strip().lower()
        if not cell_value:
            continue
        tokens = set(re.sub(r"[^a-z]", " ", cell_value).split())
        if tokens & HEADER_TOKENS:
            hits += 1
    return hits >= 2
