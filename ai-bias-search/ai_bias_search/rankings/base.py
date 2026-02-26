"""Shared models, provider protocol, and normalization helpers for ranking lists."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

DatasetFormat = Literal["csv", "tsv", "xlsx"]


class TitleNormalizationConfig(BaseModel):
    """Title normalization settings shared across ranking providers."""

    strip_parens: bool = True
    replace_ampersand: bool = True
    remove_punct: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = True


class RankingConfig(BaseModel):
    """Configuration describing how to load and interpret a ranking dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)

    enabled: bool = True

    dataset_path: Path
    dataset_path_env: str | None = None
    format: DatasetFormat = "csv"
    encoding: str = "utf-8"
    delimiter: str | None = None
    has_header: bool | Literal["auto"] = True

    sheet_name: str | None = None  # for xlsx

    edition: str | None = None
    default_rank_year: int | None = Field(default=None, ge=1800, le=2100)

    # Dataset column mappings. Values can be a column name (str) or a 0-based column index (int).
    fields: Dict[str, str | int] = Field(default_factory=dict)
    extra_fields: Dict[str, str | int] = Field(default_factory=dict)
    title_alias_fields: list[str | int] = Field(default_factory=list)

    normalization: TitleNormalizationConfig = Field(default_factory=TitleNormalizationConfig)
    validate_issn_checksum: bool = False

    rank_value_allowlist: list[str] | None = None
    rank_value_type: Literal["auto", "str", "float", "int"] = "auto"

    allow_fuzzy: bool = True
    fuzzy_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    fuzzy_top_k: int = Field(default=5, ge=1, le=25)
    reject_ambiguous_fuzzy: bool = True
    fuzzy_ambiguity_delta: float = Field(default=0.02, ge=0.0, le=1.0)

    cache_ttl_days: int = Field(default=30, ge=1)

    config_path: Path | None = Field(default=None, exclude=True)

    @field_validator("fields")
    @classmethod
    def _validate_fields(cls, value: Dict[str, str | int]) -> Dict[str, str | int]:
        required = {"title", "rank_value"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"Missing required fields mapping: {sorted(missing)}")
        return value


@dataclass(frozen=True, slots=True)
class RankingEntry:
    venue_key: str
    title: str
    title_norm: str
    issn_print: str | None
    issn_online: str | None
    issn_l: str | None
    rank_value: object | None
    rank_year: int | None
    source_id: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchResult:
    source_id: str
    rank_value: object | None
    method: str
    score: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    matched: bool = False


class RankingProvider(Protocol):
    id: str
    label: str

    def load(self) -> None: ...

    def match(self, venue_title: str | None, issn_list: list[str] | None) -> MatchResult: ...

    def stats(self) -> dict[str, Any]: ...


ISSN_CLEAN_RE = re.compile(r"[^0-9X]")
ISSN_NORM_RE = re.compile(r"^(?P<a>[0-9]{4})-(?P<b>[0-9]{3}[0-9X])$")

PARENS_RE = re.compile(r"\([^)]*\)")
PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")


def validate_issn_checksum(value: str) -> bool:
    """Validate the ISSN checksum (ISO 3297).

    Accepts normalised ISSN keys in the `NNNN-NNNN` form where the last char may be `X`.
    """

    text = value.strip().upper()
    match = ISSN_NORM_RE.fullmatch(text)
    if not match:
        return False

    digits = (match.group("a") + match.group("b")).upper()
    total = 0
    for weight, char in zip(range(8, 1, -1), digits[:7]):
        total += int(char) * weight

    remainder = total % 11
    check = 11 - remainder
    if check == 10:
        expected = "X"
    elif check == 11:
        expected = "0"
    else:
        expected = str(check)

    return digits[-1] == expected


def normalize_issn(value: str | None, *, validate_checksum: bool = False) -> str | None:
    """Normalise ISSN/eISSN to a stable `NNNN-NNNN` key."""

    if value is None:
        return None
    raw_text = str(value).strip().upper()
    if not raw_text:
        return None

    cleaned = ISSN_CLEAN_RE.sub("", raw_text)
    if len(cleaned) == 7:
        cleaned = f"0{cleaned}"
    if len(cleaned) != 8:
        return None

    normalized = f"{cleaned[:4]}-{cleaned[4:]}"
    if validate_checksum and not validate_issn_checksum(normalized):
        return None
    return normalized


def normalize_title(value: str | None, cfg: TitleNormalizationConfig | None = None) -> str:
    """Normalise a venue/journal title for matching."""

    if not value:
        return ""
    config = cfg or TitleNormalizationConfig()
    text = str(value).strip()
    if not text:
        return ""
    if config.lowercase:
        text = text.lower()
    if config.replace_ampersand:
        text = text.replace("&", " and ")
    if config.strip_parens:
        text = PARENS_RE.sub(" ", text)
    text = text.replace("_", " ")
    if config.remove_punct:
        text = PUNCT_RE.sub(" ", text)
    if config.collapse_whitespace:
        text = SPACE_RE.sub(" ", text)
    return text.strip()


def iter_normalized_issns(values: Iterable[str | None], *, validate_checksum: bool) -> Sequence[str]:
    """Normalise and de-duplicate ISSN candidates."""

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        key = normalize_issn(value, validate_checksum=validate_checksum)
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized
