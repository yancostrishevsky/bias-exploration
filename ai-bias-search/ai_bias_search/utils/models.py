"""Data models for search records."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DOI_REGEX = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)


class Record(BaseModel):
    """Standardised representation of a search result."""

    model_config = ConfigDict(extra="allow", frozen=False)

    title: str
    doi: Optional[str] = None
    url: Optional[str] = None
    rank: int = Field(ge=1)
    raw_id: Optional[str] = None
    source: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1800, le=2100)
    authors: Optional[List[str]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not DOI_REGEX.match(value):
            raise ValueError("Invalid DOI format")
        return value.lower()


class EnrichedRecord(Record):
    """Record augmented with OpenAlex metadata."""

    language: Optional[str] = None
    is_oa: Optional[bool] = None
    publication_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    host_venue: Optional[str] = None
    venue_type: Optional[str] = None
    is_core_listed: Optional[bool] = None
    core_rank: Optional[str] = None
    publisher: Optional[str] = None
    cited_by_count: Optional[int] = Field(default=None, ge=0)
