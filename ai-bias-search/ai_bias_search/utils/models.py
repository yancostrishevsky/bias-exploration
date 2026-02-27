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

    rankings: Dict[str, Any] = Field(default_factory=dict)
    scopus: Dict[str, Any] = Field(default_factory=dict)
    scopus_meta: Dict[str, Any] = Field(default_factory=dict)

    language: Optional[str] = None
    is_oa: Optional[bool] = None
    is_open_access: Optional[bool] = None
    publication_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    host_venue: Optional[str] = None
    venue_type: Optional[str] = None
    journal_title: Optional[str] = None
    source_id: Optional[str] = None
    doc_type: Optional[str] = None
    issn: Optional[str] = None
    eissn: Optional[str] = None
    issn_list: Optional[List[str]] = None
    issn_source: Optional[str] = None
    issn_provenance: Optional[str] = None
    year_raw: Optional[int] = Field(default=None, ge=1800, le=2100)
    year_enriched: Optional[int] = Field(default=None, ge=1800, le=2100)
    year_provenance: Optional[str] = None
    countries: Optional[List[str]] = None
    country_primary: Optional[str] = None
    affiliation_countries: Optional[List[str]] = None
    country_dominant: Optional[str] = None
    country_count: Optional[int] = Field(default=None, ge=0)
    country_provenance: Optional[str] = None
    country_is_fractional: bool = False
    affiliation_institutions: Optional[List[str]] = None
    affiliation_cities: Optional[List[str]] = None
    author_ids: Optional[List[str]] = None
    author_count: Optional[int] = Field(default=None, ge=0)
    scopus_enrich_view_used: Optional[str] = None
    scopus_enrich_field_used: Optional[str] = None
    scopus_enrich_downgraded: Optional[bool] = None
    is_core_listed: Optional[bool] = None
    core_rank: Optional[str] = None
    publisher: Optional[str] = None
    cited_by_count: Optional[int] = Field(default=None, ge=0)
    impact_factor: Optional[float] = None
    impact_factor_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    impact_factor_match: Optional[str] = None
    impact_factor_title_raw: Optional[str] = None
    impact_factor_title_key: Optional[str] = None
    impact_factor_source: Optional[str] = None
    jcr_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    jcr_publisher: Optional[str] = None
    jcr_issn: Optional[str] = None
    jcr_eissn: Optional[str] = None
    jcr_total_cites: Optional[int] = Field(default=None, ge=0)
    jcr_total_articles: Optional[int] = Field(default=None, ge=0)
    jcr_citable_items: Optional[int] = Field(default=None, ge=0)
    jcr_jif_5y: Optional[float] = Field(default=None, ge=0)
    jcr_jif_wo_self_cites: Optional[float] = Field(default=None, ge=0)
    jcr_jci: Optional[float] = Field(default=None, ge=0)
    jcr_quartile: Optional[str] = None
    jcr_jif_rank: Optional[str] = None
    jcr_match_type: Optional[str] = None
