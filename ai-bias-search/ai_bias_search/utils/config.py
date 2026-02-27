"""Configuration loading via Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RateLimitConfig(BaseModel):
    """Rate limiting parameters for a connector."""

    rps: float = Field(gt=0)
    burst: int = Field(gt=0)


class RetryConfig(BaseModel):
    """Retry configuration for HTTP interactions."""

    max: int = Field(default=3, ge=0)
    backoff: float = Field(default=1.5, gt=0)


class ImpactFactorConfig(BaseModel):
    """Configuration for journal impact factor enrichment."""

    enabled: bool = False
    xlsx_path: Path = Path("data/vendor/jif.xlsx")
    sheet_name: Optional[str] = None
    title_column: str = "Journal"
    jif_column: str = "JIF"
    year_column: Optional[str] = "Year"
    publisher_column: Optional[str] = "Publisher"
    issn_column: Optional[str] = "ISSN"
    eissn_column: Optional[str] = "eISSN"
    total_cites_column: Optional[str] = "Total Cites"
    citable_items_column: Optional[str] = "Citable Items"
    total_articles_column: Optional[str] = "Total Articles"
    jif_5y_column: Optional[str] = "5-Year JIF"
    jif_wo_self_cites_column: Optional[str] = "JIF Without Self-Cites"
    jci_column: Optional[str] = "JCI"
    quartile_column: Optional[str] = "JIF Quartile"
    jif_rank_column: Optional[str] = "JIF Rank"
    allow_fuzzy: bool = False
    fuzzy_threshold: int = Field(default=98, ge=0, le=100)
    min_title_len: int = Field(default=8, ge=1)
    max_len_ratio_delta: float = Field(default=0.2, ge=0.0)
    reject_ambiguous: bool = True


class ScopusRankingConfig(BaseModel):
    """Configuration for Scopus Serial Title ranking metrics enrichment."""

    enabled: bool = False
    api_key: Optional[str] = None
    insttoken: Optional[str] = None
    base_url: str = "https://api.elsevier.com"
    view_preference: List[Literal["ENHANCED", "STANDARD"]] = Field(
        default_factory=lambda: ["ENHANCED", "STANDARD"]
    )
    timeout_s: int = Field(default=30, ge=1)
    cache_ttl_days: int = Field(default=30, ge=1)
    rate_limit: float = Field(default=1.0, gt=0)
    fail_open: bool = True

    @model_validator(mode="after")
    def _normalize_view_preference(self) -> ScopusRankingConfig:
        seen: set[str] = set()
        normalized: list[Literal["ENHANCED", "STANDARD"]] = []
        for raw in self.view_preference:
            view = str(raw).strip().upper()
            if view not in {"ENHANCED", "STANDARD"} or view in seen:
                continue
            seen.add(view)
            normalized.append(view)  # type: ignore[arg-type]
        if not normalized:
            normalized = ["ENHANCED", "STANDARD"]
        self.view_preference = normalized
        return self


class ScopusEnrichConfig(BaseModel):
    """Configuration for Scopus (Elsevier) connector + enrichment."""

    enabled: bool = False
    fail_open: bool = True
    base_url: str = "https://api.elsevier.com"
    timeout: float = Field(default=20.0, gt=0)
    rps: float = Field(default=1.0, gt=0)
    burst: int = Field(default=2, gt=0)
    max_retries: Optional[int] = Field(default=None, ge=0)

    max_records_per_query: int = Field(default=100, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
    search_view: str = "STANDARD"
    search_sort: Optional[str] = None
    search_fields: Optional[str] = None

    cache_ttl_days: int = Field(default=7, ge=1)
    overwrite_existing: bool = False
    title_search_enabled: bool = False
    title_search_min_len: int = Field(default=32, ge=1)
    prefer_issn_from_scopus: bool = True
    prefer_affiliations_from_scopus: bool = True

    abstract_view: str = "FULL"
    abstract_fields: Optional[str] = None
    abstract_fallback_views: List[Optional[str]] = Field(
        default_factory=lambda: ["META", None]
    )
    abstract_fields_minimal: Optional[str] = (
        "dc:title,prism:doi,prism:publicationName,prism:issn,prism:coverDate,"
        "citedby-count,openaccessFlag,subtype,subtypeDescription,affiliation-country,affilname"
    )
    serial_title_view: str = "STANDARD"
    enable_citation_overview: bool = False
    citation_overview_exclude_self: bool = True
    enable_serial_title_metrics: bool = False
    enable_plumx: bool = False
    rankings: ScopusRankingConfig = Field(default_factory=ScopusRankingConfig)


class DiagnosticsConfig(BaseModel):
    """Configuration for diagnostics capture and redaction."""

    enabled: bool = True
    capture_samples: bool = True
    capture_requests: bool = True
    max_sample_records: int = Field(default=2, ge=1)
    max_enrich_trace_entries: int = Field(default=5, ge=0)
    max_request_logs: int = Field(default=20, ge=0)
    redact_fields: List[str] = Field(
        default_factory=lambda: ["apiKey", "insttoken", "Authorization"]
    )


class GeoBiasConfig(BaseModel):
    """Configuration for geo-bias metrics and gating."""

    top_k_country_min_coverage: float = Field(default=0.4, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="allow")

    queries_file: Path
    platforms: List[str]
    top_k: int = Field(default=25, gt=0)
    prompt_template: Optional[str] = None
    openalex_mailto: Optional[str] = None
    rate_limit: Dict[str, RateLimitConfig]
    retries: RetryConfig = Field(default_factory=RetryConfig)
    impact_factor: Optional[ImpactFactorConfig] = None
    scopus: ScopusEnrichConfig = Field(default_factory=ScopusEnrichConfig)
    scopus_enrich: ScopusEnrichConfig = Field(default_factory=ScopusEnrichConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    geo: GeoBiasConfig = Field(default_factory=GeoBiasConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_scopus_blocks(cls, data: Any) -> Any:
        """Support both `scopus:` and legacy `scopus_enrich:` config blocks."""

        if not isinstance(data, dict):
            return data

        scopus_raw = data.get("scopus")
        legacy_raw = data.get("scopus_enrich")

        if scopus_raw is None and legacy_raw is not None:
            data["scopus"] = legacy_raw
            return data
        if legacy_raw is None and scopus_raw is not None:
            data["scopus_enrich"] = scopus_raw
            return data

        if isinstance(scopus_raw, dict) and isinstance(legacy_raw, dict):
            merged = dict(legacy_raw)
            merged.update(scopus_raw)
            data["scopus"] = merged
            data["scopus_enrich"] = merged
        return data

    def resolve_queries_path(self, base_dir: Path) -> Path:
        """Return the absolute path to the queries file."""

        if self.queries_file.is_absolute():
            return self.queries_file
        return (base_dir / self.queries_file).resolve()


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
