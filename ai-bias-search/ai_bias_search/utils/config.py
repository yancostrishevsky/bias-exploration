"""Configuration loading via Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field


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

    def resolve_queries_path(self, base_dir: Path) -> Path:
        """Return the absolute path to the queries file."""

        if self.queries_file.is_absolute():
            return self.queries_file
        return (base_dir / self.queries_file).resolve()


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
