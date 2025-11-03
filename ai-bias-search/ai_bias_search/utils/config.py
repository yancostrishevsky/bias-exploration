"""Configuration loading via Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RateLimitConfig(BaseModel):
    """Rate limiting parameters for a connector."""

    rps: float = Field(gt=0)
    burst: int = Field(gt=0)


class RetryConfig(BaseModel):
    """Retry configuration for HTTP interactions."""

    max: int = Field(default=3, ge=0)
    backoff: float = Field(default=1.5, gt=0)


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

    def resolve_queries_path(self, base_dir: Path) -> Path:
        """Return the absolute path to the queries file."""

        if self.queries_file.is_absolute():
            return self.queries_file
        return (base_dir / self.queries_file).resolve()


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)
