"""Typed schemas for the LLM audit pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_bias_search.utils.ids import normalise_doi

InputMode = Literal["query_csv", "scenarios"]
ScenarioMode = Literal["article_recommendation", "ranking", "bias_probe", "generic"]
ControlLabel = Literal["control", "treatment"]


class ChatMessage(BaseModel):
    """One chat message sent to an LLM provider."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class TokenUsage(BaseModel):
    """Normalized token accounting returned by a provider."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ProviderRequest(BaseModel):
    """Provider-agnostic chat completion request."""

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=1)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    require_json: bool = True


class ProviderResponse(BaseModel):
    """Provider-agnostic chat completion response."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    response_id: str | None = None
    created: int | None = None
    output_text: str = ""
    raw_response: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)


class QueryInputRow(BaseModel):
    """One literature-retrieval query loaded from CSV."""

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    category: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_columns(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("query_id"):
            payload["query_id"] = payload.get("id")
        if not payload.get("query_text"):
            payload["query_text"] = (
                payload.get("text")
                or payload.get("query")
                or payload.get("queryText")
                or payload.get("query_text")
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        reserved = {
            "query_id",
            "query_text",
            "category",
            "language",
            "metadata",
            "id",
            "text",
            "query",
            "queryText",
        }
        for key, value in payload.items():
            if key not in reserved and key not in metadata and value not in (None, ""):
                metadata[key] = value
        payload["metadata"] = metadata
        return payload

    @model_validator(mode="after")
    def _strip_core_fields(self) -> "QueryInputRow":
        self.query_id = self.query_id.strip()
        self.query_text = self.query_text.strip()
        self.category = _strip_optional(self.category)
        self.language = _strip_optional(self.language)
        return self


class PromptScenario(BaseModel):
    """Structured scenario definition loaded from YAML or JSON."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    mode: ScenarioMode = "article_recommendation"
    prompt: str | None = None
    prompt_template: str | None = None
    template_vars: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_format: str | None = None
    language: str | None = None
    topic: str | None = None
    pair_id: str | None = None
    variant: str | None = None
    control_or_treatment: ControlLabel | None = None

    @model_validator(mode="after")
    def _render_prompt(self) -> "PromptScenario":
        if self.prompt:
            self.prompt = self.prompt.strip()
        if not self.prompt and self.prompt_template:
            try:
                self.prompt = self.prompt_template.format(**self.template_vars)
            except KeyError as exc:
                raise ValueError(
                    f"scenario {self.id!r} missing template variable: {exc.args[0]}"
                ) from exc
        if not self.prompt:
            raise ValueError(f"scenario {self.id!r} must define prompt or prompt_template")
        return self


class PromptDataset(BaseModel):
    """Top-level scenario dataset container."""

    scenarios: list[PromptScenario] = Field(default_factory=list)


class PromptTask(BaseModel):
    """One fully rendered prompt task ready for collection."""

    model_config = ConfigDict(extra="allow")

    source_mode: InputMode = "query_csv"
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    query_category: str | None = None
    query_language: str | None = None
    mode: ScenarioMode = "article_recommendation"
    prompt_text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_format: str | None = None
    topic: str | None = None
    pair_id: str | None = None
    variant: str | None = None
    control_or_treatment: ControlLabel | None = None

    @model_validator(mode="after")
    def _strip_prompt_fields(self) -> "PromptTask":
        self.query_id = self.query_id.strip()
        self.query_text = self.query_text.strip()
        self.prompt_text = self.prompt_text.strip()
        self.query_category = _strip_optional(self.query_category)
        self.query_language = _strip_optional(self.query_language)
        self.expected_format = _strip_optional(self.expected_format)
        self.topic = _strip_optional(self.topic)
        self.variant = _strip_optional(self.variant)
        self.pair_id = _strip_optional(self.pair_id)
        return self


class RawResponseRecord(BaseModel):
    """Collected raw response stored in `raw_responses.jsonl`."""

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    source_mode: InputMode = "query_csv"
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    query_category: str | None = None
    query_language: str | None = None
    mode: ScenarioMode
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    prompt_text: str = Field(min_length=1)
    input_metadata: dict[str, Any] = Field(default_factory=dict)
    expected_format: str | None = None
    topic: str | None = None
    pair_id: str | None = None
    variant: str | None = None
    control_or_treatment: ControlLabel | None = None
    request_payload: dict[str, Any] = Field(default_factory=dict)
    raw_response_text: str | None = None
    raw_response_json: dict[str, Any] | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    success: bool
    error_message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("query_id"):
            payload["query_id"] = payload.get("scenario_id")
        if not payload.get("query_text"):
            payload["query_text"] = (
                payload.get("topic")
                or payload.get("prompt_text")
                or payload.get("scenario_id")
            )
        if "query_category" not in payload:
            payload["query_category"] = payload.get("category")
        if "query_language" not in payload:
            payload["query_language"] = (
                payload.get("language") or payload.get("scenario_language")
            )
        if "input_metadata" not in payload:
            payload["input_metadata"] = (
                payload.get("scenario_metadata") or payload.get("metadata") or {}
            )
        if "source_mode" not in payload:
            payload["source_mode"] = (
                "query_csv" if payload.get("scenario_id") is None else "scenarios"
            )
        return payload


class ParsedPayload(BaseModel):
    """Intermediate JSON parsing result for a raw LLM response."""

    success: bool
    parse_method: str | None = None
    parse_error: str | None = None
    parsed_json: Any = None


class ArticleRecommendationItem(BaseModel):
    """Normalized recommendation item extracted from an LLM response."""

    rank: int | None = Field(default=None, ge=1)
    title: str | None = None
    doi: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    journal: str | None = None
    authors: list[str] | None = None
    rationale: str | None = None
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: str = "llm_output"
    raw_item: Any = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "journal" not in payload:
            payload["journal"] = payload.get("venue")
        if "rationale" not in payload:
            payload["rationale"] = payload.get("explanation")
        return payload

    @model_validator(mode="after")
    def _normalize_doi(self) -> "ArticleRecommendationItem":
        self.doi = normalise_doi(self.doi)
        self.title = _strip_optional(self.title)
        self.journal = _strip_optional(self.journal)
        self.rationale = _strip_optional(self.rationale)
        return self

    @property
    def venue(self) -> str | None:
        return self.journal

    @property
    def explanation(self) -> str | None:
        return self.rationale


class RankingItem(BaseModel):
    """Normalized ranking/judgment item extracted from an LLM response."""

    rank: int | None = Field(default=None, ge=1)
    identifier: str | None = None
    title: str | None = None
    score: float | None = None
    rationale: str | None = None
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: str = "llm_output"
    raw_item: Any = None


class NormalizedResponseRecord(BaseModel):
    """Normalized response stored in `normalized_responses.jsonl`."""

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    source_mode: InputMode = "query_csv"
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    query_category: str | None = None
    query_language: str | None = None
    mode: ScenarioMode
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    pair_id: str | None = None
    variant: str | None = None
    control_or_treatment: ControlLabel | None = None
    expected_format: str | None = None
    topic: str | None = None
    input_metadata: dict[str, Any] = Field(default_factory=dict)
    parse_success: bool
    parse_status: str = Field(min_length=1)
    parse_method: str | None = None
    parse_error: str | None = None
    parse_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_kind: Literal["article_recommendation", "ranking", "generic"]
    item_count: int = Field(default=0, ge=0)
    article_recommendations: list[ArticleRecommendationItem] = Field(default_factory=list)
    ranking_items: list[RankingItem] = Field(default_factory=list)
    parsed_payload: Any = None
    raw_response_text: str | None = None
    success: bool
    error_message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("query_id"):
            payload["query_id"] = payload.get("scenario_id")
        if not payload.get("query_text"):
            payload["query_text"] = (
                payload.get("topic")
                or payload.get("raw_response_text")
                or payload.get("scenario_id")
            )
        if "query_category" not in payload:
            payload["query_category"] = payload.get("category")
        if "query_language" not in payload:
            payload["query_language"] = payload.get("language")
        if "input_metadata" not in payload:
            payload["input_metadata"] = payload.get("scenario_metadata") or {}
        if "source_mode" not in payload:
            payload["source_mode"] = (
                "query_csv" if payload.get("scenario_id") is None else "scenarios"
            )
        if "parse_status" not in payload:
            if payload.get("parse_success"):
                payload["parse_status"] = "parsed"
            elif payload.get("success") is False:
                payload["parse_status"] = "request_failed"
            else:
                payload["parse_status"] = "parse_failed"
        return payload


class EnrichedRecommendationRecord(BaseModel):
    """Recommendation row after optional metadata enrichment."""

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    source_mode: InputMode = "query_csv"
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    query_category: str | None = None
    query_language: str | None = None
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    pair_id: str | None = None
    variant: str | None = None
    control_or_treatment: ControlLabel | None = None
    topic: str | None = None
    recommended_rank: int | None = Field(default=None, ge=1)
    llm_claimed_title: str | None = None
    llm_claimed_doi: str | None = None
    llm_claimed_year: int | None = Field(default=None, ge=1800, le=2100)
    llm_claimed_journal: str | None = None
    llm_claimed_authors: list[str] | None = None
    rationale: str | None = None
    parse_status: str = Field(default="parsed", min_length=1)
    parse_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    enriched_title: str | None = None
    enriched_doi: str | None = None
    enriched_year: int | None = Field(default=None, ge=1800, le=2100)
    enriched_journal: str | None = None
    enriched_authors: list[str] | None = None
    openalex_id: str | None = None
    openalex_match_found: bool = False
    valid_doi: bool = False
    is_oa: bool | None = None
    cited_by_count: int | None = Field(default=None, ge=0)
    publisher: str | None = None
    core_rank: str | None = None
    impact_factor: float | None = Field(default=None, ge=0.0)
    jcr_quartile: str | None = None
    language: str | None = None
    country_primary: str | None = None
    countries: list[str] | None = None
    rankings: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if not payload.get("query_id"):
            payload["query_id"] = payload.get("scenario_id")
        if not payload.get("query_text"):
            payload["query_text"] = (
                payload.get("scenario_topic")
                or payload.get("llm_claimed_title")
                or payload.get("scenario_id")
            )
        if "query_category" not in payload:
            payload["query_category"] = payload.get("category")
        if "query_language" not in payload:
            payload["query_language"] = (
                payload.get("query_language") or payload.get("scenario_language")
            )
        if "llm_claimed_journal" not in payload:
            payload["llm_claimed_journal"] = payload.get("llm_claimed_venue")
        if "enriched_journal" not in payload:
            payload["enriched_journal"] = payload.get("enriched_venue")
        if "rationale" not in payload:
            payload["rationale"] = payload.get("explanation")
        if "source_mode" not in payload:
            payload["source_mode"] = (
                "query_csv" if payload.get("scenario_id") is None else "scenarios"
            )
        return payload

    @model_validator(mode="after")
    def _normalize_identifiers(self) -> "EnrichedRecommendationRecord":
        self.llm_claimed_doi = normalise_doi(self.llm_claimed_doi)
        self.enriched_doi = normalise_doi(self.enriched_doi)
        self.valid_doi = bool(self.llm_claimed_doi)
        self.llm_claimed_journal = _strip_optional(self.llm_claimed_journal)
        self.enriched_journal = _strip_optional(self.enriched_journal)
        self.rationale = _strip_optional(self.rationale)
        return self

    @property
    def llm_claimed_venue(self) -> str | None:
        return self.llm_claimed_journal

    @property
    def enriched_venue(self) -> str | None:
        return self.enriched_journal

    @property
    def explanation(self) -> str | None:
        return self.rationale


class RunManifest(BaseModel):
    """Manifest written into every run directory."""

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    models: list[str] = Field(default_factory=list)
    mode: InputMode = "query_csv"
    input_file: str = Field(min_length=1)
    prompt_template_file: str | None = None
    output_dir: str = Field(min_length=1)
    config_snapshot: str = Field(min_length=1)
    stages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "input_file" not in payload:
            payload["input_file"] = payload.get("prompt_file")
        return payload


def compact_model_list(models: list[str]) -> list[str]:
    """Return stable unique model identifiers while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        key = model.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _strip_optional(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None
