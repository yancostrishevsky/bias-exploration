"""OpenRouter chat completions client."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ai_bias_search.llm.schemas import ProviderRequest, ProviderResponse, TokenUsage
from ai_bias_search.utils.config import RetryConfig
from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails."""


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in {408, 409, 425, 429} or 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError))


def _retrying(cfg: RetryConfig) -> Retrying:
    attempts = max(1, int(cfg.max))
    return Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, exp_base=cfg.backoff, min=1, max=30),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    )


def _extract_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
    return ""


def _usage_from_payload(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    def as_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return TokenUsage(
        prompt_tokens=as_int(prompt_tokens),
        completion_tokens=as_int(completion_tokens),
        total_tokens=as_int(total_tokens),
    )


class OpenRouterClient:
    """Typed client wrapper around the OpenRouter chat completions API."""

    provider_name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        retries: RetryConfig | None = None,
        site_url: str | None = None,
        app_name: str = "ai-bias-search",
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not self.api_key:
            raise OpenRouterError("Set OPENROUTER_API_KEY to use the LLM audit pipeline")

        self.base_url = base_url.rstrip("/")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL")
        self.app_name = app_name
        self.retry_cfg = retries or RetryConfig()
        self.retrying = _retrying(self.retry_cfg)
        self._owns_client = client is None
        self.client = client or httpx.Client(base_url=self.base_url, timeout=60.0)

    def close(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""

        if self._owns_client:
            self.client.close()

    def build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        """Build the OpenRouter request payload for a chat completion."""

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        if request.require_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Execute a chat completion request with retry handling."""

        payload = self.build_payload(request)
        started = time.perf_counter()

        def execute() -> httpx.Response:
            response = self.client.post(
                "/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=request.timeout_seconds,
            )
            response.raise_for_status()
            return response

        try:
            response = self.retrying(execute)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            LOGGER.error(
                "OpenRouter request failed model=%s status=%s body=%s",
                request.model,
                exc.response.status_code,
                body,
            )
            raise OpenRouterError(
                f"OpenRouter request failed with HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            LOGGER.error("OpenRouter transport error model=%s error=%s", request.model, exc)
            raise OpenRouterError(f"OpenRouter transport error: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            LOGGER.error("OpenRouter returned invalid JSON model=%s", request.model)
            raise OpenRouterError("OpenRouter returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter response must be a JSON object")

        choices = data.get("choices")
        finish_reason = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                finish_reason = str(finish_reason)

        return ProviderResponse(
            provider=self.provider_name,
            model=request.model,
            response_id=str(data.get("id")) if data.get("id") is not None else None,
            created=int(data["created"]) if isinstance(data.get("created"), int) else None,
            output_text=_extract_output_text(data),
            raw_response=data,
            token_usage=_usage_from_payload(data),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )


__all__ = ["OpenRouterClient", "OpenRouterError"]
