from __future__ import annotations

import json

import httpx

from ai_bias_search.llm.schemas import ChatMessage, ProviderRequest
from ai_bias_search.providers.openrouter import OpenRouterClient
from ai_bias_search.utils.config import RetryConfig


def test_openrouter_client_builds_request_and_parses_response() -> None:
    seen_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request["headers"] = dict(request.headers)
        seen_request["json"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "created": 1735689600,
                "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"articles":[{"rank":1,"title":"Paper",'
                                    '"doi":"10.1000/test"}]}'
                                )
                            },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 30,
                    "total_tokens": 50,
                },
            },
        )

    client = OpenRouterClient(
        api_key="test-key",
        retries=RetryConfig(max=1, backoff=1.0),
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://openrouter.ai/api/v1"
        ),
    )
    response = client.complete(
        ProviderRequest(
            model="openai/gpt-4o-mini",
            messages=[ChatMessage(role="user", content="Return JSON")],
            temperature=0.2,
            max_tokens=120,
            top_p=1.0,
            timeout_seconds=30,
            require_json=True,
        )
    )
    client.close()

    assert response.provider == "openrouter"
    assert response.model == "openai/gpt-4o-mini"
    assert response.output_text.startswith('{"articles"')
    assert response.token_usage.total_tokens == 50
    headers = seen_request["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer test-key"
    payload = json.loads(str(seen_request["json"]))
    assert payload["response_format"] == {"type": "json_object"}
