from __future__ import annotations

from ai_bias_search.diagnostics.capture import (
    capture_request,
    configure_request_capture,
    request_capture_snapshot,
    reset_request_capture,
)


def test_request_capture_keeps_last_n_and_redacts() -> None:
    configure_request_capture(
        enabled=True,
        max_logs=2,
        redact_fields=["apiKey", "Authorization"],
    )
    reset_request_capture()

    capture_request(
        platform="openalex",
        stage="collect",
        endpoint="https://api.openalex.org/works?apiKey=secret",
        method="GET",
        params={"apiKey": "secret", "page": 1},
        headers={"Authorization": "Bearer secret", "X-ELS-APIKey": "secret"},
        status_code=200,
        duration_ms=10,
        response_payload={"results": [{"id": "W1"}]},
    )
    capture_request(
        platform="openalex",
        stage="collect",
        endpoint="https://api.openalex.org/works",
        method="GET",
        params={"apiKey": "secret", "page": 2},
        headers={"Authorization": "Bearer secret", "X-ELS-APIKey": "secret"},
        status_code=200,
        duration_ms=11,
        response_payload={"results": [{"id": "W2"}]},
    )
    capture_request(
        platform="openalex",
        stage="collect",
        endpoint="https://api.openalex.org/works",
        method="GET",
        params={"apiKey": "secret", "page": 3},
        headers={"Authorization": "Bearer secret", "X-ELS-APIKey": "secret"},
        status_code=200,
        duration_ms=12,
        response_payload={"results": [{"id": "W3"}]},
    )

    snapshot = request_capture_snapshot()
    assert len(snapshot["openalex"]) == 2
    assert snapshot["openalex"][0]["params"]["page"] == 2
    assert snapshot["openalex"][0]["params"]["apiKey"] == "****"
    assert snapshot["openalex"][0]["headers"]["Authorization"] == "****"
    assert snapshot["openalex"][0]["headers"]["X-ELS-APIKey"] == "****"
    assert snapshot["openalex"][0]["endpoint"] == "https://api.openalex.org/works"
