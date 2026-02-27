"""Request-level diagnostics capture with basic redaction."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

DEFAULT_REDACT_FIELDS = {
    "api_key",
    "apikey",
    "api-key",
    "insttoken",
    "authorization",
    "x-api-key",
    "x-els-apikey",
    "x-els-insttoken",
    "token",
}


def _now_iso() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _safe_json_snippet(payload: object, *, limit: int = 2048) -> str | None:
    if payload is None:
        return None
    try:
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)
    except Exception:
        text = str(payload)
    if not text:
        return None
    if len(text) > limit:
        return text[:limit]
    return text


def _response_keys(payload: object) -> list[str]:
    if isinstance(payload, dict):
        return sorted(str(key) for key in payload.keys())
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return sorted(str(key) for key in payload[0].keys())
    return []


def _sanitize_endpoint(endpoint: str) -> str:
    if not endpoint:
        return endpoint
    try:
        parts = urlsplit(endpoint)
    except Exception:
        return endpoint.split("?", 1)[0]
    if not parts.scheme and not parts.netloc:
        return endpoint.split("?", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class RequestCapture:
    """In-memory rolling request logs grouped by platform."""

    def __init__(self) -> None:
        self._enabled = False
        self._max_logs = 20
        self._redact_fields = {field.lower() for field in DEFAULT_REDACT_FIELDS}
        self._store: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))

    def configure(
        self,
        *,
        enabled: bool,
        max_logs: int = 20,
        redact_fields: Iterable[str] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._max_logs = max(int(max_logs), 0)
        fields = {field.lower() for field in DEFAULT_REDACT_FIELDS}
        if redact_fields:
            fields.update(str(field).strip().lower() for field in redact_fields if str(field).strip())
        self._redact_fields = fields
        if self._max_logs <= 0:
            self._store = defaultdict(lambda: deque(maxlen=1))
            return
        refreshed: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._max_logs)
        )
        for platform, entries in self._store.items():
            refreshed[platform].extend(list(entries)[-self._max_logs :])
        self._store = refreshed

    def clear(self) -> None:
        self._store = defaultdict(lambda: deque(maxlen=max(self._max_logs, 1)))

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if lowered in self._redact_fields:
                    redacted[key_text] = "****"
                    continue
                redacted[key_text] = self._redact(item)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def log(
        self,
        *,
        platform: str,
        stage: str,
        endpoint: str,
        method: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, Any] | None = None,
        status_code: int | None = None,
        duration_ms: int | float | None = None,
        response_payload: Any = None,
    ) -> None:
        if not self._enabled or self._max_logs <= 0:
            return
        platform_name = str(platform or "unknown")
        duration = _to_int(duration_ms)
        safe_params = self._redact(dict(params or {}))
        safe_headers = self._redact(dict(headers or {}))
        safe_payload = self._redact(response_payload)
        entry = {
            "stage": str(stage),
            "endpoint": _sanitize_endpoint(str(endpoint)),
            "method": str(method).upper(),
            "params": safe_params,
            "headers": safe_headers,
            "status_code": status_code,
            "duration_ms": duration,
            "response_keys": _response_keys(safe_payload),
            "response_snippet": _safe_json_snippet(safe_payload),
            "ts": _now_iso(),
        }
        if platform_name not in self._store:
            self._store[platform_name] = deque(maxlen=self._max_logs)
        self._store[platform_name].append(entry)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {platform: list(entries) for platform, entries in sorted(self._store.items())}

    def merge_snapshot(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        for platform, raw_entries in payload.items():
            if not isinstance(raw_entries, list):
                continue
            if platform not in self._store:
                self._store[str(platform)] = deque(maxlen=max(self._max_logs, 1))
            target = self._store[str(platform)]
            for entry in raw_entries:
                if isinstance(entry, dict):
                    target.append(dict(entry))


REQUEST_CAPTURE = RequestCapture()


def configure_request_capture(
    *,
    enabled: bool,
    max_logs: int = 20,
    redact_fields: Iterable[str] | None = None,
) -> None:
    REQUEST_CAPTURE.configure(
        enabled=enabled,
        max_logs=max_logs,
        redact_fields=redact_fields,
    )


def reset_request_capture() -> None:
    REQUEST_CAPTURE.clear()


def capture_request(
    *,
    platform: str,
    stage: str,
    endpoint: str,
    method: str,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, Any] | None = None,
    status_code: int | None = None,
    duration_ms: int | float | None = None,
    response_payload: Any = None,
) -> None:
    REQUEST_CAPTURE.log(
        platform=platform,
        stage=stage,
        endpoint=endpoint,
        method=method,
        params=params,
        headers=headers,
        status_code=status_code,
        duration_ms=duration_ms,
        response_payload=response_payload,
    )


def request_capture_snapshot() -> dict[str, list[dict[str, Any]]]:
    return REQUEST_CAPTURE.snapshot()


def load_request_capture_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    clean: dict[str, list[dict[str, Any]]] = {}
    for platform, entries in payload.items():
        if not isinstance(entries, list):
            continue
        clean_entries = [entry for entry in entries if isinstance(entry, dict)]
        clean[str(platform)] = clean_entries
    return clean


def persist_request_capture(path: Path, *, merge_existing: bool = True) -> None:
    snapshot = request_capture_snapshot()
    if merge_existing:
        existing = load_request_capture_file(path)
        merged: dict[str, list[dict[str, Any]]] = {}
        for platform in sorted(set(existing.keys()) | set(snapshot.keys())):
            combined = list(existing.get(platform, [])) + list(snapshot.get(platform, []))
            if REQUEST_CAPTURE._max_logs > 0:
                combined = combined[-REQUEST_CAPTURE._max_logs :]
            else:
                combined = []
            merged[platform] = combined
        snapshot = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
