"""Robust JSON extraction for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

from ai_bias_search.llm.schemas import ParsedPayload

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_response(text: str | None) -> ParsedPayload:
    """Parse JSON from raw LLM text using progressively looser extraction rules."""

    if not text or not text.strip():
        return ParsedPayload(success=False, parse_error="empty_response")

    candidates = [
        ("strict_json", text.strip()),
        ("fenced_json", _extract_fenced_json(text)),
        ("embedded_json", _extract_embedded_json(text)),
    ]
    for method, candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return ParsedPayload(success=True, parse_method=method, parsed_json=parsed)

    return ParsedPayload(success=False, parse_error="json_parse_failed")


def _extract_fenced_json(text: str) -> str | None:
    match = _FENCED_BLOCK_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).strip()
    return candidate or None


def _extract_embedded_json(text: str) -> str | None:
    for opening, closing in (("{", "}"), ("[", "]")):
        candidate = _extract_balanced_block(text, opening=opening, closing=closing)
        if candidate is not None:
            return candidate
    return None


def _extract_balanced_block(text: str, *, opening: str, closing: str) -> str | None:
    start = text.find(opening)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return None


def summarize_json_shape(payload: Any) -> str:
    """Return a short label for parsed JSON payload shape."""

    if isinstance(payload, list):
        return "array"
    if isinstance(payload, dict):
        return "object"
    return type(payload).__name__
