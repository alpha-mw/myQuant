"""LLM request contracts and the explicit no-tools OpenAI transport."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import ssl
from typing import Any, Mapping, Sequence

from quant_investor.v16.operator_advisory.contracts import (
    LLM_REQUEST_SCHEMA,
    LLM_RESPONSE_SCHEMA,
    AdvisoryError,
    AdvisoryProviderError,
    canonical_json_bytes,
    canonical_sha256,
    validate_llm_response,
)

OPENAI_HOST = "api.openai.com"
OPENAI_PORT = 443
OPENAI_PATH = "/v1/responses"
OPENAI_ENDPOINT = "https://api.openai.com:443/v1/responses"
OPENAI_MODEL = "gpt-5.4-2026-03-05"
CODEX_DELEGATED_MODEL = "codex-delegated-reviewer.current-session"
REQUEST_MODEL_IDS = frozenset({OPENAI_MODEL, CODEX_DELEGATED_MODEL})
MAX_PROVIDER_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROVIDER_REQUEST_BYTES = 512 * 1024
MAX_OUTPUT_TOKENS = 12_000
PROVIDER_TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = """You are the LLM branch of an isolated A-share research ranking.
Use only the supplied sealed facts. Return one review for every symbol and no others.
raw_score is an ordinal research assessment in [-1,1]. confidence only describes
fact sufficiency and does not change the score. Do not provide trading actions,
portfolio instructions, forecasts stated as probabilities, or facts not present in
the request. Rationale and risks must cite only supplied evidence_ids and must not
use action labels. Output only the strict JSON object required by the schema."""

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "request_sha256",
        "model_id",
        "prompt_sha256",
        "response_schema_sha256",
        "reviews",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "request_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "model_id": {"type": "string"},
        "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "response_schema_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "reviews": {
            "type": "array",
            "minItems": 1,
            "maxItems": 50,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "symbol",
                    "raw_score",
                    "confidence",
                    "rationale",
                    "evidence_ids",
                    "risks",
                ],
                "properties": {
                    "symbol": {"type": "string"},
                    "raw_score": {"type": "number", "minimum": -1, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                    "risks": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                    },
                },
            },
        },
    },
}
RESPONSE_SCHEMA_SHA256 = canonical_sha256(RESPONSE_JSON_SCHEMA)


def build_llm_request(
    *,
    evidence: Mapping[str, Any],
    evidence_file_sha256: str,
    model_id: str = OPENAI_MODEL,
) -> dict[str, Any]:
    items = evidence.get("items")
    if not isinstance(items, list) or not items:
        raise AdvisoryError("prepared evidence has no sealed symbols")
    resolved_model_id = str(model_id or "").strip()
    if resolved_model_id not in REQUEST_MODEL_IDS:
        raise AdvisoryError("unsupported advisory LLM request model")
    return {
        "schema_version": LLM_REQUEST_SCHEMA,
        "model_id": resolved_model_id,
        "prompt_sha256": PROMPT_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "evidence_file_sha256": evidence_file_sha256,
        "items": items,
    }


def _pairs_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AdvisoryProviderError("provider returned duplicate JSON keys")
        output[key] = value
    return output


def _parse_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AdvisoryProviderError(f"{label} contains non-finite JSON")
            ),
        )
    except AdvisoryProviderError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdvisoryProviderError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AdvisoryProviderError(f"{label} must be one JSON object")
    return value


def _extract_output_text(response: Mapping[str, Any]) -> str:
    if response.get("status") != "completed":
        raise AdvisoryProviderError("provider response did not complete")
    output = response.get("output")
    if not isinstance(output, list):
        raise AdvisoryProviderError("provider response output missing")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "refusal":
                raise AdvisoryProviderError("provider refused the sealed request")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(str(part["text"]))
    if len(texts) != 1 or not texts[0].strip():
        raise AdvisoryProviderError("provider returned an invalid output_text count")
    return texts[0]


def _read_bounded(response: http.client.HTTPResponse) -> bytes:
    chunks: list[bytes] = []
    remaining = MAX_PROVIDER_RESPONSE_BYTES + 1
    while remaining > 0:
        chunk = response.read(min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
        raise AdvisoryProviderError("provider response exceeded size limit")
    return raw


def call_openai_responses(
    *,
    request: Mapping[str, Any],
    request_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if request.get("schema_version") != LLM_REQUEST_SCHEMA:
        raise AdvisoryProviderError("LLM request schema mismatch")
    if request.get("model_id") != OPENAI_MODEL:
        raise AdvisoryProviderError("LLM request model mismatch")
    if request.get("prompt_sha256") != PROMPT_SHA256:
        raise AdvisoryProviderError("LLM request prompt mismatch")
    if request.get("response_schema_sha256") != RESPONSE_SCHEMA_SHA256:
        raise AdvisoryProviderError("LLM request response schema mismatch")
    api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise AdvisoryProviderError("OPENAI_API_KEY is not configured")

    binding = {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_file_sha256,
        "model_id": OPENAI_MODEL,
        "prompt_sha256": PROMPT_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
    }
    user_text = json.dumps(
        {"required_response_bindings": binding, "sealed_request": request},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    api_payload = {
        "model": OPENAI_MODEL,
        "store": False,
        "tools": [],
        "tool_choice": "none",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": "medium"},
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "v16_operator_advisory_reviews",
                "strict": True,
                "schema": RESPONSE_JSON_SCHEMA,
            }
        },
    }
    encoded = canonical_json_bytes(api_payload)
    if len(encoded) > MAX_PROVIDER_REQUEST_BYTES:
        raise AdvisoryProviderError("provider request exceeded size limit")
    connection = http.client.HTTPSConnection(
        OPENAI_HOST,
        OPENAI_PORT,
        timeout=PROVIDER_TIMEOUT_SECONDS,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "POST",
            OPENAI_PATH,
            body=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "myQuant-v16-operator-advisory/1",
            },
        )
        response = connection.getresponse()
        content_type = str(response.getheader("Content-Type") or "").lower()
        if response.status != 200:
            _read_bounded(response)
            raise AdvisoryProviderError(f"provider HTTP status {response.status}")
        if not content_type.startswith("application/json"):
            _read_bounded(response)
            raise AdvisoryProviderError("provider content type is not JSON")
        raw = _read_bounded(response)
    except AdvisoryProviderError:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise AdvisoryProviderError(f"provider transport failed: {type(exc).__name__}") from exc
    finally:
        connection.close()

    api_response = _parse_json_object(raw, label="provider response")
    if api_response.get("model") != OPENAI_MODEL:
        raise AdvisoryProviderError("provider response model mismatch")
    output_text = _extract_output_text(api_response)
    normalized = _parse_json_object(output_text.encode("utf-8"), label="model output")
    validate_llm_response(
        normalized,
        request=request,
        request_file_sha256=request_file_sha256,
        model_id=OPENAI_MODEL,
        prompt_sha256=PROMPT_SHA256,
        response_schema_sha256=RESPONSE_SCHEMA_SHA256,
    )
    receipt = {
        "schema_version": "v16.operator-advisory-provider-receipt.v1",
        "endpoint": OPENAI_ENDPOINT,
        "model_id": OPENAI_MODEL,
        "request_sha256": request_file_sha256,
        "prompt_sha256": PROMPT_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "provider_response_id": str(api_response.get("id") or ""),
        "provider_status": str(api_response.get("status") or ""),
        "provider_response_sha256": hashlib.sha256(raw).hexdigest(),
        "store": False,
        "tools": [],
        "tool_choice": "none",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "usage": (
            api_response.get("usage") if isinstance(api_response.get("usage"), Mapping) else {}
        ),
    }
    return normalized, receipt


__all__ = [
    "CODEX_DELEGATED_MODEL",
    "OPENAI_ENDPOINT",
    "OPENAI_MODEL",
    "PROMPT_SHA256",
    "RESPONSE_JSON_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "REQUEST_MODEL_IDS",
    "SYSTEM_PROMPT",
    "build_llm_request",
    "call_openai_responses",
]
