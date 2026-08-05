"""Pure source-bound AI draft records; this module invokes no model or provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import re
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    decimal_text,
    decimal_value,
    exact_ref,
    seal_content_addressed,
    timestamp,
)

AI_DRAFT_VERSION: Final = "myquant.v17.research-intelligence.ai-draft.v1"
AI_DRAFT_KINDS: Final = {
    "CONTRARY_EVIDENCE_DRAFT",
    "EXTRACTION",
    "HYPOTHESIS_DRAFT",
    "SUMMARY",
}
FORBIDDEN_AI_KEYS: Final = {
    "allocation",
    "broker",
    "execution",
    "factor_governance_write",
    "formal_activation",
    "likelihood",
    "likelihood_ratio",
    "normalized_weights",
    "order",
    "portfolio",
    "posterior",
    "provider",
    "selector",
    "trade",
    "weights",
}
FORBIDDEN_AI_KEY_PARTS: Final = {
    "allocation",
    "broker",
    "execution",
    "governance",
    "likelihood",
    "order",
    "portfolio",
    "posterior",
    "provider",
    "selector",
    "trade",
    "weight",
    "weights",
}


def _assert_safe_payload(value: Any, *, path: str = "payload", depth: int = 0) -> None:
    if depth > 12:
        raise IntelligenceContractError("AI draft payload is too deeply nested")
    if value is None or type(value) in {bool, int, float, str}:
        return
    if type(value) is list:
        if len(value) > 256:
            raise IntelligenceContractError("AI draft payload array is too large")
        for index, child in enumerate(value):
            _assert_safe_payload(child, path=f"{path}[{index}]", depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > 128:
            raise IntelligenceContractError("AI draft payload object is too large")
        for key, child in value.items():
            folded = key.casefold() if type(key) is str else ""
            parts = set(re.split(r"[^a-z0-9]+", folded))
            if (
                type(key) is not str
                or folded in FORBIDDEN_AI_KEYS
                or parts & FORBIDDEN_AI_KEY_PARTS
            ):
                raise IntelligenceContractError(f"AI draft field is forbidden at {path}")
            _assert_safe_payload(child, path=f"{path}.{key}", depth=depth + 1)
        return
    raise IntelligenceContractError(f"AI draft payload type is forbidden at {path}")


def build_ai_draft(
    *,
    kind: str,
    payload: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    generated_at: str,
    confidence: Any,
) -> dict[str, Any]:
    """Wrap externally supplied draft text without granting it inference authority."""

    if kind not in AI_DRAFT_KINDS:
        raise IntelligenceContractError("AI draft kind is not allowlisted")
    if type(payload) is not dict or not payload:
        raise IntelligenceContractError("AI draft payload is required")
    _assert_safe_payload(payload)
    if (
        isinstance(source_refs, (str, bytes))
        or not isinstance(source_refs, Sequence)
        or not source_refs
    ):
        raise IntelligenceContractError("AI drafts require explicit source refs")
    refs = [
        exact_ref(value, label=f"source_refs[{index}]") for index, value in enumerate(source_refs)
    ]
    ref_keys = [(row["relative_path"], row["byte_sha256"]) for row in refs]
    if len(ref_keys) != len(set(ref_keys)):
        raise IntelligenceContractError("AI draft source refs must be unique")
    refs.sort(key=lambda row: (row["relative_path"].encode(), row["byte_sha256"].encode()))
    return seal_content_addressed(
        {
            "authority": dict(NO_AUTHORITY),
            "confidence": decimal_text(
                decimal_value(
                    confidence,
                    label="confidence",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
            ),
            "generated_at": timestamp(generated_at, label="generated_at"),
            "kind": kind,
            "payload": dict(payload),
            "production": False,
            "research_only": True,
            "source_refs": refs,
            "version": AI_DRAFT_VERSION,
        },
        identity_field="draft_id",
    )


__all__ = ["AI_DRAFT_KINDS", "AI_DRAFT_VERSION", "build_ai_draft"]
