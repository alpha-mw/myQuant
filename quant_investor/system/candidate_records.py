"""Exact non-authorizing records for candidate-state transaction retries."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
)

from .errors import SystemContractError
from .storage import EMPTY_POINTER_SHA256

CANDIDATE_TRANSACTION_INTENT_DOMAIN: Final = "myquant-candidate-transaction-intent"
CANDIDATE_TRANSACTION_CLOCK_SOURCE: Final = "SYSTEM_UTC"
CANDIDATE_TRANSACTION_PLAN_MAX_BYTES: Final = 1024 * 1024
CANDIDATE_TRANSACTION_INTENT_FIELDS: Final = frozenset(
    {
        "domain",
        "intent_id",
        "validation_namespace_id",
        "transaction_id",
        "expected_pointer_sha256",
        "previous_candidate_state_ref",
        "transaction_plan",
        "transaction_plan_sha256",
        "trusted_at",
        "clock_source",
        "authority",
        "semantic_sha256",
    }
)

_OBJECT_REF_FIELDS: Final = frozenset(
    {
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    }
)
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="strict")) > 512
        or any(ord(character) < 0x20 for character in value)
    ):
        raise SystemContractError(f"{label} must be canonical non-empty text")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} must be lowercase SHA-256")
    return value


def _pointer_sha256(value: Any, *, label: str) -> str:
    if value == EMPTY_POINTER_SHA256:
        return EMPTY_POINTER_SHA256
    return _sha256(value, label=label)


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} must be canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} must be canonical UTC seconds")
    return value


def _object_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_OBJECT_REF_FIELDS):
        raise SystemContractError(f"{label} fields are not exact")
    row = dict(value)
    kind = _text(row.get("kind"), label=f"{label}.kind")
    contract_sha = _sha256(row.get("contract_sha256"), label=f"{label}.contract_sha256")
    try:
        get_contract(kind, contract_sha)
    except ContractError as exc:
        raise SystemContractError(f"{label} contract pair is not compiled") from exc
    _text(row.get("artifact_id"), label=f"{label}.artifact_id")
    _sha256(row.get("semantic_sha256"), label=f"{label}.semantic_sha256")
    _sha256(row.get("byte_sha256"), label=f"{label}.byte_sha256")
    return row


def _optional_candidate_ref(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    row = _object_ref(value, label="previous_candidate_state_ref")
    if row["kind"] != "factor.composite_state":
        raise SystemContractError("previous candidate state ref has the wrong kind")
    return row


def _plan(value: Any) -> tuple[dict[str, Any], str]:
    if type(value) is not dict or not value:
        raise SystemContractError("candidate transaction plan must be a non-empty object")
    plan = dict(value)
    raw = canonical_json_bytes(plan)
    if len(raw) > CANDIDATE_TRANSACTION_PLAN_MAX_BYTES:
        raise SystemContractError("candidate transaction plan exceeds its byte bound")
    return plan, hashlib.sha256(raw).hexdigest()


def candidate_transaction_intent_id(
    validation_namespace_id: str,
    transaction_id: str,
) -> str:
    """Derive the stable identity for one namespace/transaction pair."""

    namespace = _text(validation_namespace_id, label="validation_namespace_id")
    transaction = _text(transaction_id, label="transaction_id")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant-candidate-transaction-intent-id",
                "validation_namespace_id": namespace,
                "transaction_id": transaction,
            }
        )
    ).hexdigest()


def build_candidate_transaction_intent(
    *,
    validation_namespace_id: str,
    transaction_id: str,
    expected_pointer_sha256: str,
    previous_candidate_state_ref: Mapping[str, Any] | None,
    transaction_plan: Mapping[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    """Build one exact intent; System supplies the predecessor and timestamp."""

    namespace = _text(validation_namespace_id, label="validation_namespace_id")
    transaction = _text(transaction_id, label="transaction_id")
    expected = _pointer_sha256(expected_pointer_sha256, label="expected_pointer_sha256")
    previous = _optional_candidate_ref(previous_candidate_state_ref)
    plan, plan_sha = _plan(transaction_plan)
    stamp = _timestamp(trusted_at, label="trusted_at")
    preimage: dict[str, Any] = {
        "domain": CANDIDATE_TRANSACTION_INTENT_DOMAIN,
        "intent_id": candidate_transaction_intent_id(namespace, transaction),
        "validation_namespace_id": namespace,
        "transaction_id": transaction,
        "expected_pointer_sha256": expected,
        "previous_candidate_state_ref": previous,
        "transaction_plan": plan,
        "transaction_plan_sha256": plan_sha,
        "trusted_at": stamp,
        "clock_source": CANDIDATE_TRANSACTION_CLOCK_SOURCE,
        "authority": "NON_AUTHORIZING",
    }
    document = {
        **preimage,
        "semantic_sha256": hashlib.sha256(canonical_json_bytes(preimage)).hexdigest(),
    }
    return validate_candidate_transaction_intent(document)


def validate_candidate_transaction_intent(  # noqa: C901
    value: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate exact fields, identities, opaque plan hash, and semantic hash."""

    try:
        if type(value) is bytes:
            document = parse_canonical_json_bytes(value, label="candidate transaction intent")
        elif isinstance(value, Mapping):
            document = dict(value)
            canonical_json_bytes(document)
        else:
            raise SystemContractError("candidate transaction intent must be an object")
    except ContractError as exc:
        raise SystemContractError("candidate transaction intent is not canonical") from exc
    if type(document) is not dict or set(document) != set(CANDIDATE_TRANSACTION_INTENT_FIELDS):
        raise SystemContractError("candidate transaction intent fields are not exact")
    if document.get("domain") != CANDIDATE_TRANSACTION_INTENT_DOMAIN:
        raise SystemContractError("candidate transaction intent domain differs")
    namespace = _text(document.get("validation_namespace_id"), label="validation_namespace_id")
    transaction = _text(document.get("transaction_id"), label="transaction_id")
    if document.get("intent_id") != candidate_transaction_intent_id(namespace, transaction):
        raise SystemContractError("candidate transaction intent identity differs")
    _pointer_sha256(document.get("expected_pointer_sha256"), label="expected_pointer_sha256")
    _optional_candidate_ref(document.get("previous_candidate_state_ref"))
    plan, plan_sha = _plan(document.get("transaction_plan"))
    if document.get("transaction_plan_sha256") != plan_sha:
        raise SystemContractError("candidate transaction plan hash differs")
    _timestamp(document.get("trusted_at"), label="trusted_at")
    if (
        document.get("clock_source") != CANDIDATE_TRANSACTION_CLOCK_SOURCE
        or document.get("authority") != "NON_AUTHORIZING"
    ):
        raise SystemContractError("candidate transaction intent authority differs")
    semantic = _sha256(document.get("semantic_sha256"), label="semantic_sha256")
    preimage = {key: document[key] for key in document if key != "semantic_sha256"}
    if semantic != hashlib.sha256(canonical_json_bytes(preimage)).hexdigest():
        raise SystemContractError("candidate transaction intent semantic hash differs")
    if plan != document["transaction_plan"]:
        raise SystemContractError("candidate transaction plan is not exact")
    return document


__all__ = [
    "CANDIDATE_TRANSACTION_CLOCK_SOURCE",
    "CANDIDATE_TRANSACTION_INTENT_DOMAIN",
    "CANDIDATE_TRANSACTION_INTENT_FIELDS",
    "CANDIDATE_TRANSACTION_PLAN_MAX_BYTES",
    "build_candidate_transaction_intent",
    "candidate_transaction_intent_id",
    "validate_candidate_transaction_intent",
]
