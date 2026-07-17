"""Plan-only FactorGovernanceProtocol v4 transaction contracts.

The builders in this module emit inert JSON plans.  They do not open the
registry, append a WAL, perform a CAS, apply an inverse patch, or create an
activation receipt that can pass production validation.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from quant_investor.factors.governance_protocol_v4 import (
    PROTOCOL_VERSION,
    TARGET_PRODUCTION_FACTOR_COUNT,
    assess_governance_cycle_v4,
    protocol_hash,
    semantic_sha256,
)

TRANSACTION_PLAN_SCHEMA_VERSION = "factor-governance-transaction-plan.v4"
TRANSACTION_INTENT_SCHEMA_VERSION = "factor-governance-transaction-intent.v4"
WAL_PLAN_SCHEMA_VERSION = "factor-governance-wal-plan.v4"
CAS_PLAN_SCHEMA_VERSION = "factor-governance-cas-plan.v4"
INVERSE_ROLLBACK_PLAN_SCHEMA_VERSION = "factor-governance-inverse-rollback-plan.v4"
ACTIVATION_RECEIPT_SCHEMA_VERSION = "factor-governance-activation-receipt.v4"
ACTIVATION_REQUEST_SCHEMA_VERSION = "factor-governance-activation-request.v4"
SHADOW_ACTIVATION_RECEIPT_SCHEMA_VERSION = "factor-governance-shadow-activation-receipt.v4"
SHADOW_ACTIVATION_REVOCATION_SCHEMA_VERSION = "factor-governance-shadow-activation-revocation.v4"
SHADOW_INVERSE_MANIFEST_SCHEMA_VERSION = "factor-governance-shadow-inverse-rollback-manifest.v4"
SHADOW_WAL_RECORD_SCHEMA_VERSION = "factor-governance-shadow-wal-record.v4"


class FactorGovernanceTransactionV4Error(ValueError):
    """Raised when a v4 transaction or activation contract fails closed."""


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise FactorGovernanceTransactionV4Error(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceTransactionV4Error(f"{label} must be an exact non-empty string")
    return value


def _date(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise FactorGovernanceTransactionV4Error(f"{label} must be ISO YYYY-MM-DD") from exc


def _datetime(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        observed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FactorGovernanceTransactionV4Error(f"{label} must be an ISO datetime") from exc
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise FactorGovernanceTransactionV4Error(f"{label} must be timezone-aware")
    return text


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FactorGovernanceTransactionV4Error(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise FactorGovernanceTransactionV4Error(f"{label} fields invalid: {';'.join(detail)}")
    return value


def _transaction_intent(
    *,
    transaction_id: str,
    as_of: str,
    cadence: str,
    production_factor_count: int,
    expected_registry_file_sha256: str,
    proposed_registry_file_sha256: str,
    expected_production_factor_set_sha256: str,
    proposed_production_factor_set_sha256: str,
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": TRANSACTION_INTENT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "transaction_id": _text(transaction_id, "transaction_id"),
        "as_of": _date(as_of, "as_of"),
        "cadence": cadence,
        "production_factor_count": production_factor_count,
        "expected_registry_file_sha256": _sha(
            expected_registry_file_sha256, "expected_registry_file_sha256"
        ),
        "proposed_registry_file_sha256": _sha(
            proposed_registry_file_sha256, "proposed_registry_file_sha256"
        ),
        "expected_production_factor_set_sha256": _sha(
            expected_production_factor_set_sha256,
            "expected_production_factor_set_sha256",
        ),
        "proposed_production_factor_set_sha256": _sha(
            proposed_production_factor_set_sha256,
            "proposed_production_factor_set_sha256",
        ),
        "proposals": [copy.deepcopy(dict(item)) for item in proposals],
    }


def build_factor_v4_transaction_plan(
    *,
    transaction_id: str,
    as_of: str,
    cadence: str,
    production_factor_count: int,
    expected_registry_file_sha256: str,
    proposed_registry_file_sha256: str,
    expected_production_factor_set_sha256: str,
    proposed_production_factor_set_sha256: str,
    proposals: Sequence[Mapping[str, Any]],
    wal_path: str,
    inverse_rollback_path: str,
) -> dict[str, Any]:
    """Build an inert WAL/CAS/inverse-rollback transaction plan."""

    cycle = assess_governance_cycle_v4(
        cadence=cadence,
        production_factor_count=production_factor_count,
        proposals=proposals,
    )
    intent = _transaction_intent(
        transaction_id=transaction_id,
        as_of=as_of,
        cadence=cycle["cadence"],
        production_factor_count=production_factor_count,
        expected_registry_file_sha256=expected_registry_file_sha256,
        proposed_registry_file_sha256=proposed_registry_file_sha256,
        expected_production_factor_set_sha256=expected_production_factor_set_sha256,
        proposed_production_factor_set_sha256=proposed_production_factor_set_sha256,
        proposals=proposals,
    )
    intent_sha = semantic_sha256(intent)
    blockers = list(cycle["blockers"])
    if cycle["cadence"] != "month_end":
        blockers.append("transaction_plan_requires_month_end_cadence")
    if expected_registry_file_sha256 == proposed_registry_file_sha256:
        blockers.append("proposed_registry_sha_must_differ")
    if production_factor_count >= TARGET_PRODUCTION_FACTOR_COUNT:
        replacement_count = sum(
            1 for proposal in proposals if proposal.get("action") == "replace_proposal"
        )
        if replacement_count != len(proposals):
            blockers.append("target_10_transaction_requires_only_one_in_one_out")
    blockers = list(dict.fromkeys(blockers))

    wal = {
        "schema_version": WAL_PLAN_SCHEMA_VERSION,
        "path": _text(wal_path, "wal_path"),
        "append_only": True,
        "status": "planned_not_written",
        "transaction_intent_sha256": intent_sha,
        "before_registry_file_sha256": expected_registry_file_sha256,
        "after_registry_file_sha256": proposed_registry_file_sha256,
        "write_performed": False,
    }
    cas = {
        "schema_version": CAS_PLAN_SCHEMA_VERSION,
        "compare_registry_file_sha256": expected_registry_file_sha256,
        "swap_registry_file_sha256": proposed_registry_file_sha256,
        "status": "planned_not_attempted",
        "performed": False,
    }
    rollback = {
        "schema_version": INVERSE_ROLLBACK_PLAN_SCHEMA_VERSION,
        "path": _text(inverse_rollback_path, "inverse_rollback_path"),
        "trigger_compare_registry_file_sha256": proposed_registry_file_sha256,
        "restore_registry_file_sha256": expected_registry_file_sha256,
        "restore_production_factor_set_sha256": (expected_production_factor_set_sha256),
        "requires_separate_authorization": True,
        "status": "planned_not_applied",
        "performed": False,
    }
    payload = {
        "schema_version": TRANSACTION_PLAN_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "transaction_id": intent["transaction_id"],
        "as_of": intent["as_of"],
        "cadence": cycle["cadence"],
        "status": "plan_ready" if not blockers else "plan_blocked",
        "plan_only": True,
        "production_apply_enabled": False,
        "registry_mutation_performed": False,
        "activation_receipt_required": True,
        "intent": intent,
        "transaction_intent_sha256": intent_sha,
        "cycle_assessment": cycle,
        "wal": wal,
        "cas": cas,
        "inverse_rollback_plan": rollback,
        "blockers": blockers,
    }
    payload["transaction_plan_sha256"] = semantic_sha256(payload)
    return payload


def validate_factor_v4_transaction_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a plan and prove that no side effect is represented."""

    fields = {
        "schema_version",
        "protocol_version",
        "protocol_hash",
        "transaction_id",
        "as_of",
        "cadence",
        "status",
        "plan_only",
        "production_apply_enabled",
        "registry_mutation_performed",
        "activation_receipt_required",
        "intent",
        "transaction_intent_sha256",
        "cycle_assessment",
        "wal",
        "cas",
        "inverse_rollback_plan",
        "blockers",
        "transaction_plan_sha256",
    }
    payload = _exact(dict(value), fields, "transaction plan")
    if payload["schema_version"] != TRANSACTION_PLAN_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported transaction plan schema")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceTransactionV4Error("transaction protocol version mismatch")
    if payload["protocol_hash"] != protocol_hash():
        raise FactorGovernanceTransactionV4Error("transaction protocol hash mismatch")
    _text(payload["transaction_id"], "transaction_id")
    _date(payload["as_of"], "as_of")
    if payload["cadence"] != "month_end":
        raise FactorGovernanceTransactionV4Error("transaction cadence must be month_end")
    if payload["plan_only"] is not True:
        raise FactorGovernanceTransactionV4Error("transaction must be plan_only")
    if payload["production_apply_enabled"] is not False:
        raise FactorGovernanceTransactionV4Error("production apply must remain disabled")
    if payload["registry_mutation_performed"] is not False:
        raise FactorGovernanceTransactionV4Error("registry mutation must not be performed")
    if payload["activation_receipt_required"] is not True:
        raise FactorGovernanceTransactionV4Error("activation receipt must be required")
    if payload["status"] not in {"plan_ready", "plan_blocked"}:
        raise FactorGovernanceTransactionV4Error("transaction status is invalid")
    if not isinstance(payload["blockers"], list):
        raise FactorGovernanceTransactionV4Error("transaction blockers must be a list")

    intent = _exact(
        payload["intent"],
        {
            "schema_version",
            "protocol_version",
            "protocol_hash",
            "transaction_id",
            "as_of",
            "cadence",
            "production_factor_count",
            "expected_registry_file_sha256",
            "proposed_registry_file_sha256",
            "expected_production_factor_set_sha256",
            "proposed_production_factor_set_sha256",
            "proposals",
        },
        "transaction intent",
    )
    if intent["schema_version"] != TRANSACTION_INTENT_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported transaction intent schema")
    if intent["protocol_version"] != PROTOCOL_VERSION or intent["protocol_hash"] != protocol_hash():
        raise FactorGovernanceTransactionV4Error("transaction intent protocol mismatch")
    for key in (
        "expected_registry_file_sha256",
        "proposed_registry_file_sha256",
        "expected_production_factor_set_sha256",
        "proposed_production_factor_set_sha256",
    ):
        _sha(intent[key], f"intent {key}")
    intent_sha = _sha(payload["transaction_intent_sha256"], "transaction intent SHA")
    if intent_sha != semantic_sha256(intent):
        raise FactorGovernanceTransactionV4Error("transaction intent SHA mismatch")
    if (
        payload["transaction_id"] != intent["transaction_id"]
        or payload["as_of"] != intent["as_of"]
        or payload["cadence"] != intent["cadence"]
    ):
        raise FactorGovernanceTransactionV4Error("transaction intent identity mismatch")

    expected_cycle = assess_governance_cycle_v4(
        cadence=intent["cadence"],
        production_factor_count=intent["production_factor_count"],
        proposals=intent["proposals"],
    )
    if payload["cycle_assessment"] != expected_cycle:
        raise FactorGovernanceTransactionV4Error("cycle assessment mismatch")

    wal = _exact(
        payload["wal"],
        {
            "schema_version",
            "path",
            "append_only",
            "status",
            "transaction_intent_sha256",
            "before_registry_file_sha256",
            "after_registry_file_sha256",
            "write_performed",
        },
        "WAL plan",
    )
    if wal["schema_version"] != WAL_PLAN_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported WAL plan schema")
    if wal["append_only"] is not True or wal["write_performed"] is not False:
        raise FactorGovernanceTransactionV4Error("WAL must be append-only and unwritten")
    if wal["status"] != "planned_not_written":
        raise FactorGovernanceTransactionV4Error("WAL status is invalid")
    if wal["transaction_intent_sha256"] != intent_sha:
        raise FactorGovernanceTransactionV4Error("WAL intent SHA mismatch")
    if (
        wal["before_registry_file_sha256"] != intent["expected_registry_file_sha256"]
        or wal["after_registry_file_sha256"] != intent["proposed_registry_file_sha256"]
    ):
        raise FactorGovernanceTransactionV4Error("WAL registry SHA binding mismatch")

    cas = _exact(
        payload["cas"],
        {
            "schema_version",
            "compare_registry_file_sha256",
            "swap_registry_file_sha256",
            "status",
            "performed",
        },
        "CAS plan",
    )
    if cas["schema_version"] != CAS_PLAN_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported CAS plan schema")
    if cas["status"] != "planned_not_attempted" or cas["performed"] is not False:
        raise FactorGovernanceTransactionV4Error("CAS must remain unattempted")
    if (
        cas["compare_registry_file_sha256"] != intent["expected_registry_file_sha256"]
        or cas["swap_registry_file_sha256"] != intent["proposed_registry_file_sha256"]
    ):
        raise FactorGovernanceTransactionV4Error("CAS registry SHA binding mismatch")

    rollback = _exact(
        payload["inverse_rollback_plan"],
        {
            "schema_version",
            "path",
            "trigger_compare_registry_file_sha256",
            "restore_registry_file_sha256",
            "restore_production_factor_set_sha256",
            "requires_separate_authorization",
            "status",
            "performed",
        },
        "inverse rollback plan",
    )
    if rollback["schema_version"] != INVERSE_ROLLBACK_PLAN_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported inverse rollback plan schema")
    if rollback["requires_separate_authorization"] is not True:
        raise FactorGovernanceTransactionV4Error(
            "inverse rollback must require separate authorization"
        )
    if rollback["status"] != "planned_not_applied" or rollback["performed"] is not False:
        raise FactorGovernanceTransactionV4Error("inverse rollback must remain unapplied")
    if (
        rollback["trigger_compare_registry_file_sha256"] != intent["proposed_registry_file_sha256"]
        or rollback["restore_registry_file_sha256"] != intent["expected_registry_file_sha256"]
    ):
        raise FactorGovernanceTransactionV4Error("inverse rollback registry SHA binding mismatch")
    if (
        rollback["restore_production_factor_set_sha256"]
        != intent["expected_production_factor_set_sha256"]
    ):
        raise FactorGovernanceTransactionV4Error("inverse rollback factor-set SHA binding mismatch")

    supplied_plan_sha = _sha(payload["transaction_plan_sha256"], "transaction plan SHA")
    unhashed = dict(payload)
    unhashed.pop("transaction_plan_sha256")
    if supplied_plan_sha != semantic_sha256(unhashed):
        raise FactorGovernanceTransactionV4Error("transaction plan SHA mismatch")
    rebuilt = build_factor_v4_transaction_plan(
        transaction_id=intent["transaction_id"],
        as_of=intent["as_of"],
        cadence=intent["cadence"],
        production_factor_count=intent["production_factor_count"],
        expected_registry_file_sha256=intent["expected_registry_file_sha256"],
        proposed_registry_file_sha256=intent["proposed_registry_file_sha256"],
        expected_production_factor_set_sha256=intent["expected_production_factor_set_sha256"],
        proposed_production_factor_set_sha256=intent["proposed_production_factor_set_sha256"],
        proposals=intent["proposals"],
        wal_path=wal["path"],
        inverse_rollback_path=rollback["path"],
    )
    if payload != rebuilt:
        raise FactorGovernanceTransactionV4Error(
            "transaction plan does not match deterministic rebuild"
        )
    return copy.deepcopy(payload)


def build_activation_request_v4(
    *,
    request_id: str,
    as_of: str,
    transaction_plan_sha256: str,
    proposed_registry_file_sha256: str,
    proposed_production_factor_set_sha256: str,
    runtime_contracts_sha256: str,
) -> dict[str, Any]:
    """Build a pending request which is intentionally not an activation receipt."""

    payload = {
        "schema_version": ACTIVATION_REQUEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "request_id": _text(request_id, "request_id"),
        "as_of": _date(as_of, "as_of"),
        "transaction_plan_sha256": _sha(transaction_plan_sha256, "transaction_plan_sha256"),
        "proposed_registry_file_sha256": _sha(
            proposed_registry_file_sha256, "proposed_registry_file_sha256"
        ),
        "proposed_production_factor_set_sha256": _sha(
            proposed_production_factor_set_sha256,
            "proposed_production_factor_set_sha256",
        ),
        "runtime_contracts_sha256": _sha(runtime_contracts_sha256, "runtime_contracts_sha256"),
        "status": "pending_separate_human_authorization",
        "activation_performed": False,
        "production_apply_enabled": False,
    }
    payload["request_sha256"] = semantic_sha256(payload)
    return payload


def activation_receipt_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return semantic_sha256(payload)


def validate_activation_receipt_v4(
    value: Mapping[str, Any],
    *,
    expected_as_of: str | None = None,
    expected_protocol_hash: str | None = None,
    expected_registry_file_sha256: str | None = None,
    expected_production_factor_set_sha256: str | None = None,
    expected_runtime_contracts_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an externally issued, same-day hash-bound activation receipt."""

    fields = {
        "schema_version",
        "protocol_version",
        "protocol_hash",
        "receipt_id",
        "status",
        "authorization_scope",
        "authorized_by",
        "activated_at",
        "as_of",
        "transaction_plan_sha256",
        "registry_file_sha256",
        "production_factor_set_sha256",
        "runtime_contracts_sha256",
        "activation_context_sha256",
        "activation_performed",
        "receipt_sha256",
    }
    payload = _exact(dict(value), fields, "activation receipt")
    if payload["schema_version"] != ACTIVATION_RECEIPT_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported activation receipt schema")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceTransactionV4Error("activation protocol version mismatch")
    if payload["protocol_hash"] != protocol_hash():
        raise FactorGovernanceTransactionV4Error("activation protocol hash mismatch")
    if payload["status"] != "activated" or payload["activation_performed"] is not True:
        raise FactorGovernanceTransactionV4Error("activation receipt is not activated")
    if payload["authorization_scope"] != "factor_v4_production_activation":
        raise FactorGovernanceTransactionV4Error("activation authorization scope mismatch")
    _text(payload["receipt_id"], "receipt_id")
    _text(payload["authorized_by"], "authorized_by")
    as_of = _date(payload["as_of"], "as_of")
    activated_at = _datetime(payload["activated_at"], "activated_at")
    activated_date = datetime.fromisoformat(activated_at.replace("Z", "+00:00")).date()
    if activated_date.isoformat() != as_of:
        raise FactorGovernanceTransactionV4Error(
            "activation receipt must be fresh on its as_of date"
        )
    for key in (
        "transaction_plan_sha256",
        "registry_file_sha256",
        "production_factor_set_sha256",
        "runtime_contracts_sha256",
        "activation_context_sha256",
        "receipt_sha256",
    ):
        _sha(payload[key], key)
    context = {
        "protocol_hash": payload["protocol_hash"],
        "transaction_plan_sha256": payload["transaction_plan_sha256"],
        "registry_file_sha256": payload["registry_file_sha256"],
        "production_factor_set_sha256": payload["production_factor_set_sha256"],
        "runtime_contracts_sha256": payload["runtime_contracts_sha256"],
        "as_of": as_of,
    }
    if payload["activation_context_sha256"] != semantic_sha256(context):
        raise FactorGovernanceTransactionV4Error("activation context SHA mismatch")
    if payload["receipt_sha256"] != activation_receipt_sha256(payload):
        raise FactorGovernanceTransactionV4Error("activation receipt SHA mismatch")

    expected = {
        "as_of": expected_as_of,
        "protocol_hash": expected_protocol_hash,
        "registry_file_sha256": expected_registry_file_sha256,
        "production_factor_set_sha256": expected_production_factor_set_sha256,
        "runtime_contracts_sha256": expected_runtime_contracts_sha256,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and payload[key] != expected_value:
            raise FactorGovernanceTransactionV4Error(f"activation receipt {key} mismatch")
    return copy.deepcopy(payload)


def canonical_shadow_registry_bytes_v4(value: Mapping[str, Any]) -> bytes:
    """Return the exact bytes used by the isolated v4 shadow registry."""

    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceTransactionV4Error(
            f"shadow registry is not canonical JSON: {exc}"
        ) from exc


def shadow_file_sha256_v4(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _shadow_receipt_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return semantic_sha256(payload)


def validate_shadow_activation_receipt_v4(
    value: Mapping[str, Any],
    *,
    expected_registry_file_sha256: str | None = None,
    expected_production_factor_set_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a research-shadow receipt that grants no production authority."""

    fields = {
        "schema_version",
        "protocol_version",
        "protocol_hash",
        "receipt_id",
        "status",
        "authorization_scope",
        "authorized_by",
        "activated_at",
        "as_of",
        "transaction_plan_sha256",
        "registry_file_sha256",
        "production_factor_set_sha256",
        "runtime_contracts_sha256",
        "activation_context_sha256",
        "shadow_activation_performed",
        "production_activation_performed",
        "receipt_sha256",
    }
    payload = _exact(dict(value), fields, "shadow activation receipt")
    if payload["schema_version"] != SHADOW_ACTIVATION_RECEIPT_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error("unsupported shadow activation receipt schema")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise FactorGovernanceTransactionV4Error("shadow activation protocol version mismatch")
    if payload["protocol_hash"] != protocol_hash():
        raise FactorGovernanceTransactionV4Error("shadow activation protocol hash mismatch")
    if payload["status"] != "shadow_activated":
        raise FactorGovernanceTransactionV4Error("shadow receipt is not active")
    if payload["authorization_scope"] != "factor_v4_research_shadow":
        raise FactorGovernanceTransactionV4Error("shadow activation authorization scope mismatch")
    if payload["shadow_activation_performed"] is not True:
        raise FactorGovernanceTransactionV4Error("shadow activation was not performed")
    if payload["production_activation_performed"] is not False:
        raise FactorGovernanceTransactionV4Error(
            "shadow receipt must not claim production activation"
        )
    _text(payload["receipt_id"], "receipt_id")
    _text(payload["authorized_by"], "authorized_by")
    as_of = _date(payload["as_of"], "as_of")
    activated_at = _datetime(payload["activated_at"], "activated_at")
    if datetime.fromisoformat(activated_at.replace("Z", "+00:00")).date().isoformat() != as_of:
        raise FactorGovernanceTransactionV4Error(
            "shadow activation receipt must be fresh on its as_of date"
        )
    for key in (
        "transaction_plan_sha256",
        "registry_file_sha256",
        "production_factor_set_sha256",
        "runtime_contracts_sha256",
        "activation_context_sha256",
        "receipt_sha256",
    ):
        _sha(payload[key], key)
    context = {
        "protocol_hash": payload["protocol_hash"],
        "transaction_plan_sha256": payload["transaction_plan_sha256"],
        "registry_file_sha256": payload["registry_file_sha256"],
        "production_factor_set_sha256": payload["production_factor_set_sha256"],
        "runtime_contracts_sha256": payload["runtime_contracts_sha256"],
        "as_of": as_of,
        "scope": "factor_v4_research_shadow",
    }
    if payload["activation_context_sha256"] != semantic_sha256(context):
        raise FactorGovernanceTransactionV4Error("shadow activation context SHA mismatch")
    if payload["receipt_sha256"] != _shadow_receipt_sha256(payload):
        raise FactorGovernanceTransactionV4Error("shadow activation receipt SHA mismatch")
    expected = {
        "registry_file_sha256": expected_registry_file_sha256,
        "production_factor_set_sha256": expected_production_factor_set_sha256,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and payload[key] != expected_value:
            raise FactorGovernanceTransactionV4Error(f"shadow receipt {key} mismatch")
    return copy.deepcopy(payload)


def _inverse_manifest_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return semantic_sha256(payload)


def validate_inverse_rollback_manifest_v4(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "protocol_hash",
        "transaction_id",
        "transaction_plan_sha256",
        "created_at",
        "trigger_compare_registry_file_sha256",
        "restore_registry_file_sha256",
        "restore_production_factor_set_sha256",
        "before_registry_bytes_base64",
        "requires_separate_authorization",
        "rollback_performed",
        "rolled_back_at",
        "manifest_sha256",
    }
    payload = _exact(dict(value), fields, "shadow inverse rollback manifest")
    if payload["schema_version"] != SHADOW_INVERSE_MANIFEST_SCHEMA_VERSION:
        raise FactorGovernanceTransactionV4Error(
            "unsupported shadow inverse rollback manifest schema"
        )
    if (
        payload["protocol_version"] != PROTOCOL_VERSION
        or payload["protocol_hash"] != protocol_hash()
    ):
        raise FactorGovernanceTransactionV4Error("shadow inverse rollback protocol mismatch")
    _text(payload["transaction_id"], "transaction_id")
    _datetime(payload["created_at"], "created_at")
    for key in (
        "transaction_plan_sha256",
        "trigger_compare_registry_file_sha256",
        "restore_registry_file_sha256",
        "restore_production_factor_set_sha256",
        "manifest_sha256",
    ):
        _sha(payload[key], key)
    if payload["requires_separate_authorization"] is not True:
        raise FactorGovernanceTransactionV4Error(
            "shadow rollback must require separate authorization"
        )
    if type(payload["rollback_performed"]) is not bool:
        raise FactorGovernanceTransactionV4Error("rollback_performed must be boolean")
    if payload["rollback_performed"]:
        _datetime(payload["rolled_back_at"], "rolled_back_at")
    elif payload["rolled_back_at"] is not None:
        raise FactorGovernanceTransactionV4Error(
            "unperformed rollback must not have rolled_back_at"
        )
    encoded = _text(payload["before_registry_bytes_base64"], "before_registry_bytes_base64")
    try:
        before_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise FactorGovernanceTransactionV4Error("before_registry_bytes_base64 is invalid") from exc
    if not before_bytes:
        raise FactorGovernanceTransactionV4Error("rollback before bytes are empty")
    if shadow_file_sha256_v4(before_bytes) != payload["restore_registry_file_sha256"]:
        raise FactorGovernanceTransactionV4Error("rollback before bytes SHA mismatch")
    if payload["manifest_sha256"] != _inverse_manifest_sha256(payload):
        raise FactorGovernanceTransactionV4Error("inverse manifest SHA mismatch")
    return {**copy.deepcopy(payload), "before_registry_bytes": before_bytes}


class FactorV4ShadowTransactionStore:
    """Independent v4 shadow store with atomic 0600 files and lock-bound CAS.

    The store owns only fixed files beneath the explicitly supplied
    ``shadow_root``.  It has no default path and no awareness of the current
    Factor v2/v3 registry or activation pointer.
    """

    def __init__(self, shadow_root: str | Path) -> None:
        raw_root = Path(shadow_root).expanduser()
        if raw_root.exists() and raw_root.is_symlink():
            raise FactorGovernanceTransactionV4Error("shadow_root must not be a symlink")
        raw_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw_root.chmod(0o700)
        self.root = raw_root.resolve()
        self.registry_path = self.root / "registry_v4_shadow.json"
        self.wal_path = self.root / "transaction_v4_shadow.wal.jsonl"
        self.receipt_path = self.root / "activation_v4_shadow_receipt.json"
        self.inverse_manifest_path = self.root / "inverse_v4_shadow_manifest.json"
        self.lock_path = self.root / "transaction_v4_shadow.lock"

    @contextmanager
    def _locked(self) -> Any:
        descriptor = os.open(
            self.lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise FactorGovernanceTransactionV4Error(
                    "shadow lock must be a current-user regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_regular(self, path: Path, label: str) -> bytes:
        try:
            info = path.lstat()
        except OSError as exc:
            raise FactorGovernanceTransactionV4Error(f"{label} is unavailable: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise FactorGovernanceTransactionV4Error(f"{label} must be a regular non-symlink file")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise FactorGovernanceTransactionV4Error(
                f"{label} owner/mode must be current user/0600"
            )
        if info.st_nlink != 1:
            raise FactorGovernanceTransactionV4Error(f"{label} link count must be one")
        return path.read_bytes()

    def _atomic_write(self, path: Path, raw: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
            directory_descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _append_wal(self, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": SHADOW_WAL_RECORD_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            **copy.deepcopy(dict(record)),
        }
        payload["wal_record_sha256"] = semantic_sha256(payload)
        if self.wal_path.exists():
            info = self.wal_path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise FactorGovernanceTransactionV4Error(
                    "shadow WAL must be a regular non-symlink file"
                )
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise FactorGovernanceTransactionV4Error(
                    "shadow WAL owner/mode must be current user/0600"
                )
            if info.st_nlink != 1:
                raise FactorGovernanceTransactionV4Error("shadow WAL link count must be one")
        raw = canonical_shadow_registry_bytes_v4(payload)
        descriptor = os.open(
            self.wal_path,
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            remaining = memoryview(raw)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise FactorGovernanceTransactionV4Error("shadow WAL append made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return payload

    def initialize_shadow_registry(self, value: Mapping[str, Any]) -> str:
        """Create the explicit v4 shadow registry once; never overwrite it."""

        raw = canonical_shadow_registry_bytes_v4(value)
        with self._locked():
            if self.registry_path.exists():
                raise FactorGovernanceTransactionV4Error("shadow registry is already initialized")
            self._atomic_write(self.registry_path, raw)
        return shadow_file_sha256_v4(raw)

    def registry_file_sha256(self) -> str:
        return shadow_file_sha256_v4(self._read_regular(self.registry_path, "shadow registry"))

    def apply_shadow_transaction(
        self,
        plan: Mapping[str, Any],
        proposed_registry: Mapping[str, Any],
        *,
        authorization: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply one authorized CAS to the isolated shadow registry only."""

        validated_plan = validate_factor_v4_transaction_plan(plan)
        if validated_plan["status"] != "plan_ready":
            raise FactorGovernanceTransactionV4Error(
                "blocked transaction plan cannot be applied to shadow"
            )
        intent = validated_plan["intent"]
        if authorization.get("authorization_scope") != "factor_v4_research_shadow":
            raise FactorGovernanceTransactionV4Error(
                "shadow apply requires factor_v4_research_shadow authorization"
            )
        authorized_by = _text(authorization.get("authorized_by"), "authorized_by")
        authorized_at = _datetime(authorization.get("authorized_at"), "authorized_at")
        if (
            datetime.fromisoformat(authorized_at.replace("Z", "+00:00")).date().isoformat()
            != intent["as_of"]
        ):
            raise FactorGovernanceTransactionV4Error(
                "shadow authorization must be fresh on plan as_of"
            )
        receipt_id = _text(authorization.get("receipt_id"), "receipt_id")
        runtime_contracts_sha = _sha(
            authorization.get("runtime_contracts_sha256"),
            "runtime_contracts_sha256",
        )
        proposed_raw = canonical_shadow_registry_bytes_v4(proposed_registry)
        proposed_sha = shadow_file_sha256_v4(proposed_raw)
        if proposed_sha != intent["proposed_registry_file_sha256"]:
            raise FactorGovernanceTransactionV4Error("proposed shadow registry bytes SHA mismatch")

        with self._locked():
            before_raw = self._read_regular(self.registry_path, "shadow registry")
            before_sha = shadow_file_sha256_v4(before_raw)
            if before_sha != intent["expected_registry_file_sha256"]:
                raise FactorGovernanceTransactionV4Error("shadow registry CAS compare failed")
            inverse = {
                "schema_version": SHADOW_INVERSE_MANIFEST_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(),
                "transaction_id": intent["transaction_id"],
                "transaction_plan_sha256": validated_plan["transaction_plan_sha256"],
                "created_at": authorized_at,
                "trigger_compare_registry_file_sha256": proposed_sha,
                "restore_registry_file_sha256": before_sha,
                "restore_production_factor_set_sha256": intent[
                    "expected_production_factor_set_sha256"
                ],
                "before_registry_bytes_base64": base64.b64encode(before_raw).decode("ascii"),
                "requires_separate_authorization": True,
                "rollback_performed": False,
                "rolled_back_at": None,
            }
            inverse["manifest_sha256"] = _inverse_manifest_sha256(inverse)
            validate_inverse_rollback_manifest_v4(inverse)
            self._atomic_write(
                self.inverse_manifest_path,
                canonical_shadow_registry_bytes_v4(inverse),
            )
            self._append_wal(
                {
                    "transaction_id": intent["transaction_id"],
                    "transaction_plan_sha256": validated_plan["transaction_plan_sha256"],
                    "event": "cas_prepared",
                    "recorded_at": authorized_at,
                    "before_registry_file_sha256": before_sha,
                    "after_registry_file_sha256": proposed_sha,
                }
            )
            self._atomic_write(self.registry_path, proposed_raw)
            committed_sha = shadow_file_sha256_v4(
                self._read_regular(self.registry_path, "shadow registry")
            )
            if committed_sha != proposed_sha:
                raise FactorGovernanceTransactionV4Error(
                    "shadow registry post-CAS readback mismatch"
                )
            self._append_wal(
                {
                    "transaction_id": intent["transaction_id"],
                    "transaction_plan_sha256": validated_plan["transaction_plan_sha256"],
                    "event": "cas_committed",
                    "recorded_at": authorized_at,
                    "before_registry_file_sha256": before_sha,
                    "after_registry_file_sha256": committed_sha,
                }
            )
            context = {
                "protocol_hash": protocol_hash(),
                "transaction_plan_sha256": validated_plan["transaction_plan_sha256"],
                "registry_file_sha256": committed_sha,
                "production_factor_set_sha256": intent["proposed_production_factor_set_sha256"],
                "runtime_contracts_sha256": runtime_contracts_sha,
                "as_of": intent["as_of"],
                "scope": "factor_v4_research_shadow",
            }
            receipt = {
                "schema_version": SHADOW_ACTIVATION_RECEIPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(),
                "receipt_id": receipt_id,
                "status": "shadow_activated",
                "authorization_scope": "factor_v4_research_shadow",
                "authorized_by": authorized_by,
                "activated_at": authorized_at,
                "as_of": intent["as_of"],
                "transaction_plan_sha256": validated_plan["transaction_plan_sha256"],
                "registry_file_sha256": committed_sha,
                "production_factor_set_sha256": intent["proposed_production_factor_set_sha256"],
                "runtime_contracts_sha256": runtime_contracts_sha,
                "activation_context_sha256": semantic_sha256(context),
                "shadow_activation_performed": True,
                "production_activation_performed": False,
            }
            receipt["receipt_sha256"] = _shadow_receipt_sha256(receipt)
            validate_shadow_activation_receipt_v4(receipt)
            self._atomic_write(
                self.receipt_path,
                canonical_shadow_registry_bytes_v4(receipt),
            )
        return receipt

    def rollback_shadow_transaction(
        self,
        *,
        authorization: Mapping[str, Any],
    ) -> dict[str, Any]:
        """CAS-restore exact before bytes from the independent inverse manifest."""

        if authorization.get("authorization_scope") != "factor_v4_shadow_rollback":
            raise FactorGovernanceTransactionV4Error(
                "rollback requires factor_v4_shadow_rollback authorization"
            )
        _text(authorization.get("authorized_by"), "authorized_by")
        authorized_at = _datetime(authorization.get("authorized_at"), "authorized_at")
        with self._locked():
            manifest_raw = self._read_regular(self.inverse_manifest_path, "shadow inverse manifest")
            try:
                manifest_value = json.loads(manifest_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise FactorGovernanceTransactionV4Error(
                    f"shadow inverse manifest is invalid JSON: {exc}"
                ) from exc
            manifest = validate_inverse_rollback_manifest_v4(manifest_value)
            if manifest["rollback_performed"]:
                raise FactorGovernanceTransactionV4Error(
                    "shadow inverse rollback was already performed"
                )
            current_raw = self._read_regular(self.registry_path, "shadow registry")
            current_sha = shadow_file_sha256_v4(current_raw)
            if current_sha != manifest["trigger_compare_registry_file_sha256"]:
                raise FactorGovernanceTransactionV4Error("shadow rollback CAS compare failed")
            self._append_wal(
                {
                    "transaction_id": manifest["transaction_id"],
                    "transaction_plan_sha256": manifest["transaction_plan_sha256"],
                    "event": "rollback_prepared",
                    "recorded_at": authorized_at,
                    "before_registry_file_sha256": current_sha,
                    "after_registry_file_sha256": manifest["restore_registry_file_sha256"],
                }
            )
            self._atomic_write(self.registry_path, manifest["before_registry_bytes"])
            restored_sha = shadow_file_sha256_v4(
                self._read_regular(self.registry_path, "shadow registry")
            )
            if restored_sha != manifest["restore_registry_file_sha256"]:
                raise FactorGovernanceTransactionV4Error("shadow rollback readback mismatch")
            self._append_wal(
                {
                    "transaction_id": manifest["transaction_id"],
                    "transaction_plan_sha256": manifest["transaction_plan_sha256"],
                    "event": "rollback_committed",
                    "recorded_at": authorized_at,
                    "before_registry_file_sha256": current_sha,
                    "after_registry_file_sha256": restored_sha,
                }
            )
            updated = {
                key: value
                for key, value in manifest.items()
                if key not in {"before_registry_bytes", "manifest_sha256"}
            }
            updated["rollback_performed"] = True
            updated["rolled_back_at"] = authorized_at
            updated["manifest_sha256"] = _inverse_manifest_sha256(updated)
            validate_inverse_rollback_manifest_v4(updated)
            self._atomic_write(
                self.inverse_manifest_path,
                canonical_shadow_registry_bytes_v4(updated),
            )
            revocation = {
                "schema_version": SHADOW_ACTIVATION_REVOCATION_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "transaction_id": manifest["transaction_id"],
                "status": "shadow_rolled_back",
                "rolled_back_at": authorized_at,
                "restored_registry_file_sha256": restored_sha,
                "production_activation_performed": False,
            }
            revocation["revocation_sha256"] = semantic_sha256(revocation)
            self._atomic_write(
                self.receipt_path,
                canonical_shadow_registry_bytes_v4(revocation),
            )
        return {
            "status": "shadow_rolled_back",
            "transaction_id": manifest["transaction_id"],
            "restored_registry_file_sha256": restored_sha,
            "production_activation_performed": False,
        }


__all__ = [
    "ACTIVATION_RECEIPT_SCHEMA_VERSION",
    "ACTIVATION_REQUEST_SCHEMA_VERSION",
    "CAS_PLAN_SCHEMA_VERSION",
    "FactorGovernanceTransactionV4Error",
    "FactorV4ShadowTransactionStore",
    "INVERSE_ROLLBACK_PLAN_SCHEMA_VERSION",
    "TRANSACTION_INTENT_SCHEMA_VERSION",
    "TRANSACTION_PLAN_SCHEMA_VERSION",
    "WAL_PLAN_SCHEMA_VERSION",
    "activation_receipt_sha256",
    "build_activation_request_v4",
    "build_factor_v4_transaction_plan",
    "canonical_shadow_registry_bytes_v4",
    "shadow_file_sha256_v4",
    "validate_activation_receipt_v4",
    "validate_factor_v4_transaction_plan",
    "validate_inverse_rollback_manifest_v4",
    "validate_shadow_activation_receipt_v4",
]
