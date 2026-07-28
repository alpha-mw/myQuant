"""Independent Factor-v4 production-research activation control.

This module does not alter the permanent authority fields in
``governance_protocol_v4``.  It admits an exact, healthy Factor-v4 set and
advances only its own private registry and active-set pointer under WAL, CAS,
exact readback, and inverse rollback controls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any, Final, NoReturn, cast

from quant_investor.factors.governance_protocol_v4 import (
    PRODUCTION_APPLY_ENABLED,
    assess_factor_governance_readiness_v4,
    protocol_hash,
)
from quant_investor.factors.governance_canonical_replay_v4 import (
    CanonicalReplayV4Error,
    readback_v4_evidence,
)
from quant_investor.factors.governance_transaction_v4 import (
    ACTIVATION_RECEIPT_SCHEMA_VERSION,
    activation_receipt_sha256,
    validate_factor_v4_transaction_plan,
    validate_activation_receipt_v4,
)
from quant_investor.factors.runtime import production_factor_set_sha256

PROTOCOL_ID: Final = "factor-governance-production-control.v1"
ROOT_RELATIVE_PATH: Final = (
    "data/private/factor_governance_production_control_v1"
)
REGISTRY_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.registry.v1"
)
TRANSACTION_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.transaction.schema.v1"
)
PRE_ACTIVATION_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.pre-activation-eligibility.schema.v1"
)
AUTHORIZATION_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.authorization-receipt.schema.v1"
)
ROLLBACK_AUTHORIZATION_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.rollback-authorization-receipt.schema.v1"
)
WAL_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.wal-record.schema.v1"
)
ACTIVE_SET_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.active-set-pointer.schema.v1"
)
CONTROL_RECEIPT_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.activation-receipt.schema.v1"
)
ROLLBACK_RECEIPT_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.rollback-receipt.schema.v1"
)
READINESS_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.readiness-readback.v1"
)
ARTIFACT_REF_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.artifact-ref.v1"
)
RUNTIME_CONTRACT_SET_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.runtime-contract-set.v1"
)
EVIDENCE_SET_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.v4-evidence-set.v1"
)
REPLAY_SET_SCHEMA_VERSION: Final = (
    "factor-governance-production-control.v4-replay-set.v1"
)
AUTHORIZATION_SCOPE: Final = "factor_v4_production_research_activation"
ROLLBACK_AUTHORIZATION_SCOPE: Final = (
    "factor_v4_production_research_rollback"
)
V4_ACTIVATION_SCOPE: Final = "factor_v4_production_activation"
EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SOURCE_REF_ROLES: Final = frozenset(
    {
        "runtime_contracts",
        "v4_activation_request",
        "v4_evidence",
        "v4_replay",
        "v4_transaction_plan",
    }
)
_SOURCE_SCHEMAS: Final = {
    "runtime_contracts": RUNTIME_CONTRACT_SET_SCHEMA_VERSION,
    "v4_activation_request": "factor-governance-activation-request.v4",
    "v4_evidence": EVIDENCE_SET_SCHEMA_VERSION,
    "v4_replay": REPLAY_SET_SCHEMA_VERSION,
    "v4_transaction_plan": "factor-governance-transaction-plan.v4",
}
_NO_EXTERNAL_AUTHORITY: Final = {
    "account_new_risk": False,
    "broker": False,
    "execution": False,
    "order": False,
    "trade": False,
}


class ProductionControlError(ValueError):
    """Raised when production control cannot prove an exact safe transition."""


class ProductionControlCrash(RuntimeError):
    """Testable crash boundary used to exercise idempotent recovery."""


def _blocked() -> NoReturn:
    raise ProductionControlError("FACTOR_PRODUCTION_CONTROL_BLOCKED")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError):
        _blocked()
    if len(raw) >= MAX_JSON_BYTES:
        _blocked()
    return raw


def canonical_file_bytes(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if "semantic_sha256" in value:
        _blocked()
    result = deepcopy(dict(value))
    result["semantic_sha256"] = _semantic_sha256(result)
    return result


def _sealed(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str],
    schema_version: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        _blocked()
    result = deepcopy(dict(value))
    if (
        result["schema_version"] != schema_version
        or result.get("protocol_id") != PROTOCOL_ID
        or _sha(result["semantic_sha256"]) != _semantic_sha256(result)
    ):
        _blocked()
    return result


def _sha(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        _blocked()
    return value


def _identifier(value: Any) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        _blocked()
    return value


def _text(value: Any) -> str:
    if type(value) is not str or not value or value != value.strip():
        _blocked()
    return value


def _date(value: Any) -> str:
    text = _text(value)
    try:
        if date.fromisoformat(text).isoformat() != text:
            _blocked()
    except ValueError:
        _blocked()
    return text


def _instant(value: Any) -> str:
    text = _text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _blocked()
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _blocked()
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative_path(value: Any) -> str:
    text = _text(value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _blocked()
    return text


def _authority(value: Any, *, active: bool) -> dict[str, bool]:
    expected = {
        **_NO_EXTERNAL_AUTHORITY,
        "active_for_production_research": active,
    }
    if type(value) is not dict or value != expected:
        _blocked()
    return dict(expected)


def _strict_object(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) >= MAX_JSON_BYTES:
        _blocked()

    def pairs(values: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in values:
            if key in result:
                _blocked()
            result[key] = item
        return result

    try:
        value = json.loads(
            raw,
            parse_constant=lambda _: _blocked(),
            object_pairs_hook=pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        _blocked()
    if type(value) is not dict or canonical_file_bytes(value) != raw:
        _blocked()
    return value


_REF_FIELDS: Final = frozenset(
    {
        "artifact_schema",
        "byte_sha256",
        "relative_path",
        "schema_version",
        "semantic_sha256",
    }
)


def _artifact_semantic_sha256(value: Mapping[str, Any]) -> str:
    schema = value.get("schema_version")
    if type(schema) is not str:
        _blocked()
    if "semantic_sha256" in value:
        declared = _sha(value["semantic_sha256"])
        if declared != _semantic_sha256(value):
            _blocked()
        return declared
    self_hash_field = {
        ACTIVATION_RECEIPT_SCHEMA_VERSION: "receipt_sha256",
        "factor-governance-activation-request.v4": "request_sha256",
        "factor-governance-transaction-plan.v4": "transaction_plan_sha256",
    }.get(schema)
    if self_hash_field is None:
        _blocked()
    declared = _sha(value.get(self_hash_field))
    payload = dict(value)
    payload.pop(self_hash_field, None)
    if declared != hashlib.sha256(_canonical_json_bytes(payload)).hexdigest():
        _blocked()
    return declared


def build_artifact_ref(
    value: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    if type(value) is not dict:
        _blocked()
    artifact_schema = _text(value.get("schema_version"))
    semantic_sha = _artifact_semantic_sha256(value)
    return {
        "artifact_schema": artifact_schema,
        "byte_sha256": hashlib.sha256(canonical_file_bytes(value)).hexdigest(),
        "relative_path": _relative_path(relative_path),
        "schema_version": ARTIFACT_REF_SCHEMA_VERSION,
        "semantic_sha256": semantic_sha,
    }


def validate_artifact_ref(
    value: Mapping[str, Any],
    *,
    expected_schema: str | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_REF_FIELDS):
        _blocked()
    result = {key: str(item) for key, item in value.items()}
    if result["schema_version"] != ARTIFACT_REF_SCHEMA_VERSION:
        _blocked()
    _text(result["artifact_schema"])
    _sha(result["byte_sha256"])
    _sha(result["semantic_sha256"])
    _relative_path(result["relative_path"])
    if expected_schema is not None and result["artifact_schema"] != expected_schema:
        _blocked()
    return result


def _validate_ref_payload(
    reference: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    expected_schema: str | None = None,
) -> dict[str, str]:
    normalized = validate_artifact_ref(
        reference,
        expected_schema=expected_schema,
    )
    if (
        normalized["byte_sha256"]
        != hashlib.sha256(canonical_file_bytes(payload)).hexdigest()
        or normalized["semantic_sha256"] != _artifact_semantic_sha256(payload)
    ):
        _blocked()
    return normalized


def build_runtime_contract_set(
    factor_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in factor_records:
        name = _text(record.get("name"))
        contract = record.get("runtime_contract")
        if type(contract) is not dict:
            _blocked()
        contract_sha = _sha(record.get("runtime_contract_sha256"))
        if contract_sha != hashlib.sha256(
            _canonical_json_bytes(contract)
        ).hexdigest():
            _blocked()
        rows.append(
            {
                "factor_name": name,
                "runtime_contract": deepcopy(contract),
                "runtime_contract_sha256": contract_sha,
            }
        )
    if (
        not rows
        or [row["factor_name"] for row in rows]
        != sorted(row["factor_name"] for row in rows)
    ):
        _blocked()
    runtime_sha = hashlib.sha256(
        _canonical_json_bytes(
            sorted(row["runtime_contract_sha256"] for row in rows)
        )
    ).hexdigest()
    return _seal(
        {
            "factors": rows,
            "protocol_id": PROTOCOL_ID,
            "runtime_contracts_sha256": runtime_sha,
            "schema_version": RUNTIME_CONTRACT_SET_SCHEMA_VERSION,
        }
    )


def build_v4_evidence_set(
    factor_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    readbacks = _verified_v4_evidence_readbacks(factor_records)
    rows: list[dict[str, Any]] = []
    for name, readback in readbacks:
        evidence = readback["evidence"]
        rows.append(
            {
                "complete_chain_hash_binding_verified": True,
                "context_bindings_readback_verified": True,
                "evidence_payload_sha256": hashlib.sha256(
                    canonical_file_bytes(evidence)
                ).hexdigest(),
                "factor_name": name,
                "local_bytes_readback_verified": True,
                "quantitative_evidence_hash_binding_verified": True,
                "replay_file_sha256": readback["replay_file_sha256"],
                "replay_semantic_sha256": evidence[
                    "replay_semantic_sha256"
                ],
                "runtime_contract_sha256": evidence[
                    "runtime_contract_sha256"
                ],
            }
        )
    if (
        not rows
        or [row["factor_name"] for row in rows]
        != sorted(row["factor_name"] for row in rows)
    ):
        _blocked()
    return _seal(
        {
            "factors": rows,
            "protocol_id": PROTOCOL_ID,
            "schema_version": EVIDENCE_SET_SCHEMA_VERSION,
        }
    )


def build_v4_replay_set(
    factor_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    readbacks = _verified_v4_evidence_readbacks(factor_records)
    rows: list[dict[str, Any]] = []
    for name, readback in readbacks:
        evidence = readback["evidence"]
        rows.append(
            {
                "complete_chain_hash_binding_verified": True,
                "context_bindings_readback_verified": True,
                "factor_name": name,
                "local_bytes_readback_verified": True,
                "quantitative_evidence_hash_binding_verified": True,
                "replay_file_sha256": readback["replay_file_sha256"],
                "replay_path": evidence["replay_path"],
                "replay_semantic_sha256": evidence[
                    "replay_semantic_sha256"
                ],
            }
        )
    if (
        not rows
        or [row["factor_name"] for row in rows]
        != sorted(row["factor_name"] for row in rows)
    ):
        _blocked()
    return _seal(
        {
            "factors": rows,
            "protocol_id": PROTOCOL_ID,
            "schema_version": REPLAY_SET_SCHEMA_VERSION,
        }
    )


def _verified_v4_evidence_readbacks(
    factor_records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for record in factor_records:
        name = _text(record.get("name"))
        runtime_contract_sha = _sha(
            record.get("runtime_contract_sha256")
        )
        evidence = record.get("evidence")
        if type(evidence) is not dict:
            _blocked()
        try:
            readback = readback_v4_evidence(evidence)
        except (CanonicalReplayV4Error, OSError, ValueError):
            _blocked()
        normalized_evidence = readback.get("evidence")
        replay = readback.get("replay")
        if (
            type(normalized_evidence) is not dict
            or normalized_evidence != evidence
            or type(replay) is not dict
            or normalized_evidence.get("schema_version")
            != "factor-governance-replay-evidence.v4"
            or normalized_evidence.get("status") != "verified"
            or normalized_evidence.get("factor_name") != name
            or normalized_evidence.get("runtime_contract_sha256")
            != runtime_contract_sha
            or normalized_evidence.get("replay_semantic_sha256")
            != replay.get("replay_semantic_sha256")
            or normalized_evidence.get("replay_file_sha256")
            != readback.get("replay_file_sha256")
            or any(
                readback.get(field) is not True
                for field in (
                    "complete_chain_hash_binding_verified",
                    "context_bindings_readback_verified",
                    "local_bytes_readback_verified",
                    "quantitative_evidence_hash_binding_verified",
                )
            )
        ):
            _blocked()
        _text(normalized_evidence.get("replay_path"))
        _sha(normalized_evidence.get("replay_file_sha256"))
        _sha(normalized_evidence.get("replay_semantic_sha256"))
        rows.append((name, deepcopy(readback)))
    if (
        not rows
        or [name for name, _ in rows]
        != sorted(name for name, _ in rows)
    ):
        _blocked()
    return rows


def _validate_activation_request(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "activation_performed",
        "as_of",
        "production_apply_enabled",
        "proposed_production_factor_set_sha256",
        "proposed_registry_file_sha256",
        "protocol_hash",
        "protocol_version",
        "request_id",
        "request_sha256",
        "runtime_contracts_sha256",
        "schema_version",
        "status",
        "transaction_plan_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        _blocked()
    result = deepcopy(dict(value))
    supplied_sha = _sha(result["request_sha256"])
    payload = dict(result)
    payload.pop("request_sha256")
    if (
        supplied_sha
        != hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        or result["schema_version"]
        != "factor-governance-activation-request.v4"
        or result["protocol_version"] != "v4"
        or result["protocol_hash"] != protocol_hash()
        or result["status"]
        != "pending_separate_human_authorization"
        or result["activation_performed"] is not False
        or result["production_apply_enabled"] is not False
        or _date(result["as_of"]) != registry["as_of"]
        or result["proposed_production_factor_set_sha256"]
        != registry["production_factor_set_sha256"]
        or result["runtime_contracts_sha256"]
        != registry["runtime_contracts_sha256"]
    ):
        _blocked()
    _identifier(result["request_id"])
    for field in (
        "proposed_registry_file_sha256",
        "proposed_production_factor_set_sha256",
        "runtime_contracts_sha256",
        "transaction_plan_sha256",
    ):
        _sha(result[field])
    return result


def _validate_source_artifacts(
    *,
    registry: Mapping[str, Any],
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        type(source_artifacts) is not dict
        or set(source_artifacts) != set(_SOURCE_REF_ROLES)
    ):
        _blocked()
    refs = registry["source_refs"]
    for role in _SOURCE_REF_ROLES:
        payload = source_artifacts[role]
        if type(payload) is not dict:
            _blocked()
        _validate_ref_payload(
            refs[role],
            payload,
            expected_schema=_SOURCE_SCHEMAS[role],
        )
    if source_artifacts["runtime_contracts"] != build_runtime_contract_set(
        registry["factor_records"]
    ):
        _blocked()
    if source_artifacts["v4_evidence"] != build_v4_evidence_set(
        registry["factor_records"]
    ):
        _blocked()
    if source_artifacts["v4_replay"] != build_v4_replay_set(
        registry["factor_records"]
    ):
        _blocked()
    plan = validate_factor_v4_transaction_plan(
        source_artifacts["v4_transaction_plan"]
    )
    if (
        plan["status"] != "plan_ready"
        or plan["plan_only"] is not True
        or plan["production_apply_enabled"] is not False
        or plan["registry_mutation_performed"] is not False
    ):
        _blocked()
    request = _validate_activation_request(
        source_artifacts["v4_activation_request"],
        registry=registry,
    )
    if (
        request["transaction_plan_sha256"]
        != refs["v4_transaction_plan"]["semantic_sha256"]
    ):
        _blocked()


_REGISTRY_FIELDS: Final = frozenset(
    {
        "as_of",
        "authority",
        "factor_records",
        "production_factor_names",
        "production_factor_set_sha256",
        "protocol_id",
        "runtime_contracts_sha256",
        "schema_version",
        "semantic_sha256",
        "source_refs",
    }
)


def build_production_registry(
    *,
    as_of: str,
    factor_records: Sequence[Mapping[str, Any]],
    source_refs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = [deepcopy(dict(record)) for record in factor_records]
    raw_names = [record.get("name") for record in records]
    if (
        not records
        or any(type(name) is not str or not name for name in raw_names)
    ):
        _blocked()
    names = cast(list[str], raw_names)
    if names != sorted(names) or len(names) != len(set(names)):
        _blocked()
    if type(source_refs) is not dict or set(source_refs) != set(_SOURCE_REF_ROLES):
        _blocked()
    normalized_refs = {
        role: validate_artifact_ref(
            source_refs[role],
            expected_schema=_SOURCE_SCHEMAS[role],
        )
        for role in sorted(_SOURCE_REF_ROLES)
    }
    runtime_hashes: list[str] = []
    for record in records:
        runtime_hashes.append(_sha(record.get("runtime_contract_sha256")))
    factor_set_sha = production_factor_set_sha256([str(name) for name in names])
    runtime_sha = hashlib.sha256(
        _canonical_json_bytes(sorted(runtime_hashes))
    ).hexdigest()
    return _seal(
        {
            "as_of": _date(as_of),
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": False,
            },
            "factor_records": records,
            "production_factor_names": names,
            "production_factor_set_sha256": factor_set_sha,
            "protocol_id": PROTOCOL_ID,
            "runtime_contracts_sha256": runtime_sha,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "source_refs": normalized_refs,
        }
    )


def validate_production_registry(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_REGISTRY_FIELDS,
        schema_version=REGISTRY_SCHEMA_VERSION,
    )
    _date(result["as_of"])
    _authority(result["authority"], active=False)
    records = result["factor_records"]
    names = result["production_factor_names"]
    if (
        type(records) is not list
        or type(names) is not list
        or not records
        or any(type(record) is not dict for record in records)
        or any(type(name) is not str or not name for name in names)
        or names != sorted(names)
        or len(names) != len(set(names))
        or [record.get("name") for record in records] != names
    ):
        _blocked()
    source_refs = result["source_refs"]
    if type(source_refs) is not dict or set(source_refs) != set(_SOURCE_REF_ROLES):
        _blocked()
    for role in _SOURCE_REF_ROLES:
        validate_artifact_ref(
            source_refs[role],
            expected_schema=_SOURCE_SCHEMAS[role],
        )
    if (
        _sha(result["production_factor_set_sha256"])
        != production_factor_set_sha256(names)
    ):
        _blocked()
    runtime_hashes = sorted(
        _sha(record.get("runtime_contract_sha256")) for record in records
    )
    if _sha(result["runtime_contracts_sha256"]) != hashlib.sha256(
        _canonical_json_bytes(runtime_hashes)
    ).hexdigest():
        _blocked()
    return result


_PRE_FIELDS: Final = frozenset(
    {
        "allowed_activation_blockers",
        "as_of",
        "authority",
        "blockers",
        "eligible",
        "pre_activation_healthy_factor_count",
        "production_factor_count",
        "production_factor_set_sha256",
        "proposed_registry_ref",
        "protocol_id",
        "readiness_without_activation",
        "runtime_contracts_sha256",
        "schema_version",
        "semantic_sha256",
        "source_as_of",
        "source_refs",
    }
)


def _source_freshness(
    records: Sequence[Mapping[str, Any]],
    *,
    decision_session: str,
) -> tuple[str, list[str]]:
    decision = _date(decision_session)
    calendar_sha: str | None = None
    canonical_sessions: list[str] | None = None
    source_dates: list[str] = []
    blockers: list[str] = []
    for record in records:
        name = _text(record.get("name"))
        maturity = record.get("maturity")
        calendar = (
            maturity.get("calendar")
            if isinstance(maturity, Mapping)
            else None
        )
        sessions = (
            calendar.get("open_session_dates")
            if isinstance(calendar, Mapping)
            else None
        )
        record_calendar_sha = record.get("calendar_sha256")
        if (
            type(sessions) is not list
            or any(type(session) is not str for session in sessions)
            or sessions != sorted(set(sessions))
            or type(record_calendar_sha) is not str
            or record_calendar_sha
            != hashlib.sha256(_canonical_json_bytes(calendar)).hexdigest()
        ):
            blockers.append(f"{name}:freshness_calendar_invalid")
            continue
        if calendar_sha is None:
            calendar_sha = record_calendar_sha
            canonical_sessions = list(sessions)
        elif (
            record_calendar_sha != calendar_sha
            or list(sessions) != canonical_sessions
        ):
            blockers.append(f"{name}:freshness_calendar_drift")
            continue
        health = record.get("health")
        raw_source = (
            health.get("source_as_of")
            if isinstance(health, Mapping)
            else None
        )
        try:
            source_as_of = _date(raw_source)
        except ProductionControlError:
            blockers.append(f"{name}:health_source_as_of_missing")
            continue
        source_dates.append(source_as_of)
    if canonical_sessions is None or decision not in canonical_sessions:
        blockers.append("decision_session_missing_from_canonical_calendar")
        return "", sorted(set(blockers))
    decision_index = canonical_sessions.index(decision)
    for source_as_of in source_dates:
        if source_as_of not in canonical_sessions:
            blockers.append(
                f"health_source_as_of_not_open_session:{source_as_of}"
            )
            continue
        source_index = canonical_sessions.index(source_as_of)
        open_session_lag = decision_index - source_index
        calendar_day_lag = (
            date.fromisoformat(decision)
            - date.fromisoformat(source_as_of)
        ).days
        if open_session_lag < 0:
            blockers.append(
                f"health_source_as_of_after_decision:{source_as_of}"
            )
        if open_session_lag > 3:
            blockers.append(
                f"health_source_open_session_lag_above_3:{source_as_of}"
            )
        if calendar_day_lag > 8:
            blockers.append(
                f"health_source_calendar_day_lag_above_8:{source_as_of}"
            )
    if len(source_dates) != len(records):
        blockers.append("health_source_as_of_inventory_incomplete")
    return (
        min(source_dates) if source_dates else "",
        sorted(set(blockers)),
    )


def _pre_activation_assessment(
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[str], int, str]:
    normalized = validate_production_registry(registry)
    records = normalized["factor_records"]
    readiness = assess_factor_governance_readiness_v4(
        records,
        as_of=normalized["as_of"],
        registry_file_sha256=hashlib.sha256(
            canonical_file_bytes(normalized)
        ).hexdigest(),
        production_factor_set_sha256=normalized[
            "production_factor_set_sha256"
        ],
        activation_receipt=None,
    )
    allowed = ["activation_receipt_missing"]
    allowed.extend(
        f"{name}:factor_activation_receipt_missing_or_invalid"
        for name in normalized["production_factor_names"]
    )
    allowed_set = set(allowed)
    blockers = sorted(
        blocker
        for blocker in readiness["blockers"]
        if blocker not in allowed_set
    )
    healthy = 0
    for factor in readiness["factors"]:
        factor_blockers = set(factor["blockers"])
        if factor_blockers <= {"factor_activation_receipt_missing_or_invalid"}:
            healthy += 1
        else:
            blockers.extend(
                f"{factor['name']}:{blocker}"
                for blocker in sorted(
                    factor_blockers
                    - {"factor_activation_receipt_missing_or_invalid"}
                )
            )
    source_as_of, freshness_blockers = _source_freshness(
        records,
        decision_session=normalized["as_of"],
    )
    blockers.extend(freshness_blockers)
    return (
        readiness,
        sorted(set(blockers)),
        sorted(allowed),
        healthy,
        source_as_of,
    )


def build_pre_activation_eligibility(
    *,
    registry: Mapping[str, Any],
    proposed_registry_ref: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_production_registry(registry)
    registry_ref = _validate_ref_payload(
        proposed_registry_ref,
        normalized,
        expected_schema=REGISTRY_SCHEMA_VERSION,
    )
    (
        readiness,
        blockers,
        allowed,
        healthy,
        source_as_of,
    ) = _pre_activation_assessment(normalized)
    eligible = (
        not blockers
        and healthy == len(normalized["factor_records"])
        and set(readiness["blockers"]) == set(allowed)
        and 5 <= len(normalized["factor_records"]) <= 10
    )
    if not eligible and not blockers:
        blockers = ["pre_activation_readiness_not_proven"]
    return _seal(
        {
            "allowed_activation_blockers": allowed,
            "as_of": normalized["as_of"],
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": False,
            },
            "blockers": sorted(set(blockers)),
            "eligible": eligible,
            "pre_activation_healthy_factor_count": healthy,
            "production_factor_count": len(normalized["factor_records"]),
            "production_factor_set_sha256": normalized[
                "production_factor_set_sha256"
            ],
            "proposed_registry_ref": registry_ref,
            "protocol_id": PROTOCOL_ID,
            "readiness_without_activation": readiness,
            "runtime_contracts_sha256": normalized[
                "runtime_contracts_sha256"
            ],
            "schema_version": PRE_ACTIVATION_SCHEMA_VERSION,
            "source_as_of": source_as_of,
            "source_refs": normalized["source_refs"],
        }
    )


def validate_pre_activation_eligibility(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_PRE_FIELDS,
        schema_version=PRE_ACTIVATION_SCHEMA_VERSION,
    )
    expected = build_pre_activation_eligibility(
        registry=registry,
        proposed_registry_ref=result["proposed_registry_ref"],
    )
    if result != expected:
        _blocked()
    _authority(result["authority"], active=False)
    return result


_AUTH_FIELDS: Final = frozenset(
    {
        "activation_performed",
        "authorization_scope",
        "authorized_by",
        "expires_at",
        "issued_at",
        "production_factor_set_sha256",
        "proposed_registry_sha256",
        "protocol_id",
        "receipt_id",
        "runtime_contracts_sha256",
        "schema_version",
        "semantic_sha256",
        "transaction_plan_sha256",
    }
)


def build_authorization_receipt(
    *,
    receipt_id: str,
    authorized_by: str,
    issued_at: str,
    expires_at: str,
    transaction_plan_sha256: str,
    proposed_registry_sha256: str,
    production_factor_set_sha256: str,
    runtime_contracts_sha256: str,
) -> dict[str, Any]:
    issued = _instant(issued_at)
    expires = _instant(expires_at)
    if expires <= issued:
        _blocked()
    return _seal(
        {
            "activation_performed": False,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "authorized_by": _text(authorized_by),
            "expires_at": expires,
            "issued_at": issued,
            "production_factor_set_sha256": _sha(
                production_factor_set_sha256
            ),
            "proposed_registry_sha256": _sha(proposed_registry_sha256),
            "protocol_id": PROTOCOL_ID,
            "receipt_id": _identifier(receipt_id),
            "runtime_contracts_sha256": _sha(runtime_contracts_sha256),
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "transaction_plan_sha256": _sha(transaction_plan_sha256),
        }
    )


def validate_authorization_receipt(
    value: Mapping[str, Any],
    *,
    observed_at: str,
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_AUTH_FIELDS,
        schema_version=AUTHORIZATION_SCHEMA_VERSION,
    )
    issued = _instant(result["issued_at"])
    expires = _instant(result["expires_at"])
    observed = _instant(observed_at)
    if (
        result["authorization_scope"] != AUTHORIZATION_SCOPE
        or result["activation_performed"] is not False
        or not issued <= observed <= expires
    ):
        _blocked()
    _identifier(result["receipt_id"])
    _text(result["authorized_by"])
    for field in (
        "transaction_plan_sha256",
        "proposed_registry_sha256",
        "production_factor_set_sha256",
        "runtime_contracts_sha256",
    ):
        _sha(result[field])
    return result


_ROLLBACK_AUTH_FIELDS: Final = frozenset(
    {
        "authorization_scope",
        "authorized_by",
        "control_receipt_ref",
        "current_active_set_sha256",
        "current_registry_sha256",
        "expires_at",
        "issued_at",
        "protocol_id",
        "receipt_id",
        "restore_active_set_sha256",
        "restore_registry_sha256",
        "rollback_performed",
        "schema_version",
        "semantic_sha256",
        "transaction_id",
        "transaction_ref",
    }
)


def build_rollback_authorization_receipt(
    *,
    receipt_id: str,
    authorized_by: str,
    issued_at: str,
    expires_at: str,
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _sealed(
        transaction,
        fields=_TRANSACTION_FIELDS,
        schema_version=TRANSACTION_SCHEMA_VERSION,
    )
    issued = _instant(issued_at)
    expires = _instant(expires_at)
    if expires <= issued:
        _blocked()
    control_receipt = _build_control_receipt(transaction=normalized)
    return _seal(
        {
            "authorization_scope": ROLLBACK_AUTHORIZATION_SCOPE,
            "authorized_by": _text(authorized_by),
            "control_receipt_ref": build_artifact_ref(
                control_receipt,
                relative_path=(
                    f"{ROOT_RELATIVE_PATH}/receipts/control_activations/"
                    f"{control_receipt['receipt_id']}.json"
                ),
            ),
            "current_active_set_sha256": normalized[
                "proposed_active_set_sha256"
            ],
            "current_registry_sha256": normalized[
                "proposed_registry_sha256"
            ],
            "expires_at": expires,
            "issued_at": issued,
            "protocol_id": PROTOCOL_ID,
            "receipt_id": _identifier(receipt_id),
            "restore_active_set_sha256": normalized[
                "expected_active_set_sha256"
            ],
            "restore_registry_sha256": normalized[
                "expected_registry_sha256"
            ],
            "rollback_performed": False,
            "schema_version": ROLLBACK_AUTHORIZATION_SCHEMA_VERSION,
            "transaction_id": normalized["transaction_id"],
            "transaction_ref": build_artifact_ref(
                normalized,
                relative_path=(
                    f"{ROOT_RELATIVE_PATH}/transactions/"
                    f"{normalized['transaction_id']}.json"
                ),
            ),
        }
    )


def validate_rollback_authorization_receipt(
    value: Mapping[str, Any],
    *,
    transaction: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    normalized_transaction = _sealed(
        transaction,
        fields=_TRANSACTION_FIELDS,
        schema_version=TRANSACTION_SCHEMA_VERSION,
    )
    result = _sealed(
        value,
        fields=_ROLLBACK_AUTH_FIELDS,
        schema_version=ROLLBACK_AUTHORIZATION_SCHEMA_VERSION,
    )
    issued = _instant(result["issued_at"])
    expires = _instant(result["expires_at"])
    observed = _instant(observed_at)
    if (
        result["authorization_scope"] != ROLLBACK_AUTHORIZATION_SCOPE
        or result["rollback_performed"] is not False
        or not issued <= observed <= expires
        or result["transaction_id"]
        != normalized_transaction["transaction_id"]
        or result["current_active_set_sha256"]
        != normalized_transaction["proposed_active_set_sha256"]
        or result["current_registry_sha256"]
        != normalized_transaction["proposed_registry_sha256"]
        or result["restore_active_set_sha256"]
        != normalized_transaction["expected_active_set_sha256"]
        or result["restore_registry_sha256"]
        != normalized_transaction["expected_registry_sha256"]
    ):
        _blocked()
    _identifier(result["receipt_id"])
    _text(result["authorized_by"])
    expected_transaction_ref = build_artifact_ref(
        normalized_transaction,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/transactions/"
            f"{normalized_transaction['transaction_id']}.json"
        ),
    )
    control_receipt = _build_control_receipt(
        transaction=normalized_transaction
    )
    expected_control_ref = build_artifact_ref(
        control_receipt,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/receipts/control_activations/"
            f"{control_receipt['receipt_id']}.json"
        ),
    )
    if (
        result["transaction_ref"] != expected_transaction_ref
        or result["control_receipt_ref"] != expected_control_ref
    ):
        _blocked()
    for field in (
        "current_active_set_sha256",
        "current_registry_sha256",
        "restore_active_set_sha256",
        "restore_registry_sha256",
    ):
        _sha(result[field])
    return result


_READINESS_FIELDS: Final = frozenset(
    {
        "as_of",
        "authority",
        "protocol_id",
        "readiness",
        "registry_ref",
        "schema_version",
        "semantic_sha256",
        "source_as_of",
        "v4_activation_receipt_ref",
    }
)
_ACTIVE_SET_FIELDS: Final = frozenset(
    {
        "activated_at",
        "active_set_id",
        "as_of",
        "authority",
        "production_factor_names",
        "production_factor_set_sha256",
        "protocol_id",
        "readiness_ref",
        "registry_ref",
        "runtime_contracts_sha256",
        "schema_version",
        "semantic_sha256",
        "transaction_id",
        "v4_activation_receipt_ref",
    }
)
_TRANSACTION_FIELDS: Final = frozenset(
    {
        "activated_at",
        "authority",
        "authorization_scope",
        "authorized_by",
        "control_receipt_id",
        "expected_active_set_sha256",
        "expected_registry_sha256",
        "production_control_authorization_receipt_ref",
        "proposed_active_set",
        "proposed_active_set_sha256",
        "proposed_registry_ref",
        "proposed_registry_sha256",
        "protocol_id",
        "readiness_after_activation",
        "runtime_contracts_ref",
        "runtime_contracts_sha256",
        "schema_version",
        "semantic_sha256",
        "transaction_id",
        "v4_activation_receipt",
        "v4_activation_receipt_ref",
        "v4_activation_request_ref",
        "v4_evidence_ref",
        "v4_pre_activation_eligibility_ref",
        "v4_replay_ref",
        "v4_transaction_plan_ref",
    }
)


def _build_v4_activation_receipt(
    *,
    receipt_id: str,
    registry: Mapping[str, Any],
    authorization: Mapping[str, Any],
    activated_at: str,
) -> dict[str, Any]:
    plan_ref = registry["source_refs"]["v4_transaction_plan"]
    context = {
        "protocol_hash": protocol_hash(),
        "transaction_plan_sha256": plan_ref["semantic_sha256"],
        "registry_file_sha256": hashlib.sha256(
            canonical_file_bytes(registry)
        ).hexdigest(),
        "production_factor_set_sha256": registry[
            "production_factor_set_sha256"
        ],
        "runtime_contracts_sha256": registry["runtime_contracts_sha256"],
        "as_of": registry["as_of"],
    }
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "protocol_version": "v4",
        "protocol_hash": protocol_hash(),
        "receipt_id": _identifier(receipt_id),
        "status": "activated",
        "authorization_scope": V4_ACTIVATION_SCOPE,
        "authorized_by": authorization["authorized_by"],
        "activated_at": _instant(activated_at),
        "as_of": registry["as_of"],
        "transaction_plan_sha256": plan_ref["semantic_sha256"],
        "registry_file_sha256": context["registry_file_sha256"],
        "production_factor_set_sha256": context[
            "production_factor_set_sha256"
        ],
        "runtime_contracts_sha256": context["runtime_contracts_sha256"],
        "activation_context_sha256": hashlib.sha256(
            _canonical_json_bytes(context)
        ).hexdigest(),
        "activation_performed": True,
    }
    receipt["receipt_sha256"] = activation_receipt_sha256(receipt)
    validate_activation_receipt_v4(
        receipt,
        expected_as_of=registry["as_of"],
        expected_protocol_hash=protocol_hash(),
        expected_registry_file_sha256=context["registry_file_sha256"],
        expected_production_factor_set_sha256=context[
            "production_factor_set_sha256"
        ],
        expected_runtime_contracts_sha256=context[
            "runtime_contracts_sha256"
        ],
    )
    return receipt


def _build_readiness_after_activation(
    *,
    registry: Mapping[str, Any],
    registry_ref: Mapping[str, Any],
    v4_activation_receipt: Mapping[str, Any],
    v4_activation_receipt_ref: Mapping[str, Any],
) -> dict[str, Any]:
    source_as_of, freshness_blockers = _source_freshness(
        registry["factor_records"],
        decision_session=registry["as_of"],
    )
    if freshness_blockers:
        _blocked()
    readiness = assess_factor_governance_readiness_v4(
        registry["factor_records"],
        as_of=registry["as_of"],
        registry_file_sha256=registry_ref["byte_sha256"],
        production_factor_set_sha256=registry[
            "production_factor_set_sha256"
        ],
        activation_receipt=v4_activation_receipt,
    )
    if (
        readiness["factor_governance_ready"] is not True
        or readiness["new_risk_eligible"] is not True
        or readiness["new_risk_authorized"] is not False
        or readiness["production_apply_enabled"] is not False
        or readiness["blockers"]
    ):
        _blocked()
    return _seal(
        {
            "as_of": registry["as_of"],
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": False,
            },
            "protocol_id": PROTOCOL_ID,
            "readiness": readiness,
            "registry_ref": validate_artifact_ref(
                registry_ref,
                expected_schema=REGISTRY_SCHEMA_VERSION,
            ),
            "schema_version": READINESS_SCHEMA_VERSION,
            "source_as_of": source_as_of,
            "v4_activation_receipt_ref": validate_artifact_ref(
                v4_activation_receipt_ref,
                expected_schema=ACTIVATION_RECEIPT_SCHEMA_VERSION,
            ),
        }
    )


def _validate_readiness_after_activation(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    v4_activation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_READINESS_FIELDS,
        schema_version=READINESS_SCHEMA_VERSION,
    )
    _authority(result["authority"], active=False)
    expected = _build_readiness_after_activation(
        registry=registry,
        registry_ref=result["registry_ref"],
        v4_activation_receipt=v4_activation_receipt,
        v4_activation_receipt_ref=result["v4_activation_receipt_ref"],
    )
    if result != expected:
        _blocked()
    return result


def _build_active_set_pointer(
    *,
    transaction_id: str,
    activated_at: str,
    registry: Mapping[str, Any],
    registry_ref: Mapping[str, Any],
    v4_activation_receipt_ref: Mapping[str, Any],
    readiness_ref: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal(
        {
            "activated_at": _instant(activated_at),
            "active_set_id": f"active-{_identifier(transaction_id)}",
            "as_of": registry["as_of"],
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": True,
            },
            "production_factor_names": registry["production_factor_names"],
            "production_factor_set_sha256": registry[
                "production_factor_set_sha256"
            ],
            "protocol_id": PROTOCOL_ID,
            "readiness_ref": validate_artifact_ref(
                readiness_ref,
                expected_schema=READINESS_SCHEMA_VERSION,
            ),
            "registry_ref": validate_artifact_ref(
                registry_ref,
                expected_schema=REGISTRY_SCHEMA_VERSION,
            ),
            "runtime_contracts_sha256": registry[
                "runtime_contracts_sha256"
            ],
            "schema_version": ACTIVE_SET_SCHEMA_VERSION,
            "transaction_id": _identifier(transaction_id),
            "v4_activation_receipt_ref": validate_artifact_ref(
                v4_activation_receipt_ref,
                expected_schema=ACTIVATION_RECEIPT_SCHEMA_VERSION,
            ),
        }
    )


def validate_active_set_pointer(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_ACTIVE_SET_FIELDS,
        schema_version=ACTIVE_SET_SCHEMA_VERSION,
    )
    _identifier(result["active_set_id"])
    _identifier(result["transaction_id"])
    _date(result["as_of"])
    _instant(result["activated_at"])
    _authority(result["authority"], active=True)
    names = result["production_factor_names"]
    if (
        type(names) is not list
        or not 5 <= len(names) <= 10
        or any(type(name) is not str or not name for name in names)
        or names != sorted(names)
        or len(names) != len(set(names))
        or _sha(result["production_factor_set_sha256"])
        != production_factor_set_sha256(names)
    ):
        _blocked()
    _sha(result["runtime_contracts_sha256"])
    validate_artifact_ref(
        result["registry_ref"],
        expected_schema=REGISTRY_SCHEMA_VERSION,
    )
    validate_artifact_ref(
        result["readiness_ref"],
        expected_schema=READINESS_SCHEMA_VERSION,
    )
    validate_artifact_ref(
        result["v4_activation_receipt_ref"],
        expected_schema=ACTIVATION_RECEIPT_SCHEMA_VERSION,
    )
    return result


def build_production_control_transaction(
    *,
    transaction_id: str,
    registry: Mapping[str, Any],
    pre_activation_eligibility: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
    expected_registry_sha256: str,
    expected_active_set_sha256: str,
    activated_at: str,
    v4_activation_receipt_id: str,
    control_receipt_id: str,
) -> dict[str, Any]:
    normalized_registry = validate_production_registry(registry)
    eligibility = validate_pre_activation_eligibility(
        pre_activation_eligibility,
        registry=normalized_registry,
    )
    if eligibility["eligible"] is not True:
        _blocked()
    activation_time = _instant(activated_at)
    authorization = validate_authorization_receipt(
        authorization_receipt,
        observed_at=activation_time,
    )
    registry_ref = eligibility["proposed_registry_ref"]
    proposed_registry_sha = registry_ref["byte_sha256"]
    source_refs = normalized_registry["source_refs"]
    if (
        authorization["transaction_plan_sha256"]
        != source_refs["v4_transaction_plan"]["semantic_sha256"]
        or authorization["proposed_registry_sha256"]
        != proposed_registry_sha
        or authorization["production_factor_set_sha256"]
        != normalized_registry["production_factor_set_sha256"]
        or authorization["runtime_contracts_sha256"]
        != normalized_registry["runtime_contracts_sha256"]
    ):
        _blocked()
    pre_ref = build_artifact_ref(
        eligibility,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/eligibility/"
            f"{eligibility['semantic_sha256']}.json"
        ),
    )
    authorization_ref = build_artifact_ref(
        authorization,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/authorizations/"
            f"{authorization['receipt_id']}.json"
        ),
    )
    v4_receipt = _build_v4_activation_receipt(
        receipt_id=v4_activation_receipt_id,
        registry=normalized_registry,
        authorization=authorization,
        activated_at=activation_time,
    )
    v4_receipt_ref = build_artifact_ref(
        v4_receipt,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/receipts/v4_activations/"
            f"{v4_receipt['receipt_id']}.json"
        ),
    )
    readiness = _build_readiness_after_activation(
        registry=normalized_registry,
        registry_ref=registry_ref,
        v4_activation_receipt=v4_receipt,
        v4_activation_receipt_ref=v4_receipt_ref,
    )
    readiness_ref = build_artifact_ref(
        readiness,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/readiness/"
            f"{_identifier(transaction_id)}.json"
        ),
    )
    active_set = _build_active_set_pointer(
        transaction_id=transaction_id,
        activated_at=activation_time,
        registry=normalized_registry,
        registry_ref=registry_ref,
        v4_activation_receipt_ref=v4_receipt_ref,
        readiness_ref=readiness_ref,
    )
    proposed_active_sha = hashlib.sha256(
        canonical_file_bytes(active_set)
    ).hexdigest()
    expected_registry = _sha(expected_registry_sha256)
    expected_active = _sha(expected_active_set_sha256)
    if (
        expected_registry == proposed_registry_sha
        or expected_active == proposed_active_sha
    ):
        _blocked()
    return _seal(
        {
            "activated_at": activation_time,
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": False,
            },
            "authorization_scope": AUTHORIZATION_SCOPE,
            "authorized_by": authorization["authorized_by"],
            "control_receipt_id": _identifier(control_receipt_id),
            "expected_active_set_sha256": expected_active,
            "expected_registry_sha256": expected_registry,
            "production_control_authorization_receipt_ref": authorization_ref,
            "proposed_active_set": active_set,
            "proposed_active_set_sha256": proposed_active_sha,
            "proposed_registry_ref": registry_ref,
            "proposed_registry_sha256": proposed_registry_sha,
            "protocol_id": PROTOCOL_ID,
            "readiness_after_activation": readiness,
            "runtime_contracts_ref": source_refs["runtime_contracts"],
            "runtime_contracts_sha256": normalized_registry[
                "runtime_contracts_sha256"
            ],
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "transaction_id": _identifier(transaction_id),
            "v4_activation_receipt": v4_receipt,
            "v4_activation_receipt_ref": v4_receipt_ref,
            "v4_activation_request_ref": source_refs[
                "v4_activation_request"
            ],
            "v4_evidence_ref": source_refs["v4_evidence"],
            "v4_pre_activation_eligibility_ref": pre_ref,
            "v4_replay_ref": source_refs["v4_replay"],
            "v4_transaction_plan_ref": source_refs["v4_transaction_plan"],
        }
    )


def validate_production_control_transaction(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    pre_activation_eligibility: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_TRANSACTION_FIELDS,
        schema_version=TRANSACTION_SCHEMA_VERSION,
    )
    v4_receipt = result["v4_activation_receipt"]
    if type(v4_receipt) is not dict:
        _blocked()
    rebuilt = build_production_control_transaction(
        transaction_id=result["transaction_id"],
        registry=registry,
        pre_activation_eligibility=pre_activation_eligibility,
        authorization_receipt=authorization_receipt,
        expected_registry_sha256=result["expected_registry_sha256"],
        expected_active_set_sha256=result[
            "expected_active_set_sha256"
        ],
        activated_at=result["activated_at"],
        v4_activation_receipt_id=v4_receipt["receipt_id"],
        control_receipt_id=result["control_receipt_id"],
    )
    if result != rebuilt:
        _blocked()
    _authority(result["authority"], active=False)
    validate_active_set_pointer(result["proposed_active_set"])
    _validate_readiness_after_activation(
        result["readiness_after_activation"],
        registry=registry,
        v4_activation_receipt=v4_receipt,
    )
    return result


_WAL_FIELDS: Final = frozenset(
    {
        "expected_active_set_sha256",
        "expected_registry_sha256",
        "observed_active_set_sha256",
        "observed_registry_sha256",
        "post_active_set_sha256",
        "post_registry_sha256",
        "proposed_active_set_sha256",
        "proposed_registry_sha256",
        "protocol_id",
        "recorded_at",
        "schema_version",
        "semantic_sha256",
        "state",
        "transaction_id",
    }
)
_CONTROL_RECEIPT_FIELDS: Final = frozenset(
    {
        "activated_at",
        "active_for_production_research",
        "active_set_ref",
        "active_set_readback_sha256",
        "authority",
        "authorized_by",
        "protocol_id",
        "readiness_ref",
        "receipt_id",
        "registry_readback_sha256",
        "registry_ref",
        "schema_version",
        "semantic_sha256",
        "transaction_id",
        "transaction_ref",
        "v4_activation_receipt_ref",
    }
)
_ROLLBACK_FIELDS: Final = frozenset(
    {
        "authority",
        "authorization_scope",
        "authorized_by",
        "expected_active_set_sha256",
        "expected_registry_sha256",
        "observed_active_set_sha256",
        "observed_registry_sha256",
        "protocol_id",
        "receipt_id",
        "recorded_at",
        "restored_active_set_sha256",
        "restored_registry_sha256",
        "rollback_authorization_receipt_ref",
        "rollback_performed",
        "schema_version",
        "semantic_sha256",
        "transaction_id",
    }
)
_WAL_STATES: Final = (
    "PREPARED",
    "REGISTRY_COMMITTED",
    "V4_RECEIPT_ISSUED",
    "READINESS_RECOMPUTED",
    "ACTIVE_SET_COMMITTED",
    "ROLLED_BACK",
)


def _wal_record(
    transaction: Mapping[str, Any],
    *,
    state: str,
    observed_registry_sha256: str,
    observed_active_set_sha256: str,
    post_registry_sha256: str,
    post_active_set_sha256: str,
) -> dict[str, Any]:
    if state not in _WAL_STATES:
        _blocked()
    return _seal(
        {
            "expected_active_set_sha256": transaction[
                "expected_active_set_sha256"
            ],
            "expected_registry_sha256": transaction[
                "expected_registry_sha256"
            ],
            "observed_active_set_sha256": _sha(
                observed_active_set_sha256
            ),
            "observed_registry_sha256": _sha(
                observed_registry_sha256
            ),
            "post_active_set_sha256": _sha(post_active_set_sha256),
            "post_registry_sha256": _sha(post_registry_sha256),
            "proposed_active_set_sha256": transaction[
                "proposed_active_set_sha256"
            ],
            "proposed_registry_sha256": transaction[
                "proposed_registry_sha256"
            ],
            "protocol_id": PROTOCOL_ID,
            "recorded_at": transaction["activated_at"],
            "schema_version": WAL_SCHEMA_VERSION,
            "state": state,
            "transaction_id": transaction["transaction_id"],
        }
    )


def validate_wal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_WAL_FIELDS,
        schema_version=WAL_SCHEMA_VERSION,
    )
    if result["state"] not in _WAL_STATES:
        _blocked()
    _identifier(result["transaction_id"])
    _instant(result["recorded_at"])
    for field in (
        "expected_active_set_sha256",
        "expected_registry_sha256",
        "observed_active_set_sha256",
        "observed_registry_sha256",
        "post_active_set_sha256",
        "post_registry_sha256",
        "proposed_active_set_sha256",
        "proposed_registry_sha256",
    ):
        _sha(result[field])
    return result


def _build_control_receipt(
    *,
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    transaction_ref = build_artifact_ref(
        transaction,
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/transactions/"
            f"{transaction['transaction_id']}.json"
        ),
    )
    active_ref = build_artifact_ref(
        transaction["proposed_active_set"],
        relative_path=f"{ROOT_RELATIVE_PATH}/active_sets/_active.json",
    )
    readiness_ref = build_artifact_ref(
        transaction["readiness_after_activation"],
        relative_path=(
            f"{ROOT_RELATIVE_PATH}/readiness/"
            f"{transaction['transaction_id']}.json"
        ),
    )
    return _seal(
        {
            "activated_at": transaction["activated_at"],
            "active_for_production_research": True,
            "active_set_readback_sha256": transaction[
                "proposed_active_set_sha256"
            ],
            "active_set_ref": active_ref,
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": True,
            },
            "authorized_by": transaction["authorized_by"],
            "protocol_id": PROTOCOL_ID,
            "readiness_ref": readiness_ref,
            "receipt_id": transaction["control_receipt_id"],
            "registry_readback_sha256": transaction[
                "proposed_registry_sha256"
            ],
            "registry_ref": transaction["proposed_registry_ref"],
            "schema_version": CONTROL_RECEIPT_SCHEMA_VERSION,
            "transaction_id": transaction["transaction_id"],
            "transaction_ref": transaction_ref,
            "v4_activation_receipt_ref": transaction[
                "v4_activation_receipt_ref"
            ],
        }
    )


def validate_control_receipt(
    value: Mapping[str, Any],
    *,
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_CONTROL_RECEIPT_FIELDS,
        schema_version=CONTROL_RECEIPT_SCHEMA_VERSION,
    )
    if result != _build_control_receipt(transaction=transaction):
        _blocked()
    _authority(result["authority"], active=True)
    if result["active_for_production_research"] is not True:
        _blocked()
    return result


def _build_rollback_receipt(
    *,
    transaction: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
    receipt_id: str,
    recorded_at: str,
    observed_registry_sha256: str,
    observed_active_set_sha256: str,
    restored_registry_sha256: str,
    restored_active_set_sha256: str,
) -> dict[str, Any]:
    normalized = _sealed(
        transaction,
        fields=_TRANSACTION_FIELDS,
        schema_version=TRANSACTION_SCHEMA_VERSION,
    )
    authorization = validate_rollback_authorization_receipt(
        authorization_receipt,
        transaction=normalized,
        observed_at=recorded_at,
    )
    return _seal(
        {
            "authority": {
                **_NO_EXTERNAL_AUTHORITY,
                "active_for_production_research": False,
            },
            "authorization_scope": ROLLBACK_AUTHORIZATION_SCOPE,
            "authorized_by": authorization["authorized_by"],
            "expected_active_set_sha256": normalized[
                "expected_active_set_sha256"
            ],
            "expected_registry_sha256": normalized[
                "expected_registry_sha256"
            ],
            "observed_active_set_sha256": _sha(
                observed_active_set_sha256
            ),
            "observed_registry_sha256": _sha(
                observed_registry_sha256
            ),
            "protocol_id": PROTOCOL_ID,
            "receipt_id": _identifier(receipt_id),
            "recorded_at": _instant(recorded_at),
            "restored_active_set_sha256": _sha(
                restored_active_set_sha256
            ),
            "restored_registry_sha256": _sha(
                restored_registry_sha256
            ),
            "rollback_authorization_receipt_ref": build_artifact_ref(
                authorization,
                relative_path=(
                    f"{ROOT_RELATIVE_PATH}/rollback_authorizations/"
                    f"{authorization['receipt_id']}.json"
                ),
            ),
            "rollback_performed": True,
            "schema_version": ROLLBACK_RECEIPT_SCHEMA_VERSION,
            "transaction_id": normalized["transaction_id"],
        }
    )


def validate_rollback_receipt(
    value: Mapping[str, Any],
    *,
    transaction: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        value,
        fields=_ROLLBACK_FIELDS,
        schema_version=ROLLBACK_RECEIPT_SCHEMA_VERSION,
    )
    expected = _build_rollback_receipt(
        transaction=transaction,
        authorization_receipt=authorization_receipt,
        receipt_id=result["receipt_id"],
        recorded_at=result["recorded_at"],
        observed_registry_sha256=result["observed_registry_sha256"],
        observed_active_set_sha256=result["observed_active_set_sha256"],
        restored_registry_sha256=result["restored_registry_sha256"],
        restored_active_set_sha256=result["restored_active_set_sha256"],
    )
    if result != expected:
        _blocked()
    _authority(result["authority"], active=False)
    return result


class ProductionControlStore:
    """Owner-only private WAL/CAS store for Factor-v4 research activation."""

    def __init__(
        self,
        root: str | Path,
        *,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_absolute():
            _blocked()
        self._fault_hook = fault_hook
        self._prepare_directory(self.root)
        for relative in (
            "active_sets",
            "active_sets/snapshots",
            "authorizations",
            "eligibility",
            "readiness",
            "receipts",
            "receipts/control_activations",
            "receipts/rollbacks",
            "receipts/v4_activations",
            "registry",
            "registry/snapshots",
            "rollback_authorizations",
            "transactions",
            "wal",
        ):
            self._prepare_directory(self.root / relative)

    @property
    def registry_path(self) -> Path:
        return self.root / "registry/current.json"

    @property
    def active_set_path(self) -> Path:
        return self.root / "active_sets/_active.json"

    @property
    def registry_lock_path(self) -> Path:
        return self.root / "registry/.lock"

    @property
    def active_set_lock_path(self) -> Path:
        return self.root / "active_sets/.lock"

    def _prepare_directory(self, path: Path) -> None:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            observed = path.lstat()
        except OSError:
            _blocked()
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.getuid()
        ):
            _blocked()
        try:
            path.chmod(0o700)
        except OSError:
            _blocked()

    def _validate_regular(
        self,
        path: Path,
        *,
        allow_missing: bool,
    ) -> os.stat_result | None:
        try:
            observed = path.lstat()
        except FileNotFoundError:
            if allow_missing:
                return None
            _blocked()
        except OSError:
            _blocked()
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.getuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) & 0o077
            or observed.st_size >= MAX_JSON_BYTES
        ):
            _blocked()
        return observed

    def _read_optional(self, path: Path) -> bytes | None:
        before_path = self._validate_regular(path, allow_missing=True)
        if before_path is None:
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as handle:
                before = os.fstat(handle.fileno())
                raw = handle.read(MAX_JSON_BYTES)
                after = os.fstat(handle.fileno())
        except OSError:
            _blocked()
        if len(raw) >= MAX_JSON_BYTES:
            _blocked()
        after_path = self._validate_regular(path, allow_missing=False)
        if after_path is None:
            _blocked()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ) or (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (
            before_path.st_dev,
            before_path.st_ino,
            before_path.st_size,
        ) or (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) != (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
        ) or after.st_size != len(raw):
            _blocked()
        return raw

    def _read_json(self, path: Path) -> dict[str, Any]:
        raw = self._read_optional(path)
        if raw is None:
            _blocked()
        return _strict_object(raw)

    def _hash_optional(self, raw: bytes | None) -> str:
        return EMPTY_SHA256 if raw is None else hashlib.sha256(raw).hexdigest()

    def _atomic_replace(self, path: Path, raw: bytes) -> None:
        _strict_object(raw)
        self._prepare_directory(path.parent)
        existing = self._validate_regular(path, allow_missing=True)
        if existing is not None and existing.st_nlink != 1:
            _blocked()
        temporary = path.with_name(
            f".{path.name}.tmp-{secrets.token_hex(16)}"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            _blocked()
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if self._read_optional(path) != raw:
            _blocked()

    def _write_once(self, path: Path, value: Mapping[str, Any]) -> None:
        raw = canonical_file_bytes(value)
        current = self._read_optional(path)
        if current is not None:
            if current != raw:
                _blocked()
            return
        self._atomic_replace(path, raw)

    def _remove(self, path: Path) -> None:
        if self._validate_regular(path, allow_missing=True) is None:
            return
        try:
            path.unlink()
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            _blocked()

    @contextmanager
    def _lock(self, path: Path) -> Any:
        self._prepare_directory(path.parent)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.getuid()
                or observed.st_nlink != 1
            ):
                os.close(descriptor)
                _blocked()
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except OSError:
            _blocked()
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass

    def _wal_path(self, transaction_id: str) -> Path:
        return self.root / f"wal/{_identifier(transaction_id)}.jsonl"

    def _wal_records(self, transaction_id: str) -> list[dict[str, Any]]:
        raw = self._read_optional(self._wal_path(transaction_id))
        if raw is None:
            return []
        records: list[dict[str, Any]] = []
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                _blocked()
            records.append(validate_wal_record(_strict_object(line)))
        states = [record["state"] for record in records]
        if len(states) != len(set(states)):
            _blocked()
        indices = [_WAL_STATES.index(state) for state in states]
        if indices != sorted(indices):
            _blocked()
        return records

    def _append_wal(self, record: Mapping[str, Any]) -> None:
        normalized = validate_wal_record(record)
        path = self._wal_path(normalized["transaction_id"])
        existing = self._wal_records(normalized["transaction_id"])
        for prior in existing:
            if prior["state"] == normalized["state"]:
                if prior != normalized:
                    _blocked()
                return
        if existing and _WAL_STATES.index(normalized["state"]) <= _WAL_STATES.index(
            existing[-1]["state"]
        ):
            _blocked()
        raw = canonical_file_bytes(normalized)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.getuid()
                or observed.st_nlink != 1
            ):
                os.close(descriptor)
                _blocked()
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            _blocked()
        self._wal_records(normalized["transaction_id"])

    def _fault(self, state: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(state)

    def _snapshot(
        self,
        *,
        raw: bytes | None,
        sha256: str,
        kind: str,
    ) -> None:
        if raw is None:
            if sha256 != EMPTY_SHA256:
                _blocked()
            return
        if hashlib.sha256(raw).hexdigest() != sha256:
            _blocked()
        value = _strict_object(raw)
        self._write_once(
            self.root / f"{kind}/snapshots/{sha256}.json",
            value,
        )

    def _snapshot_bytes(self, *, sha256: str, kind: str) -> bytes | None:
        if sha256 == EMPTY_SHA256:
            return None
        raw = self._read_optional(
            self.root / f"{kind}/snapshots/{sha256}.json"
        )
        if raw is None or hashlib.sha256(raw).hexdigest() != sha256:
            _blocked()
        _strict_object(raw)
        return raw

    def apply(
        self,
        transaction: Mapping[str, Any],
        *,
        registry: Mapping[str, Any],
        pre_activation_eligibility: Mapping[str, Any],
        authorization_receipt: Mapping[str, Any],
        source_artifacts: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized_registry = validate_production_registry(registry)
        _validate_source_artifacts(
            registry=normalized_registry,
            source_artifacts=source_artifacts,
        )
        normalized = validate_production_control_transaction(
            transaction,
            registry=normalized_registry,
            pre_activation_eligibility=pre_activation_eligibility,
            authorization_receipt=authorization_receipt,
        )
        transaction_id = normalized["transaction_id"]
        self._write_once(
            self.root / f"transactions/{transaction_id}.json",
            normalized,
        )
        self._write_once(
            self.root
            / (
                "eligibility/"
                f"{pre_activation_eligibility['semantic_sha256']}.json"
            ),
            pre_activation_eligibility,
        )
        self._write_once(
            self.root
            / (
                "authorizations/"
                f"{authorization_receipt['receipt_id']}.json"
            ),
            authorization_receipt,
        )
        proposed_registry_raw = canonical_file_bytes(normalized_registry)
        proposed_active_raw = canonical_file_bytes(
            normalized["proposed_active_set"]
        )
        with self._lock(self.registry_lock_path):
            with self._lock(self.active_set_lock_path):
                before_registry_raw = self._read_optional(self.registry_path)
                before_active_raw = self._read_optional(self.active_set_path)
                observed_registry = self._hash_optional(before_registry_raw)
                observed_active = self._hash_optional(before_active_raw)
                allowed_registry = {
                    normalized["expected_registry_sha256"],
                    normalized["proposed_registry_sha256"],
                }
                allowed_active = {
                    normalized["expected_active_set_sha256"],
                    normalized["proposed_active_set_sha256"],
                }
                if (
                    observed_registry not in allowed_registry
                    or observed_active not in allowed_active
                    or (
                        observed_active
                        == normalized["proposed_active_set_sha256"]
                        and observed_registry
                        != normalized["proposed_registry_sha256"]
                    )
                ):
                    _blocked()
                existing_wal = self._wal_records(transaction_id)
                if any(
                    record["state"] == "ROLLED_BACK"
                    for record in existing_wal
                ):
                    _blocked()
                prepared = next(
                    (
                        record
                        for record in existing_wal
                        if record["state"] == "PREPARED"
                    ),
                    None,
                )
                wal_observed_registry = (
                    observed_registry
                    if prepared is None
                    else prepared["observed_registry_sha256"]
                )
                wal_observed_active = (
                    observed_active
                    if prepared is None
                    else prepared["observed_active_set_sha256"]
                )
                self._snapshot(
                    raw=before_registry_raw
                    if observed_registry
                    == normalized["expected_registry_sha256"]
                    else self._snapshot_bytes(
                        sha256=normalized["expected_registry_sha256"],
                        kind="registry",
                    ),
                    sha256=normalized["expected_registry_sha256"],
                    kind="registry",
                )
                self._snapshot(
                    raw=before_active_raw
                    if observed_active
                    == normalized["expected_active_set_sha256"]
                    else self._snapshot_bytes(
                        sha256=normalized["expected_active_set_sha256"],
                        kind="active_sets",
                    ),
                    sha256=normalized["expected_active_set_sha256"],
                    kind="active_sets",
                )
                self._append_wal(
                    _wal_record(
                        normalized,
                        state="PREPARED",
                        observed_registry_sha256=wal_observed_registry,
                        observed_active_set_sha256=wal_observed_active,
                        post_registry_sha256=wal_observed_registry,
                        post_active_set_sha256=wal_observed_active,
                    )
                )
                self._fault("PREPARED")
                if observed_registry == normalized["expected_registry_sha256"]:
                    self._atomic_replace(
                        self.registry_path,
                        proposed_registry_raw,
                    )
                post_registry = self._hash_optional(
                    self._read_optional(self.registry_path)
                )
                if post_registry != normalized["proposed_registry_sha256"]:
                    _blocked()
                self._append_wal(
                    _wal_record(
                        normalized,
                        state="REGISTRY_COMMITTED",
                        observed_registry_sha256=wal_observed_registry,
                        observed_active_set_sha256=wal_observed_active,
                        post_registry_sha256=post_registry,
                        post_active_set_sha256=wal_observed_active,
                    )
                )
                self._fault("REGISTRY_COMMITTED")
                v4_receipt = normalized["v4_activation_receipt"]
                self._write_once(
                    self.root
                    / (
                        "receipts/v4_activations/"
                        f"{v4_receipt['receipt_id']}.json"
                    ),
                    v4_receipt,
                )
                self._append_wal(
                    _wal_record(
                        normalized,
                        state="V4_RECEIPT_ISSUED",
                        observed_registry_sha256=wal_observed_registry,
                        observed_active_set_sha256=wal_observed_active,
                        post_registry_sha256=post_registry,
                        post_active_set_sha256=wal_observed_active,
                    )
                )
                self._fault("V4_RECEIPT_ISSUED")
                readiness = normalized["readiness_after_activation"]
                self._write_once(
                    self.root / f"readiness/{transaction_id}.json",
                    readiness,
                )
                _validate_readiness_after_activation(
                    self._read_json(
                        self.root / f"readiness/{transaction_id}.json"
                    ),
                    registry=normalized_registry,
                    v4_activation_receipt=v4_receipt,
                )
                self._append_wal(
                    _wal_record(
                        normalized,
                        state="READINESS_RECOMPUTED",
                        observed_registry_sha256=wal_observed_registry,
                        observed_active_set_sha256=wal_observed_active,
                        post_registry_sha256=post_registry,
                        post_active_set_sha256=wal_observed_active,
                    )
                )
                self._fault("READINESS_RECOMPUTED")
                if observed_active == normalized["expected_active_set_sha256"]:
                    self._atomic_replace(
                        self.active_set_path,
                        proposed_active_raw,
                    )
                post_active = self._hash_optional(
                    self._read_optional(self.active_set_path)
                )
                if post_active != normalized["proposed_active_set_sha256"]:
                    _blocked()
                validate_active_set_pointer(
                    self._read_json(self.active_set_path)
                )
                self._append_wal(
                    _wal_record(
                        normalized,
                        state="ACTIVE_SET_COMMITTED",
                        observed_registry_sha256=wal_observed_registry,
                        observed_active_set_sha256=wal_observed_active,
                        post_registry_sha256=post_registry,
                        post_active_set_sha256=post_active,
                    )
                )
                self._fault("ACTIVE_SET_COMMITTED")
                receipt = _build_control_receipt(transaction=normalized)
                receipt_path = self.root / (
                    "receipts/control_activations/"
                    f"{receipt['receipt_id']}.json"
                )
                self._write_once(receipt_path, receipt)
                return validate_control_receipt(
                    self._read_json(receipt_path),
                    transaction=normalized,
                )

    def rollback(
        self,
        transaction: Mapping[str, Any],
        *,
        receipt_id: str,
        authorization_receipt: Mapping[str, Any],
        recorded_at: str,
    ) -> dict[str, Any]:
        normalized = _sealed(
            transaction,
            fields=_TRANSACTION_FIELDS,
            schema_version=TRANSACTION_SCHEMA_VERSION,
        )
        stored_transaction = self._read_json(
            self.root / f"transactions/{normalized['transaction_id']}.json"
        )
        if stored_transaction != normalized:
            _blocked()
        control_receipt = self._read_json(
            self.root
            / (
                "receipts/control_activations/"
                f"{normalized['control_receipt_id']}.json"
            )
        )
        validate_control_receipt(
            control_receipt,
            transaction=normalized,
        )
        authorization = validate_rollback_authorization_receipt(
            authorization_receipt,
            transaction=normalized,
            observed_at=recorded_at,
        )
        _validate_ref_payload(
            authorization["control_receipt_ref"],
            control_receipt,
            expected_schema=CONTROL_RECEIPT_SCHEMA_VERSION,
        )
        self._write_once(
            self.root
            / (
                "rollback_authorizations/"
                f"{authorization['receipt_id']}.json"
            ),
            authorization,
        )
        with self._lock(self.registry_lock_path):
            with self._lock(self.active_set_lock_path):
                registry_raw = self._read_optional(self.registry_path)
                active_raw = self._read_optional(self.active_set_path)
                observed_registry = self._hash_optional(registry_raw)
                observed_active = self._hash_optional(active_raw)
                if (
                    observed_registry
                    != normalized["proposed_registry_sha256"]
                    or observed_active
                    != normalized["proposed_active_set_sha256"]
                ):
                    _blocked()
                restore_active = self._snapshot_bytes(
                    sha256=normalized["expected_active_set_sha256"],
                    kind="active_sets",
                )
                restore_registry = self._snapshot_bytes(
                    sha256=normalized["expected_registry_sha256"],
                    kind="registry",
                )
                if restore_active is None:
                    self._remove(self.active_set_path)
                else:
                    self._atomic_replace(self.active_set_path, restore_active)
                if restore_registry is None:
                    self._remove(self.registry_path)
                else:
                    self._atomic_replace(self.registry_path, restore_registry)
                post_registry = self._hash_optional(
                    self._read_optional(self.registry_path)
                )
                post_active = self._hash_optional(
                    self._read_optional(self.active_set_path)
                )
                if (
                    post_registry != normalized["expected_registry_sha256"]
                    or post_active
                    != normalized["expected_active_set_sha256"]
                ):
                    _blocked()
                self._append_wal(
                    _wal_record(
                        normalized,
                        state="ROLLED_BACK",
                        observed_registry_sha256=observed_registry,
                        observed_active_set_sha256=observed_active,
                        post_registry_sha256=post_registry,
                        post_active_set_sha256=post_active,
                    )
                )
                receipt = _build_rollback_receipt(
                    transaction=normalized,
                    authorization_receipt=authorization,
                    receipt_id=receipt_id,
                    recorded_at=recorded_at,
                    observed_registry_sha256=observed_registry,
                    observed_active_set_sha256=observed_active,
                    restored_registry_sha256=post_registry,
                    restored_active_set_sha256=post_active,
                )
                path = self.root / (
                    f"receipts/rollbacks/{receipt['receipt_id']}.json"
                )
                self._write_once(path, receipt)
                return validate_rollback_receipt(
                    self._read_json(path),
                    transaction=normalized,
                    authorization_receipt=authorization,
                )


__all__ = [
    "ACTIVE_SET_SCHEMA_VERSION",
    "ARTIFACT_REF_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION",
    "AUTHORIZATION_SCOPE",
    "CONTROL_RECEIPT_SCHEMA_VERSION",
    "EMPTY_SHA256",
    "EVIDENCE_SET_SCHEMA_VERSION",
    "PRE_ACTIVATION_SCHEMA_VERSION",
    "PROTOCOL_ID",
    "ProductionControlCrash",
    "ProductionControlError",
    "ProductionControlStore",
    "READINESS_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "REPLAY_SET_SCHEMA_VERSION",
    "ROLLBACK_AUTHORIZATION_SCHEMA_VERSION",
    "ROLLBACK_AUTHORIZATION_SCOPE",
    "ROLLBACK_RECEIPT_SCHEMA_VERSION",
    "ROOT_RELATIVE_PATH",
    "RUNTIME_CONTRACT_SET_SCHEMA_VERSION",
    "TRANSACTION_SCHEMA_VERSION",
    "WAL_SCHEMA_VERSION",
    "build_artifact_ref",
    "build_authorization_receipt",
    "build_pre_activation_eligibility",
    "build_production_control_transaction",
    "build_production_registry",
    "build_rollback_authorization_receipt",
    "build_runtime_contract_set",
    "build_v4_evidence_set",
    "build_v4_replay_set",
    "canonical_file_bytes",
    "validate_active_set_pointer",
    "validate_artifact_ref",
    "validate_authorization_receipt",
    "validate_control_receipt",
    "validate_pre_activation_eligibility",
    "validate_production_control_transaction",
    "validate_production_registry",
    "validate_rollback_authorization_receipt",
    "validate_rollback_receipt",
    "validate_wal_record",
]
