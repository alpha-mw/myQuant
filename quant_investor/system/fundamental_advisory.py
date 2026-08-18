"""Immutable Fundamental age disclosure and build-scoped veto authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import re
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)

from .errors import SystemContractError, SystemPreconditionError
from .store import OBJECT_REF_SORT_FIELDS, validate_object_ref

FUNDAMENTAL_VETO_SUBJECT_KIND: Final = "system.fundamental_veto_subject"
FUNDAMENTAL_OPERATOR_VETO_KIND: Final = "system.fundamental_operator_veto"
FUNDAMENTAL_ADVISORY_KIND: Final = "system.fundamental_advisory_evidence"

FUNDAMENTAL_VETO_SUBJECT_CONTRACT_SHA256: Final = get_contract(
    FUNDAMENTAL_VETO_SUBJECT_KIND
).contract_sha256
FUNDAMENTAL_OPERATOR_VETO_CONTRACT_SHA256: Final = get_contract(
    FUNDAMENTAL_OPERATOR_VETO_KIND
).contract_sha256
FUNDAMENTAL_ADVISORY_CONTRACT_SHA256: Final = get_contract(
    FUNDAMENTAL_ADVISORY_KIND
).contract_sha256

FUNDAMENTAL_VETO_SUBJECT_FIELDS: Final = frozenset(
    {
        "veto_subject_id",
        "state",
        "bootstrap_admission_intent_sha256",
        "deployed_release_ref",
        "release_code_manifest_sha256",
        "system_as_of_date",
        "calendar_compilation_ref",
        "exchange_calendar_ref",
        "current_market_pointer_ref",
        "current_pit_pointer_ref",
        "current_pit_membership_ref",
        "fundamental_pointer_ref",
        "fundamental_manifest_ref",
        "fundamental_table_refs",
        "fundamental_evidence_refs",
        "fundamental_provenance_binding_sha256",
        "fundamental_target_bindings_sha256",
        "fundamental_snapshot_cutoff_date",
        "factor_set_sha256",
        "factor_dependency_rows",
        "factor_dependency_sha256",
    }
)
FUNDAMENTAL_OPERATOR_VETO_FIELDS: Final = frozenset(
    {
        "veto_id",
        "state",
        "veto_subject_ref",
        "reason_codes",
        "issued_at",
        "actor_uid",
        "os_actor",
        "human_signature_claimed",
        "system_activation_authorized",
        "factor_activation_authorized",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
        "order_authority",
        "trade_authority",
        "funds_transfer_authority",
    }
)
FUNDAMENTAL_ADVISORY_FIELDS: Final = frozenset(
    {
        "fundamental_advisory_id",
        "state",
        "veto_subject_ref",
        "operator_veto_ref",
        "integrity_status",
        "required_by_active_factor_set",
        "system_as_of_date",
        "fundamental_snapshot_cutoff_date",
        "calendar_age_days",
        "open_session_age",
        "latest_admitted_available_at",
        "last_refresh_basis",
        "disclosure_check",
        "freshness_policy",
        "default_action",
        "operator_veto_present",
        "effective_action",
        "factor_dependency_rows",
        "factor_dependency_sha256",
        "fundamental_machine_states",
        "source_limitations",
        "generic_json_max_bytes",
        "predecessor_manifest_max_bytes",
        "fundamental_parquet_max_bytes",
        "generic_replay_max_cells",
        "daily_replay_max_cells",
        "fundamental_table_source_rows",
        "predecessor_manifest_source_ref",
        "ordinary_json_source_refs",
    }
)

FUNDAMENTAL_FRESHNESS_POLICY: Final = "ADVISORY_NO_FIXED_MAXIMUM"
FUNDAMENTAL_DEFAULT_ACTION: Final = "PROCEED"
FUNDAMENTAL_VETO_ACTION: Final = "BLOCK"
FUNDAMENTAL_LAST_REFRESH_BASIS: Final = "SNAPSHOT_CUTOFF_DATE"
FUNDAMENTAL_DISCLOSURE_CHECK: Final = "PASS"
FUNDAMENTAL_INTEGRITY_STATUS: Final = "VERIFIED"
FUNDAMENTAL_VETO_REASON_CODES: Final = frozenset(
    {
        "FUNDAMENTAL_DISCLOSURE_REVIEW_REQUIRED",
        "FUNDAMENTAL_OPERATOR_HOLD",
        "FUNDAMENTAL_SOURCE_REVIEW_REQUIRED",
    }
)
FACTOR_SOURCE_ROLES: Final = frozenset(
    {"EXCHANGE_CALENDAR", "FUNDAMENTAL", "MARKET", "PIT_MEMBERSHIP"}
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical identifier")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC seconds")
    return value


def _date(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not an ISO date")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemContractError(f"{label} is not an ISO date") from exc
    if parsed.isoformat() != value:
        raise SystemContractError(f"{label} is not an ISO date")
    return value


def _refs(value: Any, *, label: str, minimum: int = 0) -> list[dict[str, str]]:
    if type(value) is not list or len(value) < minimum:
        raise SystemContractError(f"{label} is not an exact reference list")
    rows = [validate_object_ref(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    keys = [tuple(row[field] for field in OBJECT_REF_SORT_FIELDS) for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SystemContractError(f"{label} is not tuple-sorted and unique")
    return rows


def validate_factor_dependency_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemContractError("factor dependency rows are absent")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != {"factor_id", "required_source_roles"}:
            raise SystemContractError("factor dependency row fields are not exact")
        factor_id = _identifier(raw["factor_id"], label=f"factor_dependency_rows[{index}]")
        roles = raw["required_source_roles"]
        if (
            type(roles) is not list
            or not roles
            or any(type(role) is not str or role not in FACTOR_SOURCE_ROLES for role in roles)
            or roles != sorted(set(roles))
        ):
            raise SystemContractError("factor dependency roles are not exact")
        rows.append({"factor_id": factor_id, "required_source_roles": list(roles)})
    if rows != sorted(rows, key=lambda row: row["factor_id"].encode("utf-8")):
        raise SystemContractError("factor dependency rows are not sorted")
    if len({row["factor_id"] for row in rows}) != len(rows):
        raise SystemContractError("factor dependency rows are duplicated")
    return rows


def factor_dependency_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = validate_factor_dependency_rows([dict(row) for row in rows])
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _identity(prefix: str, domain: str, body: Mapping[str, Any]) -> str:
    return (
        prefix
        + hashlib.sha256(
            canonical_json_bytes({"domain": domain, "payload": dict(body)})
        ).hexdigest()
    )


def validate_fundamental_veto_subject(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FUNDAMENTAL_VETO_SUBJECT_KIND,
            expected_contract_sha256=FUNDAMENTAL_VETO_SUBJECT_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("Fundamental veto subject contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != FUNDAMENTAL_VETO_SUBJECT_FIELDS or payload["state"] != "VERIFIED":
        raise SystemContractError("Fundamental veto subject fields/state differ")
    for field in (
        "bootstrap_admission_intent_sha256",
        "release_code_manifest_sha256",
        "fundamental_provenance_binding_sha256",
        "fundamental_target_bindings_sha256",
        "factor_set_sha256",
        "factor_dependency_sha256",
    ):
        _sha(payload[field], label=field)
    _date(payload["system_as_of_date"], label="system_as_of_date")
    _date(payload["fundamental_snapshot_cutoff_date"], label="fundamental_snapshot_cutoff_date")
    for field in (
        "deployed_release_ref",
        "calendar_compilation_ref",
        "exchange_calendar_ref",
        "current_market_pointer_ref",
        "current_pit_pointer_ref",
        "current_pit_membership_ref",
        "fundamental_pointer_ref",
        "fundamental_manifest_ref",
    ):
        validate_object_ref(payload[field], label=field)
    _refs(payload["fundamental_table_refs"], label="fundamental_table_refs", minimum=3)
    _refs(payload["fundamental_evidence_refs"], label="fundamental_evidence_refs", minimum=1)
    rows = validate_factor_dependency_rows(payload["factor_dependency_rows"])
    if factor_dependency_sha256(rows) != payload["factor_dependency_sha256"]:
        raise SystemContractError("Fundamental veto subject dependency SHA differs")
    body = {key: payload[key] for key in sorted(payload) if key != "veto_subject_id"}
    if payload["veto_subject_id"] != _identity(
        "fundamental-veto-subject-", "myquant-fundamental-veto-subject", body
    ):
        raise SystemContractError("Fundamental veto subject identity differs")
    return artifact


def build_fundamental_veto_subject(*, created_at: str, **fields: Any) -> dict[str, Any]:
    body = {"state": "VERIFIED", **fields}
    artifact = seal_artifact(
        FUNDAMENTAL_VETO_SUBJECT_KIND,
        {
            "veto_subject_id": _identity(
                "fundamental-veto-subject-", "myquant-fundamental-veto-subject", body
            ),
            **body,
        },
        created_at=created_at,
    )
    return validate_fundamental_veto_subject(artifact)


def validate_fundamental_operator_veto(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FUNDAMENTAL_OPERATOR_VETO_KIND,
            expected_contract_sha256=FUNDAMENTAL_OPERATOR_VETO_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("Fundamental operator veto contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != FUNDAMENTAL_OPERATOR_VETO_FIELDS or payload["state"] != "VETO":
        raise SystemContractError("Fundamental operator veto fields/state differ")
    validate_object_ref(payload["veto_subject_ref"], label="veto_subject_ref")
    reasons = payload["reason_codes"]
    if (
        type(reasons) is not list
        or not reasons
        or any(type(row) is not str or row not in FUNDAMENTAL_VETO_REASON_CODES for row in reasons)
        or reasons != sorted(set(reasons))
    ):
        raise SystemContractError("Fundamental operator veto reasons differ")
    issued = _timestamp(payload["issued_at"], label="issued_at")
    if artifact["created_at"] != issued:
        raise SystemContractError("Fundamental operator veto issuance binding differs")
    uid = payload["actor_uid"]
    if type(uid) is not int or uid < 0 or payload["os_actor"] != f"uid:{uid}":
        raise SystemContractError("Fundamental operator veto actor differs")
    false_fields = (
        "human_signature_claimed",
        "system_activation_authorized",
        "factor_activation_authorized",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
        "order_authority",
        "trade_authority",
        "funds_transfer_authority",
    )
    if any(payload[field] is not False for field in false_fields):
        raise SystemContractError("Fundamental operator veto claims prohibited authority")
    body = {key: payload[key] for key in sorted(payload) if key != "veto_id"}
    if payload["veto_id"] != _identity(
        "fundamental-veto-", "myquant-fundamental-operator-veto", body
    ):
        raise SystemContractError("Fundamental operator veto identity differs")
    return artifact


def build_fundamental_operator_veto(
    *,
    veto_subject_ref: Mapping[str, Any],
    reason_codes: Sequence[str],
    issued_at: str,
    actor_uid: int,
) -> dict[str, Any]:
    body = {
        "state": "VETO",
        "veto_subject_ref": dict(veto_subject_ref),
        "reason_codes": list(reason_codes),
        "issued_at": issued_at,
        "actor_uid": actor_uid,
        "os_actor": f"uid:{actor_uid}",
        "human_signature_claimed": False,
        "system_activation_authorized": False,
        "factor_activation_authorized": False,
        "portfolio_authority": False,
        "strategy_record_authority": False,
        "broker_authority": False,
        "order_authority": False,
        "trade_authority": False,
        "funds_transfer_authority": False,
    }
    artifact = seal_artifact(
        FUNDAMENTAL_OPERATOR_VETO_KIND,
        {
            "veto_id": _identity("fundamental-veto-", "myquant-fundamental-operator-veto", body),
            **body,
        },
        created_at=issued_at,
    )
    return validate_fundamental_operator_veto(artifact)


def validate_fundamental_advisory(  # noqa: C901
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FUNDAMENTAL_ADVISORY_KIND,
            expected_contract_sha256=FUNDAMENTAL_ADVISORY_CONTRACT_SHA256,
        )
    except ContractError as exc:
        raise SystemContractError("Fundamental advisory contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != FUNDAMENTAL_ADVISORY_FIELDS or payload["state"] != "VERIFIED":
        raise SystemContractError("Fundamental advisory fields/state differ")
    validate_object_ref(payload["veto_subject_ref"], label="veto_subject_ref")
    veto_ref = payload["operator_veto_ref"]
    if veto_ref is not None:
        validate_object_ref(veto_ref, label="operator_veto_ref")
    if payload["integrity_status"] != FUNDAMENTAL_INTEGRITY_STATUS:
        raise SystemContractError("Fundamental advisory integrity differs")
    if type(payload["required_by_active_factor_set"]) is not bool:
        raise SystemContractError("Fundamental advisory dependency projection differs")
    system_date = _date(payload["system_as_of_date"], label="system_as_of_date")
    fundamental_date = _date(
        payload["fundamental_snapshot_cutoff_date"],
        label="fundamental_snapshot_cutoff_date",
    )
    expected_days = (date.fromisoformat(system_date) - date.fromisoformat(fundamental_date)).days
    if expected_days < 0 or payload["calendar_age_days"] != expected_days:
        raise SystemContractError("Fundamental advisory calendar age differs")
    if type(payload["open_session_age"]) is not int or payload["open_session_age"] < 0:
        raise SystemContractError("Fundamental advisory open-session age differs")
    _date(payload["latest_admitted_available_at"], label="latest_admitted_available_at")
    if (
        payload["last_refresh_basis"] != FUNDAMENTAL_LAST_REFRESH_BASIS
        or payload["disclosure_check"] != FUNDAMENTAL_DISCLOSURE_CHECK
        or payload["freshness_policy"] != FUNDAMENTAL_FRESHNESS_POLICY
        or payload["default_action"] != FUNDAMENTAL_DEFAULT_ACTION
    ):
        raise SystemContractError("Fundamental advisory policy differs")
    veto_present = veto_ref is not None
    expected_action = FUNDAMENTAL_VETO_ACTION if veto_present else FUNDAMENTAL_DEFAULT_ACTION
    if (
        payload["operator_veto_present"] is not veto_present
        or payload["effective_action"] != expected_action
    ):
        raise SystemContractError("Fundamental advisory effective action differs")
    rows = validate_factor_dependency_rows(payload["factor_dependency_rows"])
    if factor_dependency_sha256(rows) != payload["factor_dependency_sha256"]:
        raise SystemContractError("Fundamental advisory dependency SHA differs")
    machine = payload["fundamental_machine_states"]
    if type(machine) is not dict:
        raise SystemContractError("Fundamental advisory machine states are absent")
    limitations = payload["source_limitations"]
    if (
        type(limitations) is not list
        or limitations != sorted(set(limitations))
        or any(type(row) is not str or not row for row in limitations)
    ):
        raise SystemContractError("Fundamental advisory limitations differ")
    for field in (
        "generic_json_max_bytes",
        "predecessor_manifest_max_bytes",
        "fundamental_parquet_max_bytes",
        "generic_replay_max_cells",
        "daily_replay_max_cells",
    ):
        if type(payload[field]) is not int or payload[field] <= 0:
            raise SystemContractError("Fundamental advisory size policy differs")
    table_rows = payload["fundamental_table_source_rows"]
    if type(table_rows) is not list or len(table_rows) != 3:
        raise SystemContractError("Fundamental advisory table role rows differ")
    expected_names = [
        "fundamental_daily",
        "fundamental_period",
        "fundamental_quarantine",
    ]
    observed_names: list[str] = []
    for index, row in enumerate(table_rows):
        if type(row) is not dict or set(row) != {
            "table_name",
            "source_ref",
            "row_count",
            "column_count",
            "observed_cells",
            "cell_limit",
        }:
            raise SystemContractError("Fundamental advisory table row fields differ")
        observed_names.append(row["table_name"])
        validate_object_ref(row["source_ref"], label=f"fundamental table row {index}")
        expected_cell_limit = (
            payload["daily_replay_max_cells"]
            if row["table_name"] == "fundamental_daily"
            else payload["generic_replay_max_cells"]
        )
        if (
            type(row["row_count"]) is not int
            or row["row_count"] < 0
            or type(row["column_count"]) is not int
            or row["column_count"] < 0
            or row["observed_cells"] != row["row_count"] * row["column_count"]
            or row["cell_limit"] != expected_cell_limit
            or row["observed_cells"] > row["cell_limit"]
        ):
            raise SystemContractError("Fundamental advisory table cell policy differs")
    if observed_names != expected_names:
        raise SystemContractError("Fundamental advisory table roles differ")
    validate_object_ref(
        payload["predecessor_manifest_source_ref"],
        label="predecessor_manifest_source_ref",
    )
    _refs(payload["ordinary_json_source_refs"], label="ordinary_json_source_refs")
    body = {key: payload[key] for key in sorted(payload) if key != "fundamental_advisory_id"}
    if payload["fundamental_advisory_id"] != _identity(
        "fundamental-advisory-", "myquant-fundamental-advisory", body
    ):
        raise SystemContractError("Fundamental advisory identity differs")
    return artifact


def build_fundamental_advisory(*, created_at: str, **fields: Any) -> dict[str, Any]:
    body = {"state": "VERIFIED", **fields}
    artifact = seal_artifact(
        FUNDAMENTAL_ADVISORY_KIND,
        {
            "fundamental_advisory_id": _identity(
                "fundamental-advisory-", "myquant-fundamental-advisory", body
            ),
            **body,
        },
        created_at=created_at,
    )
    return validate_fundamental_advisory(artifact)


def require_fundamental_proceed(advisory: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = validate_fundamental_advisory(advisory)
    if artifact["payload"]["effective_action"] != FUNDAMENTAL_DEFAULT_ACTION:
        raise SystemPreconditionError("Fundamental operator veto blocks production admission")
    return artifact


__all__ = [
    "FACTOR_SOURCE_ROLES",
    "FUNDAMENTAL_ADVISORY_CONTRACT_SHA256",
    "FUNDAMENTAL_ADVISORY_FIELDS",
    "FUNDAMENTAL_ADVISORY_KIND",
    "FUNDAMENTAL_DEFAULT_ACTION",
    "FUNDAMENTAL_OPERATOR_VETO_CONTRACT_SHA256",
    "FUNDAMENTAL_OPERATOR_VETO_FIELDS",
    "FUNDAMENTAL_OPERATOR_VETO_KIND",
    "FUNDAMENTAL_VETO_REASON_CODES",
    "FUNDAMENTAL_VETO_SUBJECT_CONTRACT_SHA256",
    "FUNDAMENTAL_VETO_SUBJECT_FIELDS",
    "FUNDAMENTAL_VETO_SUBJECT_KIND",
    "build_fundamental_advisory",
    "build_fundamental_operator_veto",
    "build_fundamental_veto_subject",
    "factor_dependency_sha256",
    "require_fundamental_proceed",
    "validate_factor_dependency_rows",
    "validate_fundamental_advisory",
    "validate_fundamental_operator_veto",
    "validate_fundamental_veto_subject",
]
