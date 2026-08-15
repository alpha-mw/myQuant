"""Canonical primitives shared by stable Factor governance artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import math
import re
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    validate_artifact,
)

from .errors import FactorGovernanceError

DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
A_SHARE_SYMBOL_RE: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
ARTIFACT_REF_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
)

COVERAGE_MINIMUM: Final = Decimal("0.80")
TOTAL_OPEN_SESSIONS: Final = 390
SIGNAL_OPEN_SESSIONS: Final = 360
ANNUAL_OPEN_SESSIONS: Final = 252
LABEL_HORIZON_OPEN_SESSIONS: Final = 30
MIN_DAILY_RANKIC_SESSIONS: Final = 300
MIN_CLOSED_MONTH_ENDS: Final = 12
MIN_DISJOINT_COHORTS: Final = 8
CPCV_BLOCK_COUNT: Final = 10
CPCV_TEST_BLOCK_COUNT: Final = 2
CPCV_PATH_COUNT: Final = 45
CPCV_PURGE_OPEN_SESSIONS: Final = 30
CPCV_EMBARGO_OPEN_SESSIONS: Final = 30
PBO_SPLIT_COUNT: Final = 252
PBO_MIN_CONFIGURATIONS: Final = 2
T_STAT_HURDLE: Final = Decimal("3")
DSR_FLOOR: Final = Decimal("0.95")
PBO_CEILING: Final = Decimal("0.50")
BH_Q_CEILING: Final = Decimal("0.10")
POSITIVE_PATH_RATIO_FLOOR: Final = Decimal("0.55")
TURNOVER_CEILING: Final = Decimal("12")
COST_BPS: Final = Decimal("1")
REDUNDANCY_CORRELATION_FLOOR: Final = Decimal("0.70")
REDUNDANCY_MIN_OVERLAP: Final = 12
SHRINKAGE_PSEUDO_COUNT: Final = Decimal("10")
BOOTSTRAP_VALIDATION_PROFILE_ID: Final = "factor-bootstrap-contextual-validation"
PROSPECTIVE_VALIDATION_PROFILE_ID: Final = "factor-prospective-contextual-validation"


def canonical_timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise FactorGovernanceError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise FactorGovernanceError(f"{label} must be canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise FactorGovernanceError(f"{label} must be canonical UTC seconds")
    return value


def canonical_identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or IDENTIFIER_RE.fullmatch(value) is None:
        raise FactorGovernanceError(f"{label} must be a canonical identifier")
    return value


def canonical_a_share_symbol(value: Any, *, label: str) -> str:
    if type(value) is not str or A_SHARE_SYMBOL_RE.fullmatch(value) is None:
        raise FactorGovernanceError(f"{label} must be a canonical A-share symbol")
    return value


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise FactorGovernanceError(f"{label} must be lowercase SHA-256")
    return value


def decimal_value(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool):
        raise FactorGovernanceError(f"{label} must be numeric")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise FactorGovernanceError(f"{label} must be numeric") from exc
    if not parsed.is_finite():
        raise FactorGovernanceError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise FactorGovernanceError(f"{label} is below its allowed domain")
    if maximum is not None and parsed > maximum:
        raise FactorGovernanceError(f"{label} is above its allowed domain")
    return parsed


def decimal_text(value: Any, *, label: str = "decimal") -> str:
    parsed = decimal_value(value, label=label)
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        return format(parsed.quantize(DECIMAL_QUANTUM), "f")


def finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise FactorGovernanceError(f"{label} must be finite")
    return result


def canonical_sessions(values: Sequence[Any]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FactorGovernanceError("open_sessions must be a sequence")
    sessions: list[str] = []
    for index, value in enumerate(values):
        if type(value) is not str:
            raise FactorGovernanceError(f"open_sessions[{index}] must be an ISO date")
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise FactorGovernanceError(f"open_sessions[{index}] must be an ISO date") from exc
        if parsed.strftime("%Y-%m-%d") != value:
            raise FactorGovernanceError(f"open_sessions[{index}] must be canonical")
        sessions.append(value)
    if sessions != sorted(set(sessions)):
        raise FactorGovernanceError("open_sessions must be unique and increasing")
    return sessions


def business_identity(prefix: str, inputs: Mapping[str, Any]) -> str:
    """Hash declared business identity inputs, never a mutable full payload."""

    canonical_identifier(prefix, label="identity prefix")
    digest = hashlib.sha256(canonical_json_bytes(dict(inputs))).hexdigest()
    return f"{prefix}-{digest}"


def observation_lineage_identity(preregistration_id: str, selection_id: str) -> str:
    """Return the immutable prospective append-line identity."""

    return business_identity(
        "observation-lineage",
        {
            "preregistration_id": preregistration_id,
            "selection_id": selection_id,
        },
    )


def prospective_validation_namespace_id(
    *,
    exchange_calendar_ref: Mapping[str, Any],
    implementation_manifest_ref: Mapping[str, Any],
    factor_validator_manifest_ref: Mapping[str, Any],
) -> str:
    """Derive the prospective namespace solely from the fixed mine roots."""

    return business_identity(
        "factor-validation-namespace",
        {
            "validation_profile_id": PROSPECTIVE_VALIDATION_PROFILE_ID,
            "exchange_calendar_ref": validate_artifact_ref(
                dict(exchange_calendar_ref),
                label="exchange_calendar_ref",
                expected_kind="system.source_object",
            ),
            "implementation_manifest_ref": validate_artifact_ref(
                dict(implementation_manifest_ref),
                label="implementation_manifest_ref",
                expected_kind="system.source_object",
            ),
            "factor_validator_manifest_ref": validate_artifact_ref(
                dict(factor_validator_manifest_ref),
                label="factor_validator_manifest_ref",
                expected_kind="factor.validator_manifest",
            ),
        },
    )


def bootstrap_validation_namespace_id(*, intrinsic_receipt_ref: Mapping[str, Any]) -> str:
    """Derive the bootstrap namespace from its complete intrinsic receipt."""

    return business_identity(
        "factor-validation-namespace",
        {
            "validation_profile_id": BOOTSTRAP_VALIDATION_PROFILE_ID,
            "intrinsic_receipt_ref": validate_artifact_ref(
                dict(intrinsic_receipt_ref),
                label="intrinsic_receipt_ref",
                expected_kind="factor.validation_receipt",
            ),
        },
    )


def validate_governance_artifact(
    document: Mapping[str, Any] | bytes,
    *,
    expected_kind: str | None = None,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Map contract-envelope failures to the public Factor validation boundary."""

    try:
        return validate_artifact(
            document,
            expected_kind=expected_kind,
            expected_contract_sha256=expected_contract_sha256,
        )
    except ContractError as exc:
        raise FactorGovernanceError("artifact envelope is invalid") from exc


def artifact_ref(document: Mapping[str, Any] | bytes) -> dict[str, str]:
    """Return the exact five-field reference for a validated artifact."""

    envelope = validate_governance_artifact(document)
    try:
        byte_sha256 = artifact_byte_sha256(document)
    except ContractError as exc:
        raise FactorGovernanceError("artifact envelope is invalid") from exc
    return {
        "kind": envelope["kind"],
        "contract_sha256": envelope["contract_sha256"],
        "artifact_id": envelope["artifact_id"],
        "semantic_sha256": envelope["semantic_sha256"],
        "byte_sha256": byte_sha256,
    }


def validate_artifact_ref(
    value: Any,
    *,
    label: str,
    expected_kind: str | None = None,
) -> dict[str, str]:
    """Validate an exact ref and require its compiled contract pair."""

    if type(value) is not dict or set(value) != set(ARTIFACT_REF_FIELDS):
        raise FactorGovernanceError(f"{label} fields are not exact")
    kind = value.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise FactorGovernanceError(f"{label} has the wrong artifact kind")
    if type(kind) is not str:
        raise FactorGovernanceError(f"{label}.kind is invalid")
    contract_sha256 = require_sha256(value.get("contract_sha256"), label=f"{label}.contract_sha256")
    try:
        definition = get_contract(kind, contract_sha256)
    except Exception as exc:
        raise FactorGovernanceError(f"{label} contract pair is not compiled") from exc
    artifact_id = value.get("artifact_id")
    if (
        type(artifact_id) is not str
        or not artifact_id
        or artifact_id != artifact_id.strip()
        or any(ord(character) < 0x20 for character in artifact_id)
    ):
        raise FactorGovernanceError(f"{label}.artifact_id is invalid")
    return {
        "kind": definition.kind,
        "contract_sha256": definition.contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": require_sha256(
            value.get("semantic_sha256"), label=f"{label}.semantic_sha256"
        ),
        "byte_sha256": require_sha256(value.get("byte_sha256"), label=f"{label}.byte_sha256"),
    }


def exact_payload(
    document: Mapping[str, Any] | bytes,
    *,
    kind: str,
    fields: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = validate_governance_artifact(document, expected_kind=kind)
    payload = dict(envelope["payload"])
    if set(payload) != fields:
        raise FactorGovernanceError(f"{kind} payload fields are not exact")
    return envelope, payload


__all__ = [
    "ANNUAL_OPEN_SESSIONS",
    "BH_Q_CEILING",
    "BOOTSTRAP_VALIDATION_PROFILE_ID",
    "COST_BPS",
    "COVERAGE_MINIMUM",
    "CPCV_BLOCK_COUNT",
    "CPCV_EMBARGO_OPEN_SESSIONS",
    "CPCV_PATH_COUNT",
    "CPCV_PURGE_OPEN_SESSIONS",
    "CPCV_TEST_BLOCK_COUNT",
    "DSR_FLOOR",
    "LABEL_HORIZON_OPEN_SESSIONS",
    "MIN_CLOSED_MONTH_ENDS",
    "MIN_DAILY_RANKIC_SESSIONS",
    "MIN_DISJOINT_COHORTS",
    "PBO_CEILING",
    "PBO_MIN_CONFIGURATIONS",
    "PBO_SPLIT_COUNT",
    "POSITIVE_PATH_RATIO_FLOOR",
    "PROSPECTIVE_VALIDATION_PROFILE_ID",
    "REDUNDANCY_CORRELATION_FLOOR",
    "REDUNDANCY_MIN_OVERLAP",
    "SHRINKAGE_PSEUDO_COUNT",
    "SIGNAL_OPEN_SESSIONS",
    "TOTAL_OPEN_SESSIONS",
    "T_STAT_HURDLE",
    "TURNOVER_CEILING",
    "ARTIFACT_REF_FIELDS",
    "artifact_ref",
    "bootstrap_validation_namespace_id",
    "business_identity",
    "canonical_a_share_symbol",
    "canonical_identifier",
    "canonical_sessions",
    "canonical_timestamp",
    "decimal_text",
    "decimal_value",
    "exact_payload",
    "finite_float",
    "observation_lineage_identity",
    "prospective_validation_namespace_id",
    "require_sha256",
    "validate_artifact_ref",
    "validate_governance_artifact",
]
