"""Closed owner policy and risk inputs for deterministic Decision v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import wraps
from typing import Any, Callable, Final, ParamSpec, TypeVar

from .._core import (
    IntelligenceV2ContractError,
    code,
    common_fields,
    decimal_text,
    decimal_value,
    require_exact_keys,
    seal,
    timestamp,
    validate_content_ref,
    validate_seal,
)

DECISION_POLICY_V2_VERSION: Final = "myquant.v17.research-intelligence-v2.decision-policy.v1"
RISK_DIMENSIONS: Final = ("BUSINESS", "FINANCIAL", "MARKET", "THESIS")
THEME_IDENTITY_STATES: Final = frozenset({"AVAILABLE", "NO_MEMBERSHIP"})
R22_REQUIRED_STATES: Final = frozenset({"SUPPORTED", "UNCERTAIN"})

_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "frozen_v1_manifest_sha256",
    "production",
    "research_only",
    "timestamp",
}
_POLICY_FIELDS: Final = _COMMON_FIELDS | {
    "allowed_fundamental_stale_sessions",
    "fusion_threshold",
    "hard_veto_codes",
    "mandatory_industry_state",
    "mandatory_theme_states",
    "max_risk",
    "policy_id",
    "posterior_threshold",
    "required_r22_status",
    "semantic_sha256",
    "version",
}
_RISK_INPUT_FIELDS: Final = {
    "dimension",
    "evidence_refs",
    "hard_veto_codes",
    "severity",
    "status",
}


class DecisionV2ContractError(IntelligenceV2ContractError):
    """Fail-closed I4.5 contract error."""

    exit_code = 2


_P = ParamSpec("_P")
_R = TypeVar("_R")


def decision_contract(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Expose every v2-layer failure through the Decision v2 exception."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except DecisionV2ContractError:
            raise
        except (IntelligenceV2ContractError, TypeError) as exc:
            raise DecisionV2ContractError(str(exc)) from exc

    return wrapped


def _fail(message: str) -> None:
    raise DecisionV2ContractError(message)


def _codes(values: Sequence[Any], *, label: str, allow_empty: bool = True) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail(f"{label} must be a sequence")
    rows = [code(value, label=f"{label}[{index}]") for index, value in enumerate(values)]
    if (not allow_empty and not rows) or len(rows) > 64 or len(rows) != len(set(rows)):
        _fail(f"{label} cardinality or uniqueness is invalid")
    return sorted(rows, key=lambda value: value.encode("ascii"))


def _risk_evidence_refs(
    value: Any,
    *,
    label: str,
    admitted: set[tuple[tuple[str, str], ...]],
) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be a sequence")
    refs = [
        validate_content_ref(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    ]
    keys = [tuple(sorted(item.items())) for item in refs]
    if len(keys) != len(set(keys)) or any(key not in admitted for key in keys):
        _fail("risk evidence refs are duplicated or outside the admitted closure")
    refs.sort(
        key=lambda item: (
            item["artifact_id"].encode("ascii"),
            item["artifact_version"].encode("ascii"),
            item["byte_sha256"].encode("ascii"),
            item["semantic_sha256"].encode("ascii"),
        )
    )
    return refs


def _risk_row(
    value: Mapping[str, Any],
    *,
    index: int,
    admitted: set[tuple[tuple[str, str], ...]],
) -> dict[str, Any]:
    row = require_exact_keys(value, _RISK_INPUT_FIELDS, label=f"risk_rows[{index}]")
    dimension = str(row["dimension"])
    if dimension not in RISK_DIMENSIONS:
        _fail("risk dimension is invalid")
    status = str(row["status"])
    if status not in {"AVAILABLE", "UNAVAILABLE"}:
        _fail("risk status is invalid")
    refs = _risk_evidence_refs(
        row["evidence_refs"],
        label=f"risk_rows[{index}].evidence_refs",
        admitted=admitted,
    )
    vetoes = _codes(row["hard_veto_codes"], label="risk hard_veto_codes")
    if status == "UNAVAILABLE":
        if row["severity"] is not None or refs or vetoes:
            _fail("UNAVAILABLE risk dimension must be empty")
        severity = None
    else:
        if not refs:
            _fail("AVAILABLE risk dimension requires admitted evidence")
        severity = decimal_text(
            decimal_value(
                row["severity"],
                label="risk severity",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
        )
    return {
        "dimension": dimension,
        "evidence_refs": refs,
        "hard_veto_codes": vetoes,
        "severity": severity,
        "status": status,
    }


@decision_contract
def build_decision_policy_v2(
    *,
    created_at: str,
    fusion_threshold: Any,
    posterior_threshold: Any,
    max_risk: Any,
    required_r22_status: str,
    allowed_fundamental_stale_sessions: int,
    mandatory_industry_state: str,
    mandatory_theme_states: Sequence[str],
    hard_veto_codes: Sequence[str],
) -> dict[str, Any]:
    """Seal every deterministic Decision v2 gate; no defaults are inferred."""

    if type(allowed_fundamental_stale_sessions) is not int or not (
        0 <= allowed_fundamental_stale_sessions <= 1
    ):
        _fail("allowed_fundamental_stale_sessions must be zero or one")
    if mandatory_industry_state != "AVAILABLE":
        _fail("mandatory_industry_state must be AVAILABLE")
    if isinstance(mandatory_theme_states, (str, bytes)) or not isinstance(
        mandatory_theme_states, Sequence
    ):
        _fail("mandatory_theme_states must be a sequence")
    theme_states = [str(value) for value in mandatory_theme_states]
    if (
        not theme_states
        or len(theme_states) != len(set(theme_states))
        or not set(theme_states).issubset(THEME_IDENTITY_STATES)
    ):
        _fail("mandatory_theme_states is invalid")
    theme_states.sort(key=lambda value: value.encode("ascii"))
    if required_r22_status not in R22_REQUIRED_STATES:
        _fail("required_r22_status is invalid")
    unit = {"minimum": Decimal("0"), "maximum": Decimal("1")}
    return seal(
        {
            **common_fields(timestamp_value=created_at),
            "allowed_fundamental_stale_sessions": allowed_fundamental_stale_sessions,
            "fusion_threshold": decimal_text(
                decimal_value(fusion_threshold, label="fusion_threshold", **unit)
            ),
            "hard_veto_codes": _codes(hard_veto_codes, label="hard_veto_codes"),
            "mandatory_industry_state": mandatory_industry_state,
            "mandatory_theme_states": theme_states,
            "max_risk": decimal_text(decimal_value(max_risk, label="max_risk", **unit)),
            "posterior_threshold": decimal_text(
                decimal_value(posterior_threshold, label="posterior_threshold", **unit)
            ),
            "required_r22_status": required_r22_status,
            "version": DECISION_POLICY_V2_VERSION,
        },
        identity_field="policy_id",
    )


@decision_contract
def validate_decision_policy_v2(document: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_seal(document, identity_field="policy_id")
    require_exact_keys(row, _POLICY_FIELDS, label="DecisionPolicyV2")
    expected = build_decision_policy_v2(
        created_at=row["timestamp"],
        fusion_threshold=row["fusion_threshold"],
        posterior_threshold=row["posterior_threshold"],
        max_risk=row["max_risk"],
        required_r22_status=row["required_r22_status"],
        allowed_fundamental_stale_sessions=row["allowed_fundamental_stale_sessions"],
        mandatory_industry_state=row["mandatory_industry_state"],
        mandatory_theme_states=row["mandatory_theme_states"],
        hard_veto_codes=row["hard_veto_codes"],
    )
    if row != expected or row["version"] != DECISION_POLICY_V2_VERSION:
        _fail("DecisionPolicyV2 replay mismatch")
    return row


def normalize_risk_rows(
    values: Sequence[Mapping[str, Any]],
    *,
    admitted_evidence_refs: Sequence[Mapping[str, Any]],
    as_of: str,
) -> list[dict[str, Any]]:
    """Normalize four source-bound dimensions without inventing missing risk."""

    timestamp(as_of, label="as_of")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail("risk_rows must be a sequence")
    admitted = {
        tuple(sorted(validate_content_ref(value, label="admitted_evidence_ref").items()))
        for value in admitted_evidence_refs
    }
    rows = [_risk_row(value, index=index, admitted=admitted) for index, value in enumerate(values)]
    by_dimension = {row["dimension"]: row for row in rows}
    if len(rows) != len(by_dimension) or set(by_dimension) != set(RISK_DIMENSIONS):
        _fail("risk_rows must contain each dimension exactly once")
    return [by_dimension[dimension] for dimension in RISK_DIMENSIONS]


__all__ = [
    "DECISION_POLICY_V2_VERSION",
    "DecisionV2ContractError",
    "R22_REQUIRED_STATES",
    "RISK_DIMENSIONS",
    "THEME_IDENTITY_STATES",
    "build_decision_policy_v2",
    "normalize_risk_rows",
    "validate_decision_policy_v2",
]
