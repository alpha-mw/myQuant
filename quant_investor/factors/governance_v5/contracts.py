"""Sealed policy, preregistration, coverage, and substitution contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from ._core import (
    FactorGovernanceV5Error,
    common_fields,
    decimal_text,
    decimal_value,
    identifier,
    seal,
    sha256,
    sorted_unique_strings,
    timestamp,
    validate_seal,
)

POLICY_VERSION: Final = "factor-governance-policy.v5"
PREREGISTRATION_VERSION: Final = "factor-candidate-preregistration.v5"
COVERAGE_VERSION: Final = "factor-input-coverage-receipt.v5"
SUBSTITUTION_VERSION: Final = "factor-candidate-substitution-receipt.v5"


def build_governance_policy(
    *,
    created_at: str,
    coverage_threshold: Any,
    label_horizon_sessions: int,
    minimum_prospective_paths: int,
) -> dict[str, Any]:
    if type(label_horizon_sessions) is not int or label_horizon_sessions < 1:
        raise FactorGovernanceV5Error("label_horizon_sessions must be positive")
    if type(minimum_prospective_paths) is not int or minimum_prospective_paths < 1:
        raise FactorGovernanceV5Error("minimum_prospective_paths must be positive")
    return seal(
        {
            **common_fields(timestamp_value=created_at),
            "version": POLICY_VERSION,
            "coverage_threshold": decimal_text(
                decimal_value(
                    coverage_threshold,
                    label="coverage_threshold",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
            ),
            "embargo_open_sessions": 30,
            "factor_composite_max_factor_weight": "1.000000000000",
            "factor_composite_max_family_weight": "1.000000000000",
            "label_horizon_sessions": label_horizon_sessions,
            "maximum_admitted_factors": 10,
            "minimum_admitted_factors": 1,
            "minimum_admitted_families": 1,
            "minimum_prospective_paths": minimum_prospective_paths,
            "purge_open_sessions": 30,
            "weighting_method": "PURGED_OOS_SHRUNK_IC_V1",
        },
        identity_field="policy_id",
    )


def validate_governance_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_seal(document, identity_field="policy_id")
    expected = build_governance_policy(
        created_at=sealed["timestamp"],
        coverage_threshold=sealed["coverage_threshold"],
        label_horizon_sessions=sealed["label_horizon_sessions"],
        minimum_prospective_paths=sealed["minimum_prospective_paths"],
    )
    if sealed != expected:
        raise FactorGovernanceV5Error("governance policy replay mismatch")
    return sealed


def _normalize_candidate(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != {
        "candidate_id",
        "expression",
        "family",
        "implementation_sha256",
        "input_fields",
        "parameterization",
        "role",
        "source_sha256",
    }:
        raise FactorGovernanceV5Error(f"candidates[{index}] shape is invalid")
    candidate_id = identifier(raw["candidate_id"], label="candidate_id")
    family = identifier(raw["family"], label="family")
    role = str(raw["role"])
    expression = raw["expression"]
    if type(expression) is not str or not expression.strip():
        raise FactorGovernanceV5Error("candidate expression must be nonempty")
    parameterization = raw["parameterization"]
    if parameterization not in {"NONE", "PREREGISTERED"}:
        raise FactorGovernanceV5Error("candidate parameterization is invalid")
    return {
        "candidate_id": candidate_id,
        "expression": expression,
        "family": family,
        "implementation_sha256": sha256(
            raw["implementation_sha256"], label="implementation_sha256"
        ),
        "input_fields": sorted_unique_strings(
            raw["input_fields"], label="input_fields", maximum=16
        ),
        "parameterization": parameterization,
        "role": role,
        "source_sha256": sha256(raw["source_sha256"], label="source_sha256"),
    }


def _register_candidate_role(
    row: Mapping[str, Any],
    *,
    primary_families: dict[str, str],
    alternates_by_primary: dict[str, str],
) -> None:
    candidate_id = str(row["candidate_id"])
    family = str(row["family"])
    role = str(row["role"])
    if role == "PRIMARY":
        primary_families[candidate_id] = family
        return
    if not role.startswith("ALTERNATE_FOR:"):
        raise FactorGovernanceV5Error("candidate role is invalid")
    primary = identifier(role.split(":", 1)[1], label="alternate primary")
    if primary in alternates_by_primary:
        raise FactorGovernanceV5Error("a primary has more than one alternate")
    alternates_by_primary[primary] = candidate_id


def _validate_alternate_topology(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_families: Mapping[str, str],
    alternates_by_primary: Mapping[str, str],
) -> None:
    by_id = {str(row["candidate_id"]): row for row in rows}
    for primary, alternate in alternates_by_primary.items():
        if primary not in primary_families:
            raise FactorGovernanceV5Error("alternate references a non-primary")
        if by_id[alternate]["family"] != primary_families[primary]:
            raise FactorGovernanceV5Error("alternate must share the primary family")


def _candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise FactorGovernanceV5Error("candidates must be a sequence")
    if not 1 <= len(candidates) <= 20:
        raise FactorGovernanceV5Error("candidate count must be between 1 and 20")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    alternates_by_primary: dict[str, str] = {}
    primary_families: dict[str, str] = {}
    for index, raw in enumerate(candidates):
        row = _normalize_candidate(raw, index=index)
        candidate_id = row["candidate_id"]
        if candidate_id in identities:
            raise FactorGovernanceV5Error("duplicate candidate_id")
        identities.add(candidate_id)
        _register_candidate_role(
            row,
            primary_families=primary_families,
            alternates_by_primary=alternates_by_primary,
        )
        rows.append(row)
    if not primary_families or len(primary_families) > 10:
        raise FactorGovernanceV5Error("primary count must be between 1 and 10")
    _validate_alternate_topology(
        rows,
        primary_families=primary_families,
        alternates_by_primary=alternates_by_primary,
    )
    return sorted(rows, key=lambda row: row["candidate_id"].encode("ascii"))


def build_preregistration(
    *,
    policy: Mapping[str, Any],
    sealed_at: str,
    evaluation_start_session: str,
    evaluation_end_session: str,
    label_available_at: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_policy = validate_governance_policy(policy)
    sealed = timestamp(sealed_at, label="sealed_at")
    label_time = timestamp(label_available_at, label="label_available_at")
    if type(evaluation_start_session) is not str or type(evaluation_end_session) is not str:
        raise FactorGovernanceV5Error("evaluation sessions must be strings")
    if evaluation_start_session > evaluation_end_session:
        raise FactorGovernanceV5Error("evaluation session range is reversed")
    if label_time <= sealed:
        raise FactorGovernanceV5Error("prospective label was already available at seal time")
    rows = _candidate_rows(candidates)
    return seal(
        {
            **common_fields(timestamp_value=sealed),
            "version": PREREGISTRATION_VERSION,
            "candidates": rows,
            "evaluation_end_session": evaluation_end_session,
            "evaluation_start_session": evaluation_start_session,
            "label_available_at": label_time,
            "label_horizon_sessions": normalized_policy["label_horizon_sessions"],
            "policy_ref": normalized_policy["policy_id"],
            "sealed_before_label_available": True,
        },
        identity_field="preregistration_id",
    )


def validate_preregistration(
    document: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    sealed = validate_seal(document, identity_field="preregistration_id")
    expected = build_preregistration(
        policy=policy,
        sealed_at=sealed["timestamp"],
        evaluation_start_session=sealed["evaluation_start_session"],
        evaluation_end_session=sealed["evaluation_end_session"],
        label_available_at=sealed["label_available_at"],
        candidates=sealed.get("candidates", ()),
    )
    if sealed != expected:
        raise FactorGovernanceV5Error("preregistration replay mismatch")
    return sealed


def build_coverage_receipt(
    *,
    policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    candidate_id: str,
    numerator: int,
    denominator: int,
    pit_universe_sha256: str,
    input_source_sha256: str,
    cutoff: str,
    computed_at: str,
    label_reader_permitted_at: str,
) -> dict[str, Any]:
    normalized_policy = validate_governance_policy(policy)
    normalized_prereg = validate_preregistration(preregistration, policy=policy)
    candidate = identifier(candidate_id, label="candidate_id")
    declared = {row["candidate_id"] for row in normalized_prereg["candidates"]}
    if candidate not in declared:
        raise FactorGovernanceV5Error("coverage candidate is not preregistered")
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator < 1
        or numerator < 0
        or numerator > denominator
    ):
        raise FactorGovernanceV5Error("coverage counts are invalid")
    computed = timestamp(computed_at, label="computed_at")
    reader_time = timestamp(label_reader_permitted_at, label="label_reader_permitted_at")
    cutoff_time = timestamp(cutoff, label="cutoff")
    if cutoff_time > computed or computed >= reader_time:
        raise FactorGovernanceV5Error("coverage was not computed before label access")
    coverage = Decimal(numerator) / Decimal(denominator)
    threshold = decimal_value(normalized_policy["coverage_threshold"], label="coverage_threshold")
    return seal(
        {
            **common_fields(timestamp_value=computed),
            "version": COVERAGE_VERSION,
            "candidate_id": candidate,
            "coverage": decimal_text(coverage),
            "coverage_gate": "PASSED" if coverage >= threshold else "FAILED",
            "cutoff": cutoff_time,
            "denominator": denominator,
            "input_source_sha256": sha256(input_source_sha256, label="input_source_sha256"),
            "label_reader_permitted_at": reader_time,
            "numerator": numerator,
            "pit_universe_sha256": sha256(pit_universe_sha256, label="pit_universe_sha256"),
            "policy_ref": normalized_policy["policy_id"],
            "preregistration_ref": normalized_prereg["preregistration_id"],
        },
        identity_field="coverage_receipt_id",
    )


def validate_coverage_receipt(
    document: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = validate_seal(document, identity_field="coverage_receipt_id")
    expected = build_coverage_receipt(
        policy=policy,
        preregistration=preregistration,
        candidate_id=sealed["candidate_id"],
        numerator=sealed["numerator"],
        denominator=sealed["denominator"],
        pit_universe_sha256=sealed["pit_universe_sha256"],
        input_source_sha256=sealed["input_source_sha256"],
        cutoff=sealed["cutoff"],
        computed_at=sealed["timestamp"],
        label_reader_permitted_at=sealed["label_reader_permitted_at"],
    )
    if sealed != expected:
        raise FactorGovernanceV5Error("coverage receipt replay mismatch")
    return sealed


def build_substitution_receipt(
    *,
    policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    primary_coverage: Mapping[str, Any],
    alternate_coverage: Mapping[str, Any],
    substituted_at: str,
) -> dict[str, Any]:
    normalized_policy = validate_governance_policy(policy)
    normalized_prereg = validate_preregistration(preregistration, policy=policy)
    primary_receipt = validate_coverage_receipt(
        primary_coverage, policy=policy, preregistration=preregistration
    )
    alternate_receipt = validate_coverage_receipt(
        alternate_coverage, policy=policy, preregistration=preregistration
    )
    if primary_receipt["coverage_gate"] != "FAILED":
        raise FactorGovernanceV5Error("primary coverage did not fail")
    if alternate_receipt["coverage_gate"] != "PASSED":
        raise FactorGovernanceV5Error("alternate coverage did not pass")
    rows = {row["candidate_id"]: row for row in normalized_prereg["candidates"]}
    primary_id = primary_receipt["candidate_id"]
    alternate_id = alternate_receipt["candidate_id"]
    if rows[primary_id]["role"] != "PRIMARY":
        raise FactorGovernanceV5Error("substitution source is not a primary")
    if rows[alternate_id]["role"] != f"ALTERNATE_FOR:{primary_id}":
        raise FactorGovernanceV5Error("alternate is not bound to this primary")
    at = timestamp(substituted_at, label="substituted_at")
    if at >= primary_receipt["label_reader_permitted_at"]:
        raise FactorGovernanceV5Error("substitution occurred after label access")
    return seal(
        {
            **common_fields(timestamp_value=at),
            "version": SUBSTITUTION_VERSION,
            "alternate_candidate_id": alternate_id,
            "alternate_coverage_ref": alternate_receipt["coverage_receipt_id"],
            "policy_ref": normalized_policy["policy_id"],
            "preregistration_ref": normalized_prereg["preregistration_id"],
            "primary_candidate_id": primary_id,
            "primary_coverage_ref": primary_receipt["coverage_receipt_id"],
            "reason": "PREDEFINED_COVERAGE_GATE_FAILED",
        },
        identity_field="substitution_receipt_id",
    )


__all__ = [
    "COVERAGE_VERSION",
    "POLICY_VERSION",
    "PREREGISTRATION_VERSION",
    "SUBSTITUTION_VERSION",
    "build_coverage_receipt",
    "build_governance_policy",
    "build_preregistration",
    "build_substitution_receipt",
    "validate_coverage_receipt",
    "validate_governance_policy",
    "validate_preregistration",
]
