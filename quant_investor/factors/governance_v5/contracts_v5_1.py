"""Non-admission candidate registration for Factor Governance v5.1.

This additive contract records exact candidate provenance and, deliberately,
the owner policy fields that are still missing.  It cannot create prospective
evidence, admit a factor, or feed the current B0/Decision/I6/V17 chain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any, Final

from ._core import (
    FactorGovernanceV5Error,
    canonical_bytes,
    common_fields,
    identifier,
    seal,
    sha256,
    sorted_unique_strings,
    validate_seal,
)

CANDIDATE_REGISTRATION_VERSION: Final = "factor-candidate-registration.v5.1"
FUTURE_ADMISSION_EVIDENCE_LANE: Final = "FUTURE_ADMISSION_EVIDENCE_LANE"
REGISTERED_STATE: Final = "REGISTERED"

OWNER_POLICY_FIELDS: Final = (
    "benchmark_definition",
    "capacity_minimum_cny",
    "cost_bps_round_trip",
    "coverage_minimum",
    "drawdown_floor",
    "evidence_dependence_rule",
    "failure_posterior_threshold",
    "horizon_likelihood_mapping",
    "label_horizons_open_sessions",
    "neutralization_policy",
    "posterior_core_threshold",
    "posterior_entry_threshold",
    "prior_odds",
    "prospective_window",
    "purge_embargo_policy",
    "rebalance_policy",
    "regime_coverage_policy",
    "slippage_model",
    "turnover_maximum",
    "universe_policy",
)

_CANDIDATE_FIELDS: Final = {
    "candidate_id",
    "expression",
    "family",
    "input_fields",
    "role",
}


def _normalize_candidate(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != _CANDIDATE_FIELDS:
        raise FactorGovernanceV5Error(f"candidates[{index}] shape is invalid")
    expression = raw["expression"]
    if type(expression) is not str or not expression.strip():
        raise FactorGovernanceV5Error("candidate expression must be nonempty")
    return {
        "candidate_id": identifier(raw["candidate_id"], label="candidate_id"),
        "expression": expression,
        "family": identifier(raw["family"], label="family"),
        "input_fields": sorted_unique_strings(
            raw["input_fields"], label="input_fields", maximum=16
        ),
        "role": str(raw["role"]),
    }


def _register_role(
    row: Mapping[str, Any],
    *,
    primary_families: dict[str, str],
    alternate_primary: dict[str, str],
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
    if primary in alternate_primary:
        raise FactorGovernanceV5Error("a primary has more than one alternate")
    alternate_primary[primary] = candidate_id


def _validate_topology(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_families: Mapping[str, str],
    alternate_primary: Mapping[str, str],
) -> None:
    by_id = {row["candidate_id"]: row for row in rows}
    if not primary_families or len(primary_families) > 10:
        raise FactorGovernanceV5Error("primary count must be between 1 and 10")
    for primary, alternate in alternate_primary.items():
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
    seen: set[str] = set()
    primary_families: dict[str, str] = {}
    alternate_primary: dict[str, str] = {}
    for index, raw in enumerate(candidates):
        row = _normalize_candidate(raw, index=index)
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise FactorGovernanceV5Error("duplicate candidate_id")
        seen.add(candidate_id)
        _register_role(
            row,
            primary_families=primary_families,
            alternate_primary=alternate_primary,
        )
        rows.append(row)
    _validate_topology(
        rows,
        primary_families=primary_families,
        alternate_primary=alternate_primary,
    )
    return sorted(rows, key=lambda row: row["candidate_id"].encode("ascii"))


def build_candidate_registration_v5_1(
    *,
    registered_at: str,
    candidates: Sequence[Mapping[str, Any]],
    catalog_source_sha256: str,
    implementation_source_sha256: str,
    pit_universe_sha256: str,
    exchange_calendar_sha256: str,
    missing_owner_policy_fields: Sequence[str],
) -> dict[str, Any]:
    missing = sorted_unique_strings(
        missing_owner_policy_fields,
        label="missing_owner_policy_fields",
        maximum=len(OWNER_POLICY_FIELDS),
    )
    if missing != list(OWNER_POLICY_FIELDS):
        raise FactorGovernanceV5Error(
            "incomplete registration must declare the full owner policy gap"
        )
    return seal(
        {
            **common_fields(timestamp_value=registered_at),
            "admission_eligible": False,
            "b0_eligible": False,
            "candidates": _candidate_rows(candidates),
            "catalog_source_sha256": sha256(catalog_source_sha256, label="catalog_source_sha256"),
            "decision_eligible": False,
            "exchange_calendar_sha256": sha256(
                exchange_calendar_sha256, label="exchange_calendar_sha256"
            ),
            "factor_protocol_extension": "factor-governance-protocol.v5.1",
            "i6_eligible": False,
            "implementation_source_sha256": sha256(
                implementation_source_sha256, label="implementation_source_sha256"
            ),
            "lane": FUTURE_ADMISSION_EVIDENCE_LANE,
            "lifecycle_state": REGISTERED_STATE,
            "missing_owner_policy_fields": missing,
            "pit_universe_sha256": sha256(pit_universe_sha256, label="pit_universe_sha256"),
            "preregistration_valid": False,
            "production_active": False,
            "prospective_observation_authorized": False,
            "v17_eligible": False,
            "version": CANDIDATE_REGISTRATION_VERSION,
        },
        identity_field="registration_id",
    )


def validate_candidate_registration_v5_1(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="registration_id")
    expected = build_candidate_registration_v5_1(
        registered_at=normalized["timestamp"],
        candidates=normalized.get("candidates", ()),
        catalog_source_sha256=normalized["catalog_source_sha256"],
        implementation_source_sha256=normalized["implementation_source_sha256"],
        pit_universe_sha256=normalized["pit_universe_sha256"],
        exchange_calendar_sha256=normalized["exchange_calendar_sha256"],
        missing_owner_policy_fields=normalized.get("missing_owner_policy_fields", ()),
    )
    if normalized != expected:
        raise FactorGovernanceV5Error("candidate registration replay mismatch")
    return normalized


def registration_byte_sha256(document: Mapping[str, Any]) -> str:
    normalized = validate_candidate_registration_v5_1(document)
    return hashlib.sha256(canonical_bytes(normalized)).hexdigest()


__all__ = [
    "CANDIDATE_REGISTRATION_VERSION",
    "FUTURE_ADMISSION_EVIDENCE_LANE",
    "OWNER_POLICY_FIELDS",
    "REGISTERED_STATE",
    "build_candidate_registration_v5_1",
    "registration_byte_sha256",
    "validate_candidate_registration_v5_1",
]
