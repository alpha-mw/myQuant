"""Pure B0 full-universe scoring and subject binding for Factor v5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
from typing import Any, Final

from quant_investor.factors.governance_v5.weights import (
    ADMITTED_SET_VERSION,
    build_admitted_factor_set,
)

from ._core import (
    IntelligenceV2ContractError,
    canonical_bytes,
    common_fields,
    content_ref,
    decimal_text,
    decimal_value,
    exact_ref,
    identifier,
    require_exact_keys,
    seal,
    timestamp,
    validate_content_ref,
    validate_seal,
)
from .readiness import validate_investment_data_readiness

POOL_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.quant-pool-policy.v1"
INITIAL_POOL_VERSION: Final = "myquant.v17.intelligence-v2.initial-pool.v1"
QUANT_BRANCH_VERSION: Final = "myquant.v17.intelligence-v2.quant-branch.v5"
SUBJECT_BINDING_VERSION: Final = "myquant.v17.intelligence-v2.subject-branch-binding.v1"

_FACTOR_SET_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "factor_protocol",
    "factor_rows",
    "factor_set_id",
    "lane",
    "policy_ref",
    "preregistration_ref",
    "semantic_sha256",
    "timestamp",
    "version",
    "weight_total",
}
_UNIVERSE_ROW_FIELDS: Final = {
    "company_code",
    "exposures",
    "pit_active",
    "security_identity_ref",
    "tradable",
}
_FACTOR_VALIDATION_FIELDS: Final = {
    "built_at",
    "policy",
    "preregistration",
    "prospective_evaluations",
}


def build_quant_pool_policy(
    *,
    created_at: str,
    pool_size: int,
    minimum_pool_size: int,
) -> dict[str, Any]:
    if type(pool_size) is not int or not 1 <= pool_size <= 500:
        raise IntelligenceV2ContractError("pool_size must be between 1 and 500")
    if type(minimum_pool_size) is not int or not 1 <= minimum_pool_size <= pool_size:
        raise IntelligenceV2ContractError("minimum_pool_size is invalid")
    return seal(
        {
            **common_fields(timestamp_value=created_at),
            "minimum_pool_size": minimum_pool_size,
            "pool_size": pool_size,
            "version": POOL_POLICY_VERSION,
        },
        identity_field="policy_id",
    )


def validate_quant_pool_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="policy_id")
    expected = build_quant_pool_policy(
        created_at=normalized.get("timestamp"),
        pool_size=normalized.get("pool_size"),
        minimum_pool_size=normalized.get("minimum_pool_size"),
    )
    if normalized != expected or normalized.get("version") != POOL_POLICY_VERSION:
        raise IntelligenceV2ContractError("quant pool policy replay mismatch")
    return normalized


def _validated_factor_set(
    factor_admitted_set: Mapping[str, Any],
    factor_validation_closure: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Decimal]]:
    closure = require_exact_keys(
        factor_validation_closure,
        _FACTOR_VALIDATION_FIELDS,
        label="factor_validation_closure",
    )
    expected = build_admitted_factor_set(
        policy=closure["policy"],
        preregistration=closure["preregistration"],
        prospective_evaluations=closure["prospective_evaluations"],
        built_at=closure["built_at"],
    )
    if type(factor_admitted_set) is not dict or factor_admitted_set != expected:
        raise IntelligenceV2ContractError("Factor v5 admitted set replay mismatch")
    require_exact_keys(expected, _FACTOR_SET_FIELDS, label="Factor v5 admitted set")
    if expected["version"] != ADMITTED_SET_VERSION or expected["lane"] != "PROSPECTIVE_ADMISSION":
        raise IntelligenceV2ContractError("diagnostic or unsupported Factor artifact is forbidden")
    weights: dict[str, Decimal] = {}
    for row in expected["factor_rows"]:
        factor_id = identifier(row.get("candidate_id"), label="factor candidate_id")
        if factor_id in weights:
            raise IntelligenceV2ContractError("duplicate admitted factor")
        weights[factor_id] = decimal_value(
            row.get("weight"),
            label=f"factor weight {factor_id}",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    if not weights or sum(weights.values(), Decimal("0")) != Decimal("1"):
        raise IntelligenceV2ContractError("Factor weights must sum exactly to one")
    return expected, weights


def _match_exact_ref(
    reference: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    identity_field: str,
    label: str,
    as_of: str,
) -> dict[str, str]:
    result = exact_ref(reference, label=label)
    if result["available_at"] > as_of or result["cutoff"] > as_of:
        raise IntelligenceV2ContractError(f"{label} contains future evidence")
    if (
        result["artifact_id"] != str(document[identity_field])
        or result["artifact_version"] != str(document["version"])
        or result["semantic_sha256"] != str(document["semantic_sha256"])
        or result["byte_sha256"] != hashlib.sha256(canonical_bytes(document)).hexdigest()
    ):
        raise IntelligenceV2ContractError(f"{label} does not bind the exact artifact")
    return result


def _readiness_market_refs(readiness: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["name"]): dict(row["source_ref"])
        for row in readiness["rows"]
        if row["name"] in {"MARKET", "PIT_UNIVERSE"}
    }


def _validate_scoring_refs(
    *,
    readiness: Mapping[str, Any],
    market_catalog_ref: Mapping[str, Any],
    pit_universe_ref: Mapping[str, Any],
    issued_at: str,
) -> tuple[dict[str, str], dict[str, str]]:
    market_reference = exact_ref(market_catalog_ref, label="market_catalog_ref")
    pit_reference = exact_ref(pit_universe_ref, label="pit_universe_ref")
    readiness_refs = _readiness_market_refs(readiness)
    if (
        readiness.get("quant_inputs_ready") is not True
        or readiness_refs.get("MARKET") != market_reference
        or readiness_refs.get("PIT_UNIVERSE") != pit_reference
    ):
        raise IntelligenceV2ContractError("B0 readiness does not authorize Quant scoring")
    for label, reference in (
        ("market_catalog_ref", market_reference),
        ("pit_universe_ref", pit_reference),
    ):
        if reference["available_at"] > issued_at or reference["cutoff"] > issued_at:
            raise IntelligenceV2ContractError(f"{label} contains future evidence")
    return market_reference, pit_reference


def _normalized_universe_rows(
    universe_rows: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, Decimal],
    issued_at: str,
) -> list[dict[str, Any]]:
    if isinstance(universe_rows, (str, bytes)) or not isinstance(universe_rows, Sequence):
        raise IntelligenceV2ContractError("universe_rows must be a sequence")
    normalized_rows: list[dict[str, Any]] = []
    seen_companies: set[str] = set()
    for index, row in enumerate(universe_rows):
        require_exact_keys(row, _UNIVERSE_ROW_FIELDS, label=f"universe_rows[{index}]")
        company = identifier(row["company_code"], label=f"universe_rows[{index}].company_code")
        if company in seen_companies:
            raise IntelligenceV2ContractError("universe contains duplicate company_code")
        seen_companies.add(company)
        if type(row["pit_active"]) is not bool or type(row["tradable"]) is not bool:
            raise IntelligenceV2ContractError("universe eligibility flags must be boolean")
        identity_ref = exact_ref(
            row["security_identity_ref"],
            label=f"universe_rows[{index}].security_identity_ref",
        )
        if identity_ref["available_at"] > issued_at or identity_ref["cutoff"] > issued_at:
            raise IntelligenceV2ContractError("security identity contains future evidence")
        if type(row["exposures"]) is not dict:
            raise IntelligenceV2ContractError("factor exposures must be an object")
        eligible = row["pit_active"] and row["tradable"]
        if not eligible or set(row["exposures"]) != set(weights):
            continue
        exposures = {
            factor_id: decimal_value(
                row["exposures"][factor_id],
                label=f"{company}.{factor_id}",
            )
            for factor_id in sorted(weights, key=lambda value: value.encode("ascii"))
        }
        score = sum(
            (weights[factor_id] * exposures[factor_id] for factor_id in weights),
            Decimal("0"),
        )
        normalized_rows.append(
            {
                "company_code": company,
                "score": decimal_text(score),
                "security_identity_ref": identity_ref,
            }
        )
    return normalized_rows


def build_initial_pool(
    *,
    readiness_receipt: Mapping[str, Any],
    readiness_validation_closure: Mapping[str, Any],
    factor_admitted_set: Mapping[str, Any],
    factor_validation_closure: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    policy: Mapping[str, Any],
    universe_rows: Sequence[Mapping[str, Any]],
    market_catalog_ref: Mapping[str, Any],
    pit_universe_ref: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    issued_at = timestamp(as_of, label="as_of")
    if type(readiness_validation_closure) is not dict:
        raise IntelligenceV2ContractError("readiness_validation_closure must be exact")
    readiness = validate_investment_data_readiness(
        readiness_receipt,
        **dict(readiness_validation_closure),
    )
    validated_policy = validate_quant_pool_policy(policy)
    factor_set, weights = _validated_factor_set(
        factor_admitted_set,
        factor_validation_closure,
    )
    factor_reference = _match_exact_ref(
        factor_set_ref,
        factor_set,
        identity_field="factor_set_id",
        label="factor_set_ref",
        as_of=issued_at,
    )
    market_reference, pit_reference = _validate_scoring_refs(
        readiness=readiness,
        market_catalog_ref=market_catalog_ref,
        pit_universe_ref=pit_universe_ref,
        issued_at=issued_at,
    )
    normalized_rows = _normalized_universe_rows(
        universe_rows,
        weights=weights,
        issued_at=issued_at,
    )

    ranked = sorted(
        normalized_rows,
        key=lambda row: (-Decimal(row["score"]), row["company_code"].encode("ascii")),
    )
    target_size = int(validated_policy["pool_size"])
    minimum_size = int(validated_policy["minimum_pool_size"])
    blocked = len(ranked) < target_size or len(ranked) < minimum_size
    selected = [] if blocked else ranked[:target_size]
    pool_rows = [{**row, "rank": rank} for rank, row in enumerate(selected, start=1)]
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "as_of": issued_at,
            "blocker_codes": ["INSUFFICIENT_SCOREABLE_UNIVERSE"] if blocked else [],
            "eligible_count": len(ranked),
            "factor_set_ref": factor_reference,
            "market_catalog_ref": market_reference,
            "pit_universe_ref": pit_reference,
            "policy_ref": content_ref(validated_policy, identity_field="policy_id"),
            "pool_rows": pool_rows,
            "readiness_ref": content_ref(readiness, identity_field="readiness_id"),
            "status": "BLOCKED" if blocked else "AVAILABLE",
            "target_pool_size": target_size,
            "version": INITIAL_POOL_VERSION,
        },
        identity_field="pool_id",
    )


def validate_initial_pool(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="pool_id")
    expected = build_initial_pool(**closure)
    if normalized != expected or normalized.get("version") != INITIAL_POOL_VERSION:
        raise IntelligenceV2ContractError("initial pool replay mismatch")
    return normalized


def build_quant_branch_v5(
    *,
    initial_pool: Mapping[str, Any],
    pool_validation_closure: Mapping[str, Any],
    company_code: str,
    as_of: str,
) -> dict[str, Any]:
    if type(pool_validation_closure) is not dict:
        raise IntelligenceV2ContractError("pool_validation_closure must be exact")
    pool = validate_initial_pool(initial_pool, **dict(pool_validation_closure))
    issued_at = timestamp(as_of, label="as_of")
    if pool["timestamp"] != issued_at or pool["status"] != "AVAILABLE":
        raise IntelligenceV2ContractError("Quant branch requires an available same-time pool")
    company = identifier(company_code, label="company_code")
    matches = [row for row in pool["pool_rows"] if row["company_code"] == company]
    if len(matches) != 1:
        raise IntelligenceV2ContractError("company is not uniquely present in the initial pool")
    row = matches[0]
    size = len(pool["pool_rows"])
    percentile = Decimal(size - int(row["rank"]) + 1) / Decimal(size)
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "company_code": company,
            "factor_set_ref": pool["factor_set_ref"],
            "percentile": decimal_text(percentile),
            "pool_ref": content_ref(pool, identity_field="pool_id"),
            "rank": row["rank"],
            "score": row["score"],
            "security_identity_ref": row["security_identity_ref"],
            "version": QUANT_BRANCH_VERSION,
        },
        identity_field="quant_branch_id",
    )


def validate_quant_branch_v5(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="quant_branch_id")
    expected = build_quant_branch_v5(**closure)
    if normalized != expected or normalized.get("version") != QUANT_BRANCH_VERSION:
        raise IntelligenceV2ContractError("Quant branch replay mismatch")
    return normalized


def build_subject_branch_binding(
    *,
    quant_branch: Mapping[str, Any],
    quant_branch_validation_closure: Mapping[str, Any],
    frozen_v1_branch_ref: Mapping[str, Any],
    v2_manifest_ref: Mapping[str, Any],
    bound_at: str,
) -> dict[str, Any]:
    if type(quant_branch_validation_closure) is not dict:
        raise IntelligenceV2ContractError("quant_branch_validation_closure must be exact")
    branch = validate_quant_branch_v5(
        quant_branch,
        **dict(quant_branch_validation_closure),
    )
    issued_at = timestamp(bound_at, label="bound_at")
    if branch["timestamp"] != issued_at:
        raise IntelligenceV2ContractError("subject binding must share the branch cutoff")
    frozen_ref = exact_ref(frozen_v1_branch_ref, label="frozen_v1_branch_ref")
    manifest_ref = validate_content_ref(v2_manifest_ref, label="v2_manifest_ref")
    if frozen_ref["available_at"] > issued_at or frozen_ref["cutoff"] > issued_at:
        raise IntelligenceV2ContractError("frozen v1 branch ref contains future evidence")
    return seal(
        {
            **common_fields(timestamp_value=issued_at),
            "company_code": branch["company_code"],
            "data_catalog_refs": [
                branch["factor_set_ref"],
                branch["security_identity_ref"],
            ],
            "frozen_v1_branch_ref": frozen_ref,
            "quant_branch_ref": content_ref(branch, identity_field="quant_branch_id"),
            "v2_manifest_ref": manifest_ref,
            "version": SUBJECT_BINDING_VERSION,
        },
        identity_field="binding_id",
    )


def validate_subject_branch_binding(
    document: Mapping[str, Any],
    **closure: Any,
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="binding_id")
    expected = build_subject_branch_binding(**closure)
    if normalized != expected or normalized.get("version") != SUBJECT_BINDING_VERSION:
        raise IntelligenceV2ContractError("subject branch binding replay mismatch")
    return normalized


__all__ = [
    "INITIAL_POOL_VERSION",
    "POOL_POLICY_VERSION",
    "QUANT_BRANCH_VERSION",
    "SUBJECT_BINDING_VERSION",
    "build_initial_pool",
    "build_quant_branch_v5",
    "build_quant_pool_policy",
    "build_subject_branch_binding",
    "validate_initial_pool",
    "validate_quant_branch_v5",
    "validate_quant_pool_policy",
    "validate_subject_branch_binding",
]
