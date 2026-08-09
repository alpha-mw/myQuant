"""Deterministic purged/OOS shrinkage weighting for admitted factors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import Any, Final

from ._core import FactorGovernanceV5Error, common_fields, decimal_text, seal, timestamp
from .contracts import validate_governance_policy, validate_preregistration
from .prospective import validate_prospective_evaluation

ADMITTED_SET_VERSION: Final = "factor-admitted-set.v5"
_UNITS: Final = 10**12


def _largest_remainder_weights(values: Mapping[str, Decimal]) -> dict[str, str]:
    total = sum(values.values(), Decimal("0"))
    if total <= 0:
        raise FactorGovernanceV5Error("all shrunk IC values are zero")
    floors: dict[str, int] = {}
    remainders: list[tuple[Decimal, str]] = []
    with localcontext() as context:
        context.prec = 50
        for factor_id in sorted(values, key=lambda value: value.encode("ascii")):
            exact_units = values[factor_id] * Decimal(_UNITS) / total
            floor_units = int(exact_units.to_integral_value(rounding=ROUND_FLOOR))
            floors[factor_id] = floor_units
            remainders.append((exact_units - Decimal(floor_units), factor_id))
    residual = _UNITS - sum(floors.values())
    if residual < 0 or residual > len(floors):
        raise FactorGovernanceV5Error("largest-remainder residual is invalid")
    ordered = sorted(remainders, key=lambda row: (-row[0], row[1].encode("ascii")))
    for _, factor_id in ordered[:residual]:
        floors[factor_id] += 1
    result = {
        factor_id: decimal_text(Decimal(units) / Decimal(_UNITS))
        for factor_id, units in floors.items()
    }
    if sum((Decimal(value) for value in result.values()), Decimal("0")) != Decimal("1"):
        raise FactorGovernanceV5Error("factor weights do not sum to one")
    return result


def build_admitted_factor_set(
    *,
    policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    prospective_evaluations: Sequence[Mapping[str, Any]],
    built_at: str,
) -> dict[str, Any]:
    normalized_policy = validate_governance_policy(policy)
    normalized_prereg = validate_preregistration(preregistration, policy=policy)
    if isinstance(prospective_evaluations, (str, bytes)) or not isinstance(
        prospective_evaluations, Sequence
    ):
        raise FactorGovernanceV5Error("prospective_evaluations must be a sequence")
    evaluations = [
        validate_prospective_evaluation(row, policy=policy, preregistration=preregistration)
        for row in prospective_evaluations
    ]
    factor_ids = [row["candidate_id"] for row in evaluations]
    if len(factor_ids) != len(set(factor_ids)):
        raise FactorGovernanceV5Error("duplicate factor evaluation")
    admitted = [row for row in evaluations if row["admitted"] is True]
    if (
        not normalized_policy["minimum_admitted_factors"]
        <= len(admitted)
        <= normalized_policy["maximum_admitted_factors"]
    ):
        raise FactorGovernanceV5Error("admitted factor count is outside policy")
    candidates = {row["candidate_id"]: row for row in normalized_prereg["candidates"]}
    families = {candidates[row["candidate_id"]]["family"] for row in admitted}
    if len(families) < normalized_policy["minimum_admitted_families"]:
        raise FactorGovernanceV5Error("admitted family count is below policy")
    shrunk: dict[str, Decimal] = {}
    for row in admitted:
        mean_path_ic = max(Decimal("0"), Decimal(row["mean_path_ic"]))
        path_count = Decimal(row["path_count"])
        shrunk[row["candidate_id"]] = mean_path_ic * path_count / (path_count + Decimal("10"))
    weights = _largest_remainder_weights(shrunk)
    rows = [
        {
            "candidate_id": factor_id,
            "evaluation_ref": next(
                row["evaluation_receipt_id"] for row in admitted if row["candidate_id"] == factor_id
            ),
            "family": candidates[factor_id]["family"],
            "shrunk_ic": decimal_text(shrunk[factor_id]),
            "weight": weights[factor_id],
        }
        for factor_id in sorted(shrunk, key=lambda value: value.encode("ascii"))
    ]
    return seal(
        {
            **common_fields(timestamp_value=timestamp(built_at, label="built_at")),
            "version": ADMITTED_SET_VERSION,
            "factor_rows": rows,
            "lane": "PROSPECTIVE_ADMISSION",
            "policy_ref": normalized_policy["policy_id"],
            "preregistration_ref": normalized_prereg["preregistration_id"],
            "weight_total": "1.000000000000",
        },
        identity_field="factor_set_id",
    )


__all__ = ["ADMITTED_SET_VERSION", "build_admitted_factor_set"]
