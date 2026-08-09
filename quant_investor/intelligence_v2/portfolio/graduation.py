"""Research-only paper graduation evidence with closed maturity requirements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .contracts import (
    PortfolioContractError,
    content_ref,
    decimal_in_unit,
    decimal_text,
    decimal_value,
    exact_source_ref,
    portfolio_common,
    require_exact_keys,
    seal,
    timestamp,
    validate_seal,
)
from .paper import validate_paper_outcome

GRADUATION_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.graduation-policy.v2"
GRADUATION_RECEIPT_VERSION: Final = "myquant.v17.intelligence-v2.graduation-receipt.v2"

POLICY_FIELDS: Final = {
    "authority",
    "benchmark_ref",
    "decision_protocol",
    "maximum_drawdown",
    "maximum_regime_changes",
    "minimum_cost_adjusted_excess_return",
    "minimum_coverage",
    "minimum_matured_observations",
    "policy_id",
    "production",
    "require_no_hard_risk_breach",
    "required_horizons",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
RECEIPT_FIELDS: Final = {
    "authority",
    "average_cost_adjusted_excess_return",
    "benchmark_ref",
    "blocker_codes",
    "coverage",
    "decision_protocol",
    "graduation_id",
    "hard_risk_breach_count",
    "matured_by_horizon",
    "matured_observations",
    "maximum_drawdown",
    "outcome_refs",
    "policy_ref",
    "production",
    "regime_change_count",
    "research_only",
    "semantic_sha256",
    "status",
    "timestamp",
    "version",
}


def _horizons(values: Sequence[Any]) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PortfolioContractError("required_horizons must be a sequence")
    rows = list(values)
    if (
        not rows
        or any(type(value) is not int or value not in {1, 5, 20, 60} for value in rows)
        or rows != sorted(rows)
        or len(rows) != len(set(rows))
    ):
        raise PortfolioContractError("required_horizons must be sorted unique 1D/5D/20D/60D")
    return rows


def build_graduation_policy(
    *,
    created_at: str,
    required_horizons: Sequence[int],
    benchmark_ref: Mapping[str, Any],
    minimum_matured_observations: int,
    minimum_coverage: Any,
    minimum_cost_adjusted_excess_return: Any,
    maximum_drawdown: Any,
    maximum_regime_changes: int,
    require_no_hard_risk_breach: bool,
) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    horizons = _horizons(required_horizons)
    if (
        type(minimum_matured_observations) is not int
        or not 1 <= minimum_matured_observations <= 100_000
    ):
        raise PortfolioContractError("minimum_matured_observations is invalid")
    if type(maximum_regime_changes) is not int or not 0 <= maximum_regime_changes <= 100_000:
        raise PortfolioContractError("maximum_regime_changes is invalid")
    if require_no_hard_risk_breach is not True:
        raise PortfolioContractError("graduation must require no hard risk breach")
    return seal(
        {
            **portfolio_common(at=issued_at),
            "benchmark_ref": exact_source_ref(
                benchmark_ref, label="benchmark_ref", as_of=issued_at
            ),
            "maximum_drawdown": decimal_text(
                decimal_in_unit(maximum_drawdown, label="maximum_drawdown")
            ),
            "maximum_regime_changes": maximum_regime_changes,
            "minimum_cost_adjusted_excess_return": decimal_text(
                decimal_value(
                    minimum_cost_adjusted_excess_return,
                    label="minimum_cost_adjusted_excess_return",
                    minimum=Decimal("-1"),
                    maximum=Decimal("1"),
                )
            ),
            "minimum_coverage": decimal_text(
                decimal_in_unit(minimum_coverage, label="minimum_coverage")
            ),
            "minimum_matured_observations": minimum_matured_observations,
            "require_no_hard_risk_breach": True,
            "required_horizons": horizons,
            "version": GRADUATION_POLICY_VERSION,
        },
        identity_field="policy_id",
    )


def validate_graduation_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_seal(document, identity_field="policy_id")
    require_exact_keys(row, POLICY_FIELDS, label="graduation policy")
    expected = build_graduation_policy(
        created_at=row["timestamp"],
        required_horizons=row["required_horizons"],
        benchmark_ref=row["benchmark_ref"],
        minimum_matured_observations=row["minimum_matured_observations"],
        minimum_coverage=row["minimum_coverage"],
        minimum_cost_adjusted_excess_return=row["minimum_cost_adjusted_excess_return"],
        maximum_drawdown=row["maximum_drawdown"],
        maximum_regime_changes=row["maximum_regime_changes"],
        require_no_hard_risk_breach=row["require_no_hard_risk_breach"],
    )
    if row != expected or row["version"] != GRADUATION_POLICY_VERSION:
        raise PortfolioContractError("graduation policy replay mismatch")
    return row


def _validated_outcomes(
    outcomes: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
    *,
    evaluated_at: str,
) -> list[dict[str, Any]]:
    if len(outcomes) != len(closures):
        raise PortfolioContractError("outcome closure inventory mismatch")
    rows = []
    for outcome, closure in zip(outcomes, closures):
        if type(closure) is not dict:
            raise PortfolioContractError("outcome validation closure must be exact")
        row = validate_paper_outcome(outcome, **dict(closure))
        if row["timestamp"] > evaluated_at:
            raise PortfolioContractError("graduation contains future outcome")
        rows.append(row)
    identities = [row["outcome_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise PortfolioContractError("graduation contains duplicate outcomes")
    return sorted(rows, key=lambda row: (row["timestamp"], row["outcome_id"]))


def build_graduation_receipt(
    *,
    policy: Mapping[str, Any] | None,
    outcomes: Sequence[Mapping[str, Any]],
    outcome_validation_closures: Sequence[Mapping[str, Any]],
    evaluated_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(evaluated_at, label="evaluated_at")
    rows = _validated_outcomes(outcomes, outcome_validation_closures, evaluated_at=issued_at)
    count = len(rows)
    divisor = Decimal(count) if count else Decimal("1")
    average_adjusted = (
        sum((Decimal(row["cost_adjusted_excess_return"]) for row in rows), Decimal("0")) / divisor
    )
    maximum_observed_drawdown = max(
        (Decimal(row["maximum_drawdown"]) for row in rows), default=Decimal("0")
    )
    regime_ids = [row["regime_ref"]["artifact_id"] for row in rows]
    regime_changes = sum(left != right for left, right in zip(regime_ids, regime_ids[1:]))
    hard_breaches = sum(row["hard_risk_breach"] for row in rows)
    counts = {
        horizon: sum(row["horizon_sessions"] == horizon for row in rows)
        for horizon in (1, 5, 20, 60)
    }
    blockers = []
    policy_ref = None
    benchmark_ref = None
    required: list[int] = []
    if policy is None:
        blockers.append("GRADUATION_POLICY_UNAVAILABLE")
    else:
        policy_row = validate_graduation_policy(policy)
        policy_ref = content_ref(policy_row, identity_field="policy_id")
        benchmark_ref = policy_row["benchmark_ref"]
        required = policy_row["required_horizons"]
        if any(row["benchmark_ref"] != benchmark_ref for row in rows):
            raise PortfolioContractError("outcomes do not share the owner benchmark closure")
        if count < policy_row["minimum_matured_observations"]:
            blockers.append("INSUFFICIENT_MATURED_OBSERVATIONS")
        missing = [value for value in required if counts[value] == 0]
        blockers.extend(f"REQUIRED_{value}D_NOT_MATURED" for value in missing)
        coverage = Decimal(len(required) - len(missing)) / Decimal(len(required))
        if coverage < Decimal(policy_row["minimum_coverage"]):
            blockers.append("COVERAGE_BELOW_MINIMUM")
        if average_adjusted < Decimal(policy_row["minimum_cost_adjusted_excess_return"]):
            blockers.append("COST_ADJUSTED_EXCESS_BELOW_MINIMUM")
        if maximum_observed_drawdown > Decimal(policy_row["maximum_drawdown"]):
            blockers.append("DRAWDOWN_ABOVE_MAXIMUM")
        if regime_changes > policy_row["maximum_regime_changes"]:
            blockers.append("REGIME_STABILITY_FAILED")
        if hard_breaches:
            blockers.append("HARD_RISK_BREACH_OBSERVED")
    coverage = (
        Decimal(sum(counts[value] > 0 for value in required)) / Decimal(len(required))
        if required
        else Decimal("0")
    )
    status = "ELIGIBLE_FOR_OWNER_REVIEW" if policy is not None and not blockers else "NOT_ELIGIBLE"
    return seal(
        {
            **portfolio_common(at=issued_at),
            "average_cost_adjusted_excess_return": decimal_text(average_adjusted),
            "benchmark_ref": benchmark_ref,
            "blocker_codes": sorted(set(blockers), key=lambda value: value.encode("ascii")),
            "coverage": decimal_text(coverage),
            "hard_risk_breach_count": hard_breaches,
            "matured_by_horizon": [
                {"count": counts[value], "horizon_sessions": value} for value in (1, 5, 20, 60)
            ],
            "matured_observations": count,
            "maximum_drawdown": decimal_text(maximum_observed_drawdown),
            "outcome_refs": [content_ref(row, identity_field="outcome_id") for row in rows],
            "policy_ref": policy_ref,
            "regime_change_count": regime_changes,
            "status": status,
            "version": GRADUATION_RECEIPT_VERSION,
        },
        identity_field="graduation_id",
    )


def validate_graduation_receipt(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="graduation_id")
    require_exact_keys(row, RECEIPT_FIELDS, label="graduation receipt")
    expected = build_graduation_receipt(**closure)
    if row != expected or row["version"] != GRADUATION_RECEIPT_VERSION:
        raise PortfolioContractError("graduation receipt replay mismatch")
    return row


__all__ = [
    "GRADUATION_POLICY_VERSION",
    "GRADUATION_RECEIPT_VERSION",
    "build_graduation_policy",
    "build_graduation_receipt",
    "validate_graduation_policy",
    "validate_graduation_receipt",
]
