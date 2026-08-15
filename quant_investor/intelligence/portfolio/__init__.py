"""Deterministic research portfolio, paper observation, and graduation gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .._common import (
    IntelligenceError,
    NO_AUTHORITY,
    artifact_payload,
    artifact_ref,
    build_artifact,
    business_identity,
    company_code,
    decimal_text,
    decimal_value,
    identifier,
    require_no_future,
    timestamp,
    validate_artifact_ref,
)
from ..investment_decision import validate_investment_decision

PORTFOLIO_STATUSES: Final = frozenset({"AVAILABLE", "BLOCKED"})
GRADUATION_STATUSES: Final = frozenset({"ELIGIBLE_FOR_OWNER_REVIEW", "NOT_ELIGIBLE"})


def _policy(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "cash_floor",
        "minimum_adv_cny",
        "per_security_cap",
        "target_gross",
        "target_positions",
        "turnover_cap",
    }
    if type(value) is not dict or set(value) != required:
        raise IntelligenceError(
            "portfolio policy shape is invalid; implicit defaults are forbidden"
        )
    target_positions = value["target_positions"]
    if type(target_positions) is not int or not 1 <= target_positions <= 100:
        raise IntelligenceError("target_positions is invalid")
    result: dict[str, Any] = {"target_positions": target_positions}
    for field in (
        "cash_floor",
        "per_security_cap",
        "target_gross",
        "turnover_cap",
    ):
        result[field] = decimal_value(
            value[field],
            label=f"portfolio_policy.{field}",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    result["minimum_adv_cny"] = decimal_value(
        value["minimum_adv_cny"],
        label="portfolio_policy.minimum_adv_cny",
        minimum=Decimal("0"),
    )
    if result["target_gross"] + result["cash_floor"] > Decimal("1"):
        raise IntelligenceError("portfolio gross and cash constraints are infeasible")
    return result


def _market_limits(
    policy: Mapping[str, Any],
    value: Mapping[str, Any] | None,
) -> tuple[Decimal, Decimal, Decimal, list[str], list[str]]:
    gross = policy["target_gross"]
    cash = policy["cash_floor"]
    security = policy["per_security_cap"]
    blockers: list[str] = []
    vetoes: list[str] = []
    if value is None:
        return gross, cash, security, blockers, vetoes
    required = {
        "blocker_codes",
        "effective_cash_floor",
        "effective_gross_cap",
        "effective_security_cap",
        "hard_veto_codes",
        "status",
    }
    if type(value) is not dict or set(value) != required:
        raise IntelligenceError("market risk projection shape is invalid")
    if value["status"] != "AVAILABLE":
        blockers.extend(str(code) for code in value["blocker_codes"])
        blockers.append("MARKET_RISK_UNAVAILABLE")
        return Decimal("0"), Decimal("1"), Decimal("0"), blockers, []
    projected_gross = decimal_value(
        value["effective_gross_cap"],
        label="market_risk.effective_gross_cap",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    projected_cash = decimal_value(
        value["effective_cash_floor"],
        label="market_risk.effective_cash_floor",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    projected_security = decimal_value(
        value["effective_security_cap"],
        label="market_risk.effective_security_cap",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if projected_gross > gross or projected_cash < cash or projected_security > security:
        raise IntelligenceError("market risk may only tighten owner limits")
    vetoes = sorted(
        {str(code) for code in value["hard_veto_codes"]},
        key=lambda item: item.encode("ascii"),
    )
    return projected_gross, projected_cash, projected_security, blockers, vetoes


def _candidate_rows(
    decisions: Sequence[Mapping[str, Any] | bytes],
    candidate_data: Mapping[str, Mapping[str, Any]],
    *,
    as_of: str,
    minimum_adv: Decimal,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise IntelligenceError("decisions must be a sequence")
    if type(candidate_data) is not dict:
        raise IntelligenceError("candidate_data must be an object")
    rows: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, value in enumerate(decisions):
        decision = validate_investment_decision(value)
        require_no_future(decision, as_of=as_of, label=f"decisions[{index}]")
        payload = decision["payload"]
        code = company_code(payload.get("company_code"))
        if code in seen:
            raise IntelligenceError("portfolio decision closure is duplicated")
        seen.add(code)
        refs.append(artifact_ref(decision))
        if payload.get("state") != "PAPER_CANDIDATE":
            continue
        data = candidate_data.get(code)
        if type(data) is not dict or set(data) != {"adv_cny", "current_weight"}:
            raise IntelligenceError(f"candidate_data for {code} is missing or invalid")
        adv = decimal_value(data["adv_cny"], label=f"{code}.adv_cny", minimum=Decimal("0"))
        current = decimal_value(
            data["current_weight"],
            label=f"{code}.current_weight",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if adv < minimum_adv:
            continue
        percentile = decimal_value(
            payload.get("deterministic_percentile"),
            label=f"{code}.deterministic_percentile",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        rows.append(
            {
                "adv_cny": adv,
                "company_code": code,
                "current_weight": current,
                "percentile": percentile,
            }
        )
    rows.sort(key=lambda row: (-row["percentile"], row["company_code"].encode("ascii")))
    refs.sort(key=lambda row: row["artifact_id"].encode("ascii"))
    return rows, refs


def construct_research_portfolio(
    *,
    strategy_id: str,
    decisions: Sequence[Mapping[str, Any] | bytes],
    candidate_data: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    as_of: str,
    market_risk: Mapping[str, Any] | None = None,
    portfolio_id: str | None = None,
) -> dict[str, Any]:
    """Construct an inactive research portfolio from PAPER_CANDIDATE only."""

    strategy = identifier(strategy_id, label="strategy_id")
    cutoff = timestamp(as_of, label="as_of")
    owner = _policy(policy)
    gross_cap, cash_floor, security_cap, blockers, vetoes = _market_limits(owner, market_risk)
    candidates, decision_refs = _candidate_rows(
        decisions,
        candidate_data,
        as_of=cutoff,
        minimum_adv=owner["minimum_adv_cny"],
    )
    selected = candidates[: owner["target_positions"]]
    if vetoes:
        blockers.append("MARKET_HARD_VETO")
    if not selected:
        blockers.append("NO_PAPER_CANDIDATES")
    targets: list[dict[str, Any]] = []
    if selected and not blockers:
        equal = min(security_cap, gross_cap / Decimal(len(selected)))
        desired = [equal for _ in selected]
        desired_gross = sum(desired, Decimal("0"))
        if desired_gross > Decimal("1") - cash_floor:
            scale = (Decimal("1") - cash_floor) / desired_gross
            desired = [weight * scale for weight in desired]
        turnover = sum(
            (abs(weight - row["current_weight"]) for weight, row in zip(desired, selected)),
            Decimal("0"),
        ) / Decimal("2")
        if turnover > owner["turnover_cap"] and turnover > Decimal("0"):
            interpolation = owner["turnover_cap"] / turnover
            desired = [
                row["current_weight"] + (weight - row["current_weight"]) * interpolation
                for weight, row in zip(desired, selected)
            ]
            blockers.append("TURNOVER_CONSTRAINED")
        targets = [
            {
                "company_code": row["company_code"],
                "final_weight": decimal_text(max(Decimal("0"), weight)),
                "rank": index,
            }
            for index, (row, weight) in enumerate(zip(selected, desired), start=1)
        ]
    gross = sum((Decimal(row["final_weight"]) for row in targets), Decimal("0"))
    cash = Decimal("1") - gross
    fatal = any(code not in {"TURNOVER_CONSTRAINED"} for code in blockers)
    status = "BLOCKED" if fatal else "AVAILABLE"
    if status == "BLOCKED":
        targets = []
        gross = Decimal("0")
        cash = Decimal("1")
    return build_artifact(
        kind="research_portfolio",
        identity_field="portfolio_id",
        identity=portfolio_id
        or business_identity(
            kind="research_portfolio",
            identity_inputs={"as_of": cutoff, "strategy_id": strategy},
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "blocker_codes": sorted(set(blockers), key=lambda item: item.encode("ascii")),
            "cash_weight": decimal_text(cash),
            "decision_refs": decision_refs,
            "gross_weight": decimal_text(gross),
            "hard_veto_codes": vetoes,
            "status": status,
            "strategy_id": strategy,
            "targets": targets,
        },
    )


def validate_research_portfolio(  # noqa: C901 - portfolio closure replay gate
    artifact: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    normalized, payload = artifact_payload(artifact, expected_kind="research_portfolio")
    if payload.get("status") not in PORTFOLIO_STATUSES:
        raise IntelligenceError("research portfolio status is invalid")
    if (
        payload.get("authority") != NO_AUTHORITY
        or payload.get("research_only") is not True
        or payload.get("production") is not False
        or payload.get("run_state") != "INACTIVE"
    ):
        raise IntelligenceError("research portfolio authority is invalid")
    identifier(payload.get("strategy_id"), label="portfolio.strategy_id")
    timestamp(payload.get("as_of"), label="portfolio.as_of")
    decision_refs = payload.get("decision_refs")
    if type(decision_refs) is not list:
        raise IntelligenceError("research portfolio decision refs are invalid")
    normalized_refs = [
        validate_artifact_ref(ref, label=f"decision_refs[{index}]")
        for index, ref in enumerate(decision_refs)
    ]
    if (
        normalized_refs != decision_refs
        or any(ref["kind"] != "investment_decision" for ref in normalized_refs)
        or len({ref["artifact_id"] for ref in normalized_refs}) != len(normalized_refs)
    ):
        raise IntelligenceError("research portfolio decision closure is invalid")
    gross = decimal_value(payload.get("gross_weight"), label="portfolio.gross_weight")
    cash = decimal_value(payload.get("cash_weight"), label="portfolio.cash_weight")
    if gross + cash != Decimal("1"):
        raise IntelligenceError("research portfolio weights do not close")
    targets = payload.get("targets")
    if type(targets) is not list:
        raise IntelligenceError("research portfolio targets are invalid")
    target_total = Decimal("0")
    target_codes: list[str] = []
    for index, row in enumerate(targets, start=1):
        if type(row) is not dict or set(row) != {
            "company_code",
            "final_weight",
            "rank",
        }:
            raise IntelligenceError("research portfolio target shape is invalid")
        target_codes.append(company_code(row["company_code"]))
        if row["rank"] != index:
            raise IntelligenceError("research portfolio ranks are not contiguous")
        target_total += decimal_value(
            row["final_weight"],
            label=f"targets[{index - 1}].final_weight",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    if len(target_codes) != len(set(target_codes)) or target_total != gross:
        raise IntelligenceError("research portfolio target weights do not close")
    if payload["status"] == "BLOCKED" and (payload.get("targets") or gross != Decimal("0")):
        raise IntelligenceError("blocked research portfolio cannot carry targets")
    return normalized


def observe_paper_portfolio(
    *,
    portfolio: Mapping[str, Any] | bytes,
    as_of: str,
    gross_return: Any,
    benchmark_return: Any,
    estimated_cost: Any,
    drawdown: Any,
    observation_id: str | None = None,
) -> dict[str, Any]:
    """Record a supplied paper outcome; this performs no order or trade call."""

    cutoff = timestamp(as_of, label="as_of")
    portfolio_artifact = validate_research_portfolio(portfolio)
    require_no_future(portfolio_artifact, as_of=cutoff, label="portfolio")
    if portfolio_artifact["payload"].get("status") != "AVAILABLE":
        raise IntelligenceError("paper observation requires an available research portfolio")
    gross = decimal_value(gross_return, label="gross_return")
    benchmark = decimal_value(benchmark_return, label="benchmark_return")
    cost = decimal_value(estimated_cost, label="estimated_cost", minimum=Decimal("0"))
    drawdown_value = decimal_value(
        drawdown, label="drawdown", minimum=Decimal("0"), maximum=Decimal("1")
    )
    net = gross - cost
    strategy = portfolio_artifact["payload"]["strategy_id"]
    return build_artifact(
        kind="paper_observation",
        identity_field="observation_id",
        identity=observation_id
        or business_identity(
            kind="paper_observation",
            identity_inputs={
                "as_of": cutoff,
                "portfolio_id": portfolio_artifact["artifact_id"],
            },
        ),
        created_at=cutoff,
        fields={
            "as_of": cutoff,
            "benchmark_return": decimal_text(benchmark),
            "drawdown": decimal_text(drawdown_value),
            "estimated_cost": decimal_text(cost),
            "excess_return": decimal_text(net - benchmark),
            "gross_return": decimal_text(gross),
            "net_return": decimal_text(net),
            "portfolio_ref": artifact_ref(portfolio_artifact),
            "status": "OBSERVED",
            "strategy_id": strategy,
        },
    )


def assess_graduation(
    *,
    strategy_id: str,
    observations: Sequence[Mapping[str, Any] | bytes],
    minimum_observations: int,
    minimum_excess_return: Any,
    maximum_drawdown: Any,
    assessed_at: str,
    assessment_id: str | None = None,
) -> dict[str, Any]:
    """Assess paper evidence for owner review; never activate a strategy."""

    strategy = identifier(strategy_id, label="strategy_id")
    cutoff = timestamp(assessed_at, label="assessed_at")
    if type(minimum_observations) is not int or minimum_observations < 1:
        raise IntelligenceError("minimum_observations must be positive")
    excess_floor = decimal_value(minimum_excess_return, label="minimum_excess_return")
    drawdown_cap = decimal_value(
        maximum_drawdown,
        label="maximum_drawdown",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, value in enumerate(observations):
        artifact, payload = artifact_payload(value, expected_kind="paper_observation")
        require_no_future(artifact, as_of=cutoff, label=f"observations[{index}]")
        if payload.get("strategy_id") != strategy:
            raise IntelligenceError("paper observation belongs to another strategy")
        rows.append((artifact, payload))
    rows.sort(key=lambda item: item[1]["as_of"])
    blockers: list[str] = []
    if len(rows) < minimum_observations:
        blockers.append("INSUFFICIENT_PAPER_OBSERVATIONS")
    cumulative_excess = sum(
        (Decimal(payload["excess_return"]) for _, payload in rows), Decimal("0")
    )
    worst_drawdown = max(
        (Decimal(payload["drawdown"]) for _, payload in rows), default=Decimal("0")
    )
    if cumulative_excess < excess_floor:
        blockers.append("COST_ADJUSTED_EXCESS_RETURN_BELOW_POLICY")
    if worst_drawdown > drawdown_cap:
        blockers.append("PAPER_DRAWDOWN_ABOVE_POLICY")
    status = "NOT_ELIGIBLE" if blockers else "ELIGIBLE_FOR_OWNER_REVIEW"
    return build_artifact(
        kind="graduation_assessment",
        identity_field="assessment_id",
        identity=assessment_id
        or business_identity(
            kind="graduation_assessment",
            identity_inputs={"assessed_at": cutoff, "strategy_id": strategy},
        ),
        created_at=cutoff,
        fields={
            "assessed_at": cutoff,
            "blocker_codes": blockers,
            "cumulative_excess_return": decimal_text(cumulative_excess),
            "observation_refs": [artifact_ref(artifact) for artifact, _ in rows],
            "observation_count": len(rows),
            "status": status,
            "strategy_id": strategy,
            "worst_drawdown": decimal_text(worst_drawdown),
        },
    )


__all__ = [
    "GRADUATION_STATUSES",
    "IntelligenceError",
    "PORTFOLIO_STATUSES",
    "assess_graduation",
    "construct_research_portfolio",
    "observe_paper_portfolio",
    "validate_research_portfolio",
]
