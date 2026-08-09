"""Exact owner-authored policies for deterministic I6 portfolio research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from .contracts import (
    PortfolioContractError,
    decimal_in_unit,
    decimal_text,
    decimal_value,
    identifier,
    portfolio_common,
    positive_decimal,
    require_quantum_multiple,
    require_exact_keys,
    seal,
    sorted_codes,
    timestamp,
    validate_content_ref,
    validate_seal,
)

PORTFOLIO_RISK_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.portfolio-risk-policy.v2"

MACRO_RULE_FIELDS: Final = {
    "cash_floor",
    "gross_cap",
    "regime",
    "risk_multiplier",
    "veto_codes",
}

POLICY_FIELDS: Final = {
    "authority",
    "cash_floor",
    "decision_protocol",
    "drawdown_threshold",
    "fundamental_staleness_allowance_sessions",
    "graduation_policy_ref",
    "hard_veto_codes",
    "industry_cap",
    "macro_regime_rules",
    "max_adv_participation",
    "paper_execution_policy_ref",
    "per_security_cap",
    "policy_id",
    "production",
    "research_only",
    "risk_threshold",
    "semantic_sha256",
    "target_gross",
    "target_positions",
    "theme_cap",
    "timestamp",
    "turnover_cap",
    "version",
    "weight_quantum",
}


def _macro_rules(
    values: Sequence[Mapping[str, Any]],
    *,
    target_gross: Decimal,
    cash_floor: Decimal,
    hard_veto_codes: Sequence[str],
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise PortfolioContractError("macro_regime_rules must be a nonempty sequence")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    baseline_vetoes = set(hard_veto_codes)
    for index, value in enumerate(values):
        row = require_exact_keys(value, MACRO_RULE_FIELDS, label=f"macro_regime_rules[{index}]")
        regime = identifier(row["regime"], label=f"macro_regime_rules[{index}].regime")
        if regime in seen:
            raise PortfolioContractError("macro_regime_rules contains duplicate regimes")
        seen.add(regime)
        gross_cap = decimal_in_unit(row["gross_cap"], label=f"{regime}.gross_cap")
        rule_cash = decimal_in_unit(row["cash_floor"], label=f"{regime}.cash_floor")
        multiplier = decimal_value(
            row["risk_multiplier"],
            label=f"{regime}.risk_multiplier",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if gross_cap > target_gross or rule_cash < cash_floor:
            raise PortfolioContractError("macro rules may only tighten gross and cash")
        if gross_cap + rule_cash > Decimal("1"):
            raise PortfolioContractError("macro gross/cash rule is infeasible")
        vetoes = sorted_codes(
            row["veto_codes"],
            label=f"{regime}.veto_codes",
            allow_empty=True,
        )
        if not baseline_vetoes.issubset(vetoes):
            raise PortfolioContractError("macro rules may not remove owner hard vetoes")
        rows.append(
            {
                "cash_floor": decimal_text(rule_cash),
                "gross_cap": decimal_text(gross_cap),
                "regime": regime,
                "risk_multiplier": decimal_text(multiplier),
                "veto_codes": vetoes,
            }
        )
    expected = sorted(rows, key=lambda item: item["regime"].encode("ascii"))
    if rows != expected:
        raise PortfolioContractError("macro_regime_rules must be ASCII sorted")
    return rows


def build_portfolio_risk_policy(
    *,
    created_at: str,
    target_positions: int,
    target_gross: Any,
    cash_floor: Any,
    per_security_cap: Any,
    industry_cap: Any,
    theme_cap: Any,
    max_adv_participation: Any,
    turnover_cap: Any,
    weight_quantum: Any,
    drawdown_threshold: Any,
    risk_threshold: Any,
    hard_veto_codes: Sequence[Any],
    macro_regime_rules: Sequence[Mapping[str, Any]],
    fundamental_staleness_allowance_sessions: int,
    paper_execution_policy_ref: Mapping[str, Any],
    graduation_policy_ref: Mapping[str, Any],
) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    if type(target_positions) is not int or not 1 <= target_positions <= 100:
        raise PortfolioContractError("target_positions must be between 1 and 100")
    if (
        type(fundamental_staleness_allowance_sessions) is not int
        or not 0 <= fundamental_staleness_allowance_sessions <= 1
    ):
        raise PortfolioContractError("fundamental staleness allowance must be 0 or 1")

    gross = decimal_in_unit(target_gross, label="target_gross")
    cash = decimal_in_unit(cash_floor, label="cash_floor")
    security = positive_decimal(per_security_cap, label="per_security_cap", maximum=Decimal("1"))
    industry = positive_decimal(industry_cap, label="industry_cap", maximum=Decimal("1"))
    theme = positive_decimal(theme_cap, label="theme_cap", maximum=Decimal("1"))
    adv = positive_decimal(
        max_adv_participation,
        label="max_adv_participation",
        maximum=Decimal("0.10"),
    )
    turnover = decimal_in_unit(turnover_cap, label="turnover_cap")
    quantum = positive_decimal(weight_quantum, label="weight_quantum", maximum=Decimal("1"))
    drawdown = decimal_in_unit(drawdown_threshold, label="drawdown_threshold")
    risk = decimal_in_unit(risk_threshold, label="risk_threshold")
    if gross + cash > Decimal("1"):
        raise PortfolioContractError("target gross and cash floor are infeasible")
    if quantum > min(security, industry, theme, gross if gross else quantum):
        raise PortfolioContractError("weight quantum exceeds an active portfolio cap")
    vetoes = sorted_codes(hard_veto_codes, label="hard_veto_codes", allow_empty=True)
    rules = _macro_rules(
        macro_regime_rules,
        target_gross=gross,
        cash_floor=cash,
        hard_veto_codes=vetoes,
    )
    for label, value in (
        ("target_gross", gross),
        ("per_security_cap", security),
        ("industry_cap", industry),
        ("theme_cap", theme),
        ("turnover_cap", turnover),
    ):
        require_quantum_multiple(value, quantum, label=label)
    for row in rules:
        require_quantum_multiple(
            Decimal(row["gross_cap"]), quantum, label=f"{row['regime']}.gross_cap"
        )
    paper_ref = validate_content_ref(
        paper_execution_policy_ref,
        label="paper_execution_policy_ref",
    )
    graduation_ref = validate_content_ref(
        graduation_policy_ref,
        label="graduation_policy_ref",
    )
    return seal(
        {
            **portfolio_common(at=issued_at),
            "cash_floor": decimal_text(cash),
            "drawdown_threshold": decimal_text(drawdown),
            "fundamental_staleness_allowance_sessions": (fundamental_staleness_allowance_sessions),
            "graduation_policy_ref": graduation_ref,
            "hard_veto_codes": vetoes,
            "industry_cap": decimal_text(industry),
            "macro_regime_rules": rules,
            "max_adv_participation": decimal_text(adv),
            "paper_execution_policy_ref": paper_ref,
            "per_security_cap": decimal_text(security),
            "risk_threshold": decimal_text(risk),
            "target_gross": decimal_text(gross),
            "target_positions": target_positions,
            "theme_cap": decimal_text(theme),
            "turnover_cap": decimal_text(turnover),
            "version": PORTFOLIO_RISK_POLICY_VERSION,
            "weight_quantum": decimal_text(quantum),
        },
        identity_field="policy_id",
    )


def validate_portfolio_risk_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="policy_id")
    require_exact_keys(normalized, POLICY_FIELDS, label="portfolio risk policy")
    expected = build_portfolio_risk_policy(
        created_at=normalized["timestamp"],
        target_positions=normalized["target_positions"],
        target_gross=normalized["target_gross"],
        cash_floor=normalized["cash_floor"],
        per_security_cap=normalized["per_security_cap"],
        industry_cap=normalized["industry_cap"],
        theme_cap=normalized["theme_cap"],
        max_adv_participation=normalized["max_adv_participation"],
        turnover_cap=normalized["turnover_cap"],
        weight_quantum=normalized["weight_quantum"],
        drawdown_threshold=normalized["drawdown_threshold"],
        risk_threshold=normalized["risk_threshold"],
        hard_veto_codes=normalized["hard_veto_codes"],
        macro_regime_rules=normalized["macro_regime_rules"],
        fundamental_staleness_allowance_sessions=(
            normalized["fundamental_staleness_allowance_sessions"]
        ),
        paper_execution_policy_ref=normalized["paper_execution_policy_ref"],
        graduation_policy_ref=normalized["graduation_policy_ref"],
    )
    if normalized != expected or normalized["version"] != PORTFOLIO_RISK_POLICY_VERSION:
        raise PortfolioContractError("portfolio risk policy replay mismatch")
    return normalized


def macro_rule_for(policy: Mapping[str, Any], regime: str) -> dict[str, Any]:
    validated = validate_portfolio_risk_policy(policy)
    name = identifier(regime, label="macro_regime")
    rows = [row for row in validated["macro_regime_rules"] if row["regime"] == name]
    if len(rows) != 1:
        raise PortfolioContractError("macro regime has no exact owner policy rule")
    return dict(rows[0])


__all__ = [
    "MACRO_RULE_FIELDS",
    "POLICY_FIELDS",
    "PORTFOLIO_RISK_POLICY_VERSION",
    "build_portfolio_risk_policy",
    "macro_rule_for",
    "validate_portfolio_risk_policy",
]
