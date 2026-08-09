"""Deterministic P_D/P_A research portfolio construction for I6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

from ..decision_v2 import validate_decision_receipt_v2
from .contracts import (
    CURRENT_POSITION_FIELDS,
    SUBJECT_FIELDS,
    PortfolioContractError,
    company_code,
    content_ref,
    decimal_in_unit,
    decimal_text,
    exact_source_ref,
    identifier,
    portfolio_common,
    quantum_floor,
    require_exact_keys,
    require_quantum_multiple,
    seal,
    sorted_codes,
    target_row,
    validate_seal,
)
from .policies import macro_rule_for, validate_portfolio_risk_policy

PORTFOLIO_CONSTRUCTION_VERSION: Final = (
    "myquant.v17.intelligence-v2.portfolio-construction-receipt.v2"
)
ADVISORY_TV_CEILING: Final = Decimal("0.10")

CONSTRUCTION_FIELDS: Final = {
    "admitted_decision_refs",
    "advisory_capital_tv",
    "advisory_fallback",
    "authority",
    "blocker_codes",
    "current_position_rows",
    "decision_protocol",
    "final_portfolio",
    "macro_ref",
    "p_a",
    "p_d",
    "policy_ref",
    "production",
    "receipt_id",
    "research_only",
    "semantic_sha256",
    "status",
    "timestamp",
    "version",
}

SUBPORTFOLIO_FIELDS: Final = {
    "blocker_codes",
    "cash_weight",
    "gross_weight",
    "ordering",
    "status",
    "targets",
}


def _replay_decisions(  # noqa: C901 - closed replay has intentional contract branches
    subjects: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(subjects, (str, bytes)) or not isinstance(subjects, Sequence):
        raise PortfolioContractError("subjects must be a sequence")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(subjects):
        row = require_exact_keys(source, SUBJECT_FIELDS, label=f"subjects[{index}]")
        company = company_code(row["company_code"], label=f"subjects[{index}].company_code")
        if company in seen:
            raise PortfolioContractError("subjects contains duplicate company_code")
        seen.add(company)
        if type(row["decision_validation_closure"]) is not dict:
            raise PortfolioContractError("decision validation closure must be an exact object")
        try:
            decision = dict(
                validate_decision_receipt_v2(
                    row["decision_receipt"],
                    **dict(row["decision_validation_closure"]),
                )
            )
        except Exception as exc:
            raise PortfolioContractError("Decision v2 receipt replay failed") from exc
        if decision.get("state") != "PAPER_CANDIDATE" or decision.get("timestamp") != as_of:
            raise PortfolioContractError("portfolio admission requires same-time PAPER_CANDIDATE")
        decision_ref = content_ref(decision, identity_field="decision_id")
        deterministic = decimal_in_unit(
            row["deterministic_percentile"], label=f"{company}.deterministic_percentile"
        )
        if Decimal(decision.get("deterministic_percentile", "-1")) != deterministic:
            raise PortfolioContractError(
                "subject deterministic percentile does not match replayed Decision v2"
            )
        advisory_value = row["advisory_percentile"]
        advisory = (
            None
            if advisory_value is None
            else decimal_in_unit(advisory_value, label=f"{company}.advisory_percentile")
        )
        themes = sorted_codes(row["theme_codes"], label=f"{company}.theme_codes", allow_empty=False)
        theme_refs = row["theme_refs"]
        if not isinstance(theme_refs, Sequence) or isinstance(theme_refs, (str, bytes)):
            raise PortfolioContractError("theme_refs must be a sequence")
        normalized_theme_refs = [
            exact_source_ref(value, label=f"{company}.theme_refs[{position}]", as_of=as_of)
            for position, value in enumerate(theme_refs)
        ]
        if len(normalized_theme_refs) != len(themes):
            raise PortfolioContractError("theme codes and refs must be one-to-one")
        rows.append(
            {
                "advisory_percentile": advisory,
                "adv_weight_capacity": decimal_in_unit(
                    row["adv_weight_capacity"], label=f"{company}.adv_weight_capacity"
                ),
                "company_code": company,
                "decision_ref": decision_ref,
                "deterministic_percentile": deterministic,
                "drawdown": decimal_in_unit(row["drawdown"], label=f"{company}.drawdown"),
                "fundamental_age_sessions": row["fundamental_age_sessions"],
                "hard_veto_codes": sorted_codes(
                    row["hard_veto_codes"],
                    label=f"{company}.hard_veto_codes",
                    allow_empty=True,
                ),
                "industry_code": identifier(row["industry_code"], label=f"{company}.industry_code"),
                "industry_ref": exact_source_ref(
                    row["industry_ref"], label=f"{company}.industry_ref", as_of=as_of
                ),
                "liquidity_ref": exact_source_ref(
                    row["liquidity_ref"], label=f"{company}.liquidity_ref", as_of=as_of
                ),
                "risk_score": decimal_in_unit(row["risk_score"], label=f"{company}.risk_score"),
                "security_ref": exact_source_ref(
                    row["security_ref"], label=f"{company}.security_ref", as_of=as_of
                ),
                "theme_codes": themes,
                "theme_refs": normalized_theme_refs,
            }
        )
    return rows, []


def _current_positions(values: Sequence[Mapping[str, Any]], *, as_of: str) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PortfolioContractError("current_positions must be a sequence")
    rows = []
    seen: set[str] = set()
    for index, source in enumerate(values):
        row = require_exact_keys(
            source, CURRENT_POSITION_FIELDS, label=f"current_positions[{index}]"
        )
        company = company_code(row["company_code"], label="current company_code")
        if company in seen:
            raise PortfolioContractError("current_positions contains duplicate company")
        seen.add(company)
        themes = sorted_codes(row["theme_codes"], label=f"{company}.theme_codes", allow_empty=False)
        theme_refs = row["theme_refs"]
        if not isinstance(theme_refs, Sequence) or isinstance(theme_refs, (str, bytes)):
            raise PortfolioContractError("current theme_refs must be a sequence")
        normalized_theme_refs = [
            exact_source_ref(value, label=f"{company}.theme_ref", as_of=as_of)
            for value in theme_refs
        ]
        if len(normalized_theme_refs) != len(themes):
            raise PortfolioContractError("current theme codes and refs must be one-to-one")
        rows.append(
            {
                "adv_weight_capacity": decimal_in_unit(
                    row["adv_weight_capacity"], label=f"{company}.adv_weight_capacity"
                ),
                "company_code": company,
                "current_weight": decimal_in_unit(
                    row["current_weight"], label=f"{company}.current_weight"
                ),
                "industry_code": identifier(row["industry_code"], label=f"{company}.industry_code"),
                "industry_ref": exact_source_ref(
                    row["industry_ref"], label=f"{company}.industry_ref", as_of=as_of
                ),
                "liquidity_ref": exact_source_ref(
                    row["liquidity_ref"], label=f"{company}.liquidity_ref", as_of=as_of
                ),
                "security_ref": exact_source_ref(
                    row["security_ref"], label=f"{company}.security_ref", as_of=as_of
                ),
                "theme_codes": themes,
                "theme_refs": normalized_theme_refs,
            }
        )
    return rows


def _effective_limits(
    policy: Mapping[str, Any], macro_regime: str
) -> tuple[dict[str, Decimal], set[str]]:
    rule = macro_rule_for(policy, macro_regime)
    risk_threshold = Decimal(policy["risk_threshold"]) * Decimal(rule["risk_multiplier"])
    values = {
        "cash_floor": max(Decimal(policy["cash_floor"]), Decimal(rule["cash_floor"])),
        "gross": min(Decimal(policy["target_gross"]), Decimal(rule["gross_cap"])),
        "industry": Decimal(policy["industry_cap"]),
        "risk": risk_threshold,
        "security": Decimal(policy["per_security_cap"]),
        "theme": Decimal(policy["theme_cap"]),
    }
    values["gross"] = min(values["gross"], Decimal("1") - values["cash_floor"])
    vetoes = set(policy["hard_veto_codes"]) | set(rule["veto_codes"])
    return values, vetoes


def _eligible(
    subjects: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    limits: Mapping[str, Decimal],
    vetoes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in subjects:
        age = value["fundamental_age_sessions"]
        if type(age) is not int or age < 0:
            raise PortfolioContractError("fundamental_age_sessions must be a nonnegative int")
        if age > policy["fundamental_staleness_allowance_sessions"]:
            continue
        if value["risk_score"] > limits["risk"]:
            continue
        if value["drawdown"] > Decimal(policy["drawdown_threshold"]):
            continue
        if set(value["hard_veto_codes"]) & vetoes:
            continue
        rows.append(dict(value))
    return rows


def _ordered(subjects: Sequence[Mapping[str, Any]], *, advisory: bool) -> list[dict[str, Any]]:
    if advisory:
        if any(row["advisory_percentile"] is None for row in subjects):
            return _ordered(subjects, advisory=False)
        return sorted(
            (dict(row) for row in subjects),
            key=lambda row: (
                -row["advisory_percentile"],
                -row["deterministic_percentile"],
                row["company_code"].encode("ascii"),
            ),
        )
    return sorted(
        (dict(row) for row in subjects),
        key=lambda row: (-row["deterministic_percentile"], row["company_code"].encode("ascii")),
    )


def _allocation_candidate_allowed(
    *,
    row: Mapping[str, Any],
    proposed: Decimal,
    industry_used: Mapping[str, Decimal],
    theme_used: Mapping[str, Decimal],
    gross_used: Decimal,
    participation: Decimal,
    limits: Mapping[str, Decimal],
    quantum: Decimal,
) -> bool:
    if proposed > limits["security"]:
        return False
    liquidity_cap = quantum_floor(row["adv_weight_capacity"] * participation, quantum)
    if proposed > liquidity_cap:
        return False
    industry = row["industry_code"]
    if industry_used.get(industry, Decimal("0")) + quantum > limits["industry"]:
        return False
    if any(
        theme_used.get(theme, Decimal("0")) + quantum > limits["theme"]
        for theme in row["theme_codes"]
    ):
        return False
    if gross_used + quantum > limits["gross"]:
        return False
    return Decimal("1") - gross_used - quantum >= limits["cash_floor"]


def _allocate_round_robin(
    selected: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    limits: Mapping[str, Decimal],
    quantum: Decimal,
) -> dict[str, Decimal]:
    """Allocate one quantum per stable candidate pass without stranded cash."""

    weights = {row["company_code"]: Decimal("0") for row in selected}
    industry_used: dict[str, Decimal] = {}
    theme_used: dict[str, Decimal] = {}
    participation = Decimal(policy["max_adv_participation"])
    gross_used = Decimal("0")
    gross_cap = min(limits["gross"], Decimal("1") - limits["cash_floor"])
    effective_limits = {**limits, "gross": gross_cap}
    maximum_steps = int((gross_cap / quantum).to_integral_value())
    if maximum_steps > 1_000_000:
        raise PortfolioContractError("weight quantum creates excessive allocation steps")
    while gross_used + quantum <= gross_cap:
        progressed = False
        for row in selected:
            company = row["company_code"]
            proposed = weights[company] + quantum
            # SECURITY -> LIQUIDITY -> INDUSTRY -> THEME -> GROSS -> CASH.
            if not _allocation_candidate_allowed(
                row=row,
                proposed=proposed,
                industry_used=industry_used,
                theme_used=theme_used,
                gross_used=gross_used,
                participation=participation,
                limits=effective_limits,
                quantum=quantum,
            ):
                continue
            industry = row["industry_code"]
            weights[company] = proposed
            gross_used += quantum
            industry_used[industry] = industry_used.get(industry, Decimal("0")) + quantum
            for theme in row["theme_codes"]:
                theme_used[theme] = theme_used.get(theme, Decimal("0")) + quantum
            progressed = True
            if gross_used == gross_cap:
                break
        if not progressed:
            break
    return weights


def _capital_tv(
    left: Mapping[str, Decimal],
    right: Mapping[str, Decimal],
) -> Decimal:
    symbols = set(left) | set(right)
    left_cash = Decimal("1") - sum(left.values(), Decimal("0"))
    right_cash = Decimal("1") - sum(right.values(), Decimal("0"))
    distance = sum(
        (abs(left.get(code, Decimal("0")) - right.get(code, Decimal("0"))) for code in symbols),
        Decimal("0"),
    )
    return (distance + abs(left_cash - right_cash)) / Decimal("2")


def _constraints_hold(
    weights: Mapping[str, Decimal],
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    limits: Mapping[str, Decimal],
    quantum: Decimal,
) -> bool:
    industry: dict[str, Decimal] = {}
    themes: dict[str, Decimal] = {}
    participation = Decimal(policy["max_adv_participation"])
    for company, weight in weights.items():
        row = metadata[company]
        # SECURITY -> LIQUIDITY -> INDUSTRY -> THEME.
        if weight > limits["security"]:
            return False
        liquidity_cap = quantum_floor(row["adv_weight_capacity"] * participation, quantum)
        if weight > liquidity_cap:
            return False
        group = row["industry_code"]
        industry[group] = industry.get(group, Decimal("0")) + weight
        if industry[group] > limits["industry"]:
            return False
        for theme in row["theme_codes"]:
            themes[theme] = themes.get(theme, Decimal("0")) + weight
            if themes[theme] > limits["theme"]:
                return False
    # GROSS -> CASH.
    gross = sum(weights.values(), Decimal("0"))
    return gross <= limits["gross"] and Decimal("1") - gross >= limits["cash_floor"]


def _reduce_positions(
    current: Mapping[str, Decimal],
    desired: Mapping[str, Decimal],
    *,
    turnover_cap: Decimal,
    quantum: Decimal,
) -> dict[str, Decimal]:
    result = dict(current)
    reduction_order = sorted(
        set(current),
        key=lambda company: (company in desired, company.encode("ascii")),
    )
    # Reduction-only steps cannot worsen any upper-bound constraint. Non-admitted
    # current holdings are cleared first and are never candidates for an increase.
    for company in reduction_order:
        floor = desired.get(company, Decimal("0"))
        while result.get(company, Decimal("0")) > floor:
            candidate = dict(result)
            next_weight = candidate[company] - quantum
            if next_weight > 0:
                candidate[company] = next_weight
            else:
                candidate.pop(company, None)
            if _capital_tv(current, candidate) > turnover_cap:
                break
            result = candidate
    return result


def _increase_positions(
    result: Mapping[str, Decimal],
    *,
    current: Mapping[str, Decimal],
    desired: Mapping[str, Decimal],
    ordering: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    limits: Mapping[str, Decimal],
    turnover_cap: Decimal,
    quantum: Decimal,
) -> dict[str, Decimal]:
    increased = dict(result)
    # Every increase is checked in the locked order, then against TURNOVER last.
    while True:
        progressed = False
        for company in ordering:
            if increased.get(company, Decimal("0")) >= desired.get(company, Decimal("0")):
                continue
            candidate = dict(increased)
            candidate[company] = candidate.get(company, Decimal("0")) + quantum
            if not _constraints_hold(
                candidate,
                metadata=metadata,
                policy=policy,
                limits=limits,
                quantum=quantum,
            ):
                continue
            if _capital_tv(current, candidate) > turnover_cap:
                continue
            increased = candidate
            progressed = True
        if not progressed:
            break
    return increased


def _transition_with_turnover(
    current: Mapping[str, Decimal],
    desired: Mapping[str, Decimal],
    *,
    ordering: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    limits: Mapping[str, Decimal],
    turnover_cap: Decimal,
    quantum: Decimal,
) -> tuple[dict[str, Decimal], Decimal]:
    reduced = _reduce_positions(
        current,
        desired,
        turnover_cap=turnover_cap,
        quantum=quantum,
    )
    result = _increase_positions(
        reduced,
        current=current,
        desired=desired,
        ordering=ordering,
        metadata=metadata,
        policy=policy,
        limits=limits,
        turnover_cap=turnover_cap,
        quantum=quantum,
    )
    return result, _capital_tv(current, result)


def _portfolio(  # noqa: C901 - frozen constraint sequence is kept locally auditable
    subjects: Sequence[Mapping[str, Any]],
    *,
    current_positions: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    limits: Mapping[str, Decimal],
    advisory: bool,
) -> dict[str, Any]:
    quantum = Decimal(policy["weight_quantum"])
    for row in current_positions:
        require_quantum_multiple(
            row["current_weight"], quantum, label=f"{row['company_code']}.current_weight"
        )
    if sum((row["current_weight"] for row in current_positions), Decimal("0")) > Decimal("1"):
        raise PortfolioContractError("current portfolio gross exceeds total capital")
    ordered = _ordered(subjects, advisory=advisory)
    selected = ordered[: policy["target_positions"]]
    blockers: list[str] = []
    if len(selected) != policy["target_positions"]:
        blockers.append("INSUFFICIENT_ADMITTED_SUBJECTS")
    constrained = _allocate_round_robin(
        selected,
        policy=policy,
        limits=limits,
        quantum=quantum,
    )
    current = {
        row["company_code"]: row["current_weight"]
        for row in current_positions
        if row["current_weight"] > 0
    }
    metadata = {row["company_code"]: row for row in current_positions}
    metadata.update({row["company_code"]: row for row in selected})
    final, turnover = _transition_with_turnover(
        current,
        constrained,
        ordering=[row["company_code"] for row in selected],
        metadata=metadata,
        policy=policy,
        limits=limits,
        turnover_cap=Decimal(policy["turnover_cap"]),
        quantum=quantum,
    )
    gross = sum(final.values(), Decimal("0"))
    cash = Decimal("1") - gross
    if sum(constrained.values(), Decimal("0")) != limits["gross"]:
        blockers.append("INFEASIBLE_CASH")
    if turnover > Decimal(policy["turnover_cap"]):
        blockers.append("TURNOVER_CAP_EXCEEDED")
    if final != constrained:
        blockers.append("TURNOVER_CONSTRAINED")
    if cash < limits["cash_floor"]:
        blockers.append("CASH_FLOOR_BREACHED")
    rows = []
    for company in sorted(final, key=lambda item: item.encode("ascii")):
        subject = metadata[company]
        liquidity_cap = quantum_floor(
            subject["adv_weight_capacity"] * Decimal(policy["max_adv_participation"]),
            quantum,
        )
        rows.append(
            target_row(
                company=company,
                current=current.get(company, Decimal("0")),
                final=final[company],
                industry=subject["industry_code"],
                themes=subject["theme_codes"],
                liquidity_cap=liquidity_cap,
            )
        )
    industry_totals: dict[str, Decimal] = {}
    theme_totals: dict[str, Decimal] = {}
    for row in rows:
        weight = Decimal(row["final_weight"])
        if weight > limits["security"]:
            blockers.append("SECURITY_CAP_BREACHED")
        if weight > Decimal(row["liquidity_cap"]):
            blockers.append("LIQUIDITY_CAP_BREACHED")
        industry_totals[row["industry_code"]] = (
            industry_totals.get(row["industry_code"], Decimal("0")) + weight
        )
        for theme in row["theme_codes"]:
            theme_totals[theme] = theme_totals.get(theme, Decimal("0")) + weight
    if any(value > limits["industry"] for value in industry_totals.values()):
        blockers.append("INDUSTRY_CAP_BREACHED")
    if any(value > limits["theme"] for value in theme_totals.values()):
        blockers.append("THEME_CAP_BREACHED")
    if gross > limits["gross"]:
        blockers.append("GROSS_CAP_BREACHED")
    return {
        "blocker_codes": sorted(set(blockers), key=lambda item: item.encode("ascii")),
        "cash_weight": decimal_text(cash),
        "gross_weight": decimal_text(gross),
        "ordering": [row["company_code"] for row in ordered],
        "status": "BLOCKED" if blockers else "AVAILABLE",
        "targets": rows,
    }


def build_portfolio_construction(
    *,
    subjects: Sequence[Mapping[str, Any]],
    current_positions: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    macro_regime: str,
    macro_ref: Mapping[str, Any],
    as_of: str,
) -> dict[str, Any]:
    validated_policy = validate_portfolio_risk_policy(policy)
    issued_at = validated_policy["timestamp"]
    if issued_at != as_of:
        raise PortfolioContractError("portfolio policy must share the construction cutoff")
    macro_reference = exact_source_ref(macro_ref, label="macro_ref", as_of=issued_at)
    replayed, admission_blockers = _replay_decisions(subjects, as_of=issued_at)
    current = _current_positions(current_positions, as_of=issued_at)
    limits, vetoes = _effective_limits(validated_policy, macro_regime)
    eligible = _eligible(replayed, policy=validated_policy, limits=limits, vetoes=vetoes)
    p_d: dict[str, Any]
    p_a: dict[str, Any]
    final: dict[str, Any]
    if admission_blockers:
        blocked = {
            "blocker_codes": admission_blockers,
            "cash_weight": decimal_text(Decimal("1")),
            "gross_weight": decimal_text(Decimal("0")),
            "ordering": [],
            "status": "BLOCKED",
            "targets": [],
        }
        p_d = blocked
        p_a = blocked
        advisory_tv = Decimal("0")
        fallback = True
        final = blocked
    else:
        p_d = _portfolio(
            eligible,
            current_positions=current,
            policy=validated_policy,
            limits=limits,
            advisory=False,
        )
        p_a = _portfolio(
            eligible,
            current_positions=current,
            policy=validated_policy,
            limits=limits,
            advisory=True,
        )
        p_d_weights = {row["company_code"]: Decimal(row["final_weight"]) for row in p_d["targets"]}
        p_a_weights = {row["company_code"]: Decimal(row["final_weight"]) for row in p_a["targets"]}
        advisory_tv = _capital_tv(p_d_weights, p_a_weights)
        fallback = p_a["status"] != "AVAILABLE" or advisory_tv > ADVISORY_TV_CEILING
        final = p_d if fallback else p_a
    blockers = sorted(
        set(admission_blockers) | set(final["blocker_codes"]),
        key=lambda item: item.encode("ascii"),
    )
    decision_refs = sorted(
        (row["decision_ref"] for row in replayed),
        key=lambda row: (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        ),
    )
    current_rows = [
        {
            **{
                key: row[key]
                for key in CURRENT_POSITION_FIELDS
                if key not in {"adv_weight_capacity", "current_weight"}
            },
            "adv_weight_capacity": decimal_text(row["adv_weight_capacity"]),
            "current_weight": decimal_text(row["current_weight"]),
        }
        for row in sorted(current, key=lambda value: value["company_code"].encode("ascii"))
    ]
    return seal(
        {
            **portfolio_common(at=issued_at),
            "admitted_decision_refs": decision_refs,
            "advisory_capital_tv": decimal_text(advisory_tv),
            "advisory_fallback": fallback,
            "blocker_codes": blockers,
            "current_position_rows": current_rows,
            "final_portfolio": final,
            "macro_ref": macro_reference,
            "p_a": p_a,
            "p_d": p_d,
            "policy_ref": content_ref(validated_policy, identity_field="policy_id"),
            "status": "BLOCKED" if blockers else "AVAILABLE",
            "version": PORTFOLIO_CONSTRUCTION_VERSION,
        },
        identity_field="receipt_id",
    )


def validate_portfolio_construction(
    document: Mapping[str, Any],
    **closure: Any,
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="receipt_id")
    require_exact_keys(normalized, CONSTRUCTION_FIELDS, label="portfolio construction receipt")
    require_exact_keys(normalized["p_d"], SUBPORTFOLIO_FIELDS, label="P_D")
    require_exact_keys(normalized["p_a"], SUBPORTFOLIO_FIELDS, label="P_A")
    require_exact_keys(normalized["final_portfolio"], SUBPORTFOLIO_FIELDS, label="final portfolio")
    expected = build_portfolio_construction(**closure)
    if normalized != expected or normalized["version"] != PORTFOLIO_CONSTRUCTION_VERSION:
        raise PortfolioContractError("portfolio construction replay mismatch")
    return normalized


__all__ = [
    "ADVISORY_TV_CEILING",
    "CONSTRUCTION_FIELDS",
    "PORTFOLIO_CONSTRUCTION_VERSION",
    "build_portfolio_construction",
    "validate_portfolio_construction",
]
