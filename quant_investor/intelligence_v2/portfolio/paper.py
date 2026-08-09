"""Effective-dated, source-bound A-share research-paper execution contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from typing import Any, Final

from .contracts import (
    PortfolioContractError,
    company_code,
    content_ref,
    decimal_in_unit,
    decimal_text,
    decimal_value,
    exact_source_ref,
    identifier,
    portfolio_common,
    positive_decimal,
    require_exact_keys,
    seal,
    timestamp,
    validate_content_ref,
    validate_seal,
)

PAPER_EXECUTION_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.paper-execution-policy.v2"
PAPER_ORDER_VERSION: Final = "myquant.v17.intelligence-v2.paper-order.v2"
PAPER_FILL_VERSION: Final = "myquant.v17.intelligence-v2.paper-fill-receipt.v2"
PAPER_LEDGER_VERSION: Final = "myquant.v17.intelligence-v2.paper-ledger.v2"
PAPER_OUTCOME_VERSION: Final = "myquant.v17.intelligence-v2.paper-outcome.v2"

PRICE_LIMIT_RULE_FIELDS: Final = {
    "board",
    "effective_from_session",
    "effective_through_session",
    "ipo_no_limit_sessions",
    "limit_ratio",
    "rule_id",
    "source_ref",
    "st",
}
CALENDAR_SESSION_FIELDS: Final = {"session", "source_ref", "status"}
POLICY_FIELDS: Final = {
    "allow_odd_lot_full_exit",
    "allow_partial_fills",
    "authority",
    "buy_commission_rate",
    "corporate_action_policy",
    "decision_protocol",
    "effective_from_session",
    "effective_through_session",
    "exchange_calendar_ref",
    "exchange_calendar_sessions",
    "fee_rounding_mode",
    "fee_rounding_quantum_cny",
    "listing_policy",
    "lot_size",
    "max_fill_adv_participation",
    "minimum_commission_cny",
    "order_expiry_rule",
    "partial_fill_ordering",
    "policy_id",
    "price_limit_rules",
    "price_rounding_quantum_cny",
    "production",
    "research_only",
    "sell_commission_rate",
    "sell_stamp_duty_rate",
    "semantic_sha256",
    "settlement_rule",
    "slippage_rate",
    "timestamp",
    "transfer_fee_rate",
    "version",
}
ORDER_FIELDS: Final = {
    "authority",
    "blocker_codes",
    "cancellation_ref",
    "company_code",
    "decision_protocol",
    "decision_session",
    "execution_session",
    "expires_session",
    "market_ref",
    "order_id",
    "policy_ref",
    "position_shares",
    "production",
    "queue_priority",
    "requested_shares",
    "research_only",
    "semantic_sha256",
    "side",
    "simulation_only",
    "status",
    "timestamp",
    "version",
}
MARKET_OBSERVATION_FIELDS: Final = {
    "available_volume_shares",
    "board",
    "company_code",
    "corporate_action_refs",
    "delisting_session",
    "execution_price",
    "is_st",
    "listing_session",
    "lower_limit",
    "previous_close",
    "session",
    "sessions_since_listing",
    "source_ref",
    "suspended",
    "upper_limit",
}
FILL_FIELDS: Final = {
    "applied_price_limit_rule_id",
    "authority",
    "blocker_codes",
    "commission_cny",
    "company_code",
    "corporate_action_refs",
    "decision_protocol",
    "execution_session",
    "fill_id",
    "fill_price",
    "filled_shares",
    "gross_value_cny",
    "market_ref",
    "order_ref",
    "production",
    "queue_priority",
    "research_only",
    "semantic_sha256",
    "side",
    "simulation_only",
    "stamp_duty_cny",
    "status",
    "timestamp",
    "total_cost_cny",
    "transfer_fee_cny",
    "version",
}
LEDGER_FIELDS: Final = {
    "authority",
    "closing_cash_cny",
    "decision_protocol",
    "fill_refs",
    "ledger_id",
    "opening_cash_cny",
    "positions",
    "production",
    "research_only",
    "semantic_sha256",
    "simulation_only",
    "timestamp",
    "version",
}
OUTCOME_FIELDS: Final = {
    "authority",
    "benchmark_ref",
    "benchmark_return",
    "cost_adjusted_excess_return",
    "cost_ratio",
    "decision_protocol",
    "entry_price_ref",
    "excess_return",
    "hard_risk_breach",
    "horizon_sessions",
    "ledger_ref",
    "maximum_drawdown",
    "observed_return",
    "outcome_id",
    "outcome_price_ref",
    "production",
    "regime_ref",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "turnover",
    "version",
}


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise PortfolioContractError(f"{label} must be YYYYMMDD")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise PortfolioContractError(f"{label} must be YYYYMMDD") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise PortfolioContractError(f"{label} must be YYYYMMDD")
    return value


def _round_money(value: Decimal, quantum: Decimal = Decimal("0.0001")) -> str:
    with localcontext() as context:
        context.prec = 50
        normalized_quantum = quantum.normalize()
        return format(value.quantize(normalized_quantum, rounding=ROUND_HALF_EVEN), "f")


def _price_rules(values: Sequence[Mapping[str, Any]], *, as_of: str) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise PortfolioContractError("price_limit_rules must be a nonempty sequence")
    rows = []
    for index, source in enumerate(values):
        row = require_exact_keys(
            source, PRICE_LIMIT_RULE_FIELDS, label=f"price_limit_rules[{index}]"
        )
        board = identifier(row["board"], label=f"price_limit_rules[{index}].board")
        if board not in {"BSE", "CHINEXT", "MAIN", "STAR"} or type(row["st"]) is not bool:
            raise PortfolioContractError("price-limit classification is invalid")
        starts = _session(row["effective_from_session"], label="rule.effective_from_session")
        ends = _session(row["effective_through_session"], label="rule.effective_through_session")
        if starts > ends:
            raise PortfolioContractError("price-limit effective range is invalid")
        window = row["ipo_no_limit_sessions"]
        if type(window) is not int or not 0 <= window <= 20:
            raise PortfolioContractError("IPO no-limit window is invalid")
        rows.append(
            {
                "board": board,
                "effective_from_session": starts,
                "effective_through_session": ends,
                "ipo_no_limit_sessions": window,
                "limit_ratio": decimal_text(
                    positive_decimal(
                        row["limit_ratio"], label="rule.limit_ratio", maximum=Decimal("1")
                    )
                ),
                "rule_id": identifier(row["rule_id"], label="rule.rule_id"),
                "source_ref": exact_source_ref(
                    row["source_ref"], label="rule.source_ref", as_of=as_of
                ),
                "st": row["st"],
            }
        )
    keys = [
        (row["board"], row["st"], row["effective_from_session"], row["rule_id"]) for row in rows
    ]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise PortfolioContractError("price_limit_rules must be sorted and unique")
    return rows


def _calendar_sessions(values: Sequence[Mapping[str, Any]], *, as_of: str) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise PortfolioContractError("exchange_calendar_sessions must be nonempty")
    rows = []
    for index, source in enumerate(values):
        row = require_exact_keys(
            source, CALENDAR_SESSION_FIELDS, label=f"calendar_sessions[{index}]"
        )
        if row["status"] not in {"CLOSED", "OPEN"}:
            raise PortfolioContractError("calendar session status is invalid")
        rows.append(
            {
                "session": _session(row["session"], label="calendar.session"),
                "source_ref": exact_source_ref(
                    row["source_ref"], label="calendar.source_ref", as_of=as_of
                ),
                "status": row["status"],
            }
        )
    sessions = [row["session"] for row in rows]
    if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise PortfolioContractError("calendar sessions must be sorted and unique")
    return rows


def build_paper_execution_policy(
    *,
    created_at: str,
    effective_from_session: str,
    effective_through_session: str,
    lot_size: int,
    settlement_rule: str,
    buy_commission_rate: Any,
    sell_commission_rate: Any,
    minimum_commission_cny: Any,
    transfer_fee_rate: Any,
    sell_stamp_duty_rate: Any,
    slippage_rate: Any,
    max_fill_adv_participation: Any,
    fee_rounding_quantum_cny: Any,
    fee_rounding_mode: str,
    price_rounding_quantum_cny: Any,
    allow_partial_fills: bool,
    allow_odd_lot_full_exit: bool,
    order_expiry_rule: str,
    partial_fill_ordering: str,
    corporate_action_policy: str,
    listing_policy: str,
    exchange_calendar_ref: Mapping[str, Any],
    exchange_calendar_sessions: Sequence[Mapping[str, Any]],
    price_limit_rules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    starts = _session(effective_from_session, label="effective_from_session")
    ends = _session(effective_through_session, label="effective_through_session")
    if starts > ends or type(lot_size) is not int or lot_size != 100:
        raise PortfolioContractError("paper policy effective range or board lot is invalid")
    expected_enums = {
        "settlement_rule": (settlement_rule, "T_PLUS_ONE"),
        "fee_rounding_mode": (fee_rounding_mode, "ROUND_HALF_EVEN"),
        "order_expiry_rule": (order_expiry_rule, "EXPLICIT_SESSION"),
        "partial_fill_ordering": (partial_fill_ordering, "QUEUE_PRIORITY_THEN_ORDER_ID"),
        "corporate_action_policy": (corporate_action_policy, "EXACT_SOURCE_CHRONOLOGY"),
        "listing_policy": (listing_policy, "LISTED_NOT_DELISTED"),
    }
    if any(actual != expected for actual, expected in expected_enums.values()):
        raise PortfolioContractError("paper execution policy enum is invalid")
    if type(allow_partial_fills) is not bool or type(allow_odd_lot_full_exit) is not bool:
        raise PortfolioContractError("paper execution flags must be booleans")
    fee_quantum = positive_decimal(fee_rounding_quantum_cny, label="fee_rounding_quantum_cny")
    price_quantum = positive_decimal(price_rounding_quantum_cny, label="price_rounding_quantum_cny")
    calendar = exact_source_ref(
        exchange_calendar_ref, label="exchange_calendar_ref", as_of=issued_at
    )
    calendar_rows = _calendar_sessions(exchange_calendar_sessions, as_of=issued_at)
    rules = _price_rules(price_limit_rules, as_of=issued_at)
    return seal(
        {
            **portfolio_common(at=issued_at),
            "allow_odd_lot_full_exit": allow_odd_lot_full_exit,
            "allow_partial_fills": allow_partial_fills,
            "buy_commission_rate": decimal_text(
                decimal_in_unit(buy_commission_rate, label="buy_commission_rate")
            ),
            "corporate_action_policy": corporate_action_policy,
            "effective_from_session": starts,
            "effective_through_session": ends,
            "exchange_calendar_ref": calendar,
            "exchange_calendar_sessions": calendar_rows,
            "fee_rounding_mode": fee_rounding_mode,
            "fee_rounding_quantum_cny": decimal_text(fee_quantum),
            "listing_policy": listing_policy,
            "lot_size": lot_size,
            "max_fill_adv_participation": decimal_text(
                positive_decimal(
                    max_fill_adv_participation,
                    label="max_fill_adv_participation",
                    maximum=Decimal("0.10"),
                )
            ),
            "minimum_commission_cny": decimal_text(
                decimal_value(
                    minimum_commission_cny,
                    label="minimum_commission_cny",
                    minimum=Decimal("0"),
                )
            ),
            "order_expiry_rule": order_expiry_rule,
            "partial_fill_ordering": partial_fill_ordering,
            "price_limit_rules": rules,
            "price_rounding_quantum_cny": decimal_text(price_quantum),
            "sell_commission_rate": decimal_text(
                decimal_in_unit(sell_commission_rate, label="sell_commission_rate")
            ),
            "sell_stamp_duty_rate": decimal_text(
                decimal_in_unit(sell_stamp_duty_rate, label="sell_stamp_duty_rate")
            ),
            "settlement_rule": settlement_rule,
            "slippage_rate": decimal_text(decimal_in_unit(slippage_rate, label="slippage_rate")),
            "transfer_fee_rate": decimal_text(
                decimal_in_unit(transfer_fee_rate, label="transfer_fee_rate")
            ),
            "version": PAPER_EXECUTION_POLICY_VERSION,
        },
        identity_field="policy_id",
    )


def validate_paper_execution_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_seal(document, identity_field="policy_id")
    require_exact_keys(row, POLICY_FIELDS, label="paper execution policy")
    expected = build_paper_execution_policy(
        created_at=row["timestamp"],
        effective_from_session=row["effective_from_session"],
        effective_through_session=row["effective_through_session"],
        lot_size=row["lot_size"],
        settlement_rule=row["settlement_rule"],
        buy_commission_rate=row["buy_commission_rate"],
        sell_commission_rate=row["sell_commission_rate"],
        minimum_commission_cny=row["minimum_commission_cny"],
        transfer_fee_rate=row["transfer_fee_rate"],
        sell_stamp_duty_rate=row["sell_stamp_duty_rate"],
        slippage_rate=row["slippage_rate"],
        max_fill_adv_participation=row["max_fill_adv_participation"],
        fee_rounding_quantum_cny=row["fee_rounding_quantum_cny"],
        fee_rounding_mode=row["fee_rounding_mode"],
        price_rounding_quantum_cny=row["price_rounding_quantum_cny"],
        allow_partial_fills=row["allow_partial_fills"],
        allow_odd_lot_full_exit=row["allow_odd_lot_full_exit"],
        order_expiry_rule=row["order_expiry_rule"],
        partial_fill_ordering=row["partial_fill_ordering"],
        corporate_action_policy=row["corporate_action_policy"],
        listing_policy=row["listing_policy"],
        exchange_calendar_ref=row["exchange_calendar_ref"],
        exchange_calendar_sessions=row["exchange_calendar_sessions"],
        price_limit_rules=row["price_limit_rules"],
    )
    if row != expected or row["version"] != PAPER_EXECUTION_POLICY_VERSION:
        raise PortfolioContractError("paper execution policy replay mismatch")
    return row


def build_paper_order(  # noqa: C901
    *,
    policy: Mapping[str, Any] | None,
    company: str,
    side: str,
    requested_shares: int,
    position_shares: int,
    acquired_session: str | None,
    decision_session: str,
    execution_session: str,
    expires_session: str,
    queue_priority: int,
    cancellation_ref: Mapping[str, Any] | None,
    market_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    decision_date = _session(decision_session, label="decision_session")
    execution_date = _session(execution_session, label="execution_session")
    expires = _session(expires_session, label="expires_session")
    if execution_date < decision_date or expires < execution_date:
        raise PortfolioContractError("paper order session chronology is invalid")
    if side not in {"PAPER_BUY", "PAPER_SELL"}:
        raise PortfolioContractError("paper side is invalid")
    if type(requested_shares) is not int or requested_shares <= 0:
        raise PortfolioContractError("requested_shares must be positive int")
    if type(position_shares) is not int or position_shares < 0:
        raise PortfolioContractError("position_shares must be nonnegative int")
    if type(queue_priority) is not int or queue_priority < 0:
        raise PortfolioContractError("queue_priority must be a nonnegative int")
    cancellation = None
    blockers: list[str] = []
    if cancellation_ref is not None:
        cancellation = exact_source_ref(cancellation_ref, label="cancellation_ref", as_of=issued_at)
        blockers.append("ORDER_CANCELLED")
    policy_ref = None
    if policy is None:
        blockers.append("PAPER_EXECUTION_POLICY_UNAVAILABLE")
    else:
        policy_row = validate_paper_execution_policy(policy)
        policy_ref = content_ref(policy_row, identity_field="policy_id")
        if (
            not policy_row["effective_from_session"]
            <= execution_date
            <= policy_row["effective_through_session"]
        ):
            blockers.append("PAPER_EXECUTION_POLICY_NOT_EFFECTIVE")
        calendar = {
            calendar_row["session"]: calendar_row["status"]
            for calendar_row in policy_row["exchange_calendar_sessions"]
        }
        open_sessions = [value for value, status in sorted(calendar.items()) if status == "OPEN"]
        required_sessions = {decision_date, execution_date, expires}
        if not required_sessions.issubset(calendar):
            blockers.append("CALENDAR_SESSION_COVERAGE_UNAVAILABLE")
        elif any(calendar[value] != "OPEN" for value in required_sessions):
            blockers.append("ORDER_SESSION_NOT_OPEN")
        lot = policy_row["lot_size"]
        if side == "PAPER_BUY" and requested_shares % lot:
            blockers.append("BUY_LOT_INVALID")
        if side == "PAPER_SELL":
            if requested_shares > position_shares:
                blockers.append("SELL_EXCEEDS_POSITION")
            odd_exit = policy_row["allow_odd_lot_full_exit"] and requested_shares == position_shares
            if requested_shares % lot and not odd_exit:
                blockers.append("SELL_LOT_INVALID")
            if acquired_session is None:
                blockers.append("ACQUISITION_SESSION_UNAVAILABLE")
            else:
                acquired = _session(acquired_session, label="acquired_session")
                if acquired not in calendar:
                    blockers.append("CALENDAR_SESSION_COVERAGE_UNAVAILABLE")
                elif calendar[acquired] != "OPEN":
                    blockers.append("ACQUISITION_SESSION_NOT_OPEN")
                elif execution_date in open_sessions and (
                    open_sessions.index(acquired) >= open_sessions.index(execution_date)
                ):
                    blockers.append("T_PLUS_ONE_BLOCKED")
    return seal(
        {
            **portfolio_common(at=issued_at),
            "blocker_codes": sorted(set(blockers), key=lambda value: value.encode("ascii")),
            "cancellation_ref": cancellation,
            "company_code": company_code(company, label="company"),
            "decision_session": decision_date,
            "execution_session": execution_date,
            "expires_session": expires,
            "market_ref": exact_source_ref(market_ref, label="market_ref", as_of=issued_at),
            "policy_ref": policy_ref,
            "position_shares": position_shares,
            "queue_priority": queue_priority,
            "requested_shares": requested_shares,
            "side": side,
            "simulation_only": True,
            "status": "BLOCKED" if blockers else "READY",
            "version": PAPER_ORDER_VERSION,
        },
        identity_field="order_id",
    )


def validate_paper_order(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="order_id")
    require_exact_keys(row, ORDER_FIELDS, label="paper order")
    expected = build_paper_order(**closure)
    if row != expected or row["version"] != PAPER_ORDER_VERSION:
        raise PortfolioContractError("paper order replay mismatch")
    return row


def _sorted_exact_refs(
    values: Sequence[Mapping[str, Any]], *, label: str, as_of: str
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PortfolioContractError(f"{label} must be a sequence")
    rows = [
        exact_source_ref(value, label=f"{label}[{index}]", as_of=as_of)
        for index, value in enumerate(values)
    ]
    keys = [
        (row["artifact_id"], row["artifact_version"], row["byte_sha256"], row["semantic_sha256"])
        for row in rows
    ]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise PortfolioContractError(f"{label} must be sorted and unique")
    return rows


def _market_observation(
    value: Mapping[str, Any], *, company: str, session: str, as_of: str
) -> dict[str, Any]:
    row = require_exact_keys(value, MARKET_OBSERVATION_FIELDS, label="market_observation")
    if company_code(row["company_code"], label="market.company_code") != company:
        raise PortfolioContractError("market observation company mismatch")
    if _session(row["session"], label="market.session") != session:
        raise PortfolioContractError("market observation session mismatch")
    if type(row["suspended"]) is not bool or type(row["is_st"]) is not bool:
        raise PortfolioContractError("market classification flags must be booleans")
    if type(row["available_volume_shares"]) is not int or row["available_volume_shares"] < 0:
        raise PortfolioContractError("available volume must be nonnegative int")
    age = row["sessions_since_listing"]
    if type(age) is not int or age < 0:
        raise PortfolioContractError("sessions_since_listing must be nonnegative int")
    listing = _session(row["listing_session"], label="listing_session")
    delisting = (
        None
        if row["delisting_session"] is None
        else _session(row["delisting_session"], label="delisting_session")
    )
    if listing > session or (delisting is not None and delisting < listing):
        raise PortfolioContractError("listing chronology is invalid")
    limits = {}
    for name in ("lower_limit", "upper_limit"):
        limits[name] = (
            None if row[name] is None else positive_decimal(row[name], label=f"market.{name}")
        )
    return {
        "available_volume_shares": row["available_volume_shares"],
        "board": identifier(row["board"], label="market.board"),
        "company_code": company,
        "corporate_action_refs": _sorted_exact_refs(
            row["corporate_action_refs"], label="corporate_action_refs", as_of=as_of
        ),
        "delisting_session": delisting,
        "execution_price": positive_decimal(row["execution_price"], label="market.execution_price"),
        "is_st": row["is_st"],
        "listing_session": listing,
        **limits,
        "previous_close": positive_decimal(row["previous_close"], label="market.previous_close"),
        "session": session,
        "sessions_since_listing": age,
        "source_ref": exact_source_ref(row["source_ref"], label="market.source_ref", as_of=as_of),
        "suspended": row["suspended"],
    }


def _applicable_rule(policy: Mapping[str, Any], market: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        rule
        for rule in policy["price_limit_rules"]
        if rule["board"] == market["board"]
        and rule["st"] is market["is_st"]
        and rule["effective_from_session"] <= market["session"] <= rule["effective_through_session"]
    ]
    if len(matches) != 1:
        raise PortfolioContractError("market has no unique effective price-limit rule")
    return dict(matches[0])


def _price_blockers(
    *, order: Mapping[str, Any], market: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[list[str], str]:
    rule = _applicable_rule(policy, market)
    if market["sessions_since_listing"] < rule["ipo_no_limit_sessions"]:
        if market["lower_limit"] is not None or market["upper_limit"] is not None:
            raise PortfolioContractError("IPO no-limit window must not provide daily limits")
        return [], rule["rule_id"]
    quantum = Decimal(policy["price_rounding_quantum_cny"])
    ratio = Decimal(rule["limit_ratio"])
    expected_lower = Decimal(
        _round_money(market["previous_close"] * (Decimal("1") - ratio), quantum)
    )
    expected_upper = Decimal(
        _round_money(market["previous_close"] * (Decimal("1") + ratio), quantum)
    )
    if market["lower_limit"] != expected_lower or market["upper_limit"] != expected_upper:
        raise PortfolioContractError("market daily limits do not match the effective rule")
    blockers = []
    if order["side"] == "PAPER_BUY" and market["execution_price"] >= expected_upper:
        blockers.append("LIMIT_UP_BUY_BLOCKED")
    if order["side"] == "PAPER_SELL" and market["execution_price"] <= expected_lower:
        blockers.append("LIMIT_DOWN_SELL_BLOCKED")
    return blockers, rule["rule_id"]


def _fill_quantity(
    *, order: Mapping[str, Any], market: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[int, str, list[str]]:
    lot = policy["lot_size"]
    requested = order["requested_shares"]
    capacity_raw = Decimal(market["available_volume_shares"]) * Decimal(
        policy["max_fill_adv_participation"]
    )
    capacity = int((capacity_raw / Decimal(lot)).to_integral_value(rounding=ROUND_FLOOR)) * lot
    odd_exit = (
        order["side"] == "PAPER_SELL"
        and policy["allow_odd_lot_full_exit"]
        and requested == order["position_shares"]
        and requested % lot != 0
    )
    if odd_exit:
        capacity = requested if market["available_volume_shares"] >= requested else 0
    filled = min(requested, capacity)
    if filled == 0:
        return 0, "BLOCKED", ["NO_SIMULATED_LIQUIDITY"]
    if filled < requested and not policy["allow_partial_fills"]:
        return 0, "BLOCKED", ["PARTIAL_FILL_FORBIDDEN"]
    return filled, "FILLED" if filled == requested else "PARTIALLY_FILLED", []


def build_paper_fill(  # noqa: C901
    *,
    order: Mapping[str, Any],
    order_validation_closure: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    market_observation: Mapping[str, Any],
    filled_at: str,
) -> dict[str, Any]:
    if type(order_validation_closure) is not dict:
        raise PortfolioContractError("order_validation_closure must be exact")
    order_row = validate_paper_order(order, **dict(order_validation_closure))
    issued_at = timestamp(filled_at, label="filled_at")
    market = _market_observation(
        market_observation,
        company=order_row["company_code"],
        session=order_row["execution_session"],
        as_of=issued_at,
    )
    blockers = list(order_row["blocker_codes"])
    policy_row = None
    rule_id = None
    if policy is None:
        blockers.append("PAPER_EXECUTION_POLICY_UNAVAILABLE")
    else:
        policy_row = validate_paper_execution_policy(policy)
        if order_row["policy_ref"] != content_ref(policy_row, identity_field="policy_id"):
            raise PortfolioContractError("paper order policy ref mismatch")
        if market["suspended"]:
            blockers.append("SUSPENDED")
        if market["corporate_action_refs"]:
            blockers.append("CORPORATE_ACTION_SIMULATION_UNAVAILABLE")
        if (
            market["delisting_session"] is not None
            and market["session"] >= market["delisting_session"]
        ):
            blockers.append("DELISTED")
        price_blockers, rule_id = _price_blockers(order=order_row, market=market, policy=policy_row)
        blockers.extend(price_blockers)
    filled = 0
    status = "BLOCKED"
    if not blockers and policy_row is not None:
        filled, status, quantity_blockers = _fill_quantity(
            order=order_row, market=market, policy=policy_row
        )
        blockers.extend(quantity_blockers)
    fill_price = gross = commission = transfer = stamp = Decimal("0")
    if filled and policy_row is not None:
        multiplier = Decimal("1") + Decimal(policy_row["slippage_rate"])
        if order_row["side"] == "PAPER_SELL":
            multiplier = Decimal("1") - Decimal(policy_row["slippage_rate"])
        price_quantum = Decimal(policy_row["price_rounding_quantum_cny"])
        fill_price = Decimal(_round_money(market["execution_price"] * multiplier, price_quantum))
        gross = fill_price * Decimal(filled)
        fee_quantum = Decimal(policy_row["fee_rounding_quantum_cny"])
        rate_name = (
            "buy_commission_rate" if order_row["side"] == "PAPER_BUY" else "sell_commission_rate"
        )
        commission = max(
            Decimal(policy_row["minimum_commission_cny"]),
            Decimal(_round_money(gross * Decimal(policy_row[rate_name]), fee_quantum)),
        )
        transfer = Decimal(
            _round_money(gross * Decimal(policy_row["transfer_fee_rate"]), fee_quantum)
        )
        if order_row["side"] == "PAPER_SELL":
            stamp = Decimal(
                _round_money(gross * Decimal(policy_row["sell_stamp_duty_rate"]), fee_quantum)
            )
    blocker_codes = sorted(set(blockers), key=lambda value: value.encode("ascii"))
    if blocker_codes:
        status = "BLOCKED"
        filled = 0
        fill_price = gross = commission = transfer = stamp = Decimal("0")
    total = commission + transfer + stamp
    return seal(
        {
            **portfolio_common(at=issued_at),
            "applied_price_limit_rule_id": rule_id,
            "blocker_codes": blocker_codes,
            "commission_cny": _round_money(commission),
            "company_code": order_row["company_code"],
            "corporate_action_refs": market["corporate_action_refs"],
            "execution_session": order_row["execution_session"],
            "fill_price": _round_money(fill_price),
            "filled_shares": filled,
            "gross_value_cny": _round_money(gross),
            "market_ref": market["source_ref"],
            "order_ref": content_ref(order_row, identity_field="order_id"),
            "queue_priority": order_row["queue_priority"],
            "side": order_row["side"],
            "simulation_only": True,
            "stamp_duty_cny": _round_money(stamp),
            "status": status,
            "total_cost_cny": _round_money(total),
            "transfer_fee_cny": _round_money(transfer),
            "version": PAPER_FILL_VERSION,
        },
        identity_field="fill_id",
    )


def validate_paper_fill(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="fill_id")
    require_exact_keys(row, FILL_FIELDS, label="paper fill")
    expected = build_paper_fill(**closure)
    if row != expected or row["version"] != PAPER_FILL_VERSION:
        raise PortfolioContractError("paper fill replay mismatch")
    return row


def build_paper_ledger(  # noqa: C901
    *,
    fills: Sequence[Mapping[str, Any]],
    fill_validation_closures: Sequence[Mapping[str, Any]],
    opening_cash_cny: Any,
    opening_positions: Mapping[str, int],
    created_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    if len(fills) != len(fill_validation_closures):
        raise PortfolioContractError("fill closure inventory mismatch")
    cash = decimal_value(opening_cash_cny, label="opening_cash_cny", minimum=Decimal("0"))
    positions = {}
    for code, shares in opening_positions.items():
        canonical = company_code(code, label="opening_positions.company_code")
        if type(shares) is not int or shares < 0:
            raise PortfolioContractError("opening position shares must be nonnegative int")
        if shares:
            positions[canonical] = shares
    refs = []
    prior_key: tuple[str, int, str] | None = None
    for fill, closure in zip(fills, fill_validation_closures):
        if type(closure) is not dict:
            raise PortfolioContractError("fill validation closure must be exact")
        row = validate_paper_fill(fill, **dict(closure))
        if row["timestamp"] > issued_at:
            raise PortfolioContractError("paper ledger contains future fill")
        order_key = (
            row["execution_session"],
            row["queue_priority"],
            row["order_ref"]["artifact_id"],
        )
        if prior_key is not None and order_key <= prior_key:
            raise PortfolioContractError("fills violate deterministic partial-fill ordering")
        prior_key = order_key
        refs.append(content_ref(row, identity_field="fill_id"))
        if row["status"] not in {"FILLED", "PARTIALLY_FILLED"}:
            continue
        gross = Decimal(row["gross_value_cny"])
        costs = Decimal(row["total_cost_cny"])
        code = row["company_code"]
        shares = row["filled_shares"]
        if row["side"] == "PAPER_BUY":
            cash -= gross + costs
            positions[code] = positions.get(code, 0) + shares
        else:
            if positions.get(code, 0) < shares:
                raise PortfolioContractError("paper ledger sell exceeds simulated holdings")
            cash += gross - costs
            positions[code] -= shares
            if positions[code] == 0:
                del positions[code]
        if cash < 0:
            raise PortfolioContractError("paper ledger cash becomes negative")
    return seal(
        {
            **portfolio_common(at=issued_at),
            "closing_cash_cny": _round_money(cash),
            "fill_refs": refs,
            "opening_cash_cny": _round_money(
                decimal_value(opening_cash_cny, label="opening_cash_cny")
            ),
            "positions": [
                {"company_code": code, "shares": positions[code]}
                for code in sorted(positions, key=lambda value: value.encode("ascii"))
            ],
            "simulation_only": True,
            "version": PAPER_LEDGER_VERSION,
        },
        identity_field="ledger_id",
    )


def validate_paper_ledger(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="ledger_id")
    require_exact_keys(row, LEDGER_FIELDS, label="paper ledger")
    expected = build_paper_ledger(**closure)
    if row != expected or row["version"] != PAPER_LEDGER_VERSION:
        raise PortfolioContractError("paper ledger replay mismatch")
    return row


def build_paper_outcome(
    *,
    ledger_ref: Mapping[str, Any],
    horizon_sessions: int,
    observed_return: Any,
    benchmark_return: Any,
    maximum_drawdown: Any,
    turnover: Any,
    cost_ratio: Any,
    hard_risk_breach: bool,
    benchmark_ref: Mapping[str, Any],
    entry_price_ref: Mapping[str, Any],
    outcome_price_ref: Mapping[str, Any],
    regime_ref: Mapping[str, Any],
    matured_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(matured_at, label="matured_at")
    if type(horizon_sessions) is not int or horizon_sessions not in {1, 5, 20, 60}:
        raise PortfolioContractError("paper outcome horizon must be 1, 5, 20, or 60")
    if type(hard_risk_breach) is not bool:
        raise PortfolioContractError("hard_risk_breach must be boolean")
    observed = decimal_value(observed_return, label="observed_return", minimum=Decimal("-1"))
    benchmark = decimal_value(benchmark_return, label="benchmark_return", minimum=Decimal("-1"))
    costs = decimal_in_unit(cost_ratio, label="cost_ratio")
    return seal(
        {
            **portfolio_common(at=issued_at),
            "benchmark_ref": exact_source_ref(
                benchmark_ref, label="benchmark_ref", as_of=issued_at
            ),
            "benchmark_return": decimal_text(benchmark),
            "cost_adjusted_excess_return": decimal_text(observed - benchmark - costs),
            "cost_ratio": decimal_text(costs),
            "entry_price_ref": exact_source_ref(
                entry_price_ref, label="entry_price_ref", as_of=issued_at
            ),
            "excess_return": decimal_text(observed - benchmark),
            "hard_risk_breach": hard_risk_breach,
            "horizon_sessions": horizon_sessions,
            "ledger_ref": validate_content_ref(ledger_ref, label="ledger_ref"),
            "maximum_drawdown": decimal_text(
                decimal_in_unit(maximum_drawdown, label="maximum_drawdown")
            ),
            "observed_return": decimal_text(observed),
            "outcome_price_ref": exact_source_ref(
                outcome_price_ref, label="outcome_price_ref", as_of=issued_at
            ),
            "regime_ref": exact_source_ref(regime_ref, label="regime_ref", as_of=issued_at),
            "turnover": decimal_text(decimal_in_unit(turnover, label="turnover")),
            "version": PAPER_OUTCOME_VERSION,
        },
        identity_field="outcome_id",
    )


def validate_paper_outcome(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    row = validate_seal(document, identity_field="outcome_id")
    require_exact_keys(row, OUTCOME_FIELDS, label="paper outcome")
    expected = build_paper_outcome(**closure)
    if row != expected or row["version"] != PAPER_OUTCOME_VERSION:
        raise PortfolioContractError("paper outcome replay mismatch")
    return row


__all__ = [
    "PAPER_EXECUTION_POLICY_VERSION",
    "PAPER_FILL_VERSION",
    "PAPER_LEDGER_VERSION",
    "PAPER_ORDER_VERSION",
    "PAPER_OUTCOME_VERSION",
    "build_paper_execution_policy",
    "build_paper_fill",
    "build_paper_ledger",
    "build_paper_order",
    "build_paper_outcome",
    "validate_paper_execution_policy",
    "validate_paper_fill",
    "validate_paper_ledger",
    "validate_paper_order",
    "validate_paper_outcome",
]
