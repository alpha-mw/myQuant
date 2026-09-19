"""Exact nested shapes for the registered corporate reconciliation payload."""

from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import _validate_day
from quant_investor.strategy_records.corporate_contracts import KINDS, number, ordered_refs


def fields(value, expected):
    if type(value) is not dict or set(value) != set(expected.split()):
        raise ContractError("CORPORATE_REPORT_NESTED_SHAPE_INVALID")


def codes(value, allowed):
    if (
        type(value) is not list
        or any(type(v) is not str for v in value)
        or not set(value) <= allowed
    ):
        raise ContractError("CORPORATE_REPORT_BLOCKER_INVALID")


def amounts(value, unit):
    if value is None:
        return
    fields(value, "before after delta unit")
    if value["unit"] != unit or any(
        type(value[k]) is not str for k in ("before", "after", "delta")
    ):
        raise ContractError("CORPORATE_REPORT_AMOUNT_INVALID")
    if number(value["after"]) - number(value["before"]) != number(value["delta"]):
        raise ContractError("CORPORATE_REPORT_AMOUNT_IDENTITY_INVALID")


def financial(value, allowed):
    fields(
        value,
        "state before_record_id after_record_id cost_basis_adjustment shares_adjustment "
        "cash_adjustment unattributed_transition source_refs blocker_codes",
    )
    if value["state"] not in {"OBSERVED_NATIVE_POSTING", "UNCONFIRMED"}:
        raise ContractError("CORPORATE_REPORT_FINANCIAL_STATE_INVALID")
    codes(value["blocker_codes"], allowed)
    ordered_refs(value["source_refs"])
    for key, unit in (
        ("cost_basis_adjustment", "CNY"),
        ("shares_adjustment", "SHARES"),
        ("cash_adjustment", "CNY"),
    ):
        amounts(value[key], unit)
        if value["state"] == "UNCONFIRMED" and value[key] is not None:
            raise ContractError("CORPORATE_REPORT_UNPROVEN_ATTRIBUTION")
        if value["state"] == "OBSERVED_NATIVE_POSTING" and value[key] is None:
            raise ContractError("CORPORATE_REPORT_POSTING_ADJUSTMENT_MISSING")
    aggregate = value["unattributed_transition"]
    if aggregate is not None:
        fields(aggregate, "cost_basis_adjustment shares_adjustment cash_adjustment")
        for key, unit in (
            ("cost_basis_adjustment", "CNY"),
            ("shares_adjustment", "SHARES"),
            ("cash_adjustment", "CNY"),
        ):
            amounts(aggregate[key], unit)


def anchor(value, allowed):
    fields(
        value,
        "state review_ref policy_ref source_record_id old_tracking_start_date "
        "tracking_start_date blocker_codes",
    )
    if value["state"] not in {"OWNER_DECLARED_RESEARCH_RESET", "OWNER_REVIEW_REQUIRED"}:
        raise ContractError("CORPORATE_REPORT_ANCHOR_STATE_INVALID")
    validate_ref(value["policy_ref"])
    if value["review_ref"] is not None:
        validate_ref(value["review_ref"])
    for key in ("old_tracking_start_date", "tracking_start_date"):
        if value[key] is not None:
            _validate_day(value[key])
    codes(value["blocker_codes"], allowed)


def _transitions(values):
    for transition in values:
        fields(
            transition,
            "kind previous_trade_date trade_date before_factor after_factor event_ids",
        )
        if (
            transition["kind"] != "ADJUSTMENT_FACTOR_CHANGE"
            or number(transition["before_factor"]) <= 0
            or number(transition["after_factor"]) <= 0
        ):
            raise ContractError("CORPORATE_REPORT_TRANSITION_INVALID")


def validate_company_rows(rows, allowed):
    if type(rows) is not list:
        raise ContractError("CORPORATE_REPORT_COMPANIES_INVALID")
    for row in rows:
        fields(
            row,
            "symbol tracking_start_date window_state required_dates market_ref "
            "transitions events threshold_state blocker_codes",
        )
        if (
            row["window_state"]
            not in {
                "VERIFIED",
                "ANCHOR_MISSING",
                "POLICY_BASELINE_UNCONFIRMED",
                "CALENDAR_GAP",
                "MARKET_GAP",
                "MARKET_CONFLICT",
            }
            or row["threshold_state"] != "NON_EXECUTABLE"
        ):
            raise ContractError("CORPORATE_REPORT_WINDOW_STATE_INVALID")
        if row["market_ref"] is not None:
            validate_ref(row["market_ref"])
        if type(row["required_dates"]) is not list or row["required_dates"] != sorted(
            set(row["required_dates"])
        ):
            raise ContractError("CORPORATE_REPORT_DATES_INVALID")
        for day in row["required_dates"]:
            _validate_day(day)
        codes(row["blocker_codes"], allowed)
        _transitions(row["transitions"])
        for event in row["events"]:
            fields(
                event,
                "event_ref event_id kind effective_trade_date source_state financial "
                "anchor reconciliation_state blocker_codes",
            )
            validate_ref(event["event_ref"])
            _validate_day(event["effective_trade_date"])
            if (
                event["kind"] not in KINDS
                or event["source_state"] != "SOURCE_DECLARED"
                or event["reconciliation_state"]
                not in {"UNCONFIRMED", "OWNER_REVIEW_REQUIRED", "RECONCILED_RESEARCH_ONLY"}
            ):
                raise ContractError("CORPORATE_REPORT_EVENT_STATE_INVALID")
            financial(event["financial"], allowed)
            anchor(event["anchor"], allowed)
            codes(event["blocker_codes"], allowed)
