"""Shared pure owner risk-policy contracts; no policy writes or execution authority."""

from quant_investor.strategy_records import corporate_contracts as contracts
from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import _validate_day


def validate_trailing_policy(policy, *, as_of):
    contracts.shape(
        policy,
        "owner-trailing-anchor-policy.v1",
        {
            "policy_id",
            "strategy_domain",
            "owner",
            "effective_from",
            "authority",
            "store_binding",
            "calculation",
            "anchors",
            "invalidation",
        },
    )
    contracts.identifier(policy["policy_id"])
    if (
        policy["strategy_domain"] != contracts.STRATEGY
        or type(policy["owner"]) is not str
        or not policy["owner"].strip()
    ):
        raise ContractError("CORPORATE_POLICY_IDENTITY_INVALID")
    if contracts.instant(policy["effective_from"]) > contracts.instant(as_of):
        raise ContractError("CORPORATE_POLICY_NOT_EFFECTIVE")
    forbidden = (
        "store_mutation",
        "actual_holdings_mutation",
        "broker",
        "live_order",
        "live_execution",
        "trade",
    )
    if type(policy["authority"]) is not dict or any(
        policy["authority"].get(k) is not False for k in forbidden
    ):
        raise ContractError("CORPORATE_POLICY_AUTHORITY_INVALID")
    if (
        policy["invalidation"]
        != "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_NEW_ENTRY_ADD_OR_CORPORATE_ACTION"
    ):
        raise ContractError("CORPORATE_POLICY_INVALIDATION_UNSUPPORTED")
    expected_authority = {
        "research_threshold_calculation": True,
        "paper_risk_reduction_input": True,
        **dict.fromkeys(forbidden, False),
    }
    if policy["authority"] != expected_authority or any(
        type(v) is not bool for v in policy["authority"].values()
    ):
        raise ContractError("CORPORATE_POLICY_AUTHORITY_INVALID")
    if (
        policy["calculation"] != contracts.POLICY_CALCULATION
        or policy["calculation"]["automatic_order"] is not False
    ):
        raise ContractError("CORPORATE_POLICY_CALCULATION_UNSUPPORTED")
    validate_trailing_anchors(policy["anchors"])
    binding = policy["store_binding"]
    if type(binding) is not dict or set(binding) != {
        "pointer_path",
        "pointer_sha256",
        "active_record_id",
        "ledger_path",
        "ledger_sha256",
    }:
        raise ContractError("CORPORATE_POLICY_STORE_BINDING_INVALID")
    validate_ref({"path": binding["ledger_path"], "sha256": binding["ledger_sha256"]})
    validate_ref({"path": binding["pointer_path"], "sha256": binding["pointer_sha256"]})


def validate_trailing_anchors(anchors):
    if type(anchors) is not list:
        raise ContractError("CORPORATE_POLICY_ANCHORS_INVALID")
    symbols = []
    for anchor in anchors:
        required = {
            "symbol",
            "company_name",
            "tracking_start_date",
            "anchor_state",
            "exclude_pre_anchor_peaks",
        }
        allowed = required | {"anchor_ref", "retired_unexecutable_trailing_stop_cny"}
        if (
            type(anchor) is not dict
            or not required <= set(anchor) <= allowed
            or anchor["exclude_pre_anchor_peaks"] is not True
        ):
            raise ContractError("CORPORATE_POLICY_ANCHOR_INVALID")
        _validate_day(anchor["tracking_start_date"])
        symbols.append(anchor["symbol"])
    if len(symbols) != len(set(symbols)):
        raise ContractError("CORPORATE_POLICY_DUPLICATE_SYMBOL")


def validate_initial_stop_policy(policy, *, as_of):
    contracts.shape(
        policy,
        "initial-risk-stop.v1",
        {
            "policy_id",
            "strategy_domain",
            "owner",
            "owner_confirmation_recorded_at",
            "effective_from",
            "authority",
            "store_binding",
            "market_evidence",
            "stops",
        },
    )
    contracts.identifier(policy["policy_id"])
    if (
        policy["strategy_domain"] != contracts.STRATEGY
        or type(policy["owner"]) is not str
        or not policy["owner"].strip()
    ):
        raise ContractError("MORNING_STOP_POLICY_IDENTITY_INVALID")
    if any(
        contracts.instant(policy[k]) > contracts.instant(as_of)
        for k in ("effective_from", "owner_confirmation_recorded_at")
    ):
        raise ContractError("MORNING_STOP_POLICY_NOT_AVAILABLE_AT_QUOTE")
    expected = {
        "risk_policy_confirmation": True,
        **dict.fromkeys(
            (
                "broker",
                "live_order",
                "live_execution",
                "trade",
                "actual_holdings_mutation",
                "automatic_human_action",
            ),
            False,
        ),
    }
    if policy["authority"] != expected or any(
        type(v) is not bool for v in policy["authority"].values()
    ):
        raise ContractError("MORNING_STOP_POLICY_AUTHORITY_INVALID")
    binding = policy["store_binding"]
    if type(binding) is not dict or set(binding) != {
        "pointer_path",
        "pointer_sha256",
        "active_record_id",
        "ledger_path",
        "ledger_sha256",
        "valuation_trade_date",
        "total_value_after_cny",
    }:
        raise ContractError("MORNING_STOP_POLICY_STORE_BINDING_INVALID")
    for key in ("pointer", "ledger"):
        validate_ref({"path": binding[key + "_path"], "sha256": binding[key + "_sha256"]})
    _validate_day(binding["valuation_trade_date"])
    if contracts.number(binding["total_value_after_cny"]) <= 0:
        raise ContractError("MORNING_STOP_POLICY_STORE_BINDING_INVALID")
    # Calibration remains the owner's declared context, not dependencies to mutable heads.
    if type(policy["market_evidence"]) is not dict or type(policy["stops"]) is not list:
        raise ContractError("MORNING_STOP_POLICY_CONTEXT_INVALID")
    symbols = []
    for row in policy["stops"]:
        _initial_stop_row(row, policy["policy_id"])
        symbols.append(row["symbol"])
    if len(symbols) != len(set(symbols)):
        raise ContractError("MORNING_STOP_POLICY_DUPLICATE_SYMBOL")


def _initial_stop_row(row, policy_id):
    import re

    legacy = type(row) is dict and row.get("initial_stop_state") == "CONFIRMED_OWNER_REVALIDATION"
    fields = {
        "symbol",
        "company_name",
        "current_shares",
        "effective_entry_event",
        "effective_entry_trade_date",
        "fee_inclusive_avg_cost_cny",
        "initial_stop_state",
        "stop_policy_ref",
        "setting_authority",
        "initial_stop_price_cny",
        "price_tick_cny",
        "trigger",
        "calibration",
        "add_policy",
        "valid_until",
    }
    if legacy:
        fields = (fields - {"calibration"}) | {
            "entry_anchor_state",
            "legacy_stop_revalidated",
            "trailing_stop_state",
            "retired_unexecutable_trailing_stop_cny",
            "sina_price_cny_at_20260828_143135",
        }
    if (
        type(row) is not dict
        or set(row) != fields
        or type(row["symbol"]) is not str
        or not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", row["symbol"])
    ):
        raise ContractError("MORNING_STOP_POLICY_ROW_INVALID")
    _initial_stop_context(row, legacy=legacy)
    shares = contracts.number(row["current_shares"])
    price = contracts.number(row["initial_stop_price_cny"])
    tick = contracts.number(row["price_tick_cny"])
    if (
        shares <= 0
        or shares != shares.to_integral_value()
        or contracts.number(row["fee_inclusive_avg_cost_cny"]) <= 0
        or price <= 0
        or tick != contracts.number("0.01")
        or price % tick != 0
        or row["stop_policy_ref"] != policy_id + ":" + row["symbol"]
        or type(row["company_name"]) is not str
        or type(row["effective_entry_event"]) is not str
        or not row["effective_entry_event"]
        or row["add_policy"] != "NO_ADD_UNTIL_SEPARATE_I6_ELIGIBILITY_AND_NEW_ANCHOR_CONTRACT"
        or row["valid_until"]
        != "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_OR_NEW_ENTRY_ADD_ANCHOR"
    ):
        raise ContractError("MORNING_STOP_POLICY_RULE_UNSUPPORTED")
    expected = {
        "observation": "STRICT_CN_DAILY_CLOSE",
        "operator": "LESS_THAN_OR_EQUAL",
        "price_cny": row["initial_stop_price_cny"],
        "intraday_touch_only": "WARNING_NOT_BREACH",
        "confirmed_breach_state": "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW",
        "review_quantity_shares": row["current_shares"],
        "automatic_order": False,
        "automatic_execution": False,
    }
    trigger = row["trigger"]
    if (
        trigger != expected
        or trigger["automatic_order"] is not False
        or trigger["automatic_execution"] is not False
    ):
        raise ContractError("MORNING_STOP_POLICY_TRIGGER_UNSUPPORTED")


def _initial_stop_context(row, *, legacy):
    if not legacy:
        _validate_day(row["effective_entry_trade_date"])
        if (
            row["initial_stop_state"] != "CONFIRMED"
            or row["setting_authority"] != "OWNER_DELEGATED_TO_CODEX"
            or type(row["calibration"]) is not dict
        ):
            raise ContractError("MORNING_STOP_POLICY_RULE_UNSUPPORTED")
        return
    # This retained declaration confirms a fixed stop without an entry anchor.
    # Its old quote and retired trailing price are metadata, never thresholds.
    if (
        row["effective_entry_event"] != "LEGACY_POSITION_OWNER_STOP_REVALIDATION_20260828"
        or row["effective_entry_trade_date"] is not None
        or row["setting_authority"] != "OWNER_EXPLICIT_CONFIRMATION"
        or row["entry_anchor_state"] != "UNAVAILABLE_LEGACY_POSITION"
        or row["legacy_stop_revalidated"] is not True
        or row["trailing_stop_state"] != "UNAVAILABLE_MISSING_EFFECTIVE_ENTRY_ANCHOR"
        or contracts.number(row["retired_unexecutable_trailing_stop_cny"]) <= 0
        or contracts.number(row["sina_price_cny_at_20260828_143135"]) <= 0
    ):
        raise ContractError("MORNING_STOP_POLICY_RULE_UNSUPPORTED")
