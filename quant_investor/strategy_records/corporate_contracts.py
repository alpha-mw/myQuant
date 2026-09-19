"""Exact read-only corporate source declarations; no issuer or execution authority."""

from decimal import Decimal, InvalidOperation
from datetime import datetime
from zoneinfo import ZoneInfo
import re

from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import _validate_day

STRATEGY = "aggressive_tech_manufacturing"
KINDS = frozenset({"SPLIT", "DIVIDEND", "RIGHTS", "BONUS_ISSUE", "SHARE_CONVERSION"})
REVIEW_AUTHORITY = {
    "research_interpretation": True,
    **dict.fromkeys(
        (
            "store_mutation",
            "actual_holdings_mutation",
            "broker",
            "order",
            "execution",
            "trade",
            "policy_mutation",
        ),
        False,
    ),
}


POLICY_CALCULATION = {
    "peak_source": "STRICT_CN_DAILY_CLOSE_FROM_TRACKING_START",
    "review_profit_retention": "0.80",
    "reduce_profit_retention": "0.65",
    "moving_take_profit_review_price": "avg_cost + 0.80 * max(peak_price - avg_cost, 0)",
    "moving_take_profit_reduce_price": "avg_cost + 0.65 * max(peak_price - avg_cost, 0)",
    "moving_stop_price": "moving_take_profit_review_price",
    "no_positive_peak_state": "NOT_APPLICABLE_UNTIL_POSITIVE_PROFIT_PEAK",
    "rounding": "CNY_0.01_HALF_UP",
    "automatic_order": False,
}


def instant(value):
    try:
        if type(value) is not str or value != value.strip():
            raise ValueError("not timestamp text")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError("CORPORATE_TIMESTAMP_INVALID") from exc


def shape(value, schema, fields):
    if (
        type(value) is not dict
        or set(value) != {"schema_version", *fields}
        or value["schema_version"] != schema
    ):
        raise ContractError("CORPORATE_SOURCE_SCHEMA_INVALID:" + schema)
    return value


def identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ContractError("CORPORATE_SOURCE_ID_INVALID")
    return value


def number(value):
    if isinstance(value, bool) or value is None:
        raise ContractError("CORPORATE_NUMBER_INVALID")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ContractError("CORPORATE_NUMBER_INVALID") from exc
    if not result.is_finite():
        raise ContractError("CORPORATE_NUMBER_INVALID")
    return result


def ordered_refs(values):
    if type(values) is not list:
        raise ContractError("CORPORATE_SOURCE_REFS_INVALID")
    refs = [validate_ref(ref) for ref in values]
    keys = [(ref["path"], ref["sha256"]) for ref in refs]
    if keys != sorted(set(keys)):
        raise ContractError("CORPORATE_SOURCE_REFS_NOT_SORTED_UNIQUE")
    return refs


def context(value, *, as_of):
    shape(
        value,
        "cn-corporate-action-context.v1",
        {"strategy_id", "as_of", "tracking_policy_ref", "named_events_ref", "anchor_reviews_ref"},
    )
    if value["strategy_id"] != STRATEGY or value["as_of"] != as_of:
        raise ContractError("CORPORATE_CONTEXT_BINDING_INVALID")
    instant(as_of)
    for field in ("tracking_policy_ref", "named_events_ref", "anchor_reviews_ref"):
        if field == "tracking_policy_ref" or value[field] is not None:
            validate_ref(value[field])
    return value


def event_list(value, *, as_of):
    shape(value, "cn-corporate-action-events.v1", {"strategy_id", "as_of", "event_refs"})
    if value["strategy_id"] != STRATEGY or value["as_of"] != as_of:
        raise ContractError("CORPORATE_EVENT_LIST_BINDING_INVALID")
    return ordered_refs(value["event_refs"])


def event(value, *, as_of):
    shape(
        value,
        "cn-corporate-action-event.v1",
        {
            "event_id",
            "symbol",
            "kind",
            "effective_trade_date",
            "announced_at",
            "announcement_ref",
            "accounting_records",
        },
    )
    identifier(value["event_id"])
    if (
        type(value["symbol"]) is not str
        or re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", value["symbol"]) is None
    ):
        raise ContractError("CORPORATE_EVENT_SYMBOL_INVALID")
    if type(value["kind"]) is not str or value["kind"] not in KINDS:
        raise ContractError("CORPORATE_EVENT_KIND_INVALID")
    _validate_day(value["effective_trade_date"])
    if instant(value["announced_at"]) > instant(as_of):
        raise ContractError("CORPORATE_EVENT_FUTURE_ANNOUNCEMENT")
    validate_ref(value["announcement_ref"])
    records = value["accounting_records"]
    if records is not None:
        if type(records) is not dict or set(records) != {"before_record_id", "after_record_id"}:
            raise ContractError("CORPORATE_ACCOUNTING_RECORD_SHAPE_INVALID")
        for record in records.values():
            identifier(record)
    return value


def owner_reviews(value, *, as_of, owner, policy_ref, events):
    shape(
        value,
        "cn-corporate-anchor-reviews.v1",
        {"strategy_id", "owner", "declaration_id", "declared_at", "authority", "reviews"},
    )
    identifier(value["declaration_id"])
    if (
        value["strategy_id"] != STRATEGY
        or value["owner"] != owner
        or value["authority"] != REVIEW_AUTHORITY
    ):
        raise ContractError("CORPORATE_OWNER_DECLARATION_INVALID")
    # JSON bool equality is insufficient for authority checks.
    if any(type(value["authority"].get(k)) is not bool for k in REVIEW_AUTHORITY):
        raise ContractError("CORPORATE_OWNER_AUTHORITY_INVALID")
    declared = instant(value["declared_at"])
    if declared > instant(as_of) or type(value["reviews"]) is not list:
        raise ContractError("CORPORATE_OWNER_DECLARATION_TIME_INVALID")
    keys, result = [], {}
    for row in value["reviews"]:
        if type(row) is not dict or set(row) != {
            "event_ref",
            "policy_ref",
            "source_record_id",
            "reviewed_at",
            "disposition",
            "tracking_start_date",
        }:
            raise ContractError("CORPORATE_OWNER_REVIEW_SHAPE_INVALID")
        ref = validate_ref(row["event_ref"])
        key = (ref["path"], ref["sha256"])
        keys.append(key)
        if (
            key not in events
            or row["policy_ref"] != policy_ref
            or row["disposition"] != "OWNER_DECLARED_RESET"
        ):
            raise ContractError("CORPORATE_OWNER_REVIEW_BINDING_INVALID")
        source = events[key]
        identifier(row["source_record_id"])
        _validate_day(row["tracking_start_date"])
        if not (
            instant(source["announced_at"])
            <= instant(row["reviewed_at"])
            <= declared
            <= instant(as_of)
        ):
            raise ContractError("CORPORATE_OWNER_REVIEW_CHRONOLOGY_INVALID")
        if (
            not source["effective_trade_date"]
            <= row["tracking_start_date"]
            <= instant(as_of).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        ):
            raise ContractError("CORPORATE_OWNER_ANCHOR_DATE_INVALID")
        result[key] = row
    if keys != sorted(set(keys)):
        raise ContractError("CORPORATE_OWNER_REVIEW_DUPLICATE_OR_UNSORTED")
    return result
