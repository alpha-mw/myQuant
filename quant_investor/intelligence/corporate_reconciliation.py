"""Deterministic full-window corporate evidence report, with no threshold authority."""

from copy import deepcopy
from datetime import datetime, timezone

from quant_investor.strategy_records.corporate_contracts import number, STRATEGY
from quant_investor.operations.daily_contract import ContractError, utc_stamp
from quant_investor.operations.daily_journal import _validate_day
from ._common import build_artifact, business_identity
from .pcb_ai_hardware import physical_refs
from .corporate_report_contract import validate_company_rows

KIND = "corporate_action_reconciliation"
BLOCKERS = frozenset(
    {
        "ANCHOR_MISSING",
        "POLICY_BASELINE_UNCONFIRMED",
        "POSITION_LIFECYCLE_UNCONFIRMED",
        "CALENDAR_WINDOW_INCOMPLETE",
        "MARKET_SESSION_GAP",
        "MARKET_SESSION_CONFLICT",
        "NAMED_EVENT_EVIDENCE_MISSING",
        "ACCOUNTING_EVIDENCE_MISSING",
        "ACCOUNTING_ANCESTRY_UNCONFIRMED",
        "ACCOUNTING_EVENT_LINK_MISSING",
        "ACCOUNTING_MIXED_TRANSITION",
        "ACCOUNTING_ZERO_DELTA",
        "OWNER_REVIEW_REQUIRED",
        "OWNER_REVIEW_RECORD_UNCONFIRMED",
        "CURRENT_EVENT_EMPTY_CONFLICT",
        "SOURCE_CUSTODY_AFTER_DECISION",
    }
)


def _window_factors(rows, required):
    by_date = {}
    for row in rows:
        day = str(row.get("trade_date", "")).replace("-", "")
        if day not in required:
            continue
        if day in by_date:
            return None, "MARKET_SESSION_CONFLICT"
        try:
            close, factor = number(row.get("close")), number(row.get("adj_factor"))
        except ContractError:
            return None, "MARKET_SESSION_GAP"
        if close <= 0 or factor <= 0:
            return None, "MARKET_SESSION_GAP"
        by_date[day] = factor
    if set(by_date) != set(required):
        return None, "MARKET_SESSION_GAP"
    return by_date, None


def _required_sessions(calendar_dates, start, trade_date):
    for day in calendar_dates:
        _validate_day(day)
    if (
        not calendar_dates
        or calendar_dates != sorted(set(calendar_dates))
        or calendar_dates[0] > start
        or calendar_dates[-1] < trade_date
        or start > trade_date
    ):
        return None
    required = [d for d in calendar_dates if start <= d <= trade_date]
    return required if required and required[-1] == trade_date else None


def tracking_window(
    *, symbol, start, trade_date, calendar_dates, rows, market_ref, events, baseline_valid=True
):
    value = {
        "symbol": symbol,
        "tracking_start_date": start,
        "window_state": "VERIFIED",
        "required_dates": [],
        "market_ref": market_ref,
        "transitions": [],
        "events": [],
        "threshold_state": "NON_EXECUTABLE",
        "blocker_codes": [],
    }

    def missing(state, code):
        value.update(window_state=state, blocker_codes=[code])
        return value

    if start is None:
        return missing("ANCHOR_MISSING", "ANCHOR_MISSING")
    _validate_day(start)
    _validate_day(trade_date)
    if not baseline_valid:
        return missing("POLICY_BASELINE_UNCONFIRMED", "POLICY_BASELINE_UNCONFIRMED")
    required = _required_sessions(calendar_dates, start, trade_date)
    if required is None:
        return missing("CALENDAR_GAP", "CALENDAR_WINDOW_INCOMPLETE")
    value["required_dates"] = required
    by_date, error = _window_factors(rows, required)
    if error is not None:
        return missing(
            "MARKET_CONFLICT" if error == "MARKET_SESSION_CONFLICT" else "MARKET_GAP", error
        )
    for before, after in zip(required, required[1:]):
        if by_date[before] == by_date[after]:
            continue
        ids = sorted(
            e["event_id"]
            for e in events
            if e["symbol"] == symbol and e["effective_trade_date"] == after
        )
        value["transitions"].append(
            {
                "kind": "ADJUSTMENT_FACTOR_CHANGE",
                "previous_trade_date": before,
                "trade_date": after,
                "before_factor": format(by_date[before], "f"),
                "after_factor": format(by_date[after], "f"),
                "event_ids": ids,
            }
        )
        if not ids:
            value["blocker_codes"] = ["NAMED_EVENT_EVIDENCE_MISSING"]
    return value


def event_state(financial, anchor):
    if financial["state"] != "OBSERVED_NATIVE_POSTING":
        return "UNCONFIRMED"
    if anchor["state"] != "OWNER_DECLARED_RESEARCH_RESET":
        return "OWNER_REVIEW_REQUIRED"
    return "RECONCILED_RESEARCH_ONLY"


def _ordered_evidence(company_rows):
    rows = sorted(deepcopy(company_rows), key=lambda r: r["symbol"])
    if len({r["symbol"] for r in rows}) != len(rows):
        raise ContractError("CORPORATE_REPORT_DUPLICATE_SYMBOL")
    blockers = []
    for row in rows:
        row["transitions"].sort(key=lambda r: (r["trade_date"], r["previous_trade_date"]))
        row["events"].sort(key=lambda r: (r["effective_trade_date"], r["event_id"]))
        for transition in row["transitions"]:
            transition["event_ids"] = sorted(set(transition["event_ids"]))
        for event in row["events"]:
            event["reconciliation_state"] = event_state(event["financial"], event["anchor"])
            event["blocker_codes"] = sorted(
                set(event["financial"]["blocker_codes"] + event["anchor"]["blocker_codes"])
            )
            row["blocker_codes"].extend(event["blocker_codes"])
        row["threshold_state"] = "NON_EXECUTABLE"
        row["blocker_codes"] = sorted(set(row["blocker_codes"]))
        blockers.extend(row["blocker_codes"])
    codes = sorted(set(blockers))
    if not set(codes) <= BLOCKERS:
        raise ContractError("CORPORATE_REPORT_UNKNOWN_BLOCKER")
    if "CURRENT_EVENT_EMPTY_CONFLICT" in codes:
        summary = "CURRENT_FINANCIAL_CONFLICT"
    elif codes or any(r["window_state"] != "VERIFIED" for r in rows):
        summary = "UNCONFIRMED"
    elif any(r["events"] for r in rows):
        summary = "RECONCILED_RESEARCH_ONLY"
    elif any(r["transitions"] for r in rows):
        summary = "UNCONFIRMED"
    else:
        summary = "NO_ADJUSTMENT_OBSERVED"
    return rows, codes, summary


def build_reconciliation(
    *,
    as_of,
    trade_date,
    context_ref,
    decision_recipe_ref,
    store_plan_ref,
    portfolio_source_ref,
    tracking_policy_ref,
    source_refs,
    company_rows,
    custody_at,
):
    custody, cutoff = utc_stamp(custody_at), utc_stamp(as_of)
    if custody > datetime.now(timezone.utc):
        raise ContractError("CORPORATE_CUSTODY_IN_FUTURE")
    validate_company_rows(company_rows, BLOCKERS)
    rows, codes, summary = _ordered_evidence(company_rows)
    late = custody > cutoff
    if late:
        codes = sorted(set(codes) | {"SOURCE_CUSTODY_AFTER_DECISION"})
    fields = {
        "as_of": as_of,
        "trade_date": trade_date,
        "strategy_id": STRATEGY,
        "context_ref": context_ref,
        "decision_recipe_ref": decision_recipe_ref,
        "store_plan_ref": store_plan_ref,
        "portfolio_source_ref": portfolio_source_ref,
        "tracking_policy_ref": tracking_policy_ref,
        "source_refs": physical_refs(source_refs),
        "company_rows": rows,
        "summary_state": summary,
        "blocker_codes": codes,
        "custody_at": custody_at,
        "timing_status": "LATE_RECORDED" if late else "ON_TIME",
        "prospective": False,
    }
    return build_artifact(
        kind=KIND,
        identity_field="reconciliation_id",
        identity=business_identity(kind=KIND, identity_inputs=fields),
        created_at=custody_at,
        fields=fields,
    )
