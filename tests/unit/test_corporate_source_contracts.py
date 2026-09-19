"""Exact source declarations, aware-time boundaries and pure report precedence."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import pytest

from quant_investor.strategy_records import corporate_contracts as source
from quant_investor.operations.daily_contract import ContractError
from quant_investor.intelligence.corporate_reconciliation import (
    tracking_window,
    build_reconciliation,
)
from _native_corporate_fixture import STRATEGY, AS_OF

REF = {"path": "fixtures/source.json", "sha256": "a" * 64}


def declaration(kind="DIVIDEND"):
    return {
        "schema_version": "cn-corporate-action-event.v1",
        "event_id": "event1",
        "symbol": "601899.SH",
        "kind": kind,
        "effective_trade_date": "20260821",
        "announced_at": "2026-08-20T08:00:00+08:00",
        "announcement_ref": REF,
        "accounting_records": None,
    }


@pytest.mark.parametrize("kind", sorted(source.KINDS))
def test_five_named_kinds_remain_source_declarations(kind):
    value = declaration(kind)
    assert source.event(value, as_of=AS_OF) == value


@pytest.mark.parametrize("fault", ["unknown_kind", "extra_custody", "future", "naive", "bad_ref"])
def test_invalid_named_source_rejects(fault):
    value = declaration()
    if fault == "unknown_kind":
        value["kind"] = "AUTO_INFERRED_DIVIDEND"
    elif fault == "extra_custody":
        value["custody_at"] = AS_OF
    elif fault == "future":
        value["announced_at"] = "2026-09-01T00:00:00Z"
    elif fault == "naive":
        value["announced_at"] = "2026-08-20T08:00:00"
    else:
        value["announcement_ref"] = {"path": "../source.json", "sha256": "a" * 64}
    with pytest.raises(ValueError):
        source.event(value, as_of=AS_OF)


def test_aware_review_must_follow_announcement():
    event = declaration()
    event["announced_at"] = "2026-08-22T09:00:00+08:00"
    value = {
        "schema_version": "cn-corporate-anchor-reviews.v1",
        "strategy_id": STRATEGY,
        "owner": "Owner",
        "declaration_id": "review1",
        "declared_at": "2026-08-22T03:00:00Z",
        "authority": source.REVIEW_AUTHORITY,
        "reviews": [
            {
                "event_ref": REF,
                "policy_ref": REF,
                "source_record_id": "r1",
                "reviewed_at": "2026-08-22T00:59:59Z",
                "disposition": "OWNER_DECLARED_RESET",
                "tracking_start_date": "20260821",
            }
        ],
    }
    with pytest.raises(ContractError, match="CHRONOLOGY"):
        source.owner_reviews(
            value,
            as_of=AS_OF,
            owner="Owner",
            policy_ref=REF,
            events={(REF["path"], REF["sha256"]): event},
        )
    value["reviews"][0]["reviewed_at"] = "2026-08-22T01:00:00Z"
    assert source.owner_reviews(
        value,
        as_of=AS_OF,
        owner="Owner",
        policy_ref=REF,
        events={(REF["path"], REF["sha256"]): event},
    )


def window(**updates):
    args = dict(
        symbol="601899.SH",
        start="20260820",
        trade_date="20260824",
        calendar_dates=["20260820", "20260821", "20260824"],
        rows=[
            {"trade_date": day, "close": "2", "adj_factor": factor}
            for day, factor in [
                ("20260820", "1"),
                ("20260821", "1.0000000000000000001"),
                ("20260824", "1.0000000000000000001"),
            ]
        ],
        market_ref=REF,
        events=[],
    )
    args.update(updates)
    return tracking_window(**args)


def test_decimal_transition_is_not_rounded_away():
    row = window()
    assert len(row["transitions"]) == 1
    assert row["transitions"][0]["after_factor"] == "1.0000000000000000001"


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("calendar", "CALENDAR_GAP"),
        ("missing", "MARKET_GAP"),
        ("duplicate", "MARKET_CONFLICT"),
        ("infinite", "MARKET_GAP"),
    ],
)
def test_incomplete_window_never_shortens_to_available_tail(fault, expected):
    if fault == "calendar":
        row = window(calendar_dates=["20260821", "20260824"])
    else:
        rows = [
            {"trade_date": d, "close": "2", "adj_factor": "1"}
            for d in ("20260820", "20260821", "20260824")
        ]
        if fault == "missing":
            rows.pop(0)
        elif fault == "duplicate":
            rows.append(rows[0])
        else:
            rows[0]["adj_factor"] = "NaN"
        row = window(rows=rows)
    assert row["window_state"] == expected and not row["transitions"]


def report(custody, rows):
    return build_reconciliation(
        as_of=AS_OF,
        trade_date="20260824",
        context_ref=REF,
        decision_recipe_ref=REF,
        store_plan_ref=REF,
        portfolio_source_ref=REF,
        tracking_policy_ref=REF,
        source_refs=[REF],
        company_rows=rows,
        custody_at=custody,
    )


def test_cutoff_equality_late_future_and_deterministic_internal_order():
    row = window(
        rows=[
            {"trade_date": d, "close": "2", "adj_factor": "1"}
            for d in ("20260820", "20260821", "20260824")
        ]
    )
    other = deepcopy(row)
    other["symbol"] = "002463.SZ"
    first = report(AS_OF, [row, other])
    assert first == report(AS_OF, [other, row])
    assert first["payload"]["timing_status"] == "ON_TIME"
    assert first["payload"]["summary_state"] == "NO_ADJUSTMENT_OBSERVED"
    late = report("2026-08-24T13:30:01Z", [row])
    assert late["payload"]["timing_status"] == "LATE_RECORDED"
    assert late["payload"]["prospective"] is False
    assert "SOURCE_CUSTODY_AFTER_DECISION" in late["payload"]["blocker_codes"]
    future = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(ContractError, match="FUTURE"):
        report(future, [row])
    row["blocker_codes"] = ["CURRENT_EVENT_EMPTY_CONFLICT", "MARKET_SESSION_GAP"]
    assert report(AS_OF, [row])["payload"]["summary_state"] == "CURRENT_FINANCIAL_CONFLICT"


def test_late_corporate_evidence_excludes_otherwise_on_time_ledger():
    from test_daily_evidence_prospective_timing import context
    from quant_investor.operations.prospective_timing import classify_daily_evidence

    assert classify_daily_evidence(**context())["prospective"] is True
    assert classify_daily_evidence(**context(), corporate_late=True)["prospective"] is False


def test_execute_v3_and_catchup_require_explicit_corporate_context():
    from test_daily_evidence_production_request import recipe, request
    from quant_investor.operations.execution_recipe import (
        validate_execution_recipe,
        validate_catchup_template,
    )

    value = recipe()
    value.update(
        schema_version="cn-daily-execute-recipe.v3",
        theme_acquisition_ref=None,
        corporate_action_context_ref=REF,
    )
    value["research_sources"]["theme_source_ref"] = REF
    assert validate_execution_recipe(value, request=request()) == value
    template = deepcopy(value)
    template["previous_completion_ref"] = None
    template["store_preimages"]["store_pointer_ref"] = None
    assert (
        validate_catchup_template(template, request=request())["corporate_action_context_ref"]
        == REF
    )
    value["corporate_action_context_ref"] = None
    with pytest.raises(ContractError):
        validate_execution_recipe(value, request=request())
    del value["corporate_action_context_ref"]
    with pytest.raises(ContractError, match="V3_FIELDS"):
        validate_execution_recipe(value, request=request())


def test_native_v4_context_cannot_downgrade_to_legacy_shape():
    from quant_investor.operations.native_input_contract import (
        FIELDS,
        EXTRAS,
        validate_native_input_shape,
    )

    value = dict.fromkeys(FIELDS | EXTRAS)
    value.update(
        schema_version="cn-daily-native-inputs.v4",
        decision_recipe_ref=REF,
        corporate_action_context_ref=REF,
    )
    validate_native_input_shape(value)
    value["corporate_action_context_ref"] = None
    with pytest.raises(ContractError):
        validate_native_input_shape(value)
    del value["corporate_action_context_ref"]
    with pytest.raises(ContractError, match="SCHEMA"):
        validate_native_input_shape(value)
