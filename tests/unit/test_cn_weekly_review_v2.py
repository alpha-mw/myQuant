from copy import deepcopy

import pytest

from quant_investor.strategy_records.close_coverage import analyze_close_coverage, BENCHMARK_SYMBOLS
from quant_investor.strategy_records.research_risk import calculate_position_risk
from scripts.cn_weekly_review_v2 import daily_domain, period_projection
from scripts.cn_weekly_review_v2 import assess_production_days
from scripts.export_cn_weekly_review_evidence import report_window, WeeklyEvidenceError


def daily_row(**kwargs):
    return {
        "automation_id": "automation",
        "thread_id": "thread1",
        "run_id": "run1",
        "trade_date": "2026-08-31",
        "started_at": "2026-08-31T01:45:00Z",
        "completed_at": "2026-08-31T02:00:00Z",
        "run_status": "COMPLETED",
        "research_status": "COMPLETE",
        "last_run": "2026-08-30T01:45:00Z",
        **kwargs,
    }


def domain(rows):
    return daily_domain(
        {"items": rows},
        None,
        window=report_window("2026-09-06T10:00:00Z"),
        expected_trade_dates=["2026-08-31"],
    )


def test_previous_run_outside_window_does_not_exclude_real_current_run():
    d, rows = domain([daily_row()])
    assert d["status"] == "FRESH"
    assert d["evidence"]["task_review_dates"] == ["2026-08-31"]
    assert d["evidence"]["formal_closure_dates"] == []


@pytest.mark.parametrize(
    "completed,timing",
    [
        ("2026-08-31T07:00:58Z", "SAME_SESSION"),
        ("2026-08-31T07:00:59Z", "SAME_SESSION"),
        ("2026-08-31T16:00:00Z", "LATE_RECOVERY"),
    ],
)
def test_daily_cutoff_boundary(completed, timing):
    _, rows = domain([daily_row(completed_at=completed)])
    assert rows[0]["timing"] == timing


@pytest.mark.parametrize(
    "run,research,completed",
    [
        ("FAILED", "BLOCKED", "2026-08-31T02:00:00Z"),
        ("IN_PROGRESS", "PARTIAL", None),
        ("COMPLETED", "PARTIAL", "2026-08-31T02:00:00Z"),
    ],
)
def test_run_existence_does_not_prove_completed_research(run, research, completed):
    d, _ = domain([daily_row(run_status=run, research_status=research, completed_at=completed)])
    assert d["evidence"]["task_review_dates"] == ["2026-08-31"]
    assert d["evidence"]["research_complete_dates"] == []


def test_duplicate_run_replay_and_conflict():
    row = daily_row()
    assert len(domain([row, deepcopy(row)])[1]) == 1
    with pytest.raises(WeeklyEvidenceError, match="conflicting"):
        domain([row, daily_row(research_status="PARTIAL")])


def test_unexpected_automation_identity_is_not_silently_ignored():
    with pytest.raises(WeeklyEvidenceError, match="automation identity"):
        domain([daily_row(automation_id="unregistered-job")])


def test_trailing_and_owner_stop_blockers_are_independent():
    row = risk(trailing_blockers=["EXACT_ENTRY_REF_MISMATCH"], owner_stop="9.00")
    assert row["moving_take_profit_review_price"] is None
    assert row["owner_stop_trigger"] == "CLEAR"
    row = risk(owner_stop="9.00", owner_stop_blockers=["OWNER_STOP_HISTORY_GAP"])
    assert row["moving_take_profit_review_price"] is not None
    assert row["owner_stop_trigger"] == "UNCONFIRMED"
    assert "OWNER_STOP_HISTORY_GAP" in row["blockers"]
    row = risk(owner_stop="9.00", lifecycle_blockers=["LIFECYCLE_UNCONFIRMED:20260903"])
    assert row["moving_take_profit_review_price"] is None
    assert row["owner_stop_trigger"] == "UNCONFIRMED"


def test_five_and_ten_real_component_days_are_separate_from_dispatch():
    days = [f"2026-09-{day:02}" for day in [1, 2, 3, 4, 7, 8, 9, 10, 11, 14]]
    rows = [
        {
            "trade_date": d,
            "artifact_completion_at": d + "T13:00:00Z",
            "maintenance_verified": True,
            "observations_verified": True,
            "top100_verified": True,
        }
        for d in days
    ]
    a = assess_production_days(rows[:5], days)
    assert a["five_day_checkpoint"] == "MET" and a["ten_day_acceptance"] == "NOT_MET"
    assert assess_production_days(rows, days)["ten_day_acceptance"] == "MET"
    rows[4]["artifact_completion_at"] = "2026-09-08T00:00:00Z"
    a = assess_production_days(rows, days)
    assert a["late_recovery_dates"] == ["2026-09-07"]
    assert a["ten_day_acceptance"] == "NOT_MET"
    rows[9]["maintenance_verified"] = False
    assert assess_production_days(rows, days)["five_day_checkpoint"] == "NOT_MET"


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_at": "2026-08-31T01:00:00Z"},
        {"run_status": "FAILED"},
        {"completed_at": None},
        {"trade_date": "2026-08-31", "started_at": "2026-08-30T23:00:00Z"},
    ],
)
def test_invalid_run_chronology_or_status(changes):
    if changes.get("started_at"):
        changes["started_at"] = "2026-08-30T16:00:00Z"  # local8/31 is valid
        assert domain([daily_row(**changes)])[0]["status"] == "FRESH"
    else:
        with pytest.raises(WeeklyEvidenceError):
            domain([daily_row(**changes)])


def risk(**changes):
    kwargs = dict(
        position={"symbol": "000001.SZ", "avg_cost": "10.003", "shares": 100},
        anchor={"tracking_start_date": "20260901"},
        as_of="20260903",
        expected_dates=["20260901", "20260902", "20260903"],
        closes=[
            {"trade_date": d, "close": p, "adj_factor": "1"}
            for d, p in [("20260901", "12"), ("20260902", "15"), ("20260903", "11")]
        ],
    )
    kwargs.update(changes)
    return calculate_position_risk(**kwargs)


def test_policy_rounding_retention_and_stale_holdings():
    row = risk()
    assert row["moving_take_profit_review_price"] == "14.00"
    assert row["moving_take_profit_reduce_price"] == "13.25"
    assert row["moving_stop_price"] == "14.00"
    assert row["threshold_state"] == "NON_EXECUTABLE_HOLDINGS_STALE"
    assert row["trailing_trigger"] == "REDUCTION_REVIEW"
    assert row["actions"] == [] and not row["executable"]


def test_owner_reset_excludes_pre_anchor_peak_without_inventing_fill():
    before = {"trade_date": "20260831", "close": "1000", "adj_factor": "1"}
    row = risk(
        closes=[
            before,
            *[
                {"trade_date": d, "close": "12", "adj_factor": "1"}
                for d in ["20260901", "20260902", "20260903"]
            ],
        ]
    )
    assert row["peak_price"] == "12"


def test_no_positive_peak_and_owner_stop_independent_of_trailing_anchor():
    assert risk(position={"symbol": "000001.SZ", "avg_cost": 20, "shares": 100})[
        "calculation_state"
    ].startswith("NOT_APPLICABLE")
    row = risk(anchor=None, owner_stop="11.00")
    assert row["calculation_state"] == "UNCONFIRMED"
    assert row["owner_stop_trigger"] == "BREACH"
    assert row["owner_review_state"] == "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW"
    assert not row["executable"]


@pytest.mark.parametrize(
    "problem", ["gap", "duplicate", "nan", "negative", "adjustment", "new_add"]
)
def test_unsafe_inputs_never_yield_usable_threshold(problem):
    rows = [
        {"trade_date": d, "close": "12", "adj_factor": "1"}
        for d in ["20260901", "20260902", "20260903"]
    ]
    changes = {}
    if problem == "gap":
        rows.pop(1)
    if problem == "duplicate":
        rows.append(dict(rows[0]))
    if problem == "nan":
        rows[1]["close"] = "NaN"
    if problem == "negative":
        rows[1]["close"] = "-1"
    if problem == "adjustment":
        rows[1]["adj_factor"] = "2"
    if problem == "new_add":
        changes["lifecycle_blockers"] = ["NEW_ENTRY_ADD"]
    row = risk(closes=rows, **changes)
    assert row["moving_take_profit_review_price"] is None
    assert row["threshold_state"] == "NON_EXECUTABLE"


def test_official_close_reports_all_independent_missing_inputs():
    d = "2026-09-04"
    result = analyze_close_coverage(
        required_dates=[d],
        event_dates=[],
        benchmark_keys=[],
        held_close_keys=[(d, "000001.SZ")],
        symbols=["000001.SZ"],
    )
    assert result["blockers"] == [
        "EVENT_STATE_CLOSURE_MISSING:" + d,
        "BENCHMARK_EXACT_CLOSE_MISSING:" + d,
    ]
    assert result["dates"][0]["missing_held_symbols"] == []
    good = analyze_close_coverage(
        required_dates=[d],
        event_dates=[d],
        benchmark_keys=[(d, s) for s in BENCHMARK_SYMBOLS],
        held_close_keys=[(d, "000001.SZ")] * 2,
        symbols=["000001.SZ"],
    )
    assert good["status"] == "BLOCKED"


def test_week_period_uses_prior_session_and_discloses_missing_friday():
    pts = [
        {
            "date": d,
            "record": d,
            "portfolio_unit_nav": nav,
            "total_value": nav * 100,
            "csi300_nav": nav,
            "benchmark_coverage": "exact_close",
            "benchmark_value_date": d,
        }
        for d, nav in [("2026-08-28", 1), ("2026-08-31", 1.1), ("2026-09-03", 1.2)]
    ]
    bundle = {
        "report_window": report_window("2026-09-06T10:00:00Z"),
        "performance_benchmark": {"portfolio": {"performance_points": pts}},
    }
    p = period_projection(bundle, ["2026-08-31", "2026-09-03", "2026-09-04"])
    assert p["baseline_date"] == "2026-08-28"
    assert p["state"] == "PARTIAL_WEEK"
    assert p["missing_dates"] == ["2026-09-04"]
    assert p["portfolio_return"] == pytest.approx(0.2)
