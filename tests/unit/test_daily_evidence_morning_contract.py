import pytest
from quant_investor.operations.morning_contract import (
    validate_morning_request,
    expected_quote_symbols,
    require_next_open_session,
    POLICY_SCHEMA,
)
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY


def request():
    return {
        "schema_version": "morning-strategy-request.v2",
        "action": "REPLAY",
        "run_date": "20260907",
        **{
            key: {"path": key + ".json", "sha256": "a" * 64}
            for key in (
                "previous_completion_ref",
                "quote_capture_ref",
                "quote_raw_ref",
                "owner_policy_ref",
            )
        },
        "output_ref": None,
    }


def policy():
    return {
        "schema_version": POLICY_SCHEMA,
        "market": "CN",
        "strategy_id": "aggressive_tech_manufacturing",
        "effective_from": "20260901",
        "effective_through": "20260930",
        "revoked_at": None,
        "additional_symbols": ["600000.SH"],
        "authority": FALSE_AUTHORITY,
    }


def test_v2_explicit_contract_and_readonly_action():
    value = request()
    assert validate_morning_request(value) == value
    for changed in [
        {**value, "schema_version": "v1"},
        {**value, "historical": True},
        {**value, "output_ref": {"path": "report.md", "sha256": "a" * 64}},
    ]:
        with pytest.raises(ContractError):
            validate_morning_request(changed)


def test_held_symbols_cannot_be_removed_by_quote_policy():
    assert expected_quote_symbols(
        policy=policy(),
        holdings=[{"symbol": "000001.SZ", "shares": 100}, {"symbol": "000002.SZ", "shares": 0}],
        run_date="20260907",
    ) == ["000001.SZ", "600000.SH"]
    with pytest.raises(ContractError, match="NOT_EFFECTIVE"):
        expected_quote_symbols(policy=policy(), holdings=[], run_date="20261001")


@pytest.mark.parametrize("shares", [-1, "NaN", "Infinity", True])
def test_invalid_holding_amount_never_disappears_from_scope(shares):
    with pytest.raises(ContractError):
        expected_quote_symbols(
            policy=policy(),
            holdings=[{"symbol": "000001.SZ", "shares": shares}],
            run_date="20260907",
        )


def test_missing_future_calendar_cannot_infer_next_weekday():
    with pytest.raises(ContractError, match="COVERAGE_MISSING"):
        require_next_open_session(
            runtime_projection=[{"date": "2026-09-04", "status": "OPEN"}],
            previous_date="20260904",
            run_date="20260907",
        )


def test_next_open_session_requires_explicit_closed_intervening_dates():
    rows = [
        {"date": f"2026-09-{day:02d}", "status": "OPEN" if day in {4, 7} else "CLOSED"}
        for day in range(4, 8)
    ]
    require_next_open_session(
        runtime_projection=rows, previous_date="20260904", run_date="20260907"
    )
    rows[1]["status"] = "OPEN"
    with pytest.raises(ContractError, match="NOT_IMMEDIATELY"):
        require_next_open_session(
            runtime_projection=rows, previous_date="20260904", run_date="20260907"
        )
