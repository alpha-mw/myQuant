"""Closed legacy declaration grammar; synthetic values confer no owner authority."""

from copy import deepcopy

import pytest

from quant_investor.operations.daily_contract import ContractError
from quant_investor.strategy_records.risk_policy_contract import _initial_stop_row


def legacy_row(ordinary):
    """Keep fixture financial values; replace only the declared policy profile."""
    row = deepcopy(ordinary)
    row.pop("calibration")
    row.update(
        effective_entry_event="LEGACY_POSITION_OWNER_STOP_REVALIDATION_20260828",
        effective_entry_trade_date=None,
        entry_anchor_state="UNAVAILABLE_LEGACY_POSITION",
        initial_stop_state="CONFIRMED_OWNER_REVALIDATION",
        setting_authority="OWNER_EXPLICIT_CONFIRMATION",
        legacy_stop_revalidated=True,
        trailing_stop_state="UNAVAILABLE_MISSING_EFFECTIVE_ENTRY_ANCHOR",
        retired_unexecutable_trailing_stop_cny="163.51",
        sina_price_cny_at_20260828_143135="43.47",
    )
    return row


@pytest.fixture
def declaration():
    price = "35.32"
    return legacy_row(
        {
            "symbol": "605358.SH",
            "company_name": "Synthetic legacy holding",
            "current_shares": 100,
            "effective_entry_event": "SYNTHETIC_ENTRY",
            "effective_entry_trade_date": "20260821",
            "fee_inclusive_avg_cost_cny": "40.00",
            "initial_stop_state": "CONFIRMED",
            "stop_policy_ref": "synthetic-stop:605358.SH",
            "setting_authority": "OWNER_DELEGATED_TO_CODEX",
            "initial_stop_price_cny": price,
            "price_tick_cny": "0.01",
            "trigger": {
                "observation": "STRICT_CN_DAILY_CLOSE",
                "operator": "LESS_THAN_OR_EQUAL",
                "price_cny": price,
                "intraday_touch_only": "WARNING_NOT_BREACH",
                "confirmed_breach_state": "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW",
                "review_quantity_shares": 100,
                "automatic_order": False,
                "automatic_execution": False,
            },
            "calibration": {"synthetic": True},
            "add_policy": "NO_ADD_UNTIL_SEPARATE_I6_ELIGIBILITY_AND_NEW_ANCHOR_CONTRACT",
            "valid_until": "EARLIEST_OF_OWNER_REVISION_POSITION_REMOVAL_OR_NEW_ENTRY_ADD_ANCHOR",
        }
    )


def test_legacy_declaration_is_accepted_without_normalization(declaration):
    before = deepcopy(declaration)
    _initial_stop_row(declaration, "synthetic-stop")
    assert declaration == before
    assert declaration["effective_entry_trade_date"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("effective_entry_event", "OTHER_OWNER_EVENT_20260828"),
        ("effective_entry_event", "LEGACY_POSITION_OWNER_STOP_REVALIDATION_20260829"),
        ("effective_entry_trade_date", "20260828"),
        ("entry_anchor_state", "CONFIRMED"),
        ("initial_stop_state", "CONFIRMED"),
        ("setting_authority", "OWNER_DELEGATED_TO_CODEX"),
        ("legacy_stop_revalidated", False),
        ("legacy_stop_revalidated", 1),
        ("trailing_stop_state", "CONFIRMED"),
        ("retired_unexecutable_trailing_stop_cny", "NaN"),
        ("retired_unexecutable_trailing_stop_cny", "Infinity"),
        ("retired_unexecutable_trailing_stop_cny", "0"),
        ("sina_price_cny_at_20260828_143135", "NaN"),
        ("sina_price_cny_at_20260828_143135", "-1"),
        ("current_shares", "1.5"),
        ("fee_inclusive_avg_cost_cny", "0"),
        ("initial_stop_price_cny", "35.321"),
        ("price_tick_cny", "0.1"),
        ("stop_policy_ref", "foreign:605358.SH"),
        ("calibration", {}),
        ("unknown", False),
    ],
)
def test_legacy_profile_drift_rejects(declaration, field, value):
    declaration[field] = value
    with pytest.raises(ContractError):
        _initial_stop_row(declaration, "synthetic-stop")


@pytest.mark.parametrize("field", ["legacy_stop_revalidated", "trailing_stop_state"])
def test_incomplete_legacy_profile_rejects(declaration, field):
    del declaration[field]
    with pytest.raises(ContractError):
        _initial_stop_row(declaration, "synthetic-stop")


@pytest.mark.parametrize(
    "field,value",
    [
        ("automatic_order", True),
        ("automatic_execution", True),
        ("price_cny", "43.47"),
        ("review_quantity_shares", 200),
    ],
)
def test_legacy_trigger_drift_rejects(declaration, field, value):
    declaration["trigger"][field] = value
    with pytest.raises(ContractError):
        _initial_stop_row(declaration, "synthetic-stop")
