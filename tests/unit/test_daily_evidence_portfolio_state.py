"""Actual source/custody times and accounting identity for advisory portfolio context."""

from copy import deepcopy
from decimal import Decimal

import pytest

from quant_investor.intelligence.portfolio_state import (
    build_portfolio_state,
    normalize_positions,
    portfolio_timing,
)

CUTOFF = "2026-08-28T13:30:00Z"


def timing(**overrides):
    values = dict(
        as_of=CUTOFF,
        effective_date="20260827",
        sealed_at="2026-08-27T13:30:00Z",
        published_at="2026-08-27T13:31:00Z",
        created_at="2026-08-28T13:00:00Z",
    )
    return portfolio_timing(**{**values, **overrides})


def test_cutoff_equality_is_on_time_without_standalone_prospective_authority():
    assert timing(sealed_at=CUTOFF, published_at=CUTOFF, created_at=CUTOFF) == {
        "timing_status": "ON_TIME",
        "prospective": False,
        "reason_codes": [],
    }


@pytest.mark.parametrize(
    "updates,codes",
    [
        ({"created_at": "2026-08-28T13:30:01Z"}, ["PORTFOLIO_CUSTODY_AFTER_DECISION"]),
        (
            {"published_at": "2026-08-28T13:30:01Z", "created_at": "2026-08-28T13:31:00Z"},
            ["PORTFOLIO_CUSTODY_AFTER_DECISION", "PORTFOLIO_SOURCE_NOT_AVAILABLE_AT_DECISION"],
        ),
        (
            {
                "sealed_at": "2026-09-09T03:23:12Z",
                "published_at": "2026-09-09T03:23:12Z",
                "created_at": "2026-09-12T01:00:00Z",
            },
            ["PORTFOLIO_CUSTODY_AFTER_DECISION", "PORTFOLIO_SOURCE_NOT_AVAILABLE_AT_DECISION"],
        ),
    ],
)
def test_late_source_and_custody_are_retrospective_context(updates, codes):
    assert timing(**updates) == {
        "timing_status": "LATE_RECORDED",
        "prospective": False,
        "reason_codes": codes,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"effective_date": "20260828"},
        {"effective_date": "20260829"},
        {"sealed_at": "2026-08-28T13:30:01Z"},  # Seal cannot follow publication.
        {"published_at": "2026-08-28T13:30:01Z"},  # Publication cannot follow custody.
        {"sealed_at": None},
        {"published_at": "yesterday"},
        {"created_at": "2026-08-28T13:00:00"},
    ],
)
def test_future_economic_state_or_invalid_source_order_rejects(updates):
    with pytest.raises(ValueError):
        timing(**updates)


def position(**updates):
    return {
        "symbol": "000001.SZ",
        "shares": Decimal("100"),
        "avg_cost": Decimal("10"),
        "cost_basis": Decimal("1000"),
        **updates,
    }


def test_zero_shares_retained_and_native_cost_rounding_is_reused():
    rows = normalize_positions([position(), position(symbol="000002.SZ", shares=0, cost_basis=0)])
    assert rows[1]["shares"] == "0" and len(rows) == 2
    result = normalize_positions([position(avg_cost="10.000000004", cost_basis="1000.00001")])
    assert (
        result[0]["avg_cost"] == "10.000000004"
    )  # Preserve source precision, validate native units.


@pytest.mark.parametrize(
    "changes",
    [
        {"shares": -1},
        {"avg_cost": -1},
        {"cost_basis": -1},
        {"shares": True},
        {"shares": "Infinity"},
        {"avg_cost": "NaN"},
        {"cost_basis": "1001"},
    ],
)
def test_invalid_position_identity_rejects(changes):
    with pytest.raises(ValueError):
        normalize_positions([position(**changes)])
    with pytest.raises(ValueError, match="DUPLICATED"):
        normalize_positions([position(), position()])


def test_portfolio_artifact_has_actual_creation_time_and_immutable_native_refs():
    def ref(name):
        return {"path": name + ".json", "sha256": "a" * 64}

    kwargs = dict(
        as_of=CUTOFF,
        created_at="2026-09-12T01:00:00Z",
        store_plan_ref=ref("plan"),
        frozen_pointer_ref=ref("pointer"),
        catalog_ref=ref("catalog"),
        source_record_id="record",
        source_effective_trade_date="20260827",
        source_sealed_at="2026-09-09T03:23:12Z",
        pointer_published_at="2026-09-09T03:23:12Z",
        source_refs=[ref("ledger"), ref("manual")],
        positions=[position()],
        cash=Decimal("958000"),
    )
    original = deepcopy(kwargs)
    artifact = build_portfolio_state(**kwargs)
    assert kwargs == original
    assert artifact["created_at"] == kwargs["created_at"]
    assert artifact["payload"]["as_of"] == CUTOFF
    assert artifact["payload"]["timing_status"] == "LATE_RECORDED"
    assert artifact["payload"]["prospective"] is False
    assert all(value is False for value in artifact["payload"]["authority"].values())
    assert build_portfolio_state(**kwargs) == artifact
