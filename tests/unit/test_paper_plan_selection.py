"""Paper planning inputs: superseded plans and the account's own NAV history."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from quant_investor.paper.planning import outstanding_orders, value_nav_series


def _plan(root: Path, day: str, stamp: str, due: str, orders: list[dict]) -> None:
    path = root / "data/private/paper_shadow" / day / stamp / "plans.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "paper-session-plans.v1",
                "signal_date": "20260930",
                "eligible_from_trade_date": due,
                "orders": orders,
            }
        )
    )


def _order(identity: str, symbol: str = "600000.SH") -> dict:
    return {
        "source_intent_id": identity,
        "symbol": symbol,
        "side": "SELL",
        "action": "REDUCE_50",
        "shares": 100,
        "eligible_from_trade_date": "20261008",
        "signal_date": "20260930",
        "reason_codes": [],
        "requested_ratio": "0.50",
    }


def test_newest_plan_supersedes_older_snapshots_for_the_same_session(tmp_path):
    _plan(
        tmp_path,
        "2026-10-05",
        "10:00:00",
        "20261008",
        [_order("intent-a"), _order("intent-dropped", "600287.SH")],
    )
    _plan(
        tmp_path,
        "2026-10-06",
        "00:10:00",
        "20261008",
        [_order("intent-a"), _order("intent-kept", "603648.SH")],
    )
    items = outstanding_orders(workspace=tmp_path, applied=set())
    assert [item["order"]["source_intent_id"] for item in items] == ["intent-a", "intent-kept"]
    assert items[0]["due"] == "20261008"


def test_plans_for_different_sessions_coexist_and_applied_orders_drop(tmp_path):
    _plan(tmp_path, "2026-10-05", "10:00:00", "20261008", [_order("intent-a")])
    _plan(tmp_path, "2026-10-06", "00:10:00", "20261009", [_order("intent-b")])
    items = outstanding_orders(workspace=tmp_path, applied={"intent-a"})
    assert [item["order"]["source_intent_id"] for item in items] == ["intent-b"]
    assert [item["due"] for item in items] == ["20261009"]


def test_value_nav_series_reports_drawdown_between_strict_closes():
    records = [
        {"trade_date": "00000000", "cash": Decimal("0"), "positions": {}},
        {"trade_date": "20260929", "cash": Decimal("1000"), "positions": {"600000.SH": 100}},
    ]
    closes = {
        "600000.SH": {
            "20260929": Decimal("10"),
            "20260930": Decimal("8"),
            "20261008": Decimal("12"),
        }
    }
    result = value_nav_series(
        records=records,
        sessions=["20260929", "20260930", "20261008"],
        closes=closes,
    )
    assert result["status"] == "OK"
    assert [row["nav_cny"] for row in result["sessions"]] == [
        "2000.0000",
        "1800.0000",
        "2200.0000",
    ]
    assert result["max_drawdown_fraction"] == "0.100000"
    assert result["current_drawdown_fraction"] == "0.000000"
    assert result["trough_trade_date"] == "20260930"
    assert result["peak_trade_date"] == "20261008"


def test_value_nav_series_carries_a_suspended_close_and_flags_partial():
    records = [{"trade_date": "20260929", "cash": Decimal("0"), "positions": {"600000.SH": 10}}]
    closes = {"600000.SH": {"20260929": Decimal("10")}}
    result = value_nav_series(records=records, sessions=["20260929", "20260930"], closes=closes)
    assert result["status"] == "PARTIAL"
    assert result["partial_sessions"] == ["20260930"]
    assert result["sessions"][1]["partial"] is True
    assert result["sessions"][1]["nav_cny"] == "100.0000"


def test_value_nav_series_stops_when_a_held_symbol_never_has_a_close():
    records = [
        {"trade_date": "20260929", "cash": Decimal("0"), "positions": {"600000.SH": 10}},
        {
            "trade_date": "20260930",
            "cash": Decimal("0"),
            "positions": {"600000.SH": 10, "600001.SH": 5},
        },
    ]
    closes = {"600000.SH": {"20260929": Decimal("10"), "20260930": Decimal("10")}}
    result = value_nav_series(
        records=records, sessions=["20260929", "20260930", "20261008"], closes=closes
    )
    assert result["status"] == "PARTIAL"
    assert result["stopped_at"] == {
        "trade_date": "20260930",
        "symbol_without_close": "600001.SH",
    }
    assert result["observations"] == 1
    assert result["max_drawdown_fraction"] is None


def test_value_nav_series_refuses_to_claim_a_drawdown_from_one_point():
    records = [{"trade_date": "20260929", "cash": Decimal("10"), "positions": {}}]
    result = value_nav_series(records=records, sessions=["20260929"], closes={})
    assert result["status"] == "INSUFFICIENT_HISTORY"
    assert result["reason"] == "SINGLE_VALUED_SESSION"
    assert result["max_drawdown_fraction"] is None
