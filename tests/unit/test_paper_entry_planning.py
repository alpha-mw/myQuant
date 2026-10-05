"""Buy planning: cadence, caps and candidate ordering under policy v3."""

from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "paper_shadow_orders", ROOT / "scripts/operations/paper_shadow_orders.py"
)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


def _views(symbols: list[str]) -> list[dict]:
    return [{"symbol": symbol} for symbol in symbols]


POOL = [
    {"symbol": "600287.SH", "combined_percentile": "0.975438924169"},
    {"symbol": "603877.SH", "combined_percentile": "0.975345536048"},
    {"symbol": "688285.SH", "combined_percentile": "0.974598431080"},
    {"symbol": "603180.SH", "combined_percentile": "0.850000000000"},
]


def test_buys_land_on_the_first_session_of_a_week() -> None:
    orders = plan.entry_orders(
        account_views=_views(["002008.SZ"]),
        account={},
        session="20261008",
        previous_session="20260930",
        pool_rows=POOL,
    )
    assert [order["symbol"] for order in orders] == ["600287.SH", "603877.SH"]
    assert all(order["side"] == "BUY" and order["action"] == "ENTRY" for order in orders)
    assert orders[0]["target_weight"] == "0.14"
    assert orders[0]["minimum_cash_fraction"] == "0.05"


def test_no_buys_mid_week() -> None:
    orders = plan.entry_orders(
        account_views=_views(["002008.SZ"]),
        account={},
        session="20261009",
        previous_session="20261008",
        pool_rows=POOL,
    )
    assert orders == []


def test_holdings_cap_blocks_new_names() -> None:
    held = [f"00000{index}.SZ" for index in range(7)]
    orders = plan.entry_orders(
        account_views=_views(held),
        account={},
        session="20261008",
        previous_session="20260930",
        pool_rows=POOL,
    )
    assert orders == []


def test_existing_holdings_and_weak_candidates_are_excluded() -> None:
    orders = plan.entry_orders(
        account_views=_views(["600287.SH"]),
        account={},
        session="20261008",
        previous_session="20260930",
        pool_rows=POOL,
    )
    # 600287 is held and 603180 is below the 0.90 gate, so only two remain.
    assert [order["symbol"] for order in orders] == ["603877.SH", "688285.SH"]


def test_at_most_two_new_names_per_week() -> None:
    pool = [{"symbol": f"60028{index}.SH", "combined_percentile": "0.99"} for index in range(9)]
    orders = plan.entry_orders(
        account_views=_views([]),
        account={},
        session="20261008",
        previous_session="20260930",
        pool_rows=pool,
    )
    assert len(orders) == 2
