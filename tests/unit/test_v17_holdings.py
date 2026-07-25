from __future__ import annotations

import pytest

from quant_investor.v17.contracts import V17ContractError
from quant_investor.v17.holdings import (
    HoldingsSnapshot,
    build_available_holdings_snapshot,
    build_unavailable_holdings_snapshot,
    validate_holdings_snapshot,
)
from quant_investor.v17.semantic import seal_semantic


def test_available_holdings_require_positive_nav_and_reconcile() -> None:
    payload = build_available_holdings_snapshot(
        snapshot_id="holdings-20260721",
        strategy_id="cn-shadow",
        market="CN",
        pit_cutoff="2026-07-21",
        as_of="2026-07-21T15:00:00Z",
        nav=1_000_000.0,
        cash=200_000.0,
        declared_all_cash=False,
        positions=[
            {"symbol": "600000.SH", "market_value": 500_000.0},
            {"symbol": "000001.SZ", "market_value": 300_000.0},
        ],
    )
    view = HoldingsSnapshot.from_payload(payload, cutoff="2026-07-22")
    assert view.held_symbols == frozenset({"600000.SH", "000001.SZ"})

    invalid = {k: v for k, v in payload.items() if k != "semantic_sha256"}
    invalid["cash"] = 100_000.0
    with pytest.raises(V17ContractError, match="reconcile"):
        validate_holdings_snapshot(seal_semantic(invalid), cutoff=None)


def test_all_cash_must_be_explicit_and_exact() -> None:
    payload = build_available_holdings_snapshot(
        snapshot_id="holdings-cash",
        strategy_id="cn-shadow",
        market="CN",
        pit_cutoff="2026-07-21",
        as_of="2026-07-21T15:00:00Z",
        nav=1_000_000.0,
        cash=1_000_000.0,
        declared_all_cash=True,
        positions=[],
    )
    assert validate_holdings_snapshot(payload, cutoff=None) == payload
    with pytest.raises(V17ContractError, match="explicit all-cash"):
        build_available_holdings_snapshot(
            snapshot_id="holdings-ambiguous",
            strategy_id="cn-shadow",
            market="CN",
            pit_cutoff="2026-07-21",
            as_of="2026-07-21T15:00:00Z",
            nav=1_000_000.0,
            cash=1_000_000.0,
            declared_all_cash=False,
            positions=[],
        )


def test_unavailable_holdings_cannot_carry_nav_or_positions() -> None:
    payload = build_unavailable_holdings_snapshot(
        snapshot_id="holdings-missing",
        strategy_id="cn-shadow",
        market="CN",
        reason="holdings_not_provided",
    )
    unavailable = HoldingsSnapshot.from_payload(payload, cutoff=None)
    with pytest.raises(V17ContractError, match="cannot be interpreted"):
        _ = unavailable.held_symbols
    injected = {k: v for k, v in payload.items() if k != "semantic_sha256"}
    injected.update({"nav": 1.0, "positions": []})
    with pytest.raises(V17ContractError, match="shape mismatch"):
        validate_holdings_snapshot(seal_semantic(injected), cutoff=None)
