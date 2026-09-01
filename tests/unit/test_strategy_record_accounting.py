from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from quant_investor.strategy_records.store import canonical_json_bytes
from quant_investor.strategy_records.accounting import (
    PROSPECTIVE_OPENING_BALANCE,
    StrategyAccountingError,
    apply_fifo_events,
    build_daily_close,
    build_genesis,
    effective_fills,
    validate_genesis,
)

SHA = "a" * 64


def _opening() -> list[dict]:
    return [
        {
            "lot_id": "opening-000001-sz",
            "symbol": "000001.SZ",
            "remaining_shares": 100,
            "unit_cost_cny": "10.0000",
            "origin": PROSPECTIVE_OPENING_BALANCE,
        }
    ]


def _fill(
    event_id: str,
    *,
    side: str,
    shares: int,
    price: str,
    fee_status: str = "KNOWN",
    fee: str | None = "0.0000",
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "FILL",
        "corrects_event_id": None,
        "trade_date": "2026-09-01",
        "symbol": "000001.SZ",
        "side": side,
        "shares": shares,
        "price_cny": price,
        "fee_status": fee_status,
        "total_fee_cny": fee,
    }


def test_fifo_partial_sell_and_full_exit_are_deterministic() -> None:
    events = [
        _fill("fill-buy", side="BUY", shares=100, price="20.0000", fee="10.0000"),
        _fill("fill-sell-one", side="SELL", shares=150, price="30.0000", fee="15.0000"),
        _fill("fill-sell-two", side="SELL", shares=50, price="25.0000", fee="5.0000"),
    ]
    result = apply_fifo_events(_opening(), events)
    assert result["remaining_lots"] == []
    first = result["realized_rows"][0]
    assert first["consumed_lots"] == [
        {"lot_id": "opening-000001-sz", "shares": 100},
        {"lot_id": "lot-fill-buy", "shares": 50},
    ]
    assert first["cost_basis_cny"] == "2005.0000"
    assert first["gross_realized_pnl_cny"] == "2495.0000"
    assert first["net_realized_pnl_cny"] == "2480.0000"
    assert result["realized_rows"][1]["net_realized_pnl_cny"] == "240.0000"


def test_unknown_fee_propagates_null_instead_of_zero() -> None:
    result = apply_fifo_events(
        _opening(),
        [
            _fill(
                "fill-sell",
                side="SELL",
                shares=100,
                price="12.0000",
                fee_status="EVIDENCE_UNAVAILABLE",
                fee=None,
            )
        ],
    )
    assert result["fees_complete"] is False
    assert result["realized_rows"][0]["gross_realized_pnl_cny"] == "200.0000"
    assert result["realized_rows"][0]["net_realized_pnl_cny"] is None


def test_duplicate_fill_correction_and_oversell_fail_closed() -> None:
    duplicate = _fill("fill-one", side="SELL", shares=10, price="12.0000")
    with pytest.raises(StrategyAccountingError, match="duplicate"):
        effective_fills([duplicate, duplicate])
    corrected = effective_fills(
        [
            duplicate,
            {
                "event_id": "void-one",
                "event_type": "CORRECTION_VOID",
                "corrects_event_id": "fill-one",
            },
        ]
    )
    assert corrected == []
    with pytest.raises(StrategyAccountingError, match="exceeds"):
        apply_fifo_events(
            _opening(),
            [_fill("fill-too-large", side="SELL", shares=101, price="12.0000")],
        )


def test_daily_close_requires_cash_nav_and_pnl_reconciliation() -> None:
    close = build_daily_close(
        trade_date="2026-09-01",
        opening_cash_cny="1000.0000",
        opening_nav_cny="2000.0000",
        opening_realized_pnl_cny="0.0000",
        opening_unrealized_pnl_cny="0.0000",
        closing_cash_cny="199.0000",
        closing_market_value_cny="2021.0000",
        closing_realized_pnl_cny="0.0000",
        closing_unrealized_pnl_cny="220.0000",
        external_flow_cny="0.0000",
        other_pnl_cny="0.0000",
        events=[_fill("fill-buy", side="BUY", shares=80, price="10.0000", fee="1.0000")],
    )
    assert close["status"] == "VERIFIED"
    assert close["closing_nav_cny"] == "2220.0000"
    assert close["daily_pnl_cny"] == close["pnl_bridge_cny"] == "220.0000"

    blocked = build_daily_close(
        trade_date="2026-09-01",
        opening_cash_cny="1000.0000",
        opening_nav_cny="2000.0000",
        opening_realized_pnl_cny="0.0000",
        opening_unrealized_pnl_cny="0.0000",
        closing_cash_cny="200.0000",
        closing_market_value_cny="2021.0000",
        closing_realized_pnl_cny=None,
        closing_unrealized_pnl_cny="220.0000",
        external_flow_cny="0.0000",
        other_pnl_cny="0.0000",
        events=[
            _fill(
                "fill-buy",
                side="BUY",
                shares=80,
                price="10.0000",
                fee_status="EVIDENCE_UNAVAILABLE",
                fee=None,
            )
        ],
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["fees_cny"] is None
    assert blocked["pnl_bridge_cny"] is None
    assert blocked["blockers"] == [
        "FEE_EVIDENCE_UNAVAILABLE",
        "NET_REALIZED_PNL_UNAVAILABLE",
    ]


def test_genesis_is_derived_only_and_cannot_overclaim_history() -> None:
    document = build_genesis(
        generation_id="accounting-cutover-20260831-a1",
        created_at="2026-09-01T06:30:00Z",
        strategy_label="aggressive_tech_manufacturing",
        effective_date="2026-08-31",
        source_store={
            "pointer_sha256": SHA,
            "catalog_generation_id": "g-source",
            "catalog_sha256": SHA,
            "performance_generation_id": "p-source",
            "performance_manifest_sha256": SHA,
            "performance_series_sha256": SHA,
            "active_record_id": "20260901_121818-b06",
            "active_ledger_sha256": SHA,
        },
        source_refs=[{"path": "records/current.json", "sha256": SHA}],
        cash_cny="900.0000",
        nav_cny="2000.0000",
        portfolio_pnl_cny="1000.0000",
        positions=[
            {
                "symbol": "000001.SZ",
                "name": "synthetic",
                "shares": 100,
                "avg_cost_cny": "10.0000",
                "cost_basis_cny": "1000.0000",
                "market_value_cny": "1100.0000",
            }
        ],
        historical_audit_ref={"path": "audit.json", "sha256": SHA},
        industry_rows=[{"symbol": "000001.SZ", "industry_l1": "synthetic"}],
        theme_rows=[
            {
                "symbol": "000001.SZ",
                "theme_id": "OTHER_UNCLASSIFIED",
                "weight": "1",
                "confidence": "UNVERIFIED",
            }
        ],
    )
    validated = validate_genesis(document)
    assert validated["derived_only"] is True
    assert validated["coverage"] == {
        "historical": "PARTIAL",
        "prospective": "READY",
        "prospective_effective_date": "2026-08-31",
    }
    assert validated["status"] == {
        "data": "VERIFIED",
        "accounting": "PARTIAL",
        "attribution": "PARTIAL",
        "evidence": "PARTIAL",
    }
    assert Decimal(validated["historical_unallocated_pnl_cny"]) == Decimal("900.0000")

    tampered = dict(document)
    tampered["status"] = {**document["status"], "accounting": "VERIFIED"}
    with pytest.raises(StrategyAccountingError, match="content SHA"):
        validate_genesis(tampered)


def test_genesis_rejects_incomplete_industry_and_theme_weights() -> None:
    kwargs = {
        "generation_id": "accounting-cutover-20260831-a1",
        "created_at": "2026-09-01T06:30:00Z",
        "strategy_label": "aggressive_tech_manufacturing",
        "effective_date": "2026-08-31",
        "source_store": {
            "pointer_sha256": SHA,
            "catalog_generation_id": "g-source",
            "catalog_sha256": SHA,
            "performance_generation_id": "p-source",
            "performance_manifest_sha256": SHA,
            "performance_series_sha256": SHA,
            "active_record_id": "20260901_121818-b06",
            "active_ledger_sha256": SHA,
        },
        "source_refs": [{"path": "records/current.json", "sha256": SHA}],
        "cash_cny": "900.0000",
        "nav_cny": "2000.0000",
        "portfolio_pnl_cny": "1000.0000",
        "positions": [
            {
                "symbol": "000001.SZ",
                "name": "synthetic",
                "shares": 100,
                "avg_cost_cny": "10.0000",
                "cost_basis_cny": "1000.0000",
                "market_value_cny": "1100.0000",
            }
        ],
        "historical_audit_ref": {"path": "audit.json", "sha256": SHA},
        "industry_rows": [],
        "theme_rows": [],
    }
    with pytest.raises(StrategyAccountingError, match="Industry coverage"):
        build_genesis(**kwargs)
    kwargs["industry_rows"] = [{"symbol": "000001.SZ", "industry_l1": "synthetic"}]
    kwargs["theme_rows"] = [
        {"symbol": "000001.SZ", "theme_id": "OTHER_UNCLASSIFIED", "weight": "0.9"}
    ]
    with pytest.raises(StrategyAccountingError, match="sum to one"):
        build_genesis(**kwargs)


def test_immutable_write_is_exact_replay_and_conflict(tmp_path: Path) -> None:
    from quant_investor.strategy_records.accounting import immutable_write

    target = tmp_path / "generation" / "genesis.json"
    assert immutable_write(target, b"one\n") == immutable_write(target, b"one\n")
    with pytest.raises(StrategyAccountingError, match="conflicts"):
        immutable_write(target, b"two\n")


def test_accounting_pointer_cas_retains_history_and_rejects_stale_writer(
    tmp_path: Path,
) -> None:
    from quant_investor.strategy_records.accounting import seal_document
    from scripts.prepare_cn_strategy_accounting import _publish_pointer

    pointer = tmp_path / "_accounting_store" / "current.v1.json"
    first = seal_document({"schema_id": "test", "generation_id": "one"})
    second = seal_document({"schema_id": "test", "generation_id": "two"})
    first_sha = _publish_pointer(pointer, first, expected=None)
    assert _publish_pointer(pointer, first, expected=first_sha) == first_sha
    second_sha = _publish_pointer(pointer, second, expected=first_sha)
    assert second_sha != first_sha
    assert (
        pointer.parent / "pointer_history" / f"{first_sha}.json"
    ).read_bytes() == canonical_json_bytes(first)
    with pytest.raises(StrategyAccountingError, match="preimage"):
        _publish_pointer(pointer, first, expected=first_sha)
