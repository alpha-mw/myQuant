"""Historical CSV evidence cannot determine prospective positions or Store writes."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import prepare_cn_strategy_accounting as accounting


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_prepare_preserves_store_and_uses_active_parquet_for_opening_lots(tmp_path, monkeypatch):
    store = tmp_path / "results/strategy_records/CN/synthetic"
    active = store / "seed"
    active.mkdir(parents=True)
    (active / "ledger.csv").write_text("symbol,shares\n000001.SZ,1000\n")
    (active / "manifest.json").write_text('{"action_taken_today":true}')
    # Rejected manual intent must prevent an orders.csv row becoming a historical fill.
    (active / "manual_switch_and_take_profit_orders.csv").write_text(
        "symbol,shares,price,side,status\n000001.SZ,900,10,SELL,formal_order_rejected_no_manual_fill\n"
    )
    (active / "orders.csv").write_text("symbol,shares,price,side\n000001.SZ,900,10,SELL\n")
    ledger = active / "ledger_after_manual_switch.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "name": "synthetic",
                "shares": 100,
                "avg_cost": "10",
                "cost_basis": "1000",
                "current_value": "1100",
                "price_date": "2026-08-31",
            }
        ]
    ).to_parquet(ledger, index=False)
    record = {
        "record_id": "seed",
        "relative_path": "seed",
        "storage_state": "ONLINE",
        "inventory": [
            {
                "path": p.name,
                "type": "file",
                "size": p.stat().st_size,
                "sha256": _digest(p.read_bytes()),
            }
            for p in active.iterdir()
        ],
    }
    catalog = {
        "records": [record],
        "lineage_index": [{"record_id": "seed", "valuation_date": "2026-08-31"}],
        "performance_history_ref": {
            "manifest": {"path": "performance.json", "sha256": "a" * 64},
            "series": {"path": "performance.parquet", "sha256": "b" * 64},
        },
    }
    catalog_raw = accounting.canonical_json_bytes(catalog)
    (store / "catalog.json").write_bytes(catalog_raw)
    pointer = {
        "generation_id": "catalog-seed",
        "catalog_path": "catalog.json",
        "catalog_sha256": _digest(catalog_raw),
        "active_record_id": "seed",
        "active_closure": {"ledger_sha256": _digest(ledger.read_bytes())},
    }
    pointer_path = store / "_record_store/current.v1.json"
    pointer_path.parent.mkdir()
    pointer_path.write_bytes(accounting.canonical_json_bytes(pointer))
    monkeypatch.setattr(accounting, "load_registered_catalog", lambda root: (pointer, catalog))
    monkeypatch.setattr(
        accounting,
        "load_performance_history",
        lambda *args: {
            "manifest": {"performance_generation_id": "performance-seed"},
            "rows": [
                {
                    "record_id": "seed",
                    "valuation_date": "2026-08-31",
                    "cash_cny": "900",
                    "raw_nav_cny": "2000",
                    "portfolio_pnl_cny": "1000",
                }
            ],
        },
    )
    monkeypatch.setattr(
        accounting,
        "_industry_rows",
        lambda **kwargs: ([{"symbol": "000001.SZ", "industry_l1": "synthetic"}], []),
    )
    before = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    result = accounting.prepare(
        Namespace(
            project_root=str(tmp_path),
            record_root=str(store),
            expected_store_pointer_sha=_digest(pointer_path.read_bytes()),
            industry_capture="synthetic-industry.json",
            industry_capture_sha="c" * 64,
            execute=True,
            expected_accounting_pointer_sha="ABSENT",
        )
    )
    assert result["status"] == "PUBLISHED"
    assert result["reported_fill_count"] == 0
    assert result["unexplained_share_delta_count"] == 1
    assert result["historical_status"] == "PARTIAL"
    generation = store / "_accounting_store/generations" / result["generation_id"]
    genesis = json.loads((generation / "genesis.v1.json").read_bytes())
    audit = json.loads((generation / "historical-gap-audit.v1.json").read_bytes())
    assert genesis["opening_lots"][0]["remaining_shares"] == 100
    assert audit["unexplained_share_deltas"][0]["reported_path_shares"] == 1000
    assert audit["prospective_lot_authority"] is False
    assert genesis["derived_only"] is True
    after = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    assert {p: after[p] for p in before} == before
    assert all(p.parts[0] == "_accounting_store" for p in after.keys() - before.keys())
    for capability in ("provider_calls", "broker_calls", "order_calls", "trade_calls"):
        assert result[capability] is False


def test_historical_member_tamper_fails_before_csv_decode(tmp_path):
    path = tmp_path / "record/ledger.csv"
    path.parent.mkdir()
    raw = b"symbol,shares\n000001.SZ,100\n"
    path.write_bytes(raw.replace(b"100", b"900"))
    reader = accounting.RecordReader(
        project=tmp_path,
        record_root=tmp_path,
        records=[
            {
                "record_id": "record",
                "relative_path": "record",
                "storage_state": "ONLINE",
                "inventory": [
                    {"path": "ledger.csv", "type": "file", "size": len(raw), "sha256": _digest(raw)}
                ],
            }
        ],
    )
    with pytest.raises(accounting.StrategyAccountingError, match="catalog inventory"):
        reader.read("record", "ledger.csv")
