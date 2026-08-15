from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import close_cn_dashboard_official_valuation as valuation  # noqa: E402
from cn_dashboard_common import canonical_json_bytes  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    record_root = tmp_path / "records"
    source_dir = record_root / "20260810_2157"
    source_dir.mkdir(parents=True)
    rows = []
    for index, symbol in enumerate(valuation.STOCK_CODES, start=1):
        shares = index * 100
        price = 10.0 + index
        cost_basis = shares * (price - 1.0)
        rows.append(
            {
                "symbol": symbol,
                "name": f"company-{index}",
                "shares": shares,
                "avg_cost": price - 1.0,
                "cost_basis": cost_basis,
                "current_price": price,
                "current_value": shares * price,
                "unrealized_pnl": shares,
                "unrealized_pnl_pct": shares / cost_basis,
            }
        )
    ledger = pd.DataFrame(rows)
    ledger_path = source_dir / "ledger_after_manual_switch.parquet"
    ledger.to_parquet(ledger_path, index=False)
    ledger_sha = _sha(ledger_path)
    market_value = float(ledger["current_value"].sum())
    cash = 1_100_000.0
    total_value = cash + market_value
    portfolio_pnl = total_value - valuation.CAPITAL_CNY
    financial_state = {
        "capital_cny": valuation.CAPITAL_CNY,
        "cash_after": cash,
        "market_value_after": market_value,
        "total_value_after": total_value,
        "portfolio_pnl_after": portfolio_pnl,
        "portfolio_return_after": portfolio_pnl / valuation.CAPITAL_CNY,
        "ledger_sha256": ledger_sha,
    }
    financial_state_sha = hashlib.sha256(
        canonical_json_bytes(financial_state)
    ).hexdigest()
    manual = {
        "capital_cny": valuation.CAPITAL_CNY,
        "cash_after": cash,
        "market_value_after": market_value,
        "total_value_after": total_value,
        "portfolio_pnl_after": portfolio_pnl,
        "portfolio_return_after": portfolio_pnl / valuation.CAPITAL_CNY,
        "effective_manual_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_sha256": ledger_sha,
        "ledger_after_manual_switch_parquet_sha256": ledger_sha,
        "no_trade_performed": True,
        "applied_local_trades": [],
        "applied_owner_declared_trades": [],
        "rejected_or_pending_trades": [],
        "funding_events": [],
        "net_external_flow": 0.0,
        "excluded_external_flow": 0.0,
        "gross_trade_value": 0.0,
        "financial_state": financial_state,
        "financial_state_sha256": financial_state_sha,
    }
    manual_path = source_dir / "manual_execution_manifest.json"
    _write_json(manual_path, manual)
    pnl_path = source_dir / "pnl_summary.csv"
    _write_csv(
        pnl_path,
        [
            {
                "cash_after": cash,
                "market_value_after": market_value,
                "total_value_after": total_value,
                "portfolio_pnl_after": portfolio_pnl,
            }
        ],
    )
    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": source_dir.name,
        "action_taken_today": False,
        "trade_count": 0,
        "order_count": 0,
        "fill_count": 0,
        "manual_execution": manual,
    }
    manifest_path = source_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    (source_dir / "orders.csv").write_text(
        "timestamp,action,symbol,name,shares,price\n", encoding="utf-8"
    )
    (source_dir / "manual_switch_and_take_profit_orders.csv").write_text(
        "timestamp,action,symbol,name,shares,execution_price\n",
        encoding="utf-8",
    )
    closure = {
        "record_id": source_dir.name,
        "relative_path": source_dir.name,
        "manifest_path": f"{source_dir.name}/manifest.json",
        "manifest_sha256": _sha(manifest_path),
        "manual_manifest_path": (
            f"{source_dir.name}/manual_execution_manifest.json"
        ),
        "manual_manifest_sha256": _sha(manual_path),
        "ledger_path": (
            f"{source_dir.name}/ledger_after_manual_switch.parquet"
        ),
        "ledger_sha256": ledger_sha,
        "pnl_path": f"{source_dir.name}/pnl_summary.csv",
        "pnl_sha256": _sha(pnl_path),
        "financial_state_sha256": financial_state_sha,
    }
    return record_root, source_dir, closure


def _evidence() -> dict:
    return {
        "schema_version": valuation.EVIDENCE_SCHEMA,
        "provider": "tushare.pro",
        "stock_api": "daily",
        "index_api": "index_daily",
        "trade_date": "20260814",
        "coverage": "exact_close",
        "previous_trading_day_ffill": False,
        "stocks": [
            {
                "ts_code": symbol,
                "trade_date": "20260814",
                "close": 20.0 + index,
            }
            for index, symbol in enumerate(valuation.STOCK_CODES)
        ],
        "indices": [
            {
                "ts_code": symbol,
                "trade_date": "20260814",
                "close": 1000.0 + index,
            }
            for index, symbol in enumerate(valuation.INDEX_CODES)
        ],
    }


def test_build_record_is_historical_only_and_no_trade(tmp_path: Path) -> None:
    record_root, source_dir, closure = _source_fixture(tmp_path)
    staging_dir = record_root / "20260814_1815"
    staging_dir.mkdir()

    summary = valuation.build_record(
        staging_dir=staging_dir,
        record_root=record_root,
        source_dir=source_dir,
        registered_closure=closure,
        record_id=staging_dir.name,
        trade_date="20260814",
        recorded_at_iso="2026-08-14T18:15:00+08:00",
        evidence=_evidence(),
    )

    manifest = json.loads(
        (staging_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manual = json.loads(
        (staging_dir / "manual_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "canonical_v17_holdings" not in manifest
    assert "CANONICAL_V17_" not in serialized
    assert manifest["historical_lane"] == valuation.HISTORICAL_HOLDINGS_LANE
    assert manifest["historical_holdings_storage_authority"] is True
    assert manifest["v17_mainline_authority"] is False
    assert manifest["broker_order_trade_authority"] is False
    assert manual["v17_mainline_authority"] is False
    assert manual["broker_order_trade_authority"] is False
    assert manual["funding_events"] == []
    assert manual["net_external_flow"] == 0
    assert manual["excluded_external_flow"] == 0
    assert manifest["trade_count"] == 0
    assert manifest["order_count"] == 0
    assert manifest["fill_count"] == 0
    assert manifest["side_effects"]["broker"] is False
    assert manifest["side_effects"]["live_order"] is False
    assert manifest["side_effects"]["live_execution"] is False
    assert manifest["side_effects"]["v17_active_pointer_mutation"] is False
    assert manifest["side_effects"][
        "strategy_record_store_pointer_cas_by_manager"
    ] is True
    source = pd.read_parquet(
        source_dir / "ledger_after_manual_switch.parquet"
    ).sort_values("symbol")
    candidate = pd.read_parquet(
        staging_dir / "ledger_after_manual_switch.parquet"
    ).sort_values("symbol")
    for column in ("shares", "avg_cost", "cost_basis"):
        assert candidate[column].tolist() == source[column].tolist()
    assert manual["cash_after"] == 1_100_000
    assert manual["cash_after"] + manual["market_value_after"] == pytest.approx(
        manual["total_value_after"]
    )
    assert manual["total_value_after"] - valuation.CAPITAL_CNY == pytest.approx(
        manual["portfolio_pnl_after"]
    )
    assert summary["source_record"] == source_dir.name


def test_registered_source_rejects_pointer_and_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_root, source_dir, closure = _source_fixture(tmp_path)
    store = record_root / "_record_store"
    store.mkdir()
    pointer_path = store / "current.v1.json"
    pointer_path.write_bytes(b"registered-pointer\n")
    pointer_sha = _sha(pointer_path)
    pointer = {
        "active_record_id": source_dir.name,
        "active_closure": closure,
    }
    monkeypatch.setattr(
        valuation, "load_registered_catalog", lambda _root: (pointer, {})
    )
    monkeypatch.setattr(
        valuation,
        "resolve_active_record_dirs",
        lambda *_args, **_kwargs: [source_dir],
    )

    with pytest.raises(ValueError, match="expected pointer SHA drift"):
        valuation.resolve_registered_source(
            record_root=record_root, expected_pointer_sha="0" * 64
        )
    resolved, observed_closure = valuation.resolve_registered_source(
        record_root=record_root, expected_pointer_sha=pointer_sha
    )
    assert resolved == source_dir.resolve()
    assert observed_closure == closure

    other = record_root / "20260810_9999"
    other.mkdir()
    monkeypatch.setattr(
        valuation,
        "resolve_active_record_dirs",
        lambda *_args, **_kwargs: [other],
    )
    with pytest.raises(ValueError, match="record/closure identity mismatch"):
        valuation.resolve_registered_source(
            record_root=record_root, expected_pointer_sha=pointer_sha
        )


def test_source_closure_rejects_sha_flow_and_capital_drift(
    tmp_path: Path,
) -> None:
    record_root, source_dir, closure = _source_fixture(tmp_path)
    manual_path = source_dir / "manual_execution_manifest.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    manual["excluded_external_flow"] = 1
    _write_json(manual_path, manual)
    with pytest.raises(ValueError, match="manual manifest SHA mismatch"):
        valuation._source_closure(
            record_root=record_root,
            source_dir=source_dir,
            closure=closure,
        )

    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manual_execution"] = manual
    _write_json(manifest_path, manifest)
    closure["manual_manifest_sha256"] = _sha(manual_path)
    closure["manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(ValueError, match="non-zero effective external flow"):
        valuation._source_closure(
            record_root=record_root,
            source_dir=source_dir,
            closure=closure,
        )

    manual["excluded_external_flow"] = 0
    manual["capital_cny"] = 2_000_000
    _write_json(manual_path, manual)
    manifest["manual_execution"] = manual
    _write_json(manifest_path, manifest)
    closure["manual_manifest_sha256"] = _sha(manual_path)
    closure["manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(ValueError, match="one-million accounting mismatch"):
        valuation._source_closure(
            record_root=record_root,
            source_dir=source_dir,
            closure=closure,
        )
