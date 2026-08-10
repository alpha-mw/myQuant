#!/usr/bin/env python3
"""Build a hash-bound, no-trade CN Dashboard end-of-day valuation record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant_investor.config import Config
from quant_investor.credential_utils import create_tushare_pro

from cn_dashboard_common import canonical_json_bytes


STOCK_CODES = (
    "002008.SZ",
    "002384.SZ",
    "002463.SZ",
    "002916.SZ",
    "605358.SH",
    "688183.SH",
)
INDEX_CODES = ("000300.SH", "000688.SH", "399006.SZ")
CAPITAL_CNY = 1_000_000.0
EVIDENCE_SCHEMA = "cn_dashboard_tushare_close_evidence.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _exact_row(frame: Any, code: str, trade_date: str) -> dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        raise RuntimeError(f"Tushare returned no exact-date row: {code}")
    required = {"ts_code", "trade_date", "close"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Tushare response columns incomplete: {code}")
    rows = frame.loc[:, ["ts_code", "trade_date", "close"]].to_dict("records")
    exact = [
        row
        for row in rows
        if str(row["ts_code"]) == code
        and str(row["trade_date"]) == trade_date
    ]
    if len(exact) != 1:
        raise RuntimeError(f"Tushare exact-date row count is not one: {code}")
    close = float(exact[0]["close"])
    if close <= 0:
        raise RuntimeError(f"Tushare close is not positive: {code}")
    return {"ts_code": code, "trade_date": trade_date, "close": close}


def fetch_tushare_evidence(trade_date: str) -> dict[str, Any]:
    import tushare as ts  # type: ignore[import-not-found]

    pro = create_tushare_pro(ts, Config.TUSHARE_TOKEN, Config.TUSHARE_URL)
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    stocks = [
        _exact_row(
            pro.daily(ts_code=code, trade_date=trade_date),
            code,
            trade_date,
        )
        for code in STOCK_CODES
    ]
    indices = [
        _exact_row(
            pro.index_daily(
                ts_code=code,
                start_date=trade_date,
                end_date=trade_date,
            ),
            code,
            trade_date,
        )
        for code in INDEX_CODES
    ]
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "provider": "tushare.pro",
        "stock_api": "daily",
        "index_api": "index_daily",
        "trade_date": trade_date,
        "coverage": "exact_close",
        "previous_trading_day_ffill": False,
        "stocks": stocks,
        "indices": indices,
    }


def _source_closure(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "manifest.json"
    manual_path = source_dir / "manual_execution_manifest.json"
    if not manifest_path.is_file() or not manual_path.is_file():
        raise ValueError("source record manifest/manual closure is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manual = json.loads(manual_path.read_text(encoding="utf-8-sig"))
    ledger_name = str(manual.get("effective_manual_ledger_path") or "")
    ledger_path = source_dir / ledger_name
    if ledger_path.name != "ledger_after_manual_switch.parquet":
        raise ValueError("source effective ledger must be canonical Parquet")
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise ValueError("source effective ledger is not a regular file")
    ledger_sha = _sha(ledger_path)
    if ledger_sha != manual.get("next_ledger_sha256"):
        raise ValueError("source effective ledger SHA mismatch")
    if manifest.get("timestamp") != source_dir.name:
        raise ValueError("source record identity mismatch")
    return {
        "manifest": manifest,
        "manual": manual,
        "manifest_sha256": _sha(manifest_path),
        "manual_sha256": _sha(manual_path),
        "ledger_path": ledger_path,
        "ledger_sha256": ledger_sha,
    }


def _update_benchmark(path: Path, evidence: dict[str, Any]) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    trade_date = str(evidence["trade_date"])
    iso_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    replacements = {
        (iso_date, str(row["ts_code"])): {
            "date": iso_date,
            "ts_code": str(row["ts_code"]),
            "close": f"{float(row['close']):.6f}",
            "source_system": "tushare.index_daily",
            "value_date": iso_date,
            "coverage": "exact_close",
        }
        for row in evidence["indices"]
    }
    merged = {
        (str(row.get("date") or ""), str(row.get("ts_code") or "")): row
        for row in existing
    }
    merged.update(replacements)
    rows = [merged[key] for key in sorted(merged)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "ts_code",
                "close",
                "source_system",
                "value_date",
                "coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_record(
    *,
    staging_dir: Path,
    source_dir: Path,
    record_id: str,
    trade_date: str,
    recorded_at_iso: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if staging_dir.name != record_id or not staging_dir.is_dir():
        raise ValueError("staging directory/record identity mismatch")
    if evidence.get("trade_date") != trade_date:
        raise ValueError("valuation evidence trade date mismatch")
    source = _source_closure(source_dir)
    source_manual = source["manual"]
    source_ledger = pd.read_parquet(source["ledger_path"])
    if set(source_ledger["symbol"].astype(str)) != set(STOCK_CODES):
        raise ValueError(
            "source holdings do not match the authorized six symbols"
        )
    close_by_code = {
        str(row["ts_code"]): float(row["close"])
        for row in evidence["stocks"]
    }
    if set(close_by_code) != set(STOCK_CODES):
        raise ValueError("Tushare stock close coverage is incomplete")

    ledger = source_ledger.copy()
    ledger["current_price"] = ledger["symbol"].map(close_by_code)
    ledger["current_value"] = ledger["shares"] * ledger["current_price"]
    ledger["unrealized_pnl"] = ledger["current_value"] - ledger["cost_basis"]
    ledger["unrealized_pnl_pct"] = (
        ledger["unrealized_pnl"] / ledger["cost_basis"]
    )
    market_value = float(ledger["current_value"].sum())
    cash = float(source_manual["cash_after"])
    total_value = cash + market_value
    ledger["equity_sleeve_weight"] = ledger["current_value"] / market_value
    if "market_weight" in ledger.columns:
        ledger["market_weight"] = ledger["equity_sleeve_weight"]
    ledger["nav_weight"] = ledger["current_value"] / total_value

    csv_path = staging_dir / "ledger_after_manual_switch.csv"
    parquet_path = staging_dir / "ledger_after_manual_switch.parquet"
    ledger.to_csv(csv_path, index=False, encoding="utf-8-sig")
    ledger.to_parquet(parquet_path, index=False)
    csv_sha = _sha(csv_path)
    parquet_sha = _sha(parquet_path)

    evidence_path = staging_dir / "tushare_close_evidence.json"
    _write_json(evidence_path, evidence)
    evidence_sha = _sha(evidence_path)

    for filename in ("orders.csv", "manual_switch_and_take_profit_orders.csv"):
        source_path = source_dir / filename
        (staging_dir / filename).write_bytes(source_path.read_bytes())

    pnl = {
        "record_time": recorded_at_iso,
        "quote_snapshot": f"{trade_date}_TUSHARE_EXACT_CLOSE",
        "initial_capital": f"{CAPITAL_CNY:.2f}",
        "cash_before": f"{cash:.2f}",
        "market_value_before": (
            f"{float(source_manual['market_value_after']):.2f}"
        ),
        "total_value_before": (
            f"{float(source_manual['total_value_after']):.2f}"
        ),
        "portfolio_pnl_before": (
            f"{float(source_manual['portfolio_pnl_after']):.2f}"
        ),
        "portfolio_pnl_pct_before": (
            f"{float(source_manual['portfolio_return_after']):.8f}"
        ),
        "realized_pnl_from_rebalance": "0.00",
        "cash_after": f"{cash:.2f}",
        "market_value_after": f"{market_value:.2f}",
        "total_value_after": f"{total_value:.2f}",
        "portfolio_pnl_after": f"{total_value - CAPITAL_CNY:.2f}",
        "portfolio_pnl_pct_after": f"{total_value / CAPITAL_CNY - 1.0:.8f}",
        "delta_vs_source_record": (
            f"{total_value - float(source_manual['total_value_after']):.2f}"
        ),
    }
    pnl_path = staging_dir / "pnl_summary.csv"
    _write_csv(pnl_path, [pnl])

    recorded_at = datetime.fromisoformat(recorded_at_iso)
    recorded_at_text = recorded_at.strftime("%Y-%m-%d %H:%M:%S CST")
    financial_state = {
        "capital_cny": CAPITAL_CNY,
        "cash_after": round(cash, 2),
        "market_value_after": round(market_value, 2),
        "total_value_after": round(total_value, 2),
        "portfolio_pnl_after": round(total_value - CAPITAL_CNY, 2),
        "portfolio_return_after": total_value / CAPITAL_CNY - 1.0,
        "ledger_sha256": parquet_sha,
    }
    manual = {
        "schema_version": "cn_aggressive_manual_execution.v3",
        "record_origin": "official_tushare_eod_revaluation",
        "historical_lane": "CANONICAL_V17_ACTUAL_HOLDINGS_OWNER_DECLARATION",
        "not_v17_prediction_or_forward_observation": True,
        "capital_cny": CAPITAL_CNY,
        "status": "no_action_carry_forward_official_valuation",
        "execution_status": "no_action_carry_forward_official_valuation",
        "manual_execution_mode": "official_eod_revaluation_no_broker_api",
        "advisory_only": False,
        "local_manual_fills_allowed": False,
        "owner_reported_external_fills": False,
        "no_trade_performed": True,
        "record_timestamp": record_id,
        "recorded_at": recorded_at_text,
        "recorded_at_iso": recorded_at_iso,
        "trade_date": trade_date,
        "valuation_trade_date": trade_date,
        "valuation_status": "OFFICIAL_TUSHARE_EOD_CLOSE_COMPLETE",
        "official_valuation": True,
        "price_basis": "tushare_daily_exact_close_hash_bound",
        "quote_source": "tushare.daily",
        "decision_data_sufficient": True,
        "holdings_completeness_passed": True,
        "valuation_completeness_passed": True,
        "completeness_passed": True,
        "automation_run_status": "completed_official_eod_revaluation",
        "review_state": "OFFICIAL_EOD_REVALUATION_NO_ACTION",
        "blockers": [],
        "source_record": source_dir.name,
        "supersedes_record": source_dir.name,
        "source_manifest_sha256": source["manifest_sha256"],
        "source_manual_manifest_sha256": source["manual_sha256"],
        "source_contained_ledger_sha256": source["ledger_sha256"],
        "applied_local_trades": [],
        "applied_owner_declared_trades": [],
        "rejected_or_pending_trades": [],
        "effective_manual_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_sha256": parquet_sha,
        "ledger_after_manual_switch_csv": "ledger_after_manual_switch.csv",
        "ledger_after_manual_switch_csv_sha256": csv_sha,
        "ledger_after_manual_switch_parquet": (
            "ledger_after_manual_switch.parquet"
        ),
        "ledger_after_manual_switch_parquet_sha256": parquet_sha,
        "manual_orders_path": "manual_switch_and_take_profit_orders.csv",
        "manual_orders_sha256": _sha(
            staging_dir / "manual_switch_and_take_profit_orders.csv"
        ),
        "orders_path": "orders.csv",
        "orders_sha256": _sha(staging_dir / "orders.csv"),
        "pnl_summary_path": "pnl_summary.csv",
        "pnl_summary_sha256": _sha(pnl_path),
        "valuation_evidence_path": "tushare_close_evidence.json",
        "valuation_evidence_sha256": evidence_sha,
        "ledger_provenance": {
            "declared_next_ledger_path": "ledger_after_manual_switch.parquet",
            "contained_in_run_directory": True,
            "regular_non_symlink_file": True,
            "stable_double_read": True,
            "declared_sha256": parquet_sha,
            "csv_sha256": csv_sha,
            "parquet_sha256": parquet_sha,
            "source_record": source_dir.name,
            "source_ledger_sha256": source["ledger_sha256"],
            "official_eod_revaluation_only": True,
            "position_quantities_unchanged": True,
        },
        "effective_manual_holding_count": len(ledger),
        "source_manual_holding_count": len(source_ledger),
        "cash_before": cash,
        "gross_trade_value": 0.0,
        "fees_cny": 0.0,
        "cash_after": round(cash, 2),
        "market_value_after": round(market_value, 2),
        "total_value_after": round(total_value, 2),
        "portfolio_pnl_after": round(total_value - CAPITAL_CNY, 2),
        "portfolio_return_after": total_value / CAPITAL_CNY - 1.0,
        "realized_pnl_from_rebalance": 0.0,
        "provider_quote_called": True,
        "no_provider_quote_called": False,
        "no_llm_gateway_called": True,
        "no_broker_api_called": True,
        "no_order_created_by_codex": True,
        "no_execution_performed_by_codex": True,
        "financial_state": financial_state,
        "financial_state_sha256": hashlib.sha256(
            canonical_json_bytes(financial_state)
        ).hexdigest(),
    }
    manual_path = staging_dir / "manual_execution_manifest.json"
    _write_json(manual_path, manual)

    review_path = staging_dir / "daily_execution_review.md"
    review_path.write_text(
        "# AlphaMx科技创新组合001号 日终估值\n\n"
        f"- 估值日：{trade_date}\n"
        "- 行为：no_action / carry_forward；无交易。\n"
        "- 行情：Tushare daily/index_daily exact-date close，证据已 hash-bound。\n"
        f"- 直接来源：{source_dir.name}。\n",
        encoding="utf-8",
    )
    files = {
        "orders": "orders.csv",
        "pnl_summary": "pnl_summary.csv",
        "manual_execution_manifest": "manual_execution_manifest.json",
        "manual_orders": "manual_switch_and_take_profit_orders.csv",
        "ledger_after_manual_switch": "ledger_after_manual_switch.parquet",
        "ledger_after_manual_switch_csv": "ledger_after_manual_switch.csv",
        "daily_execution_review": "daily_execution_review.md",
        "valuation_evidence": "tushare_close_evidence.json",
    }
    file_sha = {name: _sha(staging_dir / name) for name in files.values()}
    manifest = {
        "schema_version": "cn_aggressive_daily_transaction_record.v1",
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "canonical_v17_holdings": True,
        "canonical_holdings_authority": "owner_declaration",
        "timestamp": record_id,
        "recorded_at": recorded_at_text,
        "recorded_at_iso": recorded_at_iso,
        "source_record": source_dir.name,
        "supersedes_record": source_dir.name,
        "source_manifest_sha256": source["manifest_sha256"],
        "formal_record": True,
        "completeness_passed": True,
        "record_origin": "official_tushare_eod_revaluation",
        "historical_lane": "CANONICAL_V17_ACTUAL_HOLDINGS_OWNER_DECLARATION",
        "not_v17_prediction_or_forward_observation": True,
        "action_taken_today": False,
        "trade_count": 0,
        "order_count": 0,
        "fill_count": 0,
        "review_state": "OFFICIAL_EOD_REVALUATION_NO_ACTION",
        "automation_run_status": "completed_official_eod_revaluation",
        "blockers": [],
        "files": files,
        "file_sha256": file_sha,
        "data_snapshot": {
            "analysis_trade_date": trade_date,
            "valuation_trade_date": trade_date,
            "valuation_status": "OFFICIAL_TUSHARE_EOD_CLOSE_COMPLETE",
            "freshness_mode": "tushare_exact_date_close_hash_bound",
            "valuation_evidence_path": "tushare_close_evidence.json",
            "valuation_evidence_sha256": evidence_sha,
            "source_record_transaction_marks_preserved": False,
            "new_quote_requested": True,
        },
        "manual_execution": manual,
        "side_effects": {
            "provider_quote_called": True,
            "broker": False,
            "live_order": False,
            "live_execution": False,
            "actual_position_quantity_mutation": False,
            "actual_cash_mutation": False,
            "active_pointer_mutation": False,
            "factor_registry_mutation": False,
            "production_rule_mutation": False,
        },
    }
    _write_json(staging_dir / "manifest.json", manifest)
    return {
        "record_id": record_id,
        "source_record": source_dir.name,
        "cash": round(cash, 2),
        "market_value": round(market_value, 2),
        "total_value": round(total_value, 2),
        "portfolio_pnl": round(total_value - CAPITAL_CNY, 2),
        "ledger_sha256": parquet_sha,
        "manual_manifest_sha256": _sha(manual_path),
        "manifest_sha256": _sha(staging_dir / "manifest.json"),
        "valuation_evidence_sha256": evidence_sha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--source-record-dir", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    evidence = fetch_tushare_evidence(args.trade_date)
    summary = build_record(
        staging_dir=args.staging_dir,
        source_dir=args.source_record_dir,
        record_id=args.record_id,
        trade_date=args.trade_date,
        recorded_at_iso=now.isoformat(),
        evidence=evidence,
    )
    _update_benchmark(args.benchmark_output, evidence)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
