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
from quant_investor.strategy_records.store import (
    load_registered_catalog,
    resolve_active_record_dirs,
)

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
HISTORICAL_HOLDINGS_LANE = "REGISTERED_HISTORICAL_HOLDINGS_STORAGE_ONLY"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_raw(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    before = path.stat()
    first = path.read_bytes()
    middle = path.stat()
    second = path.read_bytes()
    after = path.stat()
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, middle, after)
    }
    if len(identities) != 1 or first != second:
        raise ValueError(f"{label} was unstable during read")
    return first


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


def _bound_artifact(
    *,
    record_root: Path,
    source_dir: Path,
    closure: dict[str, Any],
    path_key: str,
    sha_key: str,
    label: str,
) -> tuple[Path, bytes]:
    relative_value = closure.get(path_key)
    declared_sha = closure.get(sha_key)
    if not isinstance(relative_value, str) or not isinstance(declared_sha, str):
        raise ValueError(f"registered active closure {label} binding is missing")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"registered active closure {label} path is invalid")
    candidate = record_root / relative
    raw = _stable_raw(candidate, label=f"source {label}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != source_dir.resolve(strict=True):
        raise ValueError(f"registered active closure {label} escapes source record")
    if hashlib.sha256(raw).hexdigest() != declared_sha:
        raise ValueError(f"registered active closure {label} SHA mismatch")
    return candidate, raw


def resolve_registered_source(
    *, record_root: Path, expected_pointer_sha: str
) -> tuple[Path, dict[str, Any]]:
    record_root = record_root.resolve(strict=True)
    pointer_path = record_root / "_record_store" / "current.v1.json"
    first_pointer_raw = _stable_raw(pointer_path, label="strategy-record pointer")
    observed_pointer_sha = hashlib.sha256(first_pointer_raw).hexdigest()
    if observed_pointer_sha != expected_pointer_sha:
        raise ValueError("strategy-record expected pointer SHA drift")
    registered = load_registered_catalog(record_root)
    if registered is None:
        raise ValueError("strategy-record store is unregistered")
    pointer, catalog = registered
    active_dirs = resolve_active_record_dirs(
        record_root, pointer=pointer, catalog=catalog
    )
    second_pointer_raw = _stable_raw(pointer_path, label="strategy-record pointer")
    if first_pointer_raw != second_pointer_raw:
        raise ValueError("strategy-record pointer drifted during source resolution")
    if not active_dirs:
        raise ValueError("registered active record directory is missing")
    source_dir = active_dirs[0].resolve(strict=True)
    active_record_id = pointer.get("active_record_id")
    closure = pointer.get("active_closure")
    if (
        not isinstance(active_record_id, str)
        or not isinstance(closure, dict)
        or closure.get("record_id") != active_record_id
        or closure.get("relative_path") != active_record_id
        or source_dir.name != active_record_id
        or source_dir.parent != record_root
    ):
        raise ValueError("registered active record/closure identity mismatch")
    return source_dir, dict(closure)


def _single_pnl_row(raw: bytes) -> dict[str, str]:
    try:
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    except UnicodeDecodeError as exc:
        raise ValueError("source P&L is not valid UTF-8") from exc
    if len(rows) != 1:
        raise ValueError("source P&L must contain exactly one row")
    return rows[0]


def _empty_csv_payload(path: Path, *, label: str) -> None:
    raw = _stable_raw(path, label=label)
    try:
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    if rows:
        raise ValueError(f"{label} contains order/trade rows")


def _money_close(left: float, right: float) -> bool:
    return abs(left - right) <= 0.01


def _source_closure(
    *, record_root: Path, source_dir: Path, closure: dict[str, Any]
) -> dict[str, Any]:
    _manifest_path, manifest_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="manifest_path",
        sha_key="manifest_sha256",
        label="manifest",
    )
    _manual_path, manual_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="manual_manifest_path",
        sha_key="manual_manifest_sha256",
        label="manual manifest",
    )
    ledger_path, ledger_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="ledger_path",
        sha_key="ledger_sha256",
        label="ledger",
    )
    _pnl_path, pnl_raw = _bound_artifact(
        record_root=record_root,
        source_dir=source_dir,
        closure=closure,
        path_key="pnl_path",
        sha_key="pnl_sha256",
        label="P&L",
    )
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    manual = json.loads(manual_raw.decode("utf-8-sig"))
    ledger_name = str(manual.get("effective_manual_ledger_path") or "")
    if (
        ledger_name != "ledger_after_manual_switch.parquet"
        or ledger_path.name != "ledger_after_manual_switch.parquet"
    ):
        raise ValueError("source effective ledger must be canonical Parquet")
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
    if (
        ledger_sha != manual.get("next_ledger_sha256")
        or ledger_sha != manual.get("ledger_after_manual_switch_parquet_sha256")
    ):
        raise ValueError("source effective ledger SHA mismatch")
    if (
        manifest.get("timestamp") != source_dir.name
        or manifest.get("market") != "CN"
        or manifest.get("strategy") != "aggressive_tech_manufacturing"
        or manifest.get("manual_execution") != manual
    ):
        raise ValueError("source record identity mismatch")
    if (
        manifest.get("action_taken_today") is not False
        or any(
            float(manifest.get(key, -1)) != 0
            for key in ("trade_count", "order_count", "fill_count")
        )
        or manual.get("no_trade_performed") is not True
        or any(
            manual.get(key) not in ([], None)
            for key in (
                "applied_local_trades",
                "applied_owner_declared_trades",
                "rejected_or_pending_trades",
            )
        )
        or float(manual.get("gross_trade_value", -1)) != 0
    ):
        raise ValueError("source record is not a zero-trade carry-forward closure")
    if manual.get("funding_events") not in (None, []) or float(
        manual.get("net_external_flow", 0)
    ) != 0 or float(manual.get("excluded_external_flow", 0)) != 0:
        raise ValueError("source record has a non-zero effective external flow")

    source_ledger = pd.read_parquet(ledger_path)
    required_columns = {"symbol", "shares", "avg_cost", "cost_basis", "current_value"}
    if not required_columns.issubset(source_ledger.columns):
        raise ValueError("source ledger accounting columns are incomplete")
    if source_ledger["symbol"].astype(str).duplicated().any():
        raise ValueError("source ledger contains duplicate symbols")
    capital = float(manual.get("capital_cny", 0))
    cash = float(manual.get("cash_after", "nan"))
    market_value = float(manual.get("market_value_after", "nan"))
    total_value = float(manual.get("total_value_after", "nan"))
    portfolio_pnl = float(manual.get("portfolio_pnl_after", "nan"))
    ledger_market_value = float(source_ledger["current_value"].sum())
    if (
        not _money_close(capital, CAPITAL_CNY)
        or not _money_close(ledger_market_value, market_value)
        or not _money_close(cash + market_value, total_value)
        or not _money_close(total_value - CAPITAL_CNY, portfolio_pnl)
    ):
        raise ValueError("source owner-corrected one-million accounting mismatch")
    financial_state = manual.get("financial_state")
    if not isinstance(financial_state, dict):
        raise ValueError("source financial state is missing")
    observed_financial_state_sha = hashlib.sha256(
        canonical_json_bytes(financial_state)
    ).hexdigest()
    if (
        observed_financial_state_sha != manual.get("financial_state_sha256")
        or observed_financial_state_sha != closure.get("financial_state_sha256")
        or financial_state.get("ledger_sha256") != ledger_sha
    ):
        raise ValueError("source financial-state closure mismatch")
    pnl = _single_pnl_row(pnl_raw)
    for key, expected in (
        ("cash_after", cash),
        ("market_value_after", market_value),
        ("total_value_after", total_value),
        ("portfolio_pnl_after", portfolio_pnl),
    ):
        if not _money_close(float(pnl[key]), expected):
            raise ValueError("source P&L/accounting identity mismatch")
    for filename in ("orders.csv", "manual_switch_and_take_profit_orders.csv"):
        _empty_csv_payload(source_dir / filename, label=f"source {filename}")
    return {
        "manifest": manifest,
        "manual": manual,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manual_sha256": hashlib.sha256(manual_raw).hexdigest(),
        "ledger_path": ledger_path,
        "ledger_sha256": ledger_sha,
        "source_ledger": source_ledger,
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
    record_root: Path,
    source_dir: Path,
    registered_closure: dict[str, Any],
    record_id: str,
    trade_date: str,
    recorded_at_iso: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if staging_dir.name != record_id or not staging_dir.is_dir():
        raise ValueError("staging directory/record identity mismatch")
    if evidence.get("trade_date") != trade_date:
        raise ValueError("valuation evidence trade date mismatch")
    source = _source_closure(
        record_root=record_root,
        source_dir=source_dir,
        closure=registered_closure,
    )
    source_manual = source["manual"]
    source_ledger = source["source_ledger"]
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
    for immutable_column in ("shares", "avg_cost", "cost_basis"):
        if not ledger[immutable_column].equals(source_ledger[immutable_column]):
            raise ValueError(f"no-trade valuation mutated {immutable_column}")

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
        "historical_lane": HISTORICAL_HOLDINGS_LANE,
        "historical_holdings_storage_authority": True,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
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
        "funding_events": [],
        "net_external_flow": 0.0,
        "excluded_external_flow": 0.0,
        "trade_count": 0,
        "order_count": 0,
        "fill_count": 0,
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
        "historical_holdings_storage_authority": True,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
        "timestamp": record_id,
        "recorded_at": recorded_at_text,
        "recorded_at_iso": recorded_at_iso,
        "source_record": source_dir.name,
        "supersedes_record": source_dir.name,
        "source_manifest_sha256": source["manifest_sha256"],
        "formal_record": True,
        "completeness_passed": True,
        "record_origin": "official_tushare_eod_revaluation",
        "historical_lane": HISTORICAL_HOLDINGS_LANE,
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
            "v17_active_pointer_mutation": False,
            "strategy_record_store_pointer_cas_by_manager": True,
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
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expected-pointer-sha", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    source_dir, registered_closure = resolve_registered_source(
        record_root=args.record_root,
        expected_pointer_sha=args.expected_pointer_sha,
    )
    evidence = fetch_tushare_evidence(args.trade_date)
    summary = build_record(
        staging_dir=args.staging_dir,
        record_root=args.record_root.resolve(strict=True),
        source_dir=source_dir,
        registered_closure=registered_closure,
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
