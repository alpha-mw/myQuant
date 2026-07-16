#!/usr/bin/env python3
"""Offline four-month track-record audit for CN/aggressive_tech_manufacturing.

Learned record schema, based on
``quant_investor/monitoring/cn_aggressive_portfolio_tracker.py`` and current
``results/strategy_records/CN/aggressive_tech_manufacturing`` records:

* ``pnl_summary.csv`` / ``pnl_summary.parquet`` holds the NAV snapshot. Newer
  records are one-row wide tables with ``initial_capital`` and
  ``total_value_after``; some June records use ``metric,value`` and must be
  pivoted.
* ``ledger_after_manual_switch.csv`` is the only effective execution ledger.
  ``ledger.csv`` and ``holdings_review.csv`` are not execution baselines.
* ``manual_execution_manifest.json`` stores funding/execution provenance:
  ``manual_execution_mode``, ``price_basis``, ``quote_source``,
  ``execution_price_gate``, ``applied_local_trades``,
  ``rejected_or_pending_trades``, and ``no_broker_api_called``. Latest records
  identify local/manual paper execution as
  ``paper_only_local_manual_no_broker``.
* ``manual_switch_and_take_profit_orders.csv`` is the best filled/rejected
  execution tape when present. Historical headers differ: some use
  ``execution_price``/``quote_timestamp`` while later post-review files use
  ``price``/``price_basis``/``quote_time`` and quote OHLC fields.
* ``orders.csv`` contains only realtime-price-gate accepted formal orders with
  ``timestamp,action,symbol,name,shares,price,trade_value,realized_pnl,reason``.
  Proposed orders rejected by the realtime execution-price gate are not written
  there; they appear in ``manual_switch_and_take_profit_orders.csv`` and
  ``manual_execution_manifest.execution_price_gate.rejections``.
* ``holdings_review.csv`` contains per-position review state, including
  ``current_price``, ``current_value``, ``market_weight``, LLM/review fields,
  and, from 2026-06-23 onward, ``realtime_execution_price`` plus
  ``realtime_execution_price_field``.
* ``market_snapshot.json`` stores portfolio, index snapshots, breadth,
  structured industry provenance, Markov metadata, ``execution_price_gate``
  and ``manual_execution`` mirrors.
* Current v15 records may include ``v15_run_readiness.json``. Historical mixed
  records remain readable, but their retired Theme fields are never consumed
  as current authorization evidence.

This script is deterministic, offline, and read-only for strategy records. It
only writes audit outputs under ``results/track_record_audit/<YYYYMMDD>/`` (or a
caller supplied output root) and refuses an output directory inside the record
root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "track_record_audit"
DEFAULT_BENCHMARK_FILE = PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_index_benchmark.csv"
DEFAULT_BARS_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "bars"
DEFAULT_STOCK_BASIC_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "dag_core_raw" / "table=stock_basic"
DEFAULT_FUNDAMENTALS_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "fundamental_raw"
DEFAULT_REGIME_HISTORY = PROJECT_ROOT / "results" / "regime" / "markov_regime_history.jsonl"
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / "track_record_audit" / ".matplotlib"))
PREREGISTERED_FAILURE_CONCLUSION = (
    "日频机器未在战略先验之上增值，优势主张不成立，待后续独立样本复验。"
)
BENCHMARK_CODE_TO_FIELD = {
    "000300.SH": "csi300_nav",
    "000688.SH": "star50_nav",
    "399006.SZ": "chinext_nav",
}
TECH_MANUFACTURING_INDUSTRIES = (
    "IT设备",
    "互联网",
    "元器件",
    "专用机械",
    "化工机械",
    "半导体",
    "工程机械",
    "机床制造",
    "机械基件",
    "新型电力",
    "汽车整车",
    "汽车配件",
    "电器仪表",
    "电气设备",
    "航空",
    "船舶",
    "软件服务",
    "轻工机械",
    "运输设备",
    "通信设备",
    "生物制药",
)


@contextmanager
def _suppress_native_stderr():
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


@dataclass(frozen=True)
class Record:
    run_id: str
    date: str
    path: Path
    manifest: dict[str, Any]
    manual_manifest: dict[str, Any]
    market_snapshot: dict[str, Any]
    pnl: dict[str, Any]
    holdings: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    manual_orders: list[dict[str, Any]]
    funding_cash_flows: list[dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        text = str(value).replace(",", "").strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _money(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:,.2f}"


def _iso_from_any(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _compact_date(value: str) -> str:
    return str(value or "").replace("-", "")[:8]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{str(k or "").strip(): str(v or "").strip() for k, v in row.items()} for row in reader]


def _read_pnl(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path / "pnl_summary.csv")
    if not rows:
        return {}
    if {"metric", "value"}.issubset(rows[0]):
        return {row.get("metric", ""): row.get("value", "") for row in rows if row.get("metric")}
    return rows[-1]


def _record_date(run_id: str, manifest: dict[str, Any], pnl: dict[str, Any]) -> str:
    for value in (
        pnl.get("record_time"),
        manifest.get("recorded_at"),
        manifest.get("timestamp"),
        run_id,
    ):
        parsed = _iso_from_any(value)
        if parsed:
            return parsed
    return ""


def _funding_cash_flows(path: Path, record_date: str, manual_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    embedded = manual_manifest.get("manual_funding_supplement")
    declared_path = str(manual_manifest.get("manual_funding_supplement_path") or "").strip()
    external: dict[str, Any] = {}
    if declared_path:
        evidence_path = Path(declared_path)
        if not evidence_path.is_absolute():
            evidence_path = path / evidence_path
        if not evidence_path.exists():
            raise ValueError(f"Invalid funding lineage for {path.name}: evidence file not found: {evidence_path}")
        external = _read_json(evidence_path)
        if not external:
            raise ValueError(f"Invalid funding lineage for {path.name}: unreadable funding evidence: {evidence_path}")
    if embedded is not None and not isinstance(embedded, dict):
        raise ValueError(f"Invalid funding lineage for {path.name}: manual_funding_supplement must be an object")
    if embedded and external and embedded != external:
        raise ValueError(f"Invalid funding lineage for {path.name}: embedded and external funding evidence differ")

    payloads: list[dict[str, Any]] = []
    supplement = external or (embedded if isinstance(embedded, dict) else {})
    if supplement:
        payloads.append(supplement)
    for key in ("external_funding_cash_flows", "funding_cash_flows"):
        rows = manual_manifest.get(key) or []
        if not isinstance(rows, list):
            raise ValueError(f"Invalid funding lineage for {path.name}: {key} must be a list")
        payloads.extend(row for row in rows if isinstance(row, dict))

    flows: list[dict[str, Any]] = []
    for payload in payloads:
        amount = _safe_float(payload.get("amount"), None)
        flow_date = _iso_from_any(
            payload.get("effective_date")
            or payload.get("capital_base_effective_from")
            or payload.get("date")
            or record_date
        )
        if amount is None or not math.isfinite(amount) or amount == 0:
            raise ValueError(f"Invalid funding lineage for {path.name}: funding amount must be finite and non-zero")
        if not flow_date or flow_date != record_date:
            raise ValueError(
                f"Invalid funding lineage for {path.name}: funding date {flow_date or 'missing'} "
                f"does not match record date {record_date}"
            )
        before = _safe_float(payload.get("total_value_before"), None)
        after = _safe_float(payload.get("total_value_after"), None)
        if before is not None and after is not None and not math.isclose(before + amount, after, abs_tol=0.01):
            raise ValueError(f"Invalid funding lineage for {path.name}: total_value_before + amount != total_value_after")
        flows.append(
            {
                "date": flow_date,
                "amount": amount,
                "source": str(payload.get("source") or "manual_execution_manifest"),
                "schema_version": str(payload.get("schema_version") or ""),
                "evidence_path": declared_path,
            }
        )
    return flows


def _load_records(record_root: Path) -> tuple[list[Record], list[str]]:
    warnings: list[str] = []
    if not record_root.exists():
        return [], [f"record_root not found: {record_root}"]
    records: list[Record] = []
    for path in sorted(p for p in record_root.iterdir() if p.is_dir()):
        manifest = _read_json(path / "manifest.json")
        pnl = _read_pnl(path)
        record_date = _record_date(path.name, manifest, pnl)
        if not record_date or not pnl:
            continue
        ledger_path = path / "ledger_after_manual_switch.csv"
        if not ledger_path.exists():
            warnings.append(f"Skipped {path.name}: missing required execution baseline ledger_after_manual_switch.csv")
            continue
        manual_manifest = _read_json(path / "manual_execution_manifest.json")
        records.append(
            Record(
                run_id=path.name,
                date=record_date,
                path=path,
                manifest=manifest,
                manual_manifest=manual_manifest,
                market_snapshot=_read_json(path / "market_snapshot.json"),
                pnl=pnl,
                holdings=_read_csv_rows(ledger_path),
                orders=_read_csv_rows(path / "orders.csv"),
                manual_orders=_read_csv_rows(path / "manual_switch_and_take_profit_orders.csv"),
                funding_cash_flows=_funding_cash_flows(path, record_date, manual_manifest),
            )
        )
    if not records:
        warnings.append("No auditable records with manifest/pnl_summary were found.")
    return records, warnings


def _latest_daily_records(records: Iterable[Record]) -> list[Record]:
    by_date: dict[str, Record] = {}
    funding_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.date, item.run_id)):
        by_date[record.date] = record
        funding_by_date[record.date].extend(record.funding_cash_flows)
    daily_records: list[Record] = []
    for record_date in sorted(by_date):
        seen: set[tuple[str, float, str, str]] = set()
        funding_cash_flows: list[dict[str, Any]] = []
        for flow in funding_by_date[record_date]:
            identity = (
                str(flow.get("date") or ""),
                _safe_float(flow.get("amount")),
                str(flow.get("source") or ""),
                str(flow.get("evidence_path") or ""),
            )
            if identity not in seen:
                seen.add(identity)
                funding_cash_flows.append(flow)
        daily_records.append(replace(by_date[record_date], funding_cash_flows=funding_cash_flows))
    return daily_records


def _nav_series(records: list[Record]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unit_nav: float | None = None
    previous_total: float | None = None
    for record in records:
        initial = _safe_float(record.pnl.get("initial_capital") or record.manifest.get("capital_cny"))
        total = _safe_float(record.pnl.get("total_value_after"))
        if initial <= 0 or total <= 0:
            continue
        external_flow = sum(_safe_float(flow.get("amount")) for flow in record.funding_cash_flows)
        if unit_nav is None:
            unit_nav = total / initial
            daily_return = None
        else:
            adjusted_opening_value = previous_total
            if adjusted_opening_value is None or adjusted_opening_value <= 0 or total - external_flow <= 0:
                raise ValueError(f"Invalid NAV lineage for {record.run_id}: non-positive cash-flow-adjusted value")
            daily_return = (total - external_flow) / adjusted_opening_value - 1.0
            unit_nav *= 1.0 + daily_return
        rows.append(
            {
                "date": record.date,
                "run_id": record.run_id,
                "initial_capital": initial,
                "total_value_after": total,
                "nav": unit_nav,
                "unit_nav": unit_nav,
                "daily_return": daily_return,
                "external_funding_cash_flow": external_flow,
                "raw_capital_ratio": total / initial,
                "funding_evidence": record.funding_cash_flows,
            }
        )
        previous_total = total
    return rows


def _daily_returns(nav_rows: list[dict[str, Any]]) -> list[float]:
    returns: list[float] = []
    for prev, current in zip(nav_rows, nav_rows[1:]):
        prev_nav = _safe_float(prev.get("nav"))
        current_nav = _safe_float(current.get("nav"))
        if prev_nav > 0 and current_nav > 0:
            returns.append(current_nav / prev_nav - 1.0)
    return returns


def _max_drawdown(nav_rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak = -math.inf
    peak_date = ""
    max_dd = 0.0
    trough_date = ""
    dd_start = ""
    for row in nav_rows:
        nav = _safe_float(row.get("nav"))
        if nav > peak:
            peak = nav
            peak_date = str(row.get("date") or "")
        if peak > 0:
            drawdown = nav / peak - 1.0
            if drawdown < max_dd:
                max_dd = drawdown
                dd_start = peak_date
                trough_date = str(row.get("date") or "")
    return {"max_drawdown": max_dd, "start": dd_start, "end": trough_date}


def _performance_metrics(nav_rows: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not nav_rows:
        return {}
    returns = _daily_returns(nav_rows)
    cumulative = _safe_float(nav_rows[-1].get("nav")) / _safe_float(nav_rows[0].get("nav"), 1.0) - 1.0
    annual_return = (1.0 + cumulative) ** (252.0 / max(len(nav_rows) - 1, 1)) - 1.0
    vol = statistics.pstdev(returns) * math.sqrt(252.0) if returns else 0.0
    daily_std = statistics.pstdev(returns) if returns else 0.0
    sharpe = (statistics.mean(returns) / daily_std * math.sqrt(252.0)) if daily_std > 0 else None
    drawdown = _max_drawdown(nav_rows)
    avg_nav_value = statistics.mean(_safe_float(row.get("total_value_after")) for row in nav_rows)
    filled_trades = [trade for trade in trades if trade.get("status") == "filled"]
    sell_trades = [trade for trade in filled_trades if str(trade.get("action")).lower() == "sell"]
    trade_value = sum(abs(_safe_float(trade.get("trade_value"))) for trade in filled_trades)
    week_count = max(1.0, len(nav_rows) / 5.0)
    weekly_turnover = trade_value / max(avg_nav_value, 1.0) / week_count
    realized = [_safe_float(trade.get("realized_pnl")) for trade in sell_trades if "realized_pnl" in trade]
    wins = [value for value in realized if value > 0]
    losses = [value for value in realized if value < 0]
    hit_rate = len(wins) / len(realized) if realized else None
    pl_ratio = (
        (statistics.mean(wins) / abs(statistics.mean(losses)))
        if wins and losses
        else None
    )
    return {
        "sample_trading_days": len(nav_rows),
        "cumulative_return": cumulative,
        "annualized_return": annual_return,
        "annualized_volatility": vol,
        "max_drawdown": drawdown["max_drawdown"],
        "max_drawdown_start": drawdown["start"],
        "max_drawdown_end": drawdown["end"],
        "calmar": annual_return / abs(drawdown["max_drawdown"]) if drawdown["max_drawdown"] < 0 else None,
        "daily_sharpe": sharpe,
        "weekly_turnover": weekly_turnover,
        "annualized_turnover": weekly_turnover * 52.0,
        "closed_trade_count": len(realized),
        "closed_trade_hit_rate": hit_rate,
        "closed_trade_profit_loss_ratio": pl_ratio,
    }


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"buy", "b", "买入"}:
        return "buy"
    if text in {"sell", "s", "卖出"}:
        return "sell"
    return text


def _trade_from_row(row: dict[str, str], record: Record, source: str) -> dict[str, Any]:
    price = _safe_float(row.get("execution_price") or row.get("price"))
    status = str(row.get("status") or "filled").strip().lower()
    if status in {"filled", "done", "成交"}:
        status = "filled"
    elif not status:
        status = "filled"
    else:
        status = "rejected"
    return {
        "date": record.date,
        "run_id": record.run_id,
        "timestamp": row.get("timestamp") or record.manifest.get("recorded_at") or record.run_id,
        "source": source,
        "action": _normalize_action(row.get("action")),
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "name": row.get("name") or "",
        "shares": _safe_int(row.get("shares")),
        "price": price,
        "trade_value": _safe_float(row.get("trade_value"), price * _safe_int(row.get("shares"))),
        "realized_pnl": _safe_float(row.get("realized_pnl")),
        "status": status,
        "reason": row.get("reason") or "",
        "quote_source": row.get("quote_source") or "",
        "quote_timestamp": row.get("quote_timestamp") or row.get("quote_time") or "",
        "price_basis": row.get("price_basis") or row.get("execution_price_field") or "",
    }


def _extract_trades(records: list[Record]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_trades: set[tuple[Any, ...]] = set()
    seen_rejections: set[tuple[Any, ...]] = set()
    for record in records:
        source_rows = record.manual_orders if record.manual_orders else record.orders
        source = "manual_switch_and_take_profit_orders" if record.manual_orders else "orders"
        for row in source_rows:
            trade = _trade_from_row(row, record, source)
            if not trade["symbol"]:
                continue
            if trade["status"] == "filled":
                key = (
                    trade["date"],
                    trade["action"],
                    trade["symbol"],
                    trade["shares"],
                    round(float(trade["price"]), 4),
                    round(float(trade["trade_value"]), 2),
                    trade["reason"],
                )
                if key not in seen_trades:
                    seen_trades.add(key)
                    trades.append(trade)
            else:
                key = (trade["date"], trade["action"], trade["symbol"], trade["shares"], trade["reason"])
                if key not in seen_rejections:
                    seen_rejections.add(key)
                    rejections.append(trade)
        gate = record.manual_manifest.get("execution_price_gate") or record.manifest.get("execution_price_gate") or {}
        for rejected in gate.get("rejections", []) or []:
            if isinstance(rejected, dict):
                rejection = {
                    "date": record.date,
                    "run_id": record.run_id,
                    "source": "execution_price_gate",
                    "action": _normalize_action(rejected.get("action")),
                    "symbol": str(rejected.get("symbol") or "").strip().upper(),
                    "shares": _safe_int(rejected.get("shares")),
                    "status": "rejected",
                    "reason": str(rejected.get("reason") or ""),
                }
                if rejection["symbol"]:
                    key = (rejection["date"], rejection["action"], rejection["symbol"], rejection["shares"], rejection["reason"])
                    if key not in seen_rejections:
                        seen_rejections.add(key)
                        rejections.append(rejection)
    return sorted(trades, key=lambda item: (item["date"], item["run_id"], item["symbol"], item["action"])), rejections


def _holdings_by_date(records: list[Record]) -> dict[str, dict[str, dict[str, Any]]]:
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        rows: dict[str, dict[str, Any]] = {}
        for row in record.holdings:
            symbol = str(row.get("symbol") or "").strip().upper()
            shares = _safe_int(row.get("shares") or row.get("shares_before"))
            if not symbol or shares <= 0:
                continue
            rows[symbol] = {
                "symbol": symbol,
                "name": row.get("name") or "",
                "shares": shares,
                "current_value": _safe_float(row.get("current_value")),
                "current_price": _safe_float(row.get("current_price")),
                "unrealized_pnl": _safe_float(row.get("unrealized_pnl")),
                "market_weight": _safe_float(row.get("market_weight")),
            }
        by_date[record.date] = rows
    return by_date


def _holding_periods(records: list[Record]) -> dict[str, Any]:
    holdings = _holdings_by_date(records)
    first_seen: dict[str, str] = {}
    last_seen: dict[str, str] = {}
    for row_date, rows in holdings.items():
        for symbol in rows:
            first_seen.setdefault(symbol, row_date)
            last_seen[symbol] = row_date
    periods: list[int] = []
    for symbol, start in first_seen.items():
        end = last_seen.get(symbol, start)
        try:
            periods.append((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days + 1)
        except ValueError:
            continue
    if not periods:
        return {"count": 0, "median_days": None, "min_days": None, "max_days": None}
    return {
        "count": len(periods),
        "median_days": statistics.median(periods),
        "min_days": min(periods),
        "max_days": max(periods),
    }


def _read_price_panel(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    bars_root: Path,
    *,
    prefer_adjusted: bool = True,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    warnings: list[str] = []
    symbol_set = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    if not symbol_set or not bars_root.exists():
        return {}, warnings
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return {}, ["pandas/pyarrow unavailable; local bar based attribution skipped."]
    filters = [
        ("trade_date", ">=", _compact_date(start_date)),
        ("trade_date", "<=", _compact_date(end_date)),
    ]
    try:
        with _suppress_native_stderr():
            frame = pd.read_parquet(bars_root, columns=["ts_code", "trade_date", "close", "adj_close"], filters=filters)
    except Exception as exc:
        return {}, [f"failed to read local bars: {exc}"]
    if frame.empty:
        return {}, warnings
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame = frame[frame["ts_code"].isin(symbol_set)].copy()
    if frame.empty:
        return {}, warnings
    close = pd.to_numeric(frame["close"], errors="coerce")
    adj_close = pd.to_numeric(frame.get("adj_close"), errors="coerce") if "adj_close" in frame else close
    if prefer_adjusted:
        frame["price"] = adj_close.where(adj_close.notna() & (adj_close > 0), close)
    else:
        frame["price"] = close
    frame["date"] = frame["trade_date"].astype(str).map(_iso_from_any)
    frame = frame.dropna(subset=["price"]).sort_values(["ts_code", "date"])
    prices: dict[str, dict[str, float]] = defaultdict(dict)
    for row in frame.itertuples():
        if float(row.price) > 0:
            prices[str(row.ts_code)][str(row.date)] = float(row.price)
    return dict(prices), warnings


def _read_open_dates(start_date: str, end_date: str, bars_root: Path) -> tuple[set[str], list[str]]:
    if not bars_root.exists():
        return set(), []
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return set(), ["pandas/pyarrow unavailable; local trade-calendar annotation skipped."]
    filters = [
        ("trade_date", ">=", _compact_date(start_date)),
        ("trade_date", "<=", _compact_date(end_date)),
    ]
    try:
        with _suppress_native_stderr():
            frame = pd.read_parquet(bars_root, columns=["trade_date"], filters=filters)
    except Exception as exc:
        return set(), [f"failed to read local open-date calendar from bars: {exc}"]
    if frame.empty or "trade_date" not in frame:
        return set(), []
    return {_iso_from_any(value) for value in frame["trade_date"].dropna().astype(str).unique()}, []


def _annotate_trade_calendar_status(
    trades: list[dict[str, Any]],
    dates: list[str],
    bars_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not dates:
        return [dict(trade) for trade in trades], {"available": False, "reason": "missing_nav_dates"}
    open_dates, warnings = _read_open_dates(dates[0], dates[-1], bars_root)
    annotated: list[dict[str, Any]] = []
    weekend_rows: list[dict[str, Any]] = []
    non_open_rows: list[dict[str, Any]] = []
    for trade in trades:
        row = dict(trade)
        trade_date = str(row.get("date") or "")
        status = "unclassified"
        reason = ""
        if row.get("status") != "filled":
            status = "not_filled"
        elif open_dates and trade_date not in open_dates:
            try:
                weekday = datetime.strptime(trade_date, "%Y-%m-%d").weekday()
            except ValueError:
                weekday = -1
            status = "weekend_paper_fill" if weekday >= 5 else "non_trading_paper_fill"
            reason = "date_not_in_local_bars_open_dates"
            payload = {
                "date": trade_date,
                "symbol": row.get("symbol"),
                "action": row.get("action"),
                "shares": row.get("shares"),
                "price": row.get("price"),
                "calendar_status": status,
            }
            non_open_rows.append(payload)
            if status == "weekend_paper_fill":
                weekend_rows.append(payload)
        else:
            status = "trading_day"
        row["calendar_status"] = status
        row["calendar_exclusion_reason"] = reason
        row["excluded_from_phase14_e_l_default"] = status in {"weekend_paper_fill", "non_trading_paper_fill"}
        annotated.append(row)
    diagnostics = {
        "available": bool(open_dates),
        "source": "local_bars_trade_date_set",
        "open_date_count": len(open_dates),
        "weekend_paper_fill_count": len(weekend_rows),
        "non_open_paper_fill_count": len(non_open_rows),
        "weekend_paper_fills": weekend_rows,
        "non_open_paper_fills": non_open_rows,
        "warnings": warnings,
    }
    return annotated, diagnostics


def _calendar_eligible_trade(trade: dict[str, Any]) -> bool:
    return str(trade.get("calendar_status") or "trading_day") not in {
        "weekend_paper_fill",
        "non_trading_paper_fill",
    }


def _price_at_or_after(series: dict[str, float], target: str) -> tuple[str, float] | None:
    for row_date in sorted(series):
        if row_date >= target and series[row_date] > 0:
            return row_date, series[row_date]
    return None


def _price_at_or_before(series: dict[str, float], target: str) -> tuple[str, float] | None:
    candidates = [row_date for row_date in sorted(series) if row_date <= target and series[row_date] > 0]
    if not candidates:
        return None
    row_date = candidates[-1]
    return row_date, series[row_date]


def _forward_return(series: dict[str, float], start: str, horizon: int) -> float | None:
    start_pair = _price_at_or_after(series, start)
    if start_pair is None:
        return None
    dates = sorted(date_key for date_key in series if date_key >= start_pair[0])
    if len(dates) <= horizon:
        return None
    end_price = series[dates[horizon]]
    if start_pair[1] <= 0 or end_price <= 0:
        return None
    return end_price / start_pair[1] - 1.0


def _counterfactual_nav(
    dates: list[str],
    entries: dict[str, str],
    prices: dict[str, dict[str, float]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    entry_prices: dict[str, tuple[str, float]] = {}
    for symbol, entry_date in entries.items():
        series = prices.get(symbol, {})
        pair = _price_at_or_after(series, entry_date)
        if pair is not None:
            entry_prices[symbol] = pair
    for row_date in dates:
        values: list[float] = []
        for symbol, (first_date, first_price) in entry_prices.items():
            if row_date < first_date:
                continue
            pair = _price_at_or_before(prices.get(symbol, {}), row_date)
            if pair is not None and first_price > 0:
                values.append(pair[1] / first_price)
        result[row_date] = statistics.mean(values) if values else 1.0
    return result


def _return_pairs(
    nav_rows: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, float]],
    field: str,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for prev, current in zip(nav_rows, nav_rows[1:]):
        prev_nav = _safe_float(prev.get("nav"))
        current_nav = _safe_float(current.get("nav"))
        prev_bench = _safe_float(benchmarks.get(str(prev.get("date")), {}).get(field), None)
        current_bench = _safe_float(benchmarks.get(str(current.get("date")), {}).get(field), None)
        if (
            prev_nav <= 0
            or current_nav <= 0
            or prev_bench is None
            or current_bench is None
            or prev_bench <= 0
        ):
            continue
        pairs.append(
            {
                "date": current.get("date"),
                "portfolio_return": current_nav / prev_nav - 1.0,
                "benchmark_return": current_bench / prev_bench - 1.0,
            }
        )
    return pairs


def _ols_alpha_beta(
    y_values: list[float],
    x_values: list[float],
) -> dict[str, float | int | None]:
    n = len(y_values)
    if n == 0:
        return {"n": 0, "alpha_daily": None, "beta": None, "alpha_t": None, "r_squared": None}
    mean_y = statistics.mean(y_values)
    mean_x = statistics.mean(x_values)
    sxx = sum((x - mean_x) ** 2 for x in x_values)
    if n < 2 or sxx <= 0:
        return {
            "n": n,
            "alpha_daily": mean_y,
            "beta": None,
            "alpha_t": None,
            "r_squared": None,
        }
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x
    residuals = [y - (alpha + beta * x) for x, y in zip(x_values, y_values)]
    sse = sum(value * value for value in residuals)
    sst = sum((y - mean_y) ** 2 for y in y_values)
    r_squared = 1.0 - sse / sst if sst > 0 else None
    alpha_t = None
    if n > 2:
        sigma2 = sse / (n - 2)
        se_alpha = math.sqrt(max(0.0, sigma2 * (1.0 / n + mean_x * mean_x / sxx)))
        if se_alpha > 0:
            alpha_t = alpha / se_alpha
    return {
        "n": n,
        "alpha_daily": alpha,
        "beta": beta,
        "alpha_t": alpha_t,
        "r_squared": r_squared,
    }


def _benchmark_navs(
    dates: list[str],
    benchmark_file: Path,
    bars_root: Path,
    stock_basic_root: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    by_field: dict[str, dict[str, float]] = defaultdict(dict)
    sources: set[str] = set()
    for row in _read_csv_rows(benchmark_file):
        field = BENCHMARK_CODE_TO_FIELD.get(str(row.get("ts_code") or "").strip())
        if not field:
            continue
        close = _safe_float(row.get("close"))
        row_date = _iso_from_any(row.get("date"))
        if row_date and close > 0:
            by_field[field][row_date] = close
            sources.add(row.get("source_system") or "")
    navs: dict[str, dict[str, float]] = defaultdict(dict)
    for field, closes in by_field.items():
        first = _price_at_or_after(closes, dates[0]) if dates else None
        if first is None:
            continue
        for row_date in dates:
            pair = _price_at_or_before(closes, row_date)
            if pair is not None:
                navs[row_date][field] = pair[1] / first[1]
    industry_nav, industry_meta, industry_warnings = _industry_ew_nav(dates, bars_root, stock_basic_root)
    warnings.extend(industry_warnings)
    for row_date, nav in industry_nav.items():
        navs[row_date]["industry_ew_nav"] = nav
    metadata = {
        "source_system": "+".join(sorted(source for source in sources if source)),
        "industry_equal_weight": industry_meta,
    }
    return dict(navs), metadata, warnings


def _industry_ew_nav(
    dates: list[str],
    bars_root: Path,
    stock_basic_root: Path,
) -> tuple[dict[str, float], dict[str, Any], list[str]]:
    if not dates or not bars_root.exists() or not stock_basic_root.exists():
        return {}, {}, []
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return {}, {}, ["pandas/pyarrow unavailable; industry_ew_nav skipped."]
    try:
        with _suppress_native_stderr():
            stock_basic = pd.read_parquet(stock_basic_root, columns=["ts_code", "industry"])
    except Exception as exc:
        return {}, {}, [f"failed to read stock_basic for industry_ew_nav: {exc}"]
    stock_basic["industry"] = stock_basic["industry"].astype(str).str.strip()
    stock_basic["ts_code"] = stock_basic["ts_code"].astype(str).str.strip().str.upper()
    members = set(stock_basic[stock_basic["industry"].isin(TECH_MANUFACTURING_INDUSTRIES)]["ts_code"].tolist())
    prices, warnings = _read_price_panel(members, dates[0], dates[-1], bars_root)
    if not prices:
        return {}, {"member_count": len(members), "industry_list": list(TECH_MANUFACTURING_INDUSTRIES)}, warnings
    returns_by_date: dict[str, list[float]] = defaultdict(list)
    for series in prices.values():
        ordered = sorted(series)
        for prev_date, row_date in zip(ordered, ordered[1:]):
            prev = series[prev_date]
            current = series[row_date]
            if prev > 0 and current > 0:
                returns_by_date[row_date].append(current / prev - 1.0)
    nav_by_calendar: dict[str, float] = {}
    nav = 1.0
    for row_date in sorted(set().union(*[set(series) for series in prices.values()])):
        if row_date in returns_by_date and returns_by_date[row_date]:
            nav *= 1.0 + statistics.mean(returns_by_date[row_date])
        nav_by_calendar[row_date] = nav
    result: dict[str, float] = {}
    for row_date in dates:
        pair = _price_at_or_before(nav_by_calendar, row_date)
        if pair is not None:
            result[row_date] = pair[1]
    meta = {
        "member_count": len(members),
        "industry_list": list(TECH_MANUFACTURING_INDUSTRIES),
        "valid_date_count": len(nav_by_calendar),
        "start_date": min(nav_by_calendar) if nav_by_calendar else "",
        "end_date": max(nav_by_calendar) if nav_by_calendar else "",
    }
    return result, meta, warnings


def _benchmark_metrics(
    nav_rows: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, float]],
) -> dict[str, Any]:
    actual_return = _safe_float(nav_rows[-1]["nav"]) / _safe_float(nav_rows[0]["nav"], 1.0) - 1.0
    rows: dict[str, Any] = {}
    dates = [row["date"] for row in nav_rows]
    for field in ("star50_nav", "chinext_nav", "csi300_nav", "industry_ew_nav"):
        values = [benchmarks.get(row_date, {}).get(field) for row_date in dates]
        valid = [(date_value, value) for date_value, value in zip(dates, values) if value and value > 0]
        if len(valid) < 2:
            rows[field] = {"available": False}
            continue
        aligned_series = [
            {
                "date": date_value,
                "actual_nav": _safe_float(nav_rows[dates.index(date_value)]["nav"]),
                "benchmark_nav": value,
            }
            for date_value, value in valid
        ]
        bench_return = valid[-1][1] / valid[0][1] - 1.0
        rolling: list[dict[str, Any]] = []
        for index in range(19, len(valid)):
            start_value = valid[index - 19][1]
            end_value = valid[index][1]
            actual_start = _safe_float(nav_rows[dates.index(valid[index - 19][0])]["nav"])
            actual_end = _safe_float(nav_rows[dates.index(valid[index][0])]["nav"])
            if start_value > 0 and actual_start > 0:
                rolling.append(
                    {
                        "date": valid[index][0],
                        "rolling_20d_excess": (actual_end / actual_start - 1.0) - (end_value / start_value - 1.0),
                    }
                )
        rows[field] = {
            "available": True,
            "benchmark_return": bench_return,
            "full_window_excess": actual_return - bench_return,
            "rolling_20d_excess": rolling,
            "aligned_series": aligned_series,
        }
    return rows


def _benchmark_return_between(
    benchmarks: dict[str, dict[str, float]],
    field: str,
    start_date: str,
    end_date: str,
) -> float | None:
    start = _safe_float(benchmarks.get(start_date, {}).get(field), None)
    end = _safe_float(benchmarks.get(end_date, {}).get(field), None)
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1.0


def _beta_adjusted_excess(
    nav_rows: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, float]],
) -> dict[str, Any]:
    pairs = _return_pairs(nav_rows, benchmarks, "star50_nav")
    y_values = [_safe_float(row["portfolio_return"]) for row in pairs]
    x_values = [_safe_float(row["benchmark_return"]) for row in pairs]
    regression = _ols_alpha_beta(y_values, x_values)
    alpha_daily = regression.get("alpha_daily")
    excess = [y - x for x, y in zip(x_values, y_values)]
    excess_std = statistics.pstdev(excess) if len(excess) > 1 else 0.0
    ir_daily = statistics.mean(excess) / excess_std if excess and excess_std > 0 else None
    actual_return = _safe_float(nav_rows[-1]["nav"]) / _safe_float(nav_rows[0]["nav"], 1.0) - 1.0 if nav_rows else None
    star50_return = (
        _benchmark_return_between(benchmarks, "star50_nav", nav_rows[0]["date"], nav_rows[-1]["date"])
        if nav_rows
        else None
    )
    beta = regression.get("beta")
    beta_times_star50 = (
        beta * star50_return
        if beta is not None and star50_return is not None
        else None
    )
    triggered = (
        beta_times_star50 is not None
        and actual_return is not None
        and beta_times_star50 >= actual_return
    )
    return {
        **regression,
        "alpha_annualized": alpha_daily * 244.0 if alpha_daily is not None else None,
        "standard_ir_daily": ir_daily,
        "standard_ir_annualized": ir_daily * math.sqrt(244.0) if ir_daily is not None else None,
        "actual_return": actual_return,
        "star50_return": star50_return,
        "beta_times_star50_return": beta_times_star50,
        "interpretation_triggered": triggered,
        "interpretation_line": (
            "β 调整后 vs 科创50 无正 α，原始超额来自杠杆性暴露而非风险调整后技能，待后续独立样本复验。"
            if triggered
            else "未触发"
        ),
        "fixed_sample_note": "n=69 交易日，单一政权样本，t 值与 IR 仅作描述，不作推断。",
    }


def _decomposition(
    nav_rows: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, float]],
    counterfactual_nav: dict[str, float],
) -> dict[str, Any]:
    start_date = nav_rows[0]["date"]
    end_date = nav_rows[-1]["date"]
    actual_return = _safe_float(nav_rows[-1]["nav"]) / _safe_float(nav_rows[0]["nav"], 1.0) - 1.0

    def bench_return(field: str) -> float:
        start = _safe_float(benchmarks.get(start_date, {}).get(field), 1.0)
        end = _safe_float(benchmarks.get(end_date, {}).get(field), start)
        return end / start - 1.0 if start > 0 else 0.0

    csi300_return = bench_return("csi300_nav")
    industry_return = bench_return("industry_ew_nav")
    cf_start = _safe_float(counterfactual_nav.get(start_date), 1.0)
    cf_end = _safe_float(counterfactual_nav.get(end_date), cf_start)
    cf_return = cf_end / cf_start - 1.0 if cf_start > 0 else 0.0
    config_beta = industry_return - csi300_return
    stock_alpha = cf_return - industry_return
    timing_alpha = actual_return - cf_return
    total_excess = actual_return - csi300_return
    explained = config_beta + stock_alpha + timing_alpha
    return {
        "actual_return": actual_return,
        "csi300_return": csi300_return,
        "industry_ew_return": industry_return,
        "counterfactual1_return": cf_return,
        "config_beta": config_beta,
        "stock_alpha": stock_alpha,
        "phase_timing_alpha": timing_alpha,
        "total_excess_vs_csi300": total_excess,
        "explained_sum": explained,
        "reconciliation_residual": total_excess - explained,
    }


def _counterfactual2_decomposition(
    nav_rows: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, float]],
    counterfactual1_nav: dict[str, float],
    prices: dict[str, dict[str, float]],
    symbols: Iterable[str],
) -> dict[str, Any]:
    dates = [row["date"] for row in nav_rows]
    start_date = dates[0]
    end_date = dates[-1]
    symbol_list = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    counterfactual2_nav = _counterfactual_nav(
        dates,
        {symbol: start_date for symbol in symbol_list},
        prices,
    )
    adjustments: list[dict[str, Any]] = []
    for symbol in symbol_list:
        pair = _price_at_or_after(prices.get(symbol, {}), start_date)
        if pair is not None and pair[0] > start_date:
            adjustments.append(
                {
                    "symbol": symbol,
                    "window_start": start_date,
                    "first_tradeable_date": pair[0],
                    "note": "首日未上市/停牌或本地行情不可得，使用首个可交易日。",
                }
            )
    cf2_start = _safe_float(counterfactual2_nav.get(start_date), 1.0)
    cf2_end = _safe_float(counterfactual2_nav.get(end_date), cf2_start)
    cf1_start = _safe_float(counterfactual1_nav.get(start_date), 1.0)
    cf1_end = _safe_float(counterfactual1_nav.get(end_date), cf1_start)
    cf2_return = cf2_end / cf2_start - 1.0 if cf2_start > 0 else 0.0
    cf1_return = cf1_end / cf1_start - 1.0 if cf1_start > 0 else 0.0
    industry_return = _benchmark_return_between(benchmarks, "industry_ew_nav", start_date, end_date)
    if industry_return is None:
        industry_return = 0.0
    return {
        "counterfactual2_nav": counterfactual2_nav,
        "counterfactual2_return": cf2_return,
        "industry_ew_return": industry_return,
        "pure_name_stock_alpha": cf2_return - industry_return,
        "entry_timing_contribution": cf1_return - cf2_return,
        "counterfactual1_return": cf1_return,
        "delayed_entry_adjustments": adjustments,
        "symbol_count": len(symbol_list),
    }


def _candidate_pool_rows(record: Record) -> list[dict[str, str]]:
    rows = _read_csv_rows(record.path / "candidate_pool.csv")
    if rows:
        return rows
    for path in sorted((record.path / "raw_exports").glob("*candidate_pool.csv")):
        rows = _read_csv_rows(path)
        if rows:
            return rows
    return []


def _selection_alpha(
    records: list[Record],
    trades: list[dict[str, Any]],
    prices: dict[str, dict[str, float]],
    window_end: str,
) -> dict[str, Any]:
    record_by_date = {record.date: record for record in records}
    rows: list[dict[str, Any]] = []
    missing = 0
    actual_returns: list[float] = []
    menu_returns: list[float] = []
    buy_trades = [trade for trade in trades if trade.get("action") == "buy" and trade.get("status") == "filled"]
    for trade in buy_trades:
        record = record_by_date.get(str(trade.get("date")))
        menu = _candidate_pool_rows(record) if record is not None else []
        symbols = sorted({str(row.get("symbol") or "").strip().upper() for row in menu if str(row.get("symbol") or "").strip()})
        actual_symbol = str(trade.get("symbol") or "").strip().upper()
        if not symbols:
            missing += 1
            rows.append(
                {
                    "date": trade.get("date"),
                    "symbol": actual_symbol,
                    "coverage_status": "missing_shortlist",
                    "actual_return": None,
                    "menu_equal_weight_return": None,
                    "selection_alpha": None,
                }
            )
            continue

        def hold_return(symbol: str) -> float | None:
            series = prices.get(symbol, {})
            start_pair = _price_at_or_after(series, str(trade.get("date") or ""))
            end_pair = _price_at_or_before(series, window_end)
            if start_pair is None or end_pair is None or start_pair[1] <= 0:
                return None
            return end_pair[1] / start_pair[1] - 1.0

        actual_return = hold_return(actual_symbol)
        menu_values = [value for value in (hold_return(symbol) for symbol in symbols) if value is not None]
        menu_return = statistics.mean(menu_values) if menu_values else None
        selection = actual_return - menu_return if actual_return is not None and menu_return is not None else None
        if actual_return is not None:
            actual_returns.append(actual_return)
        if menu_return is not None:
            menu_returns.append(menu_return)
        rows.append(
            {
                "date": trade.get("date"),
                "symbol": actual_symbol,
                "coverage_status": "covered" if selection is not None else "price_missing",
                "menu_size": len(symbols),
                "actual_return": actual_return,
                "menu_equal_weight_return": menu_return,
                "selection_alpha": selection,
            }
        )
    covered = [row for row in rows if row.get("selection_alpha") is not None]
    selection_alpha = (
        statistics.mean([row["selection_alpha"] for row in covered])
        if covered
        else None
    )
    triggered_line = "缺少足够菜单覆盖，无法判读"
    if selection_alpha is not None:
        if abs(selection_alpha) < 0.01:
            triggered_line = "选择 α ≈ 0 → 价值在菜单生成，不在挑选"
        elif selection_alpha > 0:
            triggered_line = "选择 α 显著为正 → 路由环节（人或会话代理）存在增量"
        else:
            triggered_line = "选择 α 为负 → 路由环节减损菜单价值"
    return {
        "buy_trade_count": len(buy_trades),
        "covered_count": len(covered),
        "missing_shortlist_count": missing,
        "coverage_rate": len(covered) / len(buy_trades) if buy_trades else None,
        "selected_mean_return": statistics.mean(actual_returns) if actual_returns else None,
        "menu_mean_return": statistics.mean(menu_returns) if menu_returns else None,
        "selection_alpha": selection_alpha,
        "interpretation_line": triggered_line,
        "rows": rows,
    }


def _symbol_return_on_date(
    prices: dict[str, dict[str, float]],
    symbol: str,
    prev_date: str,
    current_date: str,
) -> float | None:
    series = prices.get(symbol, {})
    prev_pair = _price_at_or_before(series, prev_date)
    current_pair = _price_at_or_before(series, current_date)
    if prev_pair is None or current_pair is None or prev_pair[1] <= 0:
        return None
    return current_pair[1] / prev_pair[1] - 1.0


def _weights_from_record(record: Record) -> dict[str, float]:
    exposure = _record_exposure(record)
    total_value = _safe_float(exposure.get("total_value"))
    if total_value <= 0:
        return {}
    weights: dict[str, float] = {}
    for row in record.holdings:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        value = _safe_float(row.get("current_value"))
        if value > 0:
            weights[symbol] = value / total_value
    return weights


def _cap_and_rescale_weights(weights: dict[str, float], cap: float) -> dict[str, float]:
    if not weights:
        return {}
    gross = sum(max(0.0, value) for value in weights.values())
    capped = {symbol: min(max(0.0, value), cap) for symbol, value in weights.items()}
    capped_gross = sum(capped.values())
    if capped_gross <= 0 or gross <= 0:
        return capped
    scale = min(gross / capped_gross, cap / max(capped.values(), default=cap))
    scaled = {symbol: min(value * scale, cap) for symbol, value in capped.items()}
    return scaled


def _shadow_cap_nav(
    records: list[Record],
    nav_rows: list[dict[str, Any]],
    prices: dict[str, dict[str, float]],
    cap: float,
) -> dict[str, float]:
    record_by_run_id = {record.run_id: record for record in records}
    result: dict[str, float] = {}
    if not nav_rows:
        return result
    nav = 1.0
    result[str(nav_rows[0]["date"])] = nav
    for prev_row, current_row in zip(nav_rows, nav_rows[1:]):
        prev_record = record_by_run_id.get(str(prev_row.get("run_id")))
        weights = _cap_and_rescale_weights(_weights_from_record(prev_record), cap) if prev_record else {}
        daily_return = 0.0
        covered_weight = 0.0
        for symbol, weight in weights.items():
            ret = _symbol_return_on_date(prices, symbol, str(prev_row.get("date")), str(current_row.get("date")))
            if ret is None:
                continue
            daily_return += weight * ret
            covered_weight += weight
        if covered_weight <= 0:
            prev_nav = _safe_float(prev_row.get("nav"))
            current_nav = _safe_float(current_row.get("nav"))
            daily_return = current_nav / prev_nav - 1.0 if prev_nav > 0 and current_nav > 0 else 0.0
        nav *= 1.0 + daily_return
        result[str(current_row["date"])] = nav
    return result


def _manual_proxy_trades(
    trades: list[dict[str, Any]],
    attribution: dict[str, Any],
    *,
    include_non_trading: bool = False,
) -> list[dict[str, Any]]:
    if not attribution:
        return []
    proposal_symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in trades
        if row.get("source") == "orders"
    }
    result: list[dict[str, Any]] = []
    for trade in trades:
        if trade.get("action") != "sell" or trade.get("status") != "filled":
            continue
        if not include_non_trading and not _calendar_eligible_trade(trade):
            continue
        symbol = str(trade.get("symbol") or "").strip().upper()
        if trade.get("source") != "orders" or symbol not in proposal_symbols:
            result.append(trade)
    return result


def _stop_price_by_symbol_date(records: list[Record]) -> dict[tuple[str, str], float]:
    stops: dict[tuple[str, str], float] = {}
    for record in records:
        for row in _read_csv_rows(record.path / "holdings_review.csv"):
            symbol = str(row.get("symbol") or "").strip().upper()
            stop = _safe_float(row.get("stage_stop_price"))
            if symbol and stop > 0:
                stops[(record.date, symbol)] = stop
    return stops


def _machine_exit_shadow(
    records: list[Record],
    trades: list[dict[str, Any]],
    prices: dict[str, dict[str, float]],
    nav_rows: list[dict[str, Any]],
    attribution: dict[str, Any],
    *,
    include_non_trading: bool = False,
) -> dict[str, Any]:
    stops = _stop_price_by_symbol_date(records)
    all_manual_sells = _manual_proxy_trades(trades, attribution, include_non_trading=True)
    manual_sells = _manual_proxy_trades(trades, attribution, include_non_trading=include_non_trading)
    excluded_non_trading = [
        {
            "date": trade.get("date"),
            "symbol": trade.get("symbol"),
            "shares": trade.get("shares"),
            "price": trade.get("price"),
            "calendar_status": trade.get("calendar_status"),
        }
        for trade in all_manual_sells
        if not _calendar_eligible_trade(trade)
    ]
    end_date = str(nav_rows[-1]["date"]) if nav_rows else ""
    initial = _safe_float(nav_rows[0].get("initial_capital"), 1.0) if nav_rows else 1.0
    rows: list[dict[str, Any]] = []
    total_delta = 0.0
    for trade in manual_sells:
        symbol = str(trade.get("symbol") or "").strip().upper()
        trade_date = str(trade.get("date") or "")
        trade_price = _safe_float(trade.get("price"))
        shares = _safe_int(trade.get("shares"))
        stop_price = stops.get((trade_date, symbol))
        series = prices.get(symbol, {})
        trigger_date = None
        trigger_price = None
        stop_triggered = False
        if stop_price and trade_price > 0:
            for row_date in sorted(date_key for date_key in series if date_key >= trade_date):
                price = series[row_date]
                if price <= stop_price:
                    trigger_date = row_date
                    trigger_price = price
                    stop_triggered = True
                    break
        if trigger_price is None:
            end_pair = _price_at_or_before(series, end_date)
            if end_pair is not None:
                trigger_date, trigger_price = end_pair
        delta = (trigger_price - trade_price) * shares if trigger_price is not None else None
        if delta is not None:
            total_delta += delta
        rows.append(
            {
                "date": trade_date,
                "symbol": symbol,
                "shares": shares,
                "manual_exit_price": trade_price,
                "machine_exit_date": trigger_date,
                "machine_exit_price": trigger_price,
                "stage_stop_price": stop_price,
                "delta_vs_manual_pnl": delta,
                "stop_triggered": stop_triggered,
                "exit_status": "stop_triggered_exit" if stop_triggered else "not_closed_marked_to_window_end",
                "contribution_pct_of_initial": delta / max(initial, 1.0) if delta is not None else None,
            }
        )
    for row in rows:
        delta = _safe_float(row.get("delta_vs_manual_pnl"), None)
        row["share_of_shadow_delta"] = delta / total_delta if delta is not None and total_delta else None
    return {
        "method": "manual_proxy sell is hypothetically held until recorded stage_stop_price triggers; if not triggered, window-end close is used.",
        "rows": rows,
        "current_difference_vs_actual": total_delta / max(initial, 1.0),
        "manual_proxy_sell_count": len(manual_sells),
        "include_non_trading": include_non_trading,
        "excluded_non_trading_count": 0 if include_non_trading else len(excluded_non_trading),
        "excluded_non_trading_rows": [] if include_non_trading else excluded_non_trading,
    }


def _shadow_ledgers(
    records: list[Record],
    nav_rows: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    prices: dict[str, dict[str, float]],
    execution_prices: dict[str, dict[str, float]],
    attribution: dict[str, Any],
) -> dict[str, Any]:
    cap050 = _shadow_cap_nav(records, nav_rows, prices, 0.50)
    actual_start = _safe_float(nav_rows[0].get("nav"), 1.0) if nav_rows else 1.0
    actual_end = _safe_float(nav_rows[-1].get("nav"), actual_start) if nav_rows else actual_start
    cap050_end = _safe_float(cap050.get(str(nav_rows[-1]["date"])), 1.0) if nav_rows else 1.0
    machine_exit = _machine_exit_shadow(records, trades, execution_prices, nav_rows, attribution)
    machine_exit_with_non_trading = _machine_exit_shadow(
        records,
        trades,
        execution_prices,
        nav_rows,
        attribution,
        include_non_trading=True,
    )
    return {
        "shadow_nav_cap050": cap050,
        "shadow_cap050_return": cap050_end - 1.0,
        "actual_return": actual_end / actual_start - 1.0 if actual_start > 0 else 0.0,
        "cap050_current_difference_vs_actual": cap050_end - (actual_end / actual_start if actual_start > 0 else 1.0),
        "shadow_nav_machine_exit": machine_exit,
        "machine_exit_current_difference_vs_actual": machine_exit.get("current_difference_vs_actual"),
        "machine_exit_sensitivity_including_non_trading": machine_exit_with_non_trading,
    }


def _entry_dates(records: list[Record], trades: list[dict[str, Any]]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for trade in trades:
        if trade.get("status") == "filled" and trade.get("action") == "buy":
            entries.setdefault(str(trade["symbol"]), str(trade["date"]))
    for record in records:
        for row in record.holdings:
            symbol = str(row.get("symbol") or "").strip().upper()
            shares = _safe_int(row.get("shares") or row.get("shares_before"))
            if symbol and shares > 0:
                entries.setdefault(symbol, record.date)
    return entries


def _concentration(
    records: list[Record],
    nav_rows: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    contribution: dict[str, float] = defaultdict(float)
    for trade in trades:
        if trade.get("action") == "sell":
            contribution[str(trade["symbol"])] += _safe_float(trade.get("realized_pnl"))
    final_holdings = _holdings_by_date(records).get(nav_rows[-1]["date"], {}) if nav_rows else {}
    for symbol, row in final_holdings.items():
        contribution[symbol] += _safe_float(row.get("unrealized_pnl"))
    ranked = sorted(
        [{"symbol": symbol, "contribution_pnl": pnl} for symbol, pnl in contribution.items()],
        key=lambda item: item["contribution_pnl"],
        reverse=True,
    )
    total_positive = sum(max(0.0, item["contribution_pnl"]) for item in ranked)
    top3 = ranked[:3]
    top3_pnl = sum(item["contribution_pnl"] for item in top3)
    top3_share = top3_pnl / total_positive if total_positive > 0 else None
    initial = _safe_float(nav_rows[0].get("initial_capital"), 1.0) if nav_rows else 1.0
    actual_return = _safe_float(nav_rows[-1]["nav"]) / _safe_float(nav_rows[0]["nav"], 1.0) - 1.0 if nav_rows else 0.0
    return {
        "ranked_contributors": ranked,
        "top3": top3,
        "top3_positive_contribution_share": top3_share,
        "return_excluding_top3_estimate": actual_return - top3_pnl / max(initial, 1.0),
    }


def _counterparty_quality(
    trades: list[dict[str, Any]],
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side_name, action in (("entries", "buy"), ("exits", "sell")):
        rows: list[dict[str, Any]] = []
        for trade in trades:
            if trade.get("action") != action:
                continue
            if not _calendar_eligible_trade(trade):
                continue
            series = prices.get(str(trade.get("symbol")))
            if not series:
                continue
            row = {"date": trade["date"], "symbol": trade["symbol"]}
            for horizon in (5, 10, 20):
                row[f"ret_{horizon}d"] = _forward_return(series, trade["date"], horizon)
            rows.append(row)
        summary: dict[str, Any] = {"count": len(rows), "rows": rows}
        for horizon in (5, 10, 20):
            values = [row[f"ret_{horizon}d"] for row in rows if row.get(f"ret_{horizon}d") is not None]
            summary[f"ret_{horizon}d"] = {
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "negative_share": sum(1 for value in values if value < 0) / len(values) if values else None,
            }
        result[side_name] = summary
    result["interpretation_rule"] = (
        "出场后收益显著为负 => 卖给晚期追高者；显著为正 => 我方卖早。"
    )
    return result


def _counterparty_quality_by_exit_bucket(
    trades: list[dict[str, Any]],
    prices: dict[str, dict[str, float]],
    attribution: dict[str, Any],
) -> dict[str, Any]:
    manual_keys = {
        (
            trade.get("date"),
            trade.get("symbol"),
            trade.get("shares"),
            round(_safe_float(trade.get("price")), 4),
        )
        for trade in _manual_proxy_trades(trades, attribution)
    }
    buckets = {"manual_proxy_sell": [], "system_sell": []}
    for trade in trades:
        if trade.get("action") != "sell" or trade.get("status") != "filled":
            continue
        if not _calendar_eligible_trade(trade):
            continue
        key = (
            trade.get("date"),
            trade.get("symbol"),
            trade.get("shares"),
            round(_safe_float(trade.get("price")), 4),
        )
        bucket = "manual_proxy_sell" if key in manual_keys else "system_sell"
        series = prices.get(str(trade.get("symbol")))
        if not series:
            continue
        row = {"date": trade["date"], "symbol": trade["symbol"]}
        for horizon in (5, 10, 20):
            row[f"ret_{horizon}d"] = _forward_return(series, trade["date"], horizon)
        buckets[bucket].append(row)
    result: dict[str, Any] = {}
    for bucket, rows in buckets.items():
        summary: dict[str, Any] = {"count": len(rows), "rows": rows}
        for horizon in (5, 10, 20):
            values = [row[f"ret_{horizon}d"] for row in rows if row.get(f"ret_{horizon}d") is not None]
            summary[f"ret_{horizon}d"] = {
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "negative_share": sum(1 for value in values if value < 0) / len(values) if values else None,
            }
        result[bucket] = summary
    return result


def _peak_weights(records: list[Record], symbols: Iterable[str]) -> dict[str, Any]:
    wanted = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    peaks = {
        symbol: {"peak_weight": None, "date": "", "run_id": ""}
        for symbol in sorted(wanted)
    }
    for record in records:
        weights = _weights_from_record(record)
        for symbol in wanted:
            weight = weights.get(symbol)
            if weight is None:
                continue
            current_peak = peaks[symbol]["peak_weight"]
            if current_peak is None or weight > current_peak:
                peaks[symbol] = {
                    "peak_weight": weight,
                    "date": record.date,
                    "run_id": record.run_id,
                }
    return peaks


def _read_amount_panel(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    bars_root: Path,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    warnings: list[str] = []
    symbol_set = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    if not symbol_set or not bars_root.exists():
        return {}, warnings
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return {}, ["pandas/pyarrow unavailable; execution-cost amount panel skipped."]
    filters = [
        ("trade_date", ">=", _compact_date(start_date)),
        ("trade_date", "<=", _compact_date(end_date)),
    ]
    try:
        with _suppress_native_stderr():
            frame = pd.read_parquet(bars_root, columns=["ts_code", "trade_date", "amount"], filters=filters)
    except Exception as exc:
        return {}, [f"failed to read amount panel for execution-cost estimate: {exc}"]
    if frame.empty or "amount" not in frame:
        return {}, warnings
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame = frame[frame["ts_code"].isin(symbol_set)].copy()
    frame["date"] = frame["trade_date"].astype(str).map(_iso_from_any)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    amounts: dict[str, dict[str, float]] = defaultdict(dict)
    for row in frame.dropna(subset=["amount"]).itertuples():
        if float(row.amount) > 0:
            amounts[str(row.ts_code)][str(row.date)] = float(row.amount)
    return dict(amounts), warnings


def _estimated_execution_cost(
    trades: list[dict[str, Any]],
    nav_rows: list[dict[str, Any]],
    bars_root: Path,
) -> dict[str, Any]:
    symbols = sorted({str(trade.get("symbol") or "").strip().upper() for trade in trades if trade.get("symbol")})
    dates = [str(row.get("date") or "") for row in nav_rows]
    amount_panel, warnings = _read_amount_panel(symbols, dates[0], dates[-1], bars_root) if dates else ({}, [])
    try:
        from quant_investor.factors.execution_cost import (  # type: ignore[import-not-found]
            FactorExecutionCostConfig,
            estimate_market_impact_bps,
            estimate_participation_rate,
        )
        config = FactorExecutionCostConfig(config_id="phase14-audit-estimated-cost")
    except Exception as exc:
        return {"available": False, "reason": f"execution cost helpers unavailable: {exc}", "warnings": warnings}
    nav_by_date = {str(row.get("date")): _safe_float(row.get("total_value_after")) for row in nav_rows}
    total_cost = 0.0
    rows: list[dict[str, Any]] = []
    for trade in trades:
        if trade.get("status") != "filled":
            continue
        symbol = str(trade.get("symbol") or "").strip().upper()
        trade_value = abs(_safe_float(trade.get("trade_value")))
        if trade_value <= 0:
            trade_value = abs(_safe_float(trade.get("price")) * _safe_int(trade.get("shares")))
        portfolio_value = nav_by_date.get(str(trade.get("date"))) or statistics.mean(nav_by_date.values())
        amount = _safe_float(amount_panel.get(symbol, {}).get(str(trade.get("date"))), None)
        participation = estimate_participation_rate(
            trade_weight=trade_value / portfolio_value if portfolio_value else 0.0,
            portfolio_value=portfolio_value,
            amount=amount,
        )
        impact_bps = estimate_market_impact_bps(participation_rate=participation, config=config)
        side = str(trade.get("action"))
        bps = config.commission_bps + config.exchange_fee_bps + config.slippage_bps + config.spread_bps + impact_bps
        if side == "sell" and config.apply_stamp_tax_on_sell_only:
            bps += config.stamp_tax_bps
        cost = trade_value * bps / 10000.0
        total_cost += cost
        rows.append(
            {
                "date": trade.get("date"),
                "symbol": symbol,
                "action": side,
                "trade_value": trade_value,
                "participation_rate": participation,
                "estimated_bps": bps,
                "estimated_cost": cost,
            }
        )
    initial = _safe_float(nav_rows[0].get("initial_capital"), 1.0) if nav_rows else 1.0
    actual_return = _safe_float(nav_rows[-1]["nav"]) / _safe_float(nav_rows[0]["nav"], 1.0) - 1.0 if nav_rows else 0.0
    avg_nav = statistics.mean(_safe_float(row.get("total_value_after")) for row in nav_rows) if nav_rows else 0.0
    sample_days = max(len(nav_rows) - 1, 1)
    return {
        "available": True,
        "config": config.to_dict(),
        "trade_count": len(rows),
        "gross_full_window_return": actual_return,
        "estimated_total_cost": total_cost,
        "net_full_window_return": actual_return - total_cost / max(initial, 1.0),
        "estimated_cost_drag": total_cost / avg_nav if avg_nav > 0 else None,
        "annualized_cost_drag": (total_cost / avg_nav) * (244.0 / sample_days) if avg_nav > 0 else None,
        "rows": rows,
        "warnings": warnings,
        "note": "审计层模型估计，未改变成交、组合或运行时成本模型。",
    }


def _execution_quality(records: list[Record], trades: list[dict[str, Any]], rejections: list[dict[str, Any]]) -> dict[str, Any]:
    proposal_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        for row in record.orders:
            symbol = str(row.get("symbol") or "").strip().upper()
            action = _normalize_action(row.get("action"))
            if symbol:
                proposal_by_key[(record.run_id, symbol, action)] = row
    slippage: dict[str, list[float]] = {"buy": [], "sell": []}
    total_slippage_cost = 0.0
    for trade in trades:
        proposal = proposal_by_key.get((trade["run_id"], trade["symbol"], trade["action"]))
        if not proposal:
            continue
        proposal_price = _safe_float(proposal.get("price"))
        actual_price = _safe_float(trade.get("price"))
        if proposal_price <= 0 or actual_price <= 0:
            continue
        if trade["action"] == "buy":
            bps = (actual_price / proposal_price - 1.0) * 10000.0
            total_slippage_cost += (actual_price - proposal_price) * _safe_int(trade.get("shares"))
        else:
            bps = (proposal_price / actual_price - 1.0) * 10000.0
            total_slippage_cost += (proposal_price - actual_price) * _safe_int(trade.get("shares"))
        slippage.setdefault(trade["action"], []).append(bps)
    rejection_reasons: dict[str, int] = defaultdict(int)
    for rejection in rejections:
        rejection_reasons[str(rejection.get("reason") or "unknown")] += 1
    avg_nav = statistics.mean(
        _safe_float(record.pnl.get("total_value_after"))
        for record in records
        if _safe_float(record.pnl.get("total_value_after")) > 0
    ) if records else 0.0
    return {
        "slippage_bps": {
            side: {
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "p90": _quantile(values, 0.90),
            }
            for side, values in slippage.items()
        },
        "rejection_count": len(rejections),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "estimated_cost_drag": total_slippage_cost / avg_nav if avg_nav > 0 else None,
    }


def _slippage_zero_root_cause(execution_quality: dict[str, Any]) -> dict[str, Any]:
    slippage = execution_quality.get("slippage_bps", {})
    counts = sum(int(data.get("count", 0) or 0) for data in slippage.values() if isinstance(data, dict))
    nonzero = [
        data
        for data in slippage.values()
        if isinstance(data, dict)
        and (
            abs(_safe_float(data.get("mean"))) > 1e-12
            or abs(_safe_float(data.get("median"))) > 1e-12
        )
    ]
    if counts > 0 and not nonzero:
        conclusion = (
            "F=0 is caused by the local/manual simulation design: proposal and execution rows share "
            "the same realtime/current price field for matched paper fills, so this audit has no "
            "independent broker fill price for empirical slippage."
        )
    else:
        conclusion = "F slippage is non-zero or unavailable; no zero-slippage root-cause conclusion."
    return {
        "conclusion": conclusion,
        "evidence": [
            "scripts/run_track_record_audit.py:_execution_quality compares orders.csv price to filled trade price.",
            "Record schema stores local/manual fills in manual_switch_and_take_profit_orders.csv with price/current quote basis.",
        ],
        "matched_slippage_count": counts,
    }


def _version_events() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            ["git", "log", "--format=%ci %s"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    keywords = ("kline", "v13", "runbook", "Phase 12", "retire", "merge")
    events: list[dict[str, str]] = []
    for line in output.splitlines():
        if any(keyword.lower() in line.lower() for keyword in keywords) or "2026-07-05" in line or "2026-07-06" in line:
            events.append({"date": _iso_from_any(line[:25]), "subject": line[26:]})
        if len(events) >= 40:
            break
    return events


def _version_segment_metrics(
    nav_rows: list[dict[str, Any]],
    benchmarks: dict[str, dict[str, float]],
    version_events: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if len(nav_rows) < 2:
        return []
    start = nav_rows[0]["date"]
    end = nav_rows[-1]["date"]
    cut_dates = sorted({event["date"] for event in version_events if start < event.get("date", "") < end})
    boundaries = [start, *cut_dates, end]
    segments: list[dict[str, Any]] = []
    for index, segment_start in enumerate(boundaries[:-1]):
        segment_end = boundaries[index + 1]
        rows = [row for row in nav_rows if segment_start <= row["date"] <= segment_end]
        if len(rows) < 2:
            continue
        actual = _safe_float(rows[-1]["nav"]) / _safe_float(rows[0]["nav"], 1.0) - 1.0
        segment: dict[str, Any] = {
            "start": rows[0]["date"],
            "end": rows[-1]["date"],
            "actual_return": actual,
        }
        for field in ("star50_nav", "csi300_nav", "industry_ew_nav"):
            start_value = _safe_float(benchmarks.get(rows[0]["date"], {}).get(field))
            end_value = _safe_float(benchmarks.get(rows[-1]["date"], {}).get(field))
            if start_value > 0 and end_value > 0:
                bench_return = end_value / start_value - 1.0
                segment[f"{field}_return"] = bench_return
                segment[f"excess_vs_{field}"] = actual - bench_return
        segments.append(segment)
    return segments


def _manual_system_attribution(trades: list[dict[str, Any]], records: list[Record]) -> dict[str, Any]:
    proposal_keys = {
        (record.run_id, str(row.get("symbol") or "").strip().upper(), _normalize_action(row.get("action")))
        for record in records
        for row in record.orders
    }
    buckets: dict[str, list[dict[str, Any]]] = {"system": [], "manual_proxy": []}
    for trade in trades:
        key = (trade["run_id"], trade["symbol"], trade["action"])
        if key in proposal_keys and trade.get("source") == "orders":
            buckets["system"].append(trade)
        elif key in proposal_keys and trade.get("source") == "manual_switch_and_take_profit_orders":
            buckets["system"].append(trade)
        else:
            buckets["manual_proxy"].append(trade)
    summary: dict[str, Any] = {
        "classification_method": (
            "source fields are sparse; unmatched filled manual rows or fills after rejected proposals are classified as manual_proxy."
        )
    }
    for name, rows in buckets.items():
        realized = [_safe_float(row.get("realized_pnl")) for row in rows if row.get("action") == "sell"]
        summary[name] = {
            "trade_count": len(rows),
            "realized_pnl": sum(realized),
            "hit_rate": sum(1 for value in realized if value > 0) / len(realized) if realized else None,
        }
    return summary


def _funding_nature(records: list[Record]) -> dict[str, Any]:
    modes = {
        str(record.manual_manifest.get("manual_execution_mode") or record.manifest.get("manual_execution", {}).get("manual_execution_mode") or "")
        for record in records
    }
    broker_flags = [
        record.manual_manifest.get("broker_api_called")
        for record in records
        if "broker_api_called" in record.manual_manifest
    ]
    no_broker_flags = [
        record.manual_manifest.get("no_broker_api_called")
        for record in records
        if "no_broker_api_called" in record.manual_manifest
    ]
    if any(flag is True for flag in broker_flags):
        nature = "mixed_or_real_broker_execution_detected"
    elif any("paper_only_local_manual_no_broker" in mode for mode in modes) or any(flag is True for flag in no_broker_flags):
        nature = "local_manual_paper_no_broker"
    else:
        nature = "undetermined_early_records"
    return {
        "funding_nature": nature,
        "manual_execution_modes": sorted(mode for mode in modes if mode),
        "broker_api_called_values": broker_flags,
        "no_broker_api_called_values": no_broker_flags,
        "basis": "manual_execution_manifest/manual_execution_mode, no_broker_api_called, broker_api_called, and price_basis fields.",
    }


def _record_exposure(record: Record) -> dict[str, Any]:
    total_value = _safe_float(record.pnl.get("total_value_after"))
    cash = _safe_float(record.pnl.get("cash_after"), None)
    market_value = 0.0
    holding_count = 0
    for row in record.holdings:
        shares = _safe_int(row.get("shares") or row.get("shares_before"))
        if shares <= 0:
            continue
        holding_count += 1
        market_value += _safe_float(row.get("current_value"))
    if market_value <= 0:
        market_value = _safe_float(record.pnl.get("market_value_after"))
    if cash is None and total_value > 0:
        cash = max(0.0, total_value - market_value)
    return {
        "date": record.date,
        "run_id": record.run_id,
        "total_value": total_value,
        "market_value": market_value,
        "cash": cash,
        "actual_total_exposure": market_value / total_value if total_value > 0 else None,
        "holding_count": holding_count,
        "cash_ratio": cash / total_value if cash is not None and total_value > 0 else None,
    }


def _current_status(record: Record) -> dict[str, Any]:
    return _record_exposure(record)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_regime_rows_by_date(regime_history: Path, start_date: str) -> dict[str, dict[str, Any]]:
    if not regime_history.exists():
        return {}
    rows_by_date: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    with regime_history.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            as_of = _iso_from_any(payload.get("as_of"))
            if as_of and as_of >= start_date:
                rows_by_date[as_of].append((index, payload))

    def priority(indexed_row: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int, int]:
        index, row = indexed_row
        scope = str(row.get("regime_scope") or "")
        source_count = _safe_int(row.get("source_symbol_count"))
        return (
            1 if row.get("production_eligible") is True else 0,
            2 if scope == "full_market" else (1 if scope == "market_reference" else 0),
            source_count,
            0 if row.get("sampled") else 1,
            index,
        )

    return {
        row_date: max(indexed_rows, key=priority)[1]
        for row_date, indexed_rows in rows_by_date.items()
    }


def _regime_exposure_compliance(
    records: list[Record],
    nav_rows: list[dict[str, Any]],
    regime_history: Path,
) -> dict[str, Any]:
    if not nav_rows:
        return {"available": False, "reason": "no nav rows"}
    start_date = nav_rows[0]["date"]
    regimes = _load_regime_rows_by_date(regime_history, start_date)
    record_by_run_id = {record.run_id: record for record in records}
    timeline: list[dict[str, Any]] = []
    state_counts: dict[str, int] = defaultdict(int)
    violation_margins: list[float] = []
    previous_state: str | None = None
    switch_dates: list[dict[str, str]] = []
    compliance_by_date: dict[str, str] = {}
    for nav_row in nav_rows:
        record = record_by_run_id.get(str(nav_row.get("run_id")))
        exposure = _record_exposure(record) if record is not None else {}
        row_date = str(nav_row.get("date") or "")
        regime = regimes.get(row_date)
        cap = _safe_float(regime.get("suggested_gross_exposure_cap"), None) if regime else None
        actual = exposure.get("actual_total_exposure")
        covered = regime is not None and cap is not None
        violation = bool(covered and actual is not None and actual > cap)
        compliance_status = "missing_regime_snapshot"
        margin = None
        state = None
        if covered:
            state = str(regime.get("dominant_regime") or "unknown")
            state_counts[state] += 1
            if previous_state is not None and state != previous_state:
                switch_dates.append({"date": row_date, "from": previous_state, "to": state})
            previous_state = state
            margin = actual - cap if actual is not None else None
            if violation and margin is not None:
                violation_margins.append(margin)
                compliance_status = "violation"
            else:
                compliance_status = "compliant"
        compliance_by_date[row_date] = compliance_status
        timeline.append(
            {
                "date": row_date,
                "run_id": nav_row.get("run_id"),
                "dominant_regime": state,
                "suggested_gross_exposure_cap": cap,
                "actual_total_exposure": actual,
                "compliance_status": compliance_status,
                "excess_exposure": margin if violation else None,
                "regime_scope": regime.get("regime_scope") if regime else None,
                "source_symbol_count": regime.get("source_symbol_count") if regime else None,
                "diagnostic_notes": regime.get("diagnostic_notes") if regime else ["missing_regime_snapshot"],
            }
        )
    covered_count = sum(1 for row in timeline if row["compliance_status"] in {"compliant", "violation"})
    violation_count = sum(1 for row in timeline if row["compliance_status"] == "violation")
    coverage_rate = covered_count / len(timeline) if timeline else None
    violation_ratio = violation_count / covered_count if covered_count else None
    compliant_product = 1.0
    violation_product = 1.0
    for prev, current in zip(nav_rows, nav_rows[1:]):
        prev_nav = _safe_float(prev.get("nav"))
        current_nav = _safe_float(current.get("nav"))
        if prev_nav <= 0 or current_nav <= 0:
            continue
        ret = current_nav / prev_nav - 1.0
        status = compliance_by_date.get(str(current.get("date")))
        if status == "compliant":
            compliant_product *= 1.0 + ret
        elif status == "violation":
            violation_product *= 1.0 + ret
    triggered = violation_ratio is not None and violation_ratio > 0.30
    return {
        "available": bool(regimes),
        "timeline": timeline,
        "state_day_distribution": dict(sorted(state_counts.items())),
        "state_switch_count": len(switch_dates),
        "state_switch_dates": switch_dates,
        "covered_days": covered_count,
        "total_days": len(timeline),
        "coverage_rate": coverage_rate,
        "violation_days": violation_count,
        "violation_ratio": violation_ratio,
        "excess_exposure_quantiles": {
            "p50": _quantile(violation_margins, 0.50),
            "p90": _quantile(violation_margins, 0.90),
            "max": max(violation_margins) if violation_margins else None,
        },
        "interpretation_triggered": triggered,
        "interpretation_line": (
            "窗口内战绩相当部分产生于闸门约束之外，'系统的钱'份额需按合规日子集重算"
            if triggered
            else "未触发"
        ),
        "compliant_nav_return_contribution": compliant_product - 1.0,
        "violation_nav_return_contribution": violation_product - 1.0,
    }


def _markov_diagnostics(regime_history: Path, start_date: str = "2026-06-15") -> dict[str, Any]:
    if not regime_history.exists():
        return {"available": False, "reason": f"not found: {regime_history}"}
    rows: list[dict[str, Any]] = []
    with regime_history.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            as_of = _iso_from_any(payload.get("as_of"))
            if as_of and as_of >= start_date:
                rows.append(payload)
    rows = sorted(rows, key=lambda item: (str(item.get("as_of") or ""), str(item.get("regime_scope") or "")))
    timeline = [
        {
            "as_of": _iso_from_any(row.get("as_of")),
            "dominant_regime": row.get("dominant_regime"),
            "confidence": row.get("confidence"),
            "transition_risk": row.get("transition_risk"),
            "suggested_gross_exposure_cap": row.get("suggested_gross_exposure_cap"),
            "turnover_cap": row.get("turnover_cap"),
            "regime_scope": row.get("regime_scope"),
            "diagnostic_notes": row.get("diagnostic_notes") or [],
        }
        for row in rows
    ]
    july_rows = [row for row in timeline if "2026-07-01" <= row["as_of"] <= "2026-07-07"]
    states = {row.get("dominant_regime") for row in july_rows}
    unchanged = len(states) <= 1 if july_rows else None
    latest = rows[-1] if rows else {}
    return {
        "available": True,
        "timeline": timeline,
        "july_pullback_state_unchanged": unchanged,
        "july_pullback_summary": (
            "state unchanged; feature_snapshot dumped for verification"
            if unchanged
            else "state changed or insufficient July rows"
        ),
        "latest_feature_snapshot_if_unchanged": latest.get("feature_snapshot") if unchanged else None,
    }


def _fundamentals_appendix(holdings: dict[str, dict[str, Any]], fundamentals_root: Path) -> dict[str, Any]:
    symbols = set(holdings)
    if not symbols or not fundamentals_root.exists():
        return {"available": False, "rows": [], "note": "披露日需人工补充"}
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return {"available": False, "rows": [], "note": "pandas unavailable; 披露日需人工补充"}
    candidates = [
        fundamentals_root / "table=fina_indicator",
        fundamentals_root / "table=income",
    ]
    frame = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            with _suppress_native_stderr():
                loaded = pd.read_parquet(path)
        except Exception:
            continue
        if "ts_code" in loaded.columns:
            frame = loaded
            break
    if frame is None or frame.empty:
        return {"available": False, "rows": [], "note": "local PIT fundamentals unavailable; 披露日需人工补充"}
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame = frame[frame["ts_code"].isin(symbols)].copy()
    if frame.empty:
        return {"available": False, "rows": [], "note": "current holdings not found in local PIT fundamentals; 披露日需人工补充"}
    date_col = "ann_date" if "ann_date" in frame.columns else ("end_date" if "end_date" in frame.columns else None)
    if date_col:
        frame[date_col] = frame[date_col].astype(str)
        frame = frame.sort_values(["ts_code", date_col])
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("ts_code", sort=True):
        latest = group.iloc[-1].to_dict()
        revenue_yoy = next(
            (_safe_float(latest.get(col), None) for col in ("tr_yoy", "revenue_yoy", "oper_rev_yoy") if col in latest),
            None,
        )
        profit_yoy = next(
            (_safe_float(latest.get(col), None) for col in ("netprofit_yoy", "netprofit_yoy_dt", "profit_yoy") if col in latest),
            None,
        )
        rows.append(
            {
                "symbol": symbol,
                "name": holdings.get(symbol, {}).get("name", ""),
                "period": _iso_from_any(latest.get("end_date")),
                "ann_date": _iso_from_any(latest.get("ann_date")),
                "revenue_yoy": revenue_yoy,
                "net_profit_yoy": profit_yoy,
                "h1_schedule": "披露日需人工补充",
            }
        )
    return {
        "available": True,
        "note": "产业实质判断（真订单/真产能 vs 纯故事）由人工完成，本表只提供数据。",
        "rows": rows,
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _render_report(metrics: dict[str, Any]) -> str:
    perf = metrics["performance"]
    bench = metrics["benchmarks"]
    decomp = metrics["decomposition"]
    concentration = metrics["concentration"]
    execution = metrics["execution_quality"]
    funding = metrics["funding_nature"]
    markov = metrics["markov_diagnostics"]
    counterparty = metrics["counterparty_quality"]
    current_status = metrics["current_status"]
    beta_adjusted = metrics["beta_adjusted_excess"]
    exposure = metrics["regime_exposure_compliance"]
    cf2 = metrics["counterfactual2_decomposition"]
    selection = metrics["selection_alpha"]
    shadow = metrics["shadow_ledgers"]
    exit_split = metrics["counterparty_exit_split"]
    cost = metrics["estimated_execution_cost"]
    peak_weights = metrics["top3_peak_weights"]
    f_root = metrics["slippage_zero_root_cause"]
    lines = [
        "# Phase 12 Track-Record Audit",
        "",
        f"- Funding nature: **{funding['funding_nature']}**",
        f"- 判断依据：{funding['basis']}",
        f"- 样本交易日：{perf.get('sample_trading_days', 0)}；样本 <90 交易日，Sharpe/Calmar 统计稳定性有限。",
        f"- 当前状态（{current_status.get('date')}）：实际总暴露 {_pct(current_status.get('actual_total_exposure'))}；持仓数 {current_status.get('holding_count')}；现金比例 {_pct(current_status.get('cash_ratio'))}。",
        "",
        "## A. 基础绩效",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["累计收益", _pct(perf.get("cumulative_return"))],
                ["年化波动", _pct(perf.get("annualized_volatility"))],
                [
                    "最大回撤",
                    f"{_pct(perf.get('max_drawdown'))} ({perf.get('max_drawdown_start') or 'N/A'} -> {perf.get('max_drawdown_end') or 'N/A'})",
                ],
                ["Calmar", _money(perf.get("calmar"))],
                ["日频 Sharpe", _money(perf.get("daily_sharpe"))],
                ["周换手率", _pct(perf.get("weekly_turnover"))],
                ["年化换手", _pct(perf.get("annualized_turnover"))],
                ["逐笔平仓命中率", _pct(perf.get("closed_trade_hit_rate"))],
                ["盈亏比", _money(perf.get("closed_trade_profit_loss_ratio"))],
            ],
        ),
        "",
        "## B. 基准与超额",
        "",
    ]
    bench_rows = []
    for field in ("star50_nav", "chinext_nav", "csi300_nav", "industry_ew_nav"):
        row = bench.get(field, {})
        bench_rows.append(
            [
                field,
                _pct(row.get("benchmark_return")) if row.get("available") else "N/A",
                _pct(row.get("full_window_excess")) if row.get("available") else "N/A",
                len(row.get("rolling_20d_excess", [])) if row.get("available") else 0,
            ]
        )
    lines.extend(
        [
            "科创50（`star50_nav`）是主考官。",
            _markdown_table(["Benchmark", "全窗口收益", "全窗口超额", "20日滚动点数"], bench_rows),
            "",
            "## C. 三分解",
            "",
            _markdown_table(
                ["Component", "Return"],
                [
                    ["配置 beta = industry_ew_nav - csi300_nav", _pct(decomp["config_beta"])],
                    ["组内选股 alpha = 反事实组合1 - industry_ew_nav", _pct(decomp["stock_alpha"])],
                    ["相位择时 alpha = 实际 NAV - 反事实组合1", _pct(decomp["phase_timing_alpha"])],
                    ["总超额 vs csi300", _pct(decomp["total_excess_vs_csi300"])],
                    ["三项合计", _pct(decomp["explained_sum"])],
                    ["交叉/残差项", _pct(decomp["reconciliation_residual"])],
                ],
            ),
            "",
            "## D. 集中度与稳健性",
            "",
            _markdown_table(
                ["Symbol", "Contribution P&L"],
                [[item["symbol"], _money(item["contribution_pnl"])] for item in concentration.get("top3", [])],
            ),
            f"Top3 正贡献占比：{_pct(concentration.get('top3_positive_contribution_share'))}；剔除 Top3 估算收益：{_pct(concentration.get('return_excluding_top3_estimate'))}",
            "",
            "## E. 把货卖给了谁",
            "",
            counterparty["interpretation_rule"],
            _markdown_table(
                ["Side", "Horizon", "Mean", "Median", "Negative Share"],
                [
                    [
                        side,
                        f"{horizon}d",
                        _pct(counterparty[side][f"ret_{horizon}d"].get("mean")),
                        _pct(counterparty[side][f"ret_{horizon}d"].get("median")),
                        _pct(counterparty[side][f"ret_{horizon}d"].get("negative_share")),
                    ]
                    for side in ("entries", "exits")
                    for horizon in (5, 10, 20)
                ],
            ),
            "",
            "## F. 执行质量",
            "",
            _markdown_table(
                ["Side", "Count", "Mean bps", "Median bps"],
                [
                    [
                        side,
                        data.get("count"),
                        _money(data.get("mean")),
                        _money(data.get("median")),
                    ]
                    for side, data in execution["slippage_bps"].items()
                ],
            ),
            f"门禁拒绝单量：{execution['rejection_count']}；总成本拖累估算：{_pct(execution.get('estimated_cost_drag'))}",
            _markdown_table(
                ["Reject Reason", "Count"],
                [[reason, count] for reason, count in execution.get("rejection_reasons", {}).items()],
            ),
            "",
            "## G. 归因切分",
            "",
            f"人工/系统识别方法：{metrics['manual_system_attribution']['classification_method']}",
            _markdown_table(
                ["Bucket", "Trades", "Realized P&L", "Hit Rate"],
                [
                    [
                        bucket,
                        data["trade_count"],
                        _money(data["realized_pnl"]),
                        _pct(data["hit_rate"]),
                    ]
                    for bucket, data in metrics["manual_system_attribution"].items()
                    if isinstance(data, dict) and "trade_count" in data
                ],
            ),
            "",
            "### Version Cut Candidates",
            _markdown_table(
                ["Date", "Subject"],
                [[event.get("date", ""), event.get("subject", "")] for event in metrics["version_events"][:8]],
            ),
            "",
            "### Version Segments",
            _markdown_table(
                ["Start", "End", "Actual", "Excess vs Star50", "Excess vs CSI300", "Excess vs Industry EW"],
                [
                    [
                        segment.get("start", ""),
                        segment.get("end", ""),
                        _pct(segment.get("actual_return")),
                        _pct(segment.get("excess_vs_star50_nav")),
                        _pct(segment.get("excess_vs_csi300_nav")),
                        _pct(segment.get("excess_vs_industry_ew_nav")),
                    ]
                    for segment in metrics.get("version_segments", [])
                ],
            ),
            "",
            "### Markov Regime Diagnostics",
            f"可用：{markov.get('available')}；7月初状态是否未变：{markov.get('july_pullback_state_unchanged')}",
            "",
            "## H. β 调整后的超额（vs 科创50）",
            "",
            beta_adjusted["fixed_sample_note"],
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["样本收益点数", beta_adjusted.get("n")],
                    ["β", _money(beta_adjusted.get("beta"))],
                    ["日频 α", _pct(beta_adjusted.get("alpha_daily"))],
                    ["年化 α（×244）", _pct(beta_adjusted.get("alpha_annualized"))],
                    ["α t 值", _money(beta_adjusted.get("alpha_t"))],
                    ["R²", _money(beta_adjusted.get("r_squared"))],
                    ["IR（日频）", _money(beta_adjusted.get("standard_ir_daily"))],
                    ["IR（年化 ×√244）", _money(beta_adjusted.get("standard_ir_annualized"))],
                    ["β × 科创50收益", _pct(beta_adjusted.get("beta_times_star50_return"))],
                ],
            ),
            f"判读：{beta_adjusted['interpretation_line']}",
            "",
            "## I. 全窗口 regime 时间线与暴露合规",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["覆盖天数/总天数", f"{exposure.get('covered_days')}/{exposure.get('total_days')}"],
                    ["覆盖率", _pct(exposure.get("coverage_rate"))],
                    ["违规天数", exposure.get("violation_days")],
                    ["违规占比", _pct(exposure.get("violation_ratio"))],
                    ["状态切换次数", exposure.get("state_switch_count")],
                    ["超限幅度 P50", _pct(exposure.get("excess_exposure_quantiles", {}).get("p50"))],
                    ["超限幅度 P90", _pct(exposure.get("excess_exposure_quantiles", {}).get("p90"))],
                    ["超限幅度 Max", _pct(exposure.get("excess_exposure_quantiles", {}).get("max"))],
                    ["合规日 NAV 收益贡献", _pct(exposure.get("compliant_nav_return_contribution"))],
                    ["违规日 NAV 收益贡献", _pct(exposure.get("violation_nav_return_contribution"))],
                ],
            ),
            _markdown_table(
                ["Regime", "Days"],
                [[name, count] for name, count in exposure.get("state_day_distribution", {}).items()],
            ),
            _markdown_table(
                ["Date", "From", "To"],
                [
                    [row.get("date", ""), row.get("from", ""), row.get("to", "")]
                    for row in exposure.get("state_switch_dates", [])
                ],
            ),
            f"判读：{exposure['interpretation_line']}",
            "",
            "## J. 选股 α 的第二对照",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["反事实2收益", _pct(cf2.get("counterfactual2_return"))],
                    ["选股（纯名字）= 反事实2 - industry_ew_nav", _pct(cf2.get("pure_name_stock_alpha"))],
                    ["入场时点贡献 = 反事实1 - 反事实2", _pct(cf2.get("entry_timing_contribution"))],
                    ["反事实标的数", cf2.get("symbol_count")],
                    ["首日不可交易调整数", len(cf2.get("delayed_entry_adjustments", []))],
                ],
            ),
            "",
            "## K. 选择 α（菜单 vs 所选）",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["实际买入笔数", selection.get("buy_trade_count")],
                    ["覆盖笔数", selection.get("covered_count")],
                    ["覆盖率", _pct(selection.get("coverage_rate"))],
                    ["所选均值收益", _pct(selection.get("selected_mean_return"))],
                    ["菜单均值收益", _pct(selection.get("menu_mean_return"))],
                    ["选择 α", _pct(selection.get("selection_alpha"))],
                ],
            ),
            f"判读：{selection.get('interpretation_line')}",
            "",
            "## L. 影子账本",
            "",
            _markdown_table(
                ["Shadow", "Current Difference vs Actual"],
                [
                    ["cap050", _pct(shadow.get("cap050_current_difference_vs_actual"))],
                    ["machine_exit", _pct(shadow.get("machine_exit_current_difference_vs_actual"))],
                    [
                        "machine_exit_sensitive_including_non_trading",
                        _pct(
                            shadow.get("machine_exit_sensitivity_including_non_trading", {}).get(
                                "current_difference_vs_actual"
                            )
                        ),
                    ],
                ],
            ),
            (
                "注：machine_exit 默认剔除 `weekend_paper_fill` / `non_trading_paper_fill`；"
                "敏感性行仅用于显示历史幽灵成交进入影子口径时的影响。"
            ),
            _markdown_table(
                ["Date", "Symbol", "Status", "Delta", "Contribution", "Share"],
                [
                    [
                        row.get("date"),
                        row.get("symbol"),
                        row.get("exit_status"),
                        _money(row.get("delta_vs_manual_pnl")),
                        _pct(row.get("contribution_pct_of_initial")),
                        _pct(row.get("share_of_shadow_delta")),
                    ]
                    for row in shadow.get("shadow_nav_machine_exit", {}).get("rows", [])
                ],
            ),
            "",
            "## M. 毛/净执行成本估计",
            "",
            f"F=0 根因：{f_root.get('conclusion')}",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["毛口径全窗口收益", _pct(cost.get("gross_full_window_return"))],
                    ["净口径全窗口收益", _pct(cost.get("net_full_window_return"))],
                    ["估计总成本", _money(cost.get("estimated_total_cost"))],
                    ["年化成本拖累", _pct(cost.get("annualized_cost_drag"))],
                ],
            ),
            "### E 拆分",
            _markdown_table(
                ["Bucket", "Horizon", "Mean", "Median", "Negative Share"],
                [
                    [
                        bucket,
                        f"{horizon}d",
                        _pct(exit_split[bucket][f"ret_{horizon}d"].get("mean")),
                        _pct(exit_split[bucket][f"ret_{horizon}d"].get("median")),
                        _pct(exit_split[bucket][f"ret_{horizon}d"].get("negative_share")),
                    ]
                    for bucket in ("manual_proxy_sell", "system_sell")
                    for horizon in (5, 10, 20)
                ],
            ),
            "### Top3 峰值权重",
            _markdown_table(
                ["Symbol", "Peak Weight", "Date", "Run"],
                [
                    [symbol, _pct(data.get("peak_weight")), data.get("date", ""), data.get("run_id", "")]
                    for symbol, data in peak_weights.items()
                ],
            ),
            "",
            "## 附录：中报证伪数据支持",
            "",
            "产业实质判断（真订单/真产能 vs 纯故事）由人工完成，本表只提供数据。",
            _markdown_table(
                ["Symbol", "Name", "Period", "Ann Date", "Revenue YoY", "Net Profit YoY", "H1 Schedule"],
                [
                    [
                        row.get("symbol", ""),
                        row.get("name", ""),
                        row.get("period", ""),
                        row.get("ann_date", ""),
                        _pct(row.get("revenue_yoy") / 100.0 if row.get("revenue_yoy") is not None and abs(row.get("revenue_yoy")) > 2 else row.get("revenue_yoy")),
                        _pct(row.get("net_profit_yoy") / 100.0 if row.get("net_profit_yoy") is not None and abs(row.get("net_profit_yoy")) > 2 else row.get("net_profit_yoy")),
                        row.get("h1_schedule", "披露日需人工补充"),
                    ]
                    for row in metrics["fundamentals_appendix"].get("rows", [])
                ],
            ),
        ]
    )
    star50_excess = bench.get("star50_nav", {}).get("full_window_excess")
    if star50_excess is not None and star50_excess <= 0 and decomp["phase_timing_alpha"] <= 0:
        lines.extend(["", "## 结论", "", PREREGISTERED_FAILURE_CONCLUSION])
    return "\n".join(lines) + "\n"


def _write_outputs(metrics: dict[str, Any], output_dir: Path, generate_plots: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit_report.md").write_text(_render_report(metrics), encoding="utf-8")
    if not generate_plots:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return
    nav_rows = metrics.get("nav_rows", [])
    if not nav_rows:
        return
    plt.figure(figsize=(8, 4))
    plt.plot([row["date"] for row in nav_rows], [row["nav"] for row in nav_rows], label="portfolio_nav")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir / "nav_curve.png", dpi=120, metadata={"Software": "run_track_record_audit.py"})
    plt.close()


def _assert_output_not_inside_records(record_root: Path, output_dir: Path) -> None:
    record_resolved = record_root.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == record_resolved or record_resolved in output_resolved.parents:
        raise ValueError("Audit output directory must not be inside strategy record root.")


def run_audit(
    *,
    record_root: Path = DEFAULT_RECORD_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    benchmark_file: Path = DEFAULT_BENCHMARK_FILE,
    bars_root: Path = DEFAULT_BARS_ROOT,
    stock_basic_root: Path = DEFAULT_STOCK_BASIC_ROOT,
    fundamentals_root: Path = DEFAULT_FUNDAMENTALS_ROOT,
    regime_history: Path = DEFAULT_REGIME_HISTORY,
    as_of_date: str | None = None,
    generate_plots: bool = False,
) -> dict[str, Any]:
    as_of = as_of_date or date.today().strftime("%Y%m%d")
    output_dir = output_root / as_of
    _assert_output_not_inside_records(record_root, output_dir)
    records, warnings = _load_records(record_root)
    daily_records = _latest_daily_records(records)
    nav_rows = _nav_series(daily_records)
    if len(nav_rows) < 2:
        raise ValueError("Need at least two NAV records for track-record audit.")
    dates = [row["date"] for row in nav_rows]
    trades, rejections = _extract_trades(records)
    trades, trade_calendar_diagnostics = _annotate_trade_calendar_status(trades, dates, bars_root)
    entries = _entry_dates(daily_records, trades)
    audit_symbols = set(entries)
    audit_symbols.update(str(trade.get("symbol") or "").strip().upper() for trade in trades if trade.get("symbol"))
    for record in daily_records:
        audit_symbols.update(
            str(row.get("symbol") or "").strip().upper()
            for row in record.holdings
            if str(row.get("symbol") or "").strip()
        )
        audit_symbols.update(
            str(row.get("symbol") or "").strip().upper()
            for row in _candidate_pool_rows(record)
            if str(row.get("symbol") or "").strip()
        )
    prices, price_warnings = _read_price_panel(audit_symbols, dates[0], dates[-1], bars_root)
    warnings.extend(price_warnings)
    execution_prices, execution_price_warnings = _read_price_panel(
        audit_symbols,
        dates[0],
        dates[-1],
        bars_root,
        prefer_adjusted=False,
    )
    warnings.extend(execution_price_warnings)
    benchmarks, benchmark_meta, benchmark_warnings = _benchmark_navs(dates, benchmark_file, bars_root, stock_basic_root)
    warnings.extend(benchmark_warnings)
    counterfactual_nav = _counterfactual_nav(dates, entries, prices)
    final_holdings = _holdings_by_date(daily_records).get(dates[-1], {})
    version_events = _version_events()
    latest_record = daily_records[-1]
    manual_system = _manual_system_attribution(trades, records)
    execution_quality = _execution_quality(records, trades, rejections)
    top3_symbols = [str(item.get("symbol")) for item in _concentration(daily_records, nav_rows, trades).get("top3", [])]
    concentration = _concentration(daily_records, nav_rows, trades)
    metrics = {
        "schema_version": "track_record_audit.v2",
        "record_root": str(record_root),
        "output_dir": str(output_dir),
        "warnings": warnings,
        "nav_rows": nav_rows,
        "trade_count": len(trades),
        "trade_calendar_diagnostics": trade_calendar_diagnostics,
        "rejection_count": len(rejections),
        "performance": _performance_metrics(nav_rows, trades),
        "holding_periods": _holding_periods(daily_records),
        "benchmark_metadata": benchmark_meta,
        "benchmarks": _benchmark_metrics(nav_rows, benchmarks),
        "counterfactual1_nav": counterfactual_nav,
        "decomposition": _decomposition(nav_rows, benchmarks, counterfactual_nav),
        "concentration": concentration,
        "counterparty_quality": _counterparty_quality(trades, prices),
        "counterparty_exit_split": _counterparty_quality_by_exit_bucket(trades, prices, manual_system),
        "execution_quality": execution_quality,
        "slippage_zero_root_cause": _slippage_zero_root_cause(execution_quality),
        "estimated_execution_cost": _estimated_execution_cost(trades, nav_rows, bars_root),
        "version_events": version_events,
        "version_segments": _version_segment_metrics(nav_rows, benchmarks, version_events),
        "manual_system_attribution": manual_system,
        "funding_nature": _funding_nature(records),
        "markov_diagnostics": _markov_diagnostics(regime_history),
        "fundamentals_appendix": _fundamentals_appendix(final_holdings, fundamentals_root),
        "current_status": _current_status(latest_record),
        "beta_adjusted_excess": _beta_adjusted_excess(nav_rows, benchmarks),
        "regime_exposure_compliance": _regime_exposure_compliance(daily_records, nav_rows, regime_history),
        "counterfactual2_decomposition": _counterfactual2_decomposition(
            nav_rows,
            benchmarks,
            counterfactual_nav,
            prices,
            entries.keys(),
        ),
        "selection_alpha": _selection_alpha(daily_records, trades, prices, dates[-1]),
        "shadow_ledgers": _shadow_ledgers(daily_records, nav_rows, trades, prices, execution_prices, manual_system),
        "top3_peak_weights": _peak_weights(daily_records, top3_symbols),
    }
    _write_outputs(metrics, output_dir, generate_plots=generate_plots)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS_ROOT)
    parser.add_argument("--stock-basic-root", type=Path, default=DEFAULT_STOCK_BASIC_ROOT)
    parser.add_argument("--fundamentals-root", type=Path, default=DEFAULT_FUNDAMENTALS_ROOT)
    parser.add_argument("--regime-history", type=Path, default=DEFAULT_REGIME_HISTORY)
    parser.add_argument("--as-of-date")
    parser.add_argument("--plots", action="store_true", help="Opt in to matplotlib PNG output when the local cache is safe.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    metrics = run_audit(
        record_root=args.record_root,
        output_root=args.output_root,
        benchmark_file=args.benchmark_file,
        bars_root=args.bars_root,
        stock_basic_root=args.stock_basic_root,
        fundamentals_root=args.fundamentals_root,
        regime_history=args.regime_history,
        as_of_date=args.as_of_date,
        generate_plots=bool(args.plots and not args.no_plots),
    )
    print(
        json.dumps(
            {
                "output_dir": metrics["output_dir"],
                "record_count": len(metrics["nav_rows"]),
                "production_benchmark_source": metrics["benchmark_metadata"].get("source_system", ""),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
