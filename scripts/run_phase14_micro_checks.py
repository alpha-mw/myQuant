#!/usr/bin/env python3
"""Run Phase 14.1 measurement-only micro checks.

This script is offline and read-only for strategy records and session logs. It
writes only ignored outputs under ``results/track_record_audit/<YYYYMMDD>/``.
It does not call market providers, LLMs, brokers, or execution APIs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

from print_pipeline_state import DEFAULT_RECORD_ROOT, build_state  # noqa: E402
from run_track_record_audit import (  # noqa: E402
    DEFAULT_BARS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _compact_date,
    _extract_trades,
    _load_records,
    _manual_proxy_trades,
    _manual_system_attribution,
    _safe_float,
    _safe_int,
    _suppress_native_stderr,
    run_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = Path.home() / ".codex" / "sessions" / "2026"
ADVICE_VERB_RE = re.compile(r"卖出|减仓|清仓|止盈|建议|\bshould sell\b|\bexit\b", re.IGNORECASE)


def _iso_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    compact = _compact_date(text)
    if len(compact) == 8:
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    return text[:10]


def _file_date(path: Path) -> str:
    parts = path.parts
    try:
        index = parts.index("2026")
        return f"{parts[index]}-{parts[index + 1]}-{parts[index + 2]}"
    except (ValueError, IndexError):
        match = re.search(r"(2026)-(\d{2})-(\d{2})", path.name)
        return "-".join(match.groups()) if match else ""


def _date_in_window(value: str, start: str, end: str) -> bool:
    try:
        current = datetime.strptime(value, "%Y-%m-%d").date()
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return False
    return start_date <= current <= end_date


def _symbol_aliases(symbol: str) -> list[str]:
    text = str(symbol or "").strip().upper()
    short = text.split(".")[0]
    return [item for item in [text, short] if item]


def strict_session_match_matrix(
    manual_sells: Iterable[dict[str, Any]],
    session_root: Path,
    *,
    lookback_days: int = 3,
    context_lines: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(session_root.glob("*/*/*.jsonl")) if session_root.exists() else []
    for trade in sorted(manual_sells, key=lambda item: (str(item.get("date")), str(item.get("symbol")))):
        trade_date = _iso_date(trade.get("date"))
        try:
            end_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        except ValueError:
            end_date = None
        start_date = (end_date - timedelta(days=lookback_days)).isoformat() if end_date else ""
        aliases = _symbol_aliases(str(trade.get("symbol") or ""))
        best: dict[str, Any] | None = None
        for path in files:
            current_date = _file_date(path)
            if end_date and not _date_in_window(current_date, start_date, trade_date):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                if not any(alias in line.upper() for alias in aliases):
                    continue
                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                context = "\n".join(lines[start:end])
                has_verb = bool(ADVICE_VERB_RE.search(context))
                best = {
                    "match_level": "symbol_plus_advice_verb" if has_verb else "symbol_only_no_advice_verb",
                    "match_date": current_date,
                    "match_file": path.name,
                    "match_line": index + 1,
                    "advice_verb_found": has_verb,
                }
                if has_verb:
                    break
            if best and best["advice_verb_found"]:
                break
        rows.append(
            {
                "date": trade_date,
                "symbol": str(trade.get("symbol") or "").strip().upper(),
                "shares": _safe_int(trade.get("shares") or trade.get("quantity")),
                "price": _safe_float(trade.get("price")),
                "realized_pnl": _safe_float(trade.get("realized_pnl")),
                **(best or {
                    "match_level": "no_symbol_window",
                    "match_date": "",
                    "match_file": "",
                    "match_line": None,
                    "advice_verb_found": False,
                }),
            }
        )
    return rows


def _read_bar_rows(symbols: Iterable[str], dates: Iterable[str], bars_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    symbol_set = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    date_set = {_compact_date(date_value) for date_value in dates if _compact_date(date_value)}
    if not symbol_set or not date_set or not bars_root.exists():
        return {}, []
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return {}, ["pandas/pyarrow unavailable; tradability check skipped."]
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
    filters = [
        ("trade_date", ">=", min(date_set)),
        ("trade_date", "<=", max(date_set)),
    ]
    try:
        with _suppress_native_stderr():
            frame = pd.read_parquet(bars_root, columns=columns, filters=filters)
    except Exception as exc:
        return {}, [f"failed to read bars for tradability check: {exc}"]
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame[frame["ts_code"].isin(symbol_set) & frame["trade_date"].isin(date_set)].copy()
    for field in columns[2:]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        rows[(str(payload["ts_code"]), _iso_date(payload["trade_date"]))] = payload
    return rows, []


def _normal_limit_ratio(symbol: str) -> float:
    code = str(symbol or "").split(".")[0]
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def trade_tradability_check(trades: list[dict[str, Any]], bars_root: Path) -> dict[str, Any]:
    filled = [trade for trade in trades if trade.get("status") == "filled"]
    bar_rows, warnings = _read_bar_rows(
        [str(trade.get("symbol") or "") for trade in filled],
        [str(trade.get("date") or "") for trade in filled],
        bars_root,
    )
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for trade in sorted(filled, key=lambda item: (str(item.get("date")), str(item.get("symbol")), str(item.get("action")))):
        symbol = str(trade.get("symbol") or "").strip().upper()
        trade_date = _iso_date(trade.get("date"))
        action = str(trade.get("action") or "").strip().lower()
        bar = bar_rows.get((symbol, trade_date))
        flags: list[str] = []
        if not bar:
            flags.append("missing_bar_or_possible_suspension")
        else:
            prices = [_safe_float(bar.get(field), math.nan) for field in ("open", "high", "low", "close")]
            amount = _safe_float(bar.get("amount"), 0.0)
            volume = _safe_float(bar.get("vol"), 0.0)
            if amount <= 0 or volume <= 0:
                flags.append("zero_amount_or_volume_possible_suspension")
            if all(math.isfinite(value) for value in prices) and max(prices) - min(prices) <= 1e-8:
                pct = _safe_float(bar.get("pct_chg"), 0.0) / 100.0
                limit_ratio = _normal_limit_ratio(symbol)
                if pct >= limit_ratio - 0.005:
                    flags.append("one_price_limit_up")
                    if action == "buy":
                        flags.append("buy_blocked_by_one_price_limit_up")
                if pct <= -limit_ratio + 0.005:
                    flags.append("one_price_limit_down")
                    if action == "sell":
                        flags.append("sell_blocked_by_one_price_limit_down")
        row = {
            "date": trade_date,
            "symbol": symbol,
            "action": action,
            "shares": _safe_int(trade.get("shares") or trade.get("quantity")),
            "price": _safe_float(trade.get("price")),
            "flags": flags,
            "violation": any(
                flag in {
                    "missing_bar_or_possible_suspension",
                    "zero_amount_or_volume_possible_suspension",
                    "buy_blocked_by_one_price_limit_up",
                    "sell_blocked_by_one_price_limit_down",
                }
                for flag in flags
            ),
        }
        rows.append(row)
        if row["violation"]:
            violations.append(row)
    return {"trade_count": len(filled), "rows": rows, "violations": violations, "warnings": warnings}


def machine_exit_breakdown(metrics: dict[str, Any]) -> dict[str, Any]:
    shadow = metrics.get("shadow_ledgers", {}).get("shadow_nav_machine_exit", {})
    nav_rows = metrics.get("nav_rows") or []
    initial = _safe_float(nav_rows[0].get("initial_capital"), 1.0) if nav_rows else 1.0
    total_delta = sum(_safe_float(row.get("delta_vs_manual_pnl")) for row in shadow.get("rows", []))
    rows: list[dict[str, Any]] = []
    for row in shadow.get("rows", []):
        delta = _safe_float(row.get("delta_vs_manual_pnl"))
        stop = _safe_float(row.get("stage_stop_price"), None)
        machine_price = _safe_float(row.get("machine_exit_price"), None)
        stop_triggered = bool(stop is not None and machine_price is not None and machine_price <= stop)
        rows.append(
            {
                **row,
                "stop_triggered": stop_triggered,
                "exit_status": "stop_triggered_exit" if stop_triggered else "not_closed_marked_to_window_end",
                "contribution_pct_of_initial": delta / max(initial, 1.0),
                "share_of_shadow_delta": delta / total_delta if total_delta else None,
            }
        )
    return {
        "total_delta": total_delta,
        "total_difference_vs_actual": total_delta / max(initial, 1.0),
        "rows": rows,
    }


def build_micro_report(
    *,
    record_root: Path = DEFAULT_RECORD_ROOT,
    record_id: str = "20260707_1046",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    bars_root: Path = DEFAULT_BARS_ROOT,
    session_root: Path = DEFAULT_SESSION_ROOT,
    as_of_date: str = "20260707",
) -> dict[str, Any]:
    record_dir = record_root / record_id
    if not record_dir.exists():
        raise FileNotFoundError(f"record not found: {record_dir}")
    state = build_state(record_root=record_root, record_id=record_id)
    metrics = run_audit(record_root=record_root, output_root=output_root, as_of_date=as_of_date, generate_plots=False)
    records, _warnings = _load_records(record_root)
    trades, _rejections = _extract_trades(records)
    attribution = _manual_system_attribution(trades, records)
    manual_sells = _manual_proxy_trades(trades, attribution)
    return {
        "schema_version": "phase14_1_micro_checks.v1",
        "record_id": record_id,
        "exposure_reconciliation": state.get("exposure_components"),
        "strict_forensics_match_matrix": strict_session_match_matrix(manual_sells, session_root),
        "trade_tradability_check": trade_tradability_check(trades, bars_root),
        "machine_exit_breakdown": machine_exit_breakdown(metrics),
        "shortlist_diagnosis": state.get("shortlist_diagnosis"),
    }


def _pct(value: Any) -> str:
    numeric = _safe_float(value, None)
    return "N/A" if numeric is None else f"{numeric:.2%}"


def render_markdown(payload: dict[str, Any]) -> str:
    exposure = payload["exposure_reconciliation"]
    tradability = payload["trade_tradability_check"]
    machine = payload["machine_exit_breakdown"]
    lines = [
        "# Phase 14.1 Micro Checks",
        "",
        "## Exposure Reconciliation",
        f"- Phase13 effective: {exposure.get('phase13_market_value_numerator')} / {exposure.get('denominator_total_value_after')} = {_pct(exposure.get('phase13_exposure'))}",
        f"- holdings_review: {exposure.get('holdings_review_market_value_numerator')} / {exposure.get('denominator_total_value_after')} = {_pct(exposure.get('holdings_review_exposure'))}",
        f"- review_not_effective_symbols: {', '.join(exposure.get('difference_symbols_review_not_effective') or []) or 'N/A'}",
        f"- conclusion: {exposure.get('conclusion')}",
        "",
        "## Strict Session Match Matrix",
        "| date | symbol | shares | level | verb | file:line |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in payload["strict_forensics_match_matrix"]:
        lines.append(
            f"| {row['date']} | {row['symbol']} | {row['shares']} | {row['match_level']} | "
            f"{row['advice_verb_found']} | {row.get('match_file') or 'N/A'}:{row.get('match_line') or 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "## Trade Tradability Check",
            f"- filled_trade_count: {tradability.get('trade_count')}",
            f"- violation_count: {len(tradability.get('violations', []))}",
            "| date | symbol | action | flags |",
            "|---|---:|---|---|",
        ]
    )
    for row in tradability.get("violations", []):
        lines.append(f"| {row['date']} | {row['symbol']} | {row['action']} | {', '.join(row['flags'])} |")
    if not tradability.get("violations"):
        lines.append("| N/A | N/A | N/A | no violations |")
    lines.extend(
        [
            "",
            "## Machine-exit Breakdown",
            f"- total_difference_vs_actual: {_pct(machine.get('total_difference_vs_actual'))}",
            "| date | symbol | shares | status | delta | contribution | share |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in machine.get("rows", []):
        lines.append(
            f"| {row.get('date')} | {row.get('symbol')} | {row.get('shares')} | {row.get('exit_status')} | "
            f"{_safe_float(row.get('delta_vs_manual_pnl')):.2f} | {_pct(row.get('contribution_pct_of_initial'))} | "
            f"{_pct(row.get('share_of_shadow_delta'))} |"
        )
    diagnosis = payload["shortlist_diagnosis"]
    lines.extend(
        [
            "",
            "## Shortlist Diagnosis",
            f"- status: {diagnosis.get('status')}",
            f"- conclusion: {diagnosis.get('conclusion')}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], output_root: Path, as_of_date: str) -> tuple[Path, Path]:
    out_dir = output_root / as_of_date / "phase14_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "micro_checks.json"
    md_path = out_dir / "micro_checks.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--record-id", default="20260707_1046")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS_ROOT)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--as-of-date", default="20260707")
    args = parser.parse_args()
    payload = build_micro_report(
        record_root=args.record_root,
        record_id=args.record_id,
        output_root=args.output_root,
        bars_root=args.bars_root,
        session_root=args.session_root,
        as_of_date=args.as_of_date,
    )
    json_path, md_path = write_outputs(payload, args.output_root, args.as_of_date)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
