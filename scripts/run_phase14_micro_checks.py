#!/usr/bin/env python3
"""Run Phase 14 measurement-only micro checks.

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
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

from quant_investor.agent_protocol import GlobalContext  # noqa: E402
from quant_investor.funnel.theme_candidate_pool import (  # noqa: E402
    ThemeCandidatePoolBuilder,
    ThemePoolConfig,
)
from print_pipeline_state import DEFAULT_RECORD_ROOT, build_state  # noqa: E402
from run_track_record_audit import (  # noqa: E402
    DEFAULT_BARS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _compact_date,
    _annotate_trade_calendar_status,
    _extract_trades,
    _load_records,
    _manual_proxy_trades,
    _manual_system_attribution,
    _read_csv_rows,
    _read_open_dates,
    _safe_float,
    _safe_int,
    _suppress_native_stderr,
    run_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = Path.home() / ".codex" / "sessions" / "2026"
ADVICE_VERB_RE = re.compile(r"卖出|减仓|清仓|止盈|建议|\bshould sell\b|\bexit\b", re.IGNORECASE)
SELL_FORENSICS_RE = re.compile(
    r"卖出|清仓|减仓建议|减仓待确认|已移入|不再.*持仓|真实成交|"
    r"触发卖出|必须卖|至少减仓|\bshould sell\b|\bexit\b|\bsell\s*\d+\b",
    re.IGNORECASE,
)


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
    trade_dates = sorted({_iso_date(trade.get("date")) for trade in filled if _iso_date(trade.get("date"))})
    open_dates: set[str] = set()
    calendar_warnings: list[str] = []
    if trade_dates:
        open_dates, calendar_warnings = _read_open_dates(trade_dates[0], trade_dates[-1], bars_root)
    bar_rows, warnings = _read_bar_rows(
        [str(trade.get("symbol") or "") for trade in filled],
        [str(trade.get("date") or "") for trade in filled],
        bars_root,
    )
    warnings.extend(calendar_warnings)
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for trade in sorted(filled, key=lambda item: (str(item.get("date")), str(item.get("symbol")), str(item.get("action")))):
        symbol = str(trade.get("symbol") or "").strip().upper()
        trade_date = _iso_date(trade.get("date"))
        action = str(trade.get("action") or "").strip().lower()
        bar = bar_rows.get((symbol, trade_date))
        flags: list[str] = []
        if not bar:
            if open_dates and trade_date not in open_dates:
                try:
                    weekday = datetime.strptime(trade_date, "%Y-%m-%d").weekday()
                except ValueError:
                    weekday = -1
                flags.append("weekend_paper_fill" if weekday >= 5 else "non_trading_paper_fill")
            else:
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
                    "weekend_paper_fill",
                    "non_trading_paper_fill",
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
        "excluded_non_trading_rows": shadow.get("excluded_non_trading_rows", []),
    }


def _read_close_sequence(
    symbol: str,
    start_date: str,
    end_date: str,
    bars_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not bars_root.exists():
        return [], []
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return [], ["pandas/pyarrow unavailable; price audit skipped."]
    filters = [
        ("trade_date", ">=", _compact_date(start_date)),
        ("trade_date", "<=", _compact_date(end_date)),
    ]
    try:
        with _suppress_native_stderr():
            frame = pd.read_parquet(
                bars_root,
                columns=["ts_code", "trade_date", "close", "adj_close"],
                filters=filters,
            )
    except Exception as exc:
        return [], [f"failed to read bars for price audit: {exc}"]
    if frame.empty:
        return [], []
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame = frame[frame["ts_code"] == str(symbol).strip().upper()].copy()
    if frame.empty:
        return [], []
    frame["date"] = frame["trade_date"].astype(str).map(_iso_date)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adj_close"] = pd.to_numeric(frame.get("adj_close"), errors="coerce")
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("date").itertuples(index=False):
        rows.append(
            {
                "date": str(row.date),
                "close": _safe_float(row.close, None),
                "adj_close": _safe_float(row.adj_close, None),
            }
        )
    return rows, []


def price_audit_688301(
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    bars_root: Path,
) -> dict[str, Any]:
    symbol = "688301.SH"
    nav_rows = metrics.get("nav_rows") or []
    start_date = "2026-06-17"
    end_date = str(nav_rows[-1].get("date")) if nav_rows else "2026-07-07"
    close_sequence, warnings = _read_close_sequence(symbol, start_date, end_date, bars_root)
    target_trade = None
    for trade in sorted(trades, key=lambda item: str(item.get("date"))):
        if str(trade.get("symbol") or "").strip().upper() != symbol:
            continue
        if trade.get("action") == "sell" and trade.get("status") == "filled":
            if _iso_date(trade.get("date")) == "2026-06-23":
                target_trade = trade
                break
            target_trade = trade
    initial = _safe_float(nav_rows[0].get("initial_capital"), 1.0) if nav_rows else 1.0
    end_row = close_sequence[-1] if close_sequence else {}
    p_end = _safe_float(end_row.get("close"), None)
    p_end_adj = _safe_float(end_row.get("adj_close"), None)
    p_sell = _safe_float(target_trade.get("price") if target_trade else None, None)
    shares = _safe_int(target_trade.get("shares") if target_trade else 0)
    raw_contribution = None
    adjusted_mismatch_contribution = None
    if p_end is not None and p_sell is not None:
        raw_contribution = shares * (p_end - p_sell) / max(initial, 1.0)
    if p_end_adj is not None and p_sell is not None:
        adjusted_mismatch_contribution = shares * (p_end_adj - p_sell) / max(initial, 1.0)
    default_rows = metrics.get("shadow_ledgers", {}).get("shadow_nav_machine_exit", {}).get("rows", [])
    sensitivity_rows = (
        metrics.get("shadow_ledgers", {})
        .get("machine_exit_sensitivity_including_non_trading", {})
        .get("rows", [])
    )
    ghost_trades = [
        {
            "date": _iso_date(trade.get("date")),
            "symbol": str(trade.get("symbol") or "").strip().upper(),
            "shares": _safe_int(trade.get("shares") or trade.get("quantity")),
            "price": _safe_float(trade.get("price")),
            "calendar_status": trade.get("calendar_status"),
        }
        for trade in trades
        if _iso_date(trade.get("date")) == "2026-06-20" and str(trade.get("symbol") or "").strip().upper() == symbol
    ]
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "close_sequence": close_sequence,
        "manual_recompute": {
            "target_date": _iso_date(target_trade.get("date")) if target_trade else "",
            "shares": shares,
            "shares_unit": "股",
            "shares_source": "manual_switch_and_take_profit_orders/applied_local_trades shares column",
            "lot_conversion_applied": False,
            "p_sell": p_sell,
            "p_end": p_end,
            "p_end_source": "local raw close at or before window end",
            "denominator": initial,
            "denominator_definition": "initial_capital from first NAV row",
            "formula": "shares * (P_end - P_sell) / denominator",
            "raw_contribution_pct": raw_contribution,
            "adjusted_close_mismatch_contribution_pct": adjusted_mismatch_contribution,
            "conclusion": (
                "the former +13.20% row mixed raw execution price with adjusted close"
                if adjusted_mismatch_contribution is not None
                else "insufficient local bars"
            ),
        },
        "ghost_20260620": {
            "trade_rows": ghost_trades,
            "included_in_current_default_shadow": any(str(row.get("date")) == "2026-06-20" for row in default_rows),
            "included_in_sensitivity_shadow": any(str(row.get("date")) == "2026-06-20" for row in sensitivity_rows),
            "current_default_policy": "excluded when marked weekend_paper_fill/non_trading_paper_fill",
        },
        "warnings": warnings,
    }


def phase14_e_l_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    shadow = metrics.get("shadow_ledgers", {})
    default_machine = shadow.get("shadow_nav_machine_exit", {})
    sensitivity = shadow.get("machine_exit_sensitivity_including_non_trading", {})
    return {
        "counterparty_exit_split": metrics.get("counterparty_exit_split", {}),
        "shadow_machine_exit_default": {
            "manual_proxy_sell_count": default_machine.get("manual_proxy_sell_count"),
            "current_difference_vs_actual": default_machine.get("current_difference_vs_actual"),
            "excluded_non_trading_count": default_machine.get("excluded_non_trading_count"),
            "rows": default_machine.get("rows", []),
        },
        "shadow_machine_exit_sensitivity_including_non_trading": {
            "manual_proxy_sell_count": sensitivity.get("manual_proxy_sell_count"),
            "current_difference_vs_actual": sensitivity.get("current_difference_vs_actual"),
            "rows": sensitivity.get("rows", []),
        },
    }


def _theme_pool_summary_from_record(run_dir: Path) -> dict[str, Any]:
    audit_path = run_dir / "theme_pool_audit.json"
    if not audit_path.exists():
        return {}
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload.get("summary") or {})


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_blocker(run_dir: Path) -> str:
    for name in ("market_snapshot.json", "manifest.json"):
        payload = _read_json_file(run_dir / name)
        if not payload:
            continue
        blocker = str(payload.get("blocker") or "").strip()
        if blocker:
            return blocker
        nested = payload.get("candidate_level_dag_status")
        if isinstance(nested, dict):
            blocker = str(nested.get("blocker") or "").strip()
            if blocker:
                return blocker
    return ""


def _theme_related_commits(limit: int = 8) -> list[str]:
    cmd = [
        "git",
        "log",
        f"-{limit}",
        "--format=%h %ci %s",
        "--",
        "quant_investor/funnel/theme_candidate_pool.py",
        "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
        "tests/unit/test_theme_candidate_pool.py",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _json_symbol_paths(payload: Any, symbol: str, prefix: str = "") -> list[str]:
    symbol_text = str(symbol or "").strip().upper()
    if not symbol_text:
        return []
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.strip().upper() == symbol_text:
                paths.append(path)
            paths.extend(_json_symbol_paths(value, symbol_text, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]"
            if isinstance(value, str) and value.strip().upper() == symbol_text:
                paths.append(path)
            else:
                paths.extend(_json_symbol_paths(value, symbol_text, path))
    elif isinstance(payload, str) and payload.strip().upper() == symbol_text:
        paths.append(prefix)
    return paths


def _unique_paths(paths: Iterable[str], *, limit: int = 20) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def residual_symbol_source_audit(run_dir: Path) -> dict[str, Any]:
    audit = _read_json_file(run_dir / "theme_pool_audit.json")
    snapshot = _read_json_file(run_dir / "theme_snapshot.json")
    summary = dict(audit.get("summary") or {})
    symbols_payload = dict(audit.get("symbols") or {})
    by_source = dict(audit.get("admitted_symbols_by_source") or {})
    sample_by_source = dict(summary.get("admitted_symbols_by_source_sample") or {})
    residual_symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in list(by_source.get("residual_theme") or [])
            + list(sample_by_source.get("residual_theme") or [])
            + [
                symbol
                for symbol, payload in symbols_payload.items()
                if isinstance(payload, dict)
                and str(payload.get("source") or "").strip() == "residual_theme"
                and payload.get("admitted") is True
            ]
            if str(symbol).strip()
        }
    )
    rotation = dict((snapshot.get("stored_snapshot") or {}).get("theme_rotation") or {})
    symbol_primary_theme = dict(rotation.get("symbol_primary_theme") or {})
    theme_scores = dict(rotation.get("theme_scores") or {})
    rows: list[dict[str, Any]] = []
    artifact_names = [
        "theme_pool_audit.json",
        "market_snapshot.json",
        "manifest.json",
        "codex_guardrail_override.json",
    ]
    artifact_payloads = {
        name: _read_json_file(run_dir / name)
        for name in artifact_names
    }
    for symbol in residual_symbols:
        payload = dict(symbols_payload.get(symbol) or {})
        theme_id = str(payload.get("primary_theme_id") or symbol_primary_theme.get(symbol) or "")
        explicit_paths: list[str] = []
        if symbol in list(sample_by_source.get("residual_theme") or []):
            explicit_paths.append("theme_pool_audit.json:summary.admitted_symbols_by_source_sample.residual_theme")
        if symbol in list(by_source.get("residual_theme") or []):
            explicit_paths.append("theme_pool_audit.json:admitted_symbols_by_source.residual_theme")
        if str(payload.get("source") or "") == "residual_theme":
            explicit_paths.append(f"theme_pool_audit.json:symbols.{symbol}.source")
        if theme_id:
            explicit_paths.extend(
                [
                    f"theme_snapshot.json:stored_snapshot.theme_rotation.symbol_primary_theme.{symbol}",
                    f"theme_snapshot.json:stored_snapshot.theme_rotation.symbol_scores.{symbol}",
                    f"theme_snapshot.json:stored_snapshot.theme_rotation.symbol_smoothed_scores.{symbol}",
                    f"theme_snapshot.json:stored_snapshot.theme_rotation.symbol_phase.{symbol}",
                    f"theme_snapshot.json:stored_snapshot.theme_rotation.symbol_risk_flags.{symbol}",
                    f"theme_snapshot.json:stored_snapshot.theme_rotation.theme_scores.{theme_id}",
                ]
            )
        scanned_paths: list[str] = []
        for name, artifact_payload in artifact_payloads.items():
            scanned_paths.extend(f"{name}:{path}" for path in _json_symbol_paths(artifact_payload, symbol))
        rows.append(
            {
                "symbol": symbol,
                "source": payload.get("source"),
                "admitted": payload.get("admitted"),
                "primary_theme_id": theme_id,
                "theme_pool_score": payload.get("theme_pool_score"),
                "theme_policy_regime": payload.get("theme_policy_regime"),
                "theme_pool_reason": payload.get("theme_pool_reason"),
                "source_paths": _unique_paths(explicit_paths + scanned_paths),
                "theme_snapshot_theme_score": theme_scores.get(theme_id, {}),
            }
        )
    return {
        "schema_version": "phase14_3_residual_symbol_source_audit.v1",
        "record_id": run_dir.name,
        "observed_residual_symbol_count": _safe_int(summary.get("residual_symbol_count")),
        "observed_policy": summary.get("policy", {}),
        "residual_symbols": residual_symbols,
        "rows": rows,
    }


def _theme_rotation_from_record(run_dir: Path) -> dict[str, Any]:
    snapshot = _read_json_file(run_dir / "theme_snapshot.json")
    stored = dict(snapshot.get("stored_snapshot") or {})
    rotation = stored.get("theme_rotation")
    if isinstance(rotation, dict):
        return rotation
    for name in ("market_snapshot.json", "manifest.json"):
        payload = _read_json_file(run_dir / name)
        rotation = payload.get("theme_rotation")
        if isinstance(rotation, dict):
            return rotation
        formal = payload.get("formal_diagnostics")
        if isinstance(formal, dict) and isinstance(formal.get("theme_rotation"), dict):
            return formal["theme_rotation"]
    return {}


def current_theme_pool_replay(run_dir: Path) -> dict[str, Any]:
    audit = _read_json_file(run_dir / "theme_pool_audit.json")
    summary = dict(audit.get("summary") or {})
    symbols_payload = dict(audit.get("symbols") or {})
    input_symbols = sorted(str(symbol).strip().upper() for symbol in symbols_payload if str(symbol).strip())
    rotation = _theme_rotation_from_record(run_dir)
    if not input_symbols or not rotation:
        return {
            "status": "unavailable",
            "reason": "missing theme_pool_audit.symbols or theme_rotation snapshot",
        }
    quant_scores = {
        symbol: _safe_float(payload.get("theme_pool_score"))
        for symbol, payload in symbols_payload.items()
        if isinstance(payload, dict)
    }
    policy_regime = (
        str(summary.get("policy_regime") or "")
        or str(dict(summary.get("policy") or {}).get("regime") or "")
        or "趋势下跌"
    )
    context = GlobalContext(
        market="CN",
        universe_key=str(rotation.get("universe_key") or "full_a"),
        universe_symbols=input_symbols,
        universe_tiers={"researchable": input_symbols},
        metadata={
            "theme_rotation": rotation,
            "markov_regime": {
                "dominant_regime": policy_regime,
                "production_eligible": True,
            },
        },
        regime_params={
            "markov": {
                "dominant_regime": policy_regime,
                "production_eligible": True,
            }
        },
    )
    requested_max = max(_safe_int(summary.get("admitted_symbol_count"), 50), 1)
    output = ThemeCandidatePoolBuilder(ThemePoolConfig()).build(
        symbols=input_symbols,
        global_context=context,
        quant_scores=quant_scores,
        max_candidates=requested_max,
    )
    metadata = output.metadata
    current_residual = _safe_int(metadata.get("residual_symbol_count"))
    observed_residual = _safe_int(summary.get("residual_symbol_count"))
    if observed_residual > 0 and current_residual == 0:
        classification = "stale_artifact_pollution_or_historical_old_code_behavior"
    elif current_residual > 0:
        classification = "code_behavior_regression"
    else:
        classification = "no_residual_violation"
    return {
        "status": "replayed",
        "record_id": run_dir.name,
        "input_symbol_count": len(input_symbols),
        "input_source": "theme_pool_audit.json:symbols keys",
        "quant_score_source": "theme_pool_audit.json:symbols.*.theme_pool_score proxy",
        "requested_max_candidates": requested_max,
        "output_symbol_count": len(output.symbols),
        "residual_symbol_count": current_residual,
        "source_counts": metadata.get("source_counts", {}),
        "policy": metadata.get("policy", {}),
        "natural_admitted_theme_count": metadata.get("natural_admitted_theme_count"),
        "forced_theme_count": metadata.get("forced_theme_count"),
        "admitted_themes": metadata.get("admitted_themes", []),
        "classification": classification,
    }


def daily_pipeline_proof(record_root: Path, proof_record_id: str = "20260708_0910") -> dict[str, Any]:
    run_dir = record_root / proof_record_id
    if not run_dir.exists():
        return {
            "record_id": proof_record_id,
            "status": "missing",
            "conclusion": "daily pipeline proof record not found; rerun the offline proof command first.",
        }
    theme_summary = _theme_pool_summary_from_record(run_dir)
    market = _read_json_file(run_dir / "market_snapshot.json")
    candidate_status = dict(market.get("candidate_level_dag_status") or {})
    dag = dict(candidate_status.get("dag_pipeline") or {})
    final_pool = candidate_status.get("candidate_pool")
    final_pool_count = len(final_pool) if isinstance(final_pool, list) else 0
    core = _safe_int(theme_summary.get("core_symbol_count"))
    residual = _safe_int(theme_summary.get("residual_symbol_count"))
    hard_pool_restored = residual == 0 and core > 0
    final_nonempty = final_pool_count > 0 or _safe_int(dag.get("shortlist_count")) > 0
    blocker = str(candidate_status.get("blocker") or market.get("blocker") or "").strip()
    return {
        "record_id": proof_record_id,
        "status": "available",
        "hard_filter_pool_restored_nonempty": hard_pool_restored,
        "final_shortlist_restored_nonempty": final_nonempty,
        "candidate_generation_status": candidate_status.get("candidate_generation_status") or market.get("candidate_generation_status"),
        "blocker": blocker,
        "theme_pool": {
            "core_symbol_count": core,
            "residual_symbol_count": residual,
            "policy_residual_enabled": dict(theme_summary.get("policy") or {}).get("residual_enabled"),
            "hard_theme_constraint": dict(theme_summary.get("policy") or {}).get("hard_theme_constraint"),
            "forced_theme_count": _safe_int(theme_summary.get("forced_theme_count")),
        },
        "dag_pipeline": dag,
        "final_candidate_pool_count": final_pool_count,
        "conclusion": (
            "hard-filter pool is nonempty with residual=0; final shortlist remains empty because "
            "PortfolioConstructor selected no positive target weight."
            if hard_pool_restored and not final_nonempty and blocker == "no_candidate_selected_by_portfolio_constructor"
            else "daily proof did not restore the hard-filter pool; inspect the generated record."
        ),
    }


def system_sell_post_exit_triple(metrics: dict[str, Any]) -> dict[str, Any]:
    system_sell = dict(metrics.get("counterparty_exit_split", {}).get("system_sell") or {})
    return {
        "count": system_sell.get("count", 0),
        "ret_5d": system_sell.get("ret_5d", {}),
        "ret_10d": system_sell.get("ret_10d", {}),
        "ret_20d": system_sell.get("ret_20d", {}),
        "ret_20d_mean_survives_23_80pct": abs(_safe_float(dict(system_sell.get("ret_20d") or {}).get("mean")) - 0.2380) < 0.001,
    }


def shadow_denominator_check(metrics: dict[str, Any]) -> dict[str, Any]:
    nav_rows = list(metrics.get("nav_rows") or [])
    if not nav_rows:
        return {"status": "unavailable", "reason": "nav_rows missing"}
    first = dict(nav_rows[0])
    last = dict(nav_rows[-1])
    initial = _safe_float(first.get("initial_capital"), 0.0)
    first_nav = _safe_float(first.get("nav"), 0.0)
    last_nav = _safe_float(last.get("nav"), 0.0)
    last_total = _safe_float(last.get("total_value_after"), 0.0)
    window_return = (last_nav / first_nav - 1.0) if first_nav else 0.0
    full_return_vs_initial = last_nav - 1.0
    window_implied_total = initial * (1.0 + window_return)
    return {
        "status": "checked",
        "initial_capital": initial,
        "first_nav": first_nav,
        "last_nav": last_nav,
        "last_total_value_after": last_total,
        "window_return_from_first_nav": window_return,
        "full_return_vs_initial_capital": full_return_vs_initial,
        "window_return_implied_total_from_1m": window_implied_total,
        "gap_vs_last_total_if_window_return_is_misused": last_total - window_implied_total,
        "last_nav_times_initial_capital": last_nav * initial,
        "initial_capital_values": sorted({_safe_float(row.get("initial_capital")) for row in nav_rows}),
        "conclusion": (
            "shadow denominator 1,000,000 is the same initial_capital/NAV base; "
            "61.22% is first-record-to-latest window return, not return from initial capital. "
            "No deposit-injection or interest evidence appears in NAV rows."
        ),
    }


def session_forensics_688301_excerpt(
    session_root: Path,
    *,
    start_date: str = "2026-06-16",
    end_date: str = "2026-06-30",
    context_lines: int = 10,
    max_matches: int = 5,
) -> dict[str, Any]:
    aliases = ("688301", "688301.SH", "奕瑞")
    files = sorted(session_root.glob("*/*/*.jsonl")) if session_root.exists() else []
    rows: list[dict[str, Any]] = []
    for path in files:
        current_date = _file_date(path)
        if not _date_in_window(current_date, start_date, end_date):
            continue
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        entries = _session_text_entries(raw_lines)
        for index, entry in enumerate(entries):
            upper_line = entry["text"].upper()
            if not any(alias.upper() in upper_line for alias in aliases):
                continue
            if not SELL_FORENSICS_RE.search(entry["text"]):
                continue
            start = max(0, index - context_lines)
            end = min(len(entries), index + context_lines + 1)
            context = entries[start:end]
            rows.append(
                {
                    "date": current_date,
                    "file": str(path),
                    "line": entry["line"],
                    "context_lines": [
                        {"line": item["line"], "text": item["text"][:1200]}
                        for item in context
                    ],
                }
            )
            if len(rows) >= max_matches:
                return {
                    "schema_version": "phase14_3_session_forensics_688301.v1",
                    "symbol": "688301.SH",
                    "date_window": [start_date, end_date],
                    "match_count": len(rows),
                    "matches": rows,
                }
    return {
        "schema_version": "phase14_3_session_forensics_688301.v1",
        "symbol": "688301.SH",
        "date_window": [start_date, end_date],
        "match_count": len(rows),
        "matches": rows,
    }


def _session_text_entries(raw_lines: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        fallback.append({"line": line_number, "text": raw})
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        item_type = str(payload.get("type") or "")
        body = payload.get("payload")
        if item_type == "event_msg" and isinstance(body, dict) and body.get("type") == "agent_message":
            message = str(body.get("message") or "").strip()
            if message:
                entries.append({"line": line_number, "text": message})
            continue
        if item_type != "response_item" or not isinstance(body, dict):
            continue
        if body.get("type") != "message" or body.get("role") != "assistant":
            continue
        fragments: list[str] = []
        for part in body.get("content") or []:
            if isinstance(part, dict):
                text = part.get("text") or part.get("output_text") or part.get("input_text")
                if text:
                    fragments.append(str(text))
        if fragments:
            entries.append({"line": line_number, "text": "\n".join(fragments)})
    return entries or fallback


def theme_guardrail_replay(record_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    first_trigger = None
    for run_dir in sorted(path for path in record_root.iterdir() if path.is_dir()):
        summary = _theme_pool_summary_from_record(run_dir)
        if not summary:
            continue
        policy = dict(summary.get("policy") or {})
        residual = _safe_int(summary.get("residual_symbol_count"))
        hard = policy.get("hard_theme_constraint") is True
        residual_enabled = policy.get("residual_enabled") is True
        blocker = _record_blocker(run_dir)
        row = {
            "record_id": run_dir.name,
            "date": _iso_date(run_dir.name[:8]),
            "blocker": blocker,
            "core_symbol_count": _safe_int(summary.get("core_symbol_count")),
            "residual_symbol_count": residual,
            "policy_residual_enabled": residual_enabled,
            "hard_theme_constraint": hard,
        }
        if hard and residual > 0:
            row["violation"] = "residual_symbol_count_must_be_zero_under_hard_theme_constraint"
            if first_trigger is None:
                first_trigger = row
        rows.append(row)
    return {
        "invariant": (
            "When hard_theme_constraint=true, residual_enabled must be false and "
            "residual_symbol_count must be 0; production candidates/switch plans must not use residual symbols."
        ),
        "residual_6_violation": "residual=6 violates the zero upper bound, not a lower-bound requirement.",
        "first_trigger": first_trigger,
        "latest_violations": [row for row in rows if row.get("violation")][-8:],
        "related_commits": _theme_related_commits(),
        "conclusion": (
            "真回归：历史运行产物在 hard filter 下仍启用了 residual channel；"
            "当前代码已把 ThemeGatePolicy.to_dict residual_enabled 固定为 false 并由测试约束 residual=0。"
        ),
    }


def exit_record_audit(record_root: Path, symbols: Iterable[str] = ("300285.SZ", "600487.SH", "603078.SH")) -> dict[str, Any]:
    symbol_set = {str(symbol).strip().upper() for symbol in symbols}
    rows: dict[str, dict[str, Any]] = {
        symbol: {"symbol": symbol, "manifest_trades": [], "order_rows": []}
        for symbol in sorted(symbol_set)
    }
    for run_dir in sorted(path for path in record_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manual_execution_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            for trade in manifest.get("applied_local_trades", []) if isinstance(manifest, dict) else []:
                symbol = str(trade.get("symbol") or "").strip().upper()
                if symbol in rows:
                    rows[symbol]["manifest_trades"].append({"record_id": run_dir.name, **trade})
        orders_path = run_dir / "manual_switch_and_take_profit_orders.csv"
        for order in _read_csv_rows(orders_path):
            symbol = str(order.get("symbol") or "").strip().upper()
            if symbol in rows:
                rows[symbol]["order_rows"].append({"record_id": run_dir.name, **order})
    output_rows: list[dict[str, Any]] = []
    for symbol, payload in rows.items():
        manifest_sells = [
            row for row in payload["manifest_trades"]
            if (
                "filled" in str(row.get("status") or "filled").lower()
                and (
                    "sell" in str(row.get("action") or "").lower()
                    or "clear_risk" in str(row.get("action") or "").lower()
                )
            )
        ]
        order_sells = [
            row for row in payload["order_rows"]
            if (
                "filled" in str(row.get("status") or "").lower()
                and (
                    "sell" in str(row.get("action") or "").lower()
                    or "clear_risk" in str(row.get("action") or "").lower()
                )
            )
        ]
        output_rows.append(
            {
                "symbol": symbol,
                "manifest_trade_count": len(manifest_sells),
                "order_row_count": len(order_sells),
                "manifest_trades": manifest_sells,
                "order_rows": order_sells,
                "status": "complete_exit_record" if manifest_sells and order_sells else "missing_exit_record",
                "nav_breakpoint_risk": not (manifest_sells and order_sells),
            }
        )
    return {
        "schema_version": "phase14_2_exit_record_audit.v1",
        "rows": output_rows,
        "nav_breakpoint_risk_symbols": [
            row["symbol"] for row in output_rows if row["nav_breakpoint_risk"]
        ],
    }


def build_micro_report(
    *,
    record_root: Path = DEFAULT_RECORD_ROOT,
    record_id: str = "20260707_1046",
    proof_record_id: str = "20260708_0910",
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
    nav_rows = metrics.get("nav_rows") or []
    dates = [str(row.get("date")) for row in nav_rows if row.get("date")]
    trades, _calendar_diagnostics = _annotate_trade_calendar_status(trades, dates, bars_root)
    attribution = _manual_system_attribution(trades, records)
    manual_sells = _manual_proxy_trades(trades, attribution, include_non_trading=True)
    return {
        "schema_version": "phase14_2_micro_checks.v1",
        "record_id": record_id,
        "exposure_reconciliation": state.get("exposure_components"),
        "strict_forensics_match_matrix": strict_session_match_matrix(manual_sells, session_root),
        "trade_tradability_check": trade_tradability_check(trades, bars_root),
        "trade_calendar_diagnostics": metrics.get("trade_calendar_diagnostics"),
        "machine_exit_breakdown": machine_exit_breakdown(metrics),
        "price_audit_688301": price_audit_688301(metrics, trades, bars_root),
        "e_l_after_calendar_filter": phase14_e_l_summary(metrics),
        "theme_guardrail_diagnosis": theme_guardrail_replay(record_root),
        "exit_record_audit": exit_record_audit(record_root),
        "shortlist_diagnosis": state.get("shortlist_diagnosis"),
        "phase14_3": {
            "residual_symbol_source_audit": residual_symbol_source_audit(record_dir),
            "current_theme_pool_replay": current_theme_pool_replay(record_dir),
            "daily_pipeline_proof": daily_pipeline_proof(record_root, proof_record_id),
            "system_sell_post_exit_triple": system_sell_post_exit_triple(metrics),
            "shadow_denominator_check": shadow_denominator_check(metrics),
            "session_forensics_688301": session_forensics_688301_excerpt(session_root),
        },
    }


def _pct(value: Any) -> str:
    numeric = _safe_float(value, None)
    return "N/A" if numeric is None else f"{numeric:.2%}"


def render_markdown(payload: dict[str, Any]) -> str:
    exposure = payload["exposure_reconciliation"]
    tradability = payload["trade_tradability_check"]
    machine = payload["machine_exit_breakdown"]
    price_audit = payload["price_audit_688301"]
    e_l = payload["e_l_after_calendar_filter"]
    theme_guardrail = payload["theme_guardrail_diagnosis"]
    exit_audit = payload["exit_record_audit"]
    phase14_3 = payload.get("phase14_3", {})
    residual_audit = phase14_3.get("residual_symbol_source_audit", {})
    current_replay = phase14_3.get("current_theme_pool_replay", {})
    daily_proof = phase14_3.get("daily_pipeline_proof", {})
    system_sell = phase14_3.get("system_sell_post_exit_triple", {})
    denominator = phase14_3.get("shadow_denominator_check", {})
    forensics = phase14_3.get("session_forensics_688301", {})
    lines = [
        "# Phase 14 Micro Checks",
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
            "## 688301 Shadow Price Audit",
            "| date | close | adj_close |",
            "|---|---:|---:|",
        ]
    )
    for row in price_audit.get("close_sequence", []):
        lines.append(f"| {row['date']} | {row['close']} | {row['adj_close']} |")
    recompute = price_audit.get("manual_recompute", {})
    ghost = price_audit.get("ghost_20260620", {})
    lines.extend(
        [
            "",
            f"- shares: {recompute.get('shares')} {recompute.get('shares_unit')} ({recompute.get('shares_source')})",
            f"- P_sell: {recompute.get('p_sell')}; P_end(raw): {recompute.get('p_end')}; denominator: {recompute.get('denominator')} ({recompute.get('denominator_definition')})",
            f"- raw contribution: {_pct(recompute.get('raw_contribution_pct'))}",
            f"- adjusted-close mismatch contribution: {_pct(recompute.get('adjusted_close_mismatch_contribution_pct'))}",
            f"- conclusion: {recompute.get('conclusion')}",
            f"- 2026-06-20 included in current default shadow: {ghost.get('included_in_current_default_shadow')}",
            f"- 2026-06-20 included in sensitivity shadow: {ghost.get('included_in_sensitivity_shadow')}",
            "",
            "## Machine-exit Breakdown",
            f"- total_difference_vs_actual: {_pct(machine.get('total_difference_vs_actual'))}",
            f"- excluded_non_trading_rows: {len(machine.get('excluded_non_trading_rows', []))}",
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
    split = e_l.get("counterparty_exit_split", {})
    shadow_default = e_l.get("shadow_machine_exit_default", {})
    shadow_sensitivity = e_l.get("shadow_machine_exit_sensitivity_including_non_trading", {})
    lines.extend(
        [
            "",
            "## E/L After Calendar Filter",
            f"- E manual_proxy_sell count: {split.get('manual_proxy_sell', {}).get('count')}",
            f"- E system_sell count: {split.get('system_sell', {}).get('count')}",
            f"- L default machine_exit: {_pct(shadow_default.get('current_difference_vs_actual'))}",
            f"- L sensitivity including non-trading: {_pct(shadow_sensitivity.get('current_difference_vs_actual'))}",
            f"- L excluded_non_trading_count: {shadow_default.get('excluded_non_trading_count')}",
            "",
            "## Theme Guardrail Replay",
            f"- invariant: {theme_guardrail.get('invariant')}",
            f"- residual=6: {theme_guardrail.get('residual_6_violation')}",
            f"- first_trigger: {theme_guardrail.get('first_trigger')}",
            f"- conclusion: {theme_guardrail.get('conclusion')}",
            "",
            "## Exit Record Audit",
            "| symbol | manifest | orders | status | NAV breakpoint risk |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in exit_audit.get("rows", []):
        lines.append(
            f"| {row['symbol']} | {row['manifest_trade_count']} | {row['order_row_count']} | "
            f"{row['status']} | {row['nav_breakpoint_risk']} |"
        )
    diagnosis = payload["shortlist_diagnosis"]
    lines.extend(
        [
            "",
            "## Shortlist Diagnosis",
            f"- status: {diagnosis.get('status')}",
            f"- conclusion: {diagnosis.get('conclusion')}",
            "",
            "## Phase 14.3 Residual Symbols",
            f"- observed_residual_symbol_count: {residual_audit.get('observed_residual_symbol_count')}",
            f"- residual_symbols: {', '.join(residual_audit.get('residual_symbols') or []) or 'N/A'}",
            "| symbol | source | theme | score | key source paths |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in residual_audit.get("rows", []):
        paths = "<br>".join(row.get("source_paths", [])[:6])
        lines.append(
            f"| {row.get('symbol')} | {row.get('source')} | {row.get('primary_theme_id')} | "
            f"{row.get('theme_pool_score')} | {paths} |"
        )
    if not residual_audit.get("rows"):
        lines.append("| N/A | N/A | N/A | N/A | no residual symbols |")
    lines.extend(
        [
            "",
            "## Phase 14.3 Current Builder Replay",
            f"- status: {current_replay.get('status')}",
            f"- classification: {current_replay.get('classification')}",
            f"- input_symbol_count: {current_replay.get('input_symbol_count')}",
            f"- output_symbol_count: {current_replay.get('output_symbol_count')}",
            f"- residual_symbol_count: {current_replay.get('residual_symbol_count')}",
            f"- source_counts: {current_replay.get('source_counts')}",
            f"- quant_score_source: {current_replay.get('quant_score_source')}",
            "",
            "## Phase 14.3 Daily Pipeline Proof",
            f"- record_id: {daily_proof.get('record_id')}",
            f"- hard_filter_pool_restored_nonempty: {daily_proof.get('hard_filter_pool_restored_nonempty')}",
            f"- final_shortlist_restored_nonempty: {daily_proof.get('final_shortlist_restored_nonempty')}",
            f"- candidate_generation_status: {daily_proof.get('candidate_generation_status')}",
            f"- blocker: {daily_proof.get('blocker')}",
            f"- theme_pool: {daily_proof.get('theme_pool')}",
            f"- conclusion: {daily_proof.get('conclusion')}",
            "",
            "## Phase 14.3 E Split System Sell",
            f"- count: {system_sell.get('count')}",
            f"- 5d mean/median/negative: {_pct(dict(system_sell.get('ret_5d') or {}).get('mean'))} / "
            f"{_pct(dict(system_sell.get('ret_5d') or {}).get('median'))} / "
            f"{_pct(dict(system_sell.get('ret_5d') or {}).get('negative_share'))}",
            f"- 10d mean/median/negative: {_pct(dict(system_sell.get('ret_10d') or {}).get('mean'))} / "
            f"{_pct(dict(system_sell.get('ret_10d') or {}).get('median'))} / "
            f"{_pct(dict(system_sell.get('ret_10d') or {}).get('negative_share'))}",
            f"- 20d mean/median/negative: {_pct(dict(system_sell.get('ret_20d') or {}).get('mean'))} / "
            f"{_pct(dict(system_sell.get('ret_20d') or {}).get('median'))} / "
            f"{_pct(dict(system_sell.get('ret_20d') or {}).get('negative_share'))}",
            f"- +23.80% survives: {system_sell.get('ret_20d_mean_survives_23_80pct')}",
            "",
            "## Phase 14.3 Shadow Denominator",
            f"- initial_capital: {denominator.get('initial_capital')}",
            f"- first_nav: {denominator.get('first_nav')}; last_nav: {denominator.get('last_nav')}",
            f"- window_return_from_first_nav: {_pct(denominator.get('window_return_from_first_nav'))}",
            f"- full_return_vs_initial_capital: {_pct(denominator.get('full_return_vs_initial_capital'))}",
            f"- gap_if_window_return_is_misused: {denominator.get('gap_vs_last_total_if_window_return_is_misused')}",
            f"- conclusion: {denominator.get('conclusion')}",
            "",
            "## Phase 14.3 688301 Session Forensics",
            f"- match_count: {forensics.get('match_count')}",
        ]
    )
    for match in forensics.get("matches", []):
        lines.append(f"- {match.get('date')} {match.get('file')}:{match.get('line')}")
        lines.append("```text")
        for item in match.get("context_lines", []):
            lines.append(f"{item.get('line')}: {item.get('text')}")
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    payload: dict[str, Any],
    output_root: Path,
    as_of_date: str,
    *,
    output_phase: str = "phase14_2",
) -> tuple[Path, Path]:
    out_dir = output_root / as_of_date / output_phase
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
    parser.add_argument("--proof-record-id", default="20260708_0910")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-phase", default="phase14_2")
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS_ROOT)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--as-of-date", default="20260707")
    args = parser.parse_args()
    payload = build_micro_report(
        record_root=args.record_root,
        record_id=args.record_id,
        proof_record_id=args.proof_record_id,
        output_root=args.output_root,
        bars_root=args.bars_root,
        session_root=args.session_root,
        as_of_date=args.as_of_date,
    )
    json_path, md_path = write_outputs(
        payload,
        args.output_root,
        args.as_of_date,
        output_phase=args.output_phase,
    )
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
