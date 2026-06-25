#!/usr/bin/env python3
"""Export CN aggressive strategy records into the static dashboard data bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_DASHBOARD_ROOT = PROJECT_ROOT / "portfolio_dashboard"
DEFAULT_INITIAL_BENCHMARK = 1.0
DEFAULT_STOCK_BASIC_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "dag_core_raw" / "table=stock_basic"
SNAPSHOT_BENCHMARK_STATUS = "not_production_grade"
SNAPSHOT_BENCHMARK_SOURCE_SYSTEM = "strategy_record.market_snapshot.indices"
INDEX_BENCHMARK_FIELDS = {
    "sh000300": "csi300_nav",
    "sz399905": "csi500_nav",
    "sz399852": "csi1000_nav",
    "sh000688": "star50_nav",
    "sz399006": "chinext_nav",
}
TUSHARE_INDEX_BENCHMARKS = {
    "000300.SH": "csi300_nav",
    "000905.SH": "csi500_nav",
    "000852.SH": "csi1000_nav",
    "000688.SH": "star50_nav",
    "399006.SZ": "chinext_nav",
}
BENCHMARK_FIELD_ORDER = [
    "benchmark_main_nav",
    "benchmark_nav",
    "csi300_nav",
    "csi500_nav",
    "csi1000_nav",
    "star50_nav",
    "chinext_nav",
]
TUSHARE_BENCHMARK_SOURCE_SYSTEM = "tushare.index_daily"
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")


@dataclass(frozen=True)
class RecordRun:
    run_id: str
    date: str
    path: Path
    record_time: str
    initial_capital: float
    total_value_after: float
    benchmark_values: dict[str, float]


@dataclass(frozen=True)
class BenchmarkExport:
    values_by_date: dict[str, dict[str, float]]
    raw_rows: list[dict[str, Any]]
    source_system: str
    normalization: str
    status_hint: str
    notes: list[str]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
        scale = 0.01
    else:
        scale = 1.0
    try:
        return float(text) * scale
    except ValueError:
        return None


def record_date_from_run(run_id: str) -> str:
    return f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"


def iso_to_tushare_date(value: str) -> str:
    return str(value or "").replace("-", "")


def tushare_to_iso_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def benchmarks_from_snapshot(run_dir: Path) -> dict[str, float]:
    snapshot = load_json(run_dir / "market_snapshot.json")
    indices = snapshot.get("indices")
    if not isinstance(indices, dict):
        return {}
    values: dict[str, float] = {}
    for code, field in INDEX_BENCHMARK_FIELDS.items():
        payload = indices.get(code) or {}
        value = parse_float(payload.get("current") or payload.get("realtime_price"))
        if value:
            values[field] = value
    return values


def build_tushare_benchmark_export(runs: list[RecordRun], pro: Any) -> tuple[BenchmarkExport | None, list[str]]:
    warnings: list[str] = []
    if not runs:
        return None, ["没有可用策略记录，无法拉取 Tushare benchmark。"]
    start_date = iso_to_tushare_date(runs[0].date)
    end_date = iso_to_tushare_date(runs[-1].date)
    closes_by_field: dict[str, dict[str, float]] = {}
    raw_rows: list[dict[str, Any]] = []
    for ts_code, field in TUSHARE_INDEX_BENCHMARKS.items():
        try:
            frame = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception as exc:  # pragma: no cover - live provider defensive guard.
            warnings.append(f"Tushare index_daily({ts_code}) 调用失败：{exc}")
            continue
        if frame is None or getattr(frame, "empty", True):
            warnings.append(f"Tushare index_daily({ts_code}) 未返回数据。")
            continue
        if "trade_date" not in frame.columns or "close" not in frame.columns:
            warnings.append(f"Tushare index_daily({ts_code}) 缺少 trade_date 或 close 字段。")
            continue
        close_by_date: dict[str, float] = {}
        for _, row in frame.iterrows():
            iso_date = tushare_to_iso_date(row.get("trade_date"))
            close_value = parse_float(row.get("close"))
            if not iso_date or close_value is None or close_value <= 0:
                continue
            close_by_date[iso_date] = close_value
        if not close_by_date:
            warnings.append(f"Tushare index_daily({ts_code}) 未返回有效 close。")
            continue
        first_date = min(close_by_date)
        first_close = close_by_date[first_date]
        if first_close <= 0:
            warnings.append(f"Tushare index_daily({ts_code}) 首个 close 非正数，已跳过。")
            continue
        closes_by_field[field] = close_by_date
        for iso_date in sorted(close_by_date):
            nav_value = close_by_date[iso_date] / first_close
            raw_rows.append(
                {
                    "date": iso_date,
                    "ts_code": ts_code,
                    "field": field,
                    "close": f"{close_by_date[iso_date]:.6f}",
                    "nav": f"{nav_value:.8f}",
                    "source_system": TUSHARE_BENCHMARK_SOURCE_SYSTEM,
                }
            )

    if not closes_by_field:
        return None, warnings or ["Tushare 未返回任何可用 benchmark 指数历史。"]

    first_closes = {
        field: close_by_date[min(close_by_date)]
        for field, close_by_date in closes_by_field.items()
        if close_by_date
    }
    values_by_date: dict[str, dict[str, float]] = {}
    for run in runs:
        normalized: dict[str, float] = {}
        for field, close_by_date in closes_by_field.items():
            close_value = close_by_date.get(run.date)
            first_close = first_closes.get(field)
            if close_value and first_close:
                normalized[field] = close_value / first_close
        main_components = [
            ("star50_nav", 0.50),
            ("csi300_nav", 0.30),
            ("chinext_nav", 0.20),
        ]
        used_weight = sum(weight for field, weight in main_components if field in normalized)
        if used_weight:
            normalized["benchmark_main_nav"] = (
                sum(normalized[field] * weight for field, weight in main_components if field in normalized)
                / used_weight
            )
        if "csi300_nav" in normalized:
            normalized["benchmark_nav"] = normalized["csi300_nav"]
        if normalized:
            values_by_date[run.date] = normalized

    return BenchmarkExport(
        values_by_date=values_by_date,
        raw_rows=raw_rows,
        source_system=TUSHARE_BENCHMARK_SOURCE_SYSTEM,
        normalization="tushare_index_daily_close_divided_by_first_valid_close",
        status_hint="production_source",
        notes=[
            "benchmark 来自 Tushare index_daily 连续指数 close，并按 close/first_valid_close 归一化。",
            "若 latest strategy record 日期尚无 Tushare index_daily close，则该日期 benchmark 留空并降级为 partial。",
        ],
    ), warnings


def load_tushare_benchmark_export(runs: list[RecordRun]) -> tuple[BenchmarkExport | None, list[str]]:
    try:
        import tushare as ts  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local environment.
        return None, [f"未安装或无法导入 tushare，无法拉取生产 benchmark：{exc}"]

    from quant_investor.config import Config
    from quant_investor.credential_utils import create_tushare_pro

    pro = create_tushare_pro(ts, Config.TUSHARE_TOKEN, Config.TUSHARE_URL)
    if pro is None:
        return None, ["TUSHARE_TOKEN 未设置，无法拉取生产 benchmark。"]
    return build_tushare_benchmark_export(runs, pro)


@contextmanager
def suppress_native_stderr():
    """Silence native library stderr noise while preserving script warnings."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def normalize_pnl_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """Support both current wide pnl_summary and older metric,value exports."""
    if not rows:
        return {}
    first = rows[0]
    keys = {key.strip().lower() for key in first}
    if {"metric", "value"}.issubset(keys):
        normalized: dict[str, str] = {}
        for row in rows:
            metric = str(row.get("metric") or "").strip()
            if metric:
                normalized[metric] = row.get("value", "")
        return normalized
    return first


def infer_initial_capital(run_dir: Path, pnl_row: dict[str, str]) -> float | None:
    initial_capital = parse_float(pnl_row.get("initial_capital"))
    if initial_capital:
        return initial_capital

    total_value = parse_float(pnl_row.get("total_value_after"))
    pnl_after = parse_float(pnl_row.get("portfolio_pnl_after"))
    if total_value is not None and pnl_after is not None:
        inferred = total_value - pnl_after
        if inferred > 0:
            return inferred

    pnl_pct = parse_float(pnl_row.get("portfolio_pnl_pct_after"))
    if total_value is not None and pnl_pct is not None and pnl_pct > -1:
        inferred = total_value / (1 + pnl_pct)
        if inferred > 0:
            return inferred

    snapshot_portfolio = load_json(run_dir / "market_snapshot.json").get("portfolio")
    if isinstance(snapshot_portfolio, dict):
        snapshot_total = parse_float(snapshot_portfolio.get("total_value"))
        snapshot_pnl = parse_float(snapshot_portfolio.get("portfolio_pnl"))
        if snapshot_total is not None and snapshot_pnl is not None:
            inferred = snapshot_total - snapshot_pnl
            if inferred > 0:
                return inferred
    return None


def discover_record_runs(record_root: Path) -> tuple[list[RecordRun], list[str]]:
    warnings: list[str] = []
    runs: list[RecordRun] = []
    for run_dir in sorted(record_root.iterdir() if record_root.exists() else []):
        if not run_dir.is_dir() or not run_dir.name[:8].isdigit():
            continue
        pnl_rows = read_csv_rows(run_dir / "pnl_summary.csv")
        if not pnl_rows:
            continue
        row = normalize_pnl_row(pnl_rows)
        initial_capital = infer_initial_capital(run_dir, row)
        total_value_after = parse_float(row.get("total_value_after"))
        if not initial_capital or not total_value_after:
            warnings.append(f"{run_dir.name}: pnl_summary.csv 缺少 initial_capital 或 total_value_after，已跳过。")
            continue
        runs.append(
            RecordRun(
                run_id=run_dir.name,
                date=record_date_from_run(run_dir.name),
                path=run_dir,
                record_time=row.get("record_time", ""),
                initial_capital=initial_capital,
                total_value_after=total_value_after,
                benchmark_values=benchmarks_from_snapshot(run_dir),
            )
        )
    latest_by_date: dict[str, RecordRun] = {}
    for run in runs:
        latest_by_date[run.date] = run
    return [latest_by_date[date] for date in sorted(latest_by_date)], warnings


def build_theme_map(run: RecordRun) -> dict[str, str]:
    theme_map: dict[str, str] = {}
    snapshot = load_json(run.path / "market_snapshot.json")
    for row in snapshot.get("theme_strength", []) if isinstance(snapshot.get("theme_strength"), list) else []:
        theme = str(row.get("theme") or "").strip()
        symbols = row.get("symbols")
        if not theme or not isinstance(symbols, list):
            continue
        for symbol_value in symbols:
            symbol = str(symbol_value or "").strip()
            if symbol:
                theme_map[symbol] = theme
    for row in snapshot.get("candidate_pool", []) if isinstance(snapshot.get("candidate_pool"), list) else []:
        symbol = str(row.get("symbol") or "").strip()
        theme = str(row.get("theme_label") or row.get("candidate_source") or "").strip()
        if symbol and theme:
            theme_map.setdefault(symbol, theme)
    for row in snapshot.get("switch_plan", []) if isinstance(snapshot.get("switch_plan"), list) else []:
        buy_symbol = str(row.get("buy_symbol") or "").strip()
        buy_theme = str(row.get("buy_theme") or "").strip()
        if buy_symbol and buy_theme:
            theme_map.setdefault(buy_symbol, buy_theme)
    return theme_map


def load_sector_map(stock_basic_root: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not stock_basic_root.exists():
        return {}, [f"未找到 stock_basic 行业映射目录：{stock_basic_root}，sector 留空。"]
    try:
        with suppress_native_stderr():
            import pyarrow.dataset as ds  # type: ignore[import-not-found]
    except ImportError:
        return {}, ["当前 Python 环境未安装 pyarrow，无法读取 stock_basic 行业映射，sector 留空；可用 ./.venv/bin/python 运行导出脚本。"]

    try:
        with suppress_native_stderr():
            dataset = ds.dataset(str(stock_basic_root), format="parquet")
            table = dataset.to_table(columns=["ts_code", "industry"])
    except Exception as exc:  # pragma: no cover - defensive local artifact guard.
        return {}, [f"读取 stock_basic 行业映射失败：{exc}，sector 留空。"]

    mapping: dict[str, dict[str, str]] = {}
    for row in table.to_pylist():
        symbol = str(row.get("ts_code") or "").strip()
        industry = str(row.get("industry") or "").strip()
        if symbol and industry:
            mapping[symbol] = {"sector": industry, "sub_sector": ""}
    return mapping, []


def build_nav_rows(
    runs: list[RecordRun],
    benchmark_export: BenchmarkExport | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    if benchmark_export is not None and benchmark_export.values_by_date:
        available_fields = [
            field
            for field in BENCHMARK_FIELD_ORDER
            if any(field in values for values in benchmark_export.values_by_date.values())
        ]
        for run in runs:
            row: dict[str, Any] = {
                "date": run.date,
                "portfolio_nav": f"{run.total_value_after / run.initial_capital:.8f}",
                "cash_weight": "",
                "gross_exposure": "",
                "net_exposure": "",
            }
            values = benchmark_export.values_by_date.get(run.date, {})
            for field in available_fields:
                row[field] = f"{values[field]:.8f}" if field in values else ""
            rows.append(row)
        fieldnames = [
            "date",
            "portfolio_nav",
            *available_fields,
            "cash_weight",
            "gross_exposure",
            "net_exposure",
        ]
        return rows, warnings, list(dict.fromkeys(fieldnames))

    available_fields = sorted(
        {field for run in runs for field, value in run.benchmark_values.items() if value},
        key=lambda field: list(INDEX_BENCHMARK_FIELDS.values()).index(field)
        if field in INDEX_BENCHMARK_FIELDS.values()
        else 999,
    )
    first_values: dict[str, float] = {}
    for field in available_fields:
        first = next((run.benchmark_values[field] for run in runs if run.benchmark_values.get(field)), None)
        if first:
            first_values[field] = first
    if not first_values:
        warnings.append("记录中未找到可用 benchmark 指数快照，benchmark_main_nav 使用 1.0 平线占位。")
    for run in runs:
        normalized: dict[str, float] = {}
        for field in available_fields:
            first = first_values.get(field)
            value = run.benchmark_values.get(field)
            if first and value:
                normalized[field] = value / first
        main_components = [
            ("star50_nav", 0.50),
            ("csi300_nav", 0.30),
            ("chinext_nav", 0.20),
        ]
        used_weight = sum(weight for field, weight in main_components if field in normalized)
        if used_weight:
            benchmark_main_nav = sum(normalized[field] * weight for field, weight in main_components if field in normalized) / used_weight
        elif "csi300_nav" in normalized:
            benchmark_main_nav = normalized["csi300_nav"]
        else:
            benchmark_main_nav = DEFAULT_INITIAL_BENCHMARK
        legacy_benchmark_nav = normalized.get("csi300_nav", benchmark_main_nav)
        row: dict[str, Any] = {
            "date": run.date,
            "portfolio_nav": f"{run.total_value_after / run.initial_capital:.8f}",
            "benchmark_main_nav": f"{benchmark_main_nav:.8f}",
            "benchmark_nav": f"{legacy_benchmark_nav:.8f}",
            "cash_weight": "",
            "gross_exposure": "",
            "net_exposure": "",
        }
        for field in available_fields:
            row[field] = f"{normalized[field]:.8f}" if field in normalized else ""
        rows.append(
            row
        )
    fieldnames = [
        "date",
        "portfolio_nav",
        "benchmark_main_nav",
        "benchmark_nav",
        *available_fields,
        "cash_weight",
        "gross_exposure",
        "net_exposure",
    ]
    # Preserve stable order and avoid duplicating csi300_nav if also used as legacy benchmark_nav.
    fieldnames = list(dict.fromkeys(fieldnames))
    return rows, warnings, fieldnames


def benchmark_source_summary(
    nav_rows: list[dict[str, Any]],
    fieldnames: list[str],
    benchmark_export: BenchmarkExport | None = None,
) -> dict[str, Any]:
    benchmark_fields = [
        field
        for field in fieldnames
        if field.endswith("_nav") and field != "portfolio_nav"
    ]
    actual_snapshot_fields = [
        field
        for field in benchmark_fields
        if field not in {"benchmark_main_nav", "benchmark_nav"}
        and any(str(row.get(field) or "").strip() for row in nav_rows)
    ]
    field_missing_counts = {
        field: sum(1 for row in nav_rows if not str(row.get(field) or "").strip())
        for field in benchmark_fields
    }
    valid_dates = [
        str(row.get("date") or "")
        for row in nav_rows
        if any(str(row.get(field) or "").strip() for field in actual_snapshot_fields)
    ]
    if not actual_snapshot_fields:
        return {
            "benchmark_fields": benchmark_fields,
            "benchmark_source_status": "benchmark_production_source_unavailable",
            "source_system": "none",
            "production_grade": False,
            "first_valid_date": "",
            "last_valid_date": "",
            "missing_date_count": len(nav_rows),
            "field_missing_counts": field_missing_counts,
            "normalization": "unavailable",
            "notes": [
                "未找到连续真实指数历史；benchmark_main_nav/benchmark_nav 不能作为生产 benchmark。",
            ],
        }
    if benchmark_export is not None:
        missing_dates = [
            str(row.get("date") or "")
            for row in nav_rows
            if not all(str(row.get(field) or "").strip() for field in actual_snapshot_fields)
        ]
        missing_date_count = sum(
            1
            for row in nav_rows
            if not all(str(row.get(field) or "").strip() for field in actual_snapshot_fields)
        )
        last_nav_date = str(nav_rows[-1].get("date") or "") if nav_rows else ""
        latest_only_missing = bool(missing_dates) and missing_dates == [last_nav_date]
        if missing_date_count == 0:
            status = "production_grade"
        elif latest_only_missing:
            status = "production_source_partial_latest_unavailable"
        else:
            status = "production_source_partial_missing_dates"
        notes = list(benchmark_export.notes)
        if missing_date_count:
            notes.append("Tushare benchmark 未覆盖所有策略记录日期；缺失日期不按生产级 benchmark 标记。")
        return {
            "benchmark_fields": benchmark_fields,
            "benchmark_source_status": status,
            "source_system": benchmark_export.source_system,
            "production_grade": status == "production_grade",
            "first_valid_date": min(valid_dates) if valid_dates else "",
            "last_valid_date": max(valid_dates) if valid_dates else "",
            "missing_date_count": missing_date_count,
            "missing_dates": missing_dates,
            "field_missing_counts": field_missing_counts,
            "normalization": benchmark_export.normalization,
            "notes": notes,
            "raw_row_count": len(benchmark_export.raw_rows),
        }
    return {
        "benchmark_fields": benchmark_fields,
        "benchmark_source_status": SNAPSHOT_BENCHMARK_STATUS,
        "source_system": SNAPSHOT_BENCHMARK_SOURCE_SYSTEM,
        "production_grade": False,
        "first_valid_date": min(valid_dates) if valid_dates else "",
        "last_valid_date": max(valid_dates) if valid_dates else "",
        "missing_date_count": sum(
            1
            for row in nav_rows
            if not all(str(row.get(field) or "").strip() for field in actual_snapshot_fields)
        ),
        "field_missing_counts": field_missing_counts,
        "normalization": "snapshot_close_divided_by_first_available_snapshot_close",
        "notes": [
            "当前 benchmark 来自各策略记录 market_snapshot.indices 的零散指数快照，不是连续真实指数历史。",
            "正式投委会口径需要 Wind/Choice/iFinD/Bloomberg/Tushare 或内部数据库的连续指数 close，并按 close/first_valid_close 归一化。",
        ],
    }


def build_positions_rows(
    runs: list[RecordRun],
    sector_map: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run in runs:
        ledger_path = run.path / "ledger_after_manual_switch.csv"
        if not ledger_path.exists():
            ledger_path = run.path / "ledger.csv"
        ledger_rows = read_csv_rows(ledger_path)
        if not ledger_rows:
            warnings.append(f"{run.run_id}: 未找到可用 ledger，positions 已跳过。")
            continue
        holdings_review = {
            row.get("symbol", ""): row
            for row in read_csv_rows(run.path / "holdings_review.csv")
            if row.get("symbol")
        }
        theme_map = build_theme_map(run)
        for row in ledger_rows:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            value = parse_float(row.get("current_value"))
            weight = parse_float(row.get("market_weight"))
            if weight is None and value is not None and run.total_value_after:
                weight = value / run.total_value_after
            review = holdings_review.get(symbol, {})
            daily_return = parse_float(review.get("today_change_pct"))
            if daily_return is not None and abs(daily_return) > 1:
                daily_return = daily_return / 100.0
            contribution = weight * daily_return if weight is not None and daily_return is not None else None
            sector_info = sector_map.get(symbol, {})
            explicit_theme = theme_map.get(symbol, "")
            theme = explicit_theme
            if not theme and sector_info.get("sector"):
                theme = f"行业: {sector_info['sector']}"
            if not theme:
                theme = "UNSPECIFIED_RECORD_THEME"
            rows.append(
                {
                    "date": run.date,
                    "ticker": symbol,
                    "name": row.get("name") or review.get("name") or "UNKNOWN_NAME",
                    "weight": f"{weight:.8f}" if weight is not None else "",
                    "theme": theme,
                    "sector": sector_info.get("sector", ""),
                    "sub_sector": sector_info.get("sub_sector", ""),
                    "daily_return": f"{daily_return:.8f}" if daily_return is not None else "",
                    "contribution": f"{contribution:.8f}" if contribution is not None else "",
                    "market_value": f"{value:.2f}" if value is not None else "",
                }
            )
    unspecified_count = sum(1 for row in rows if row["theme"] == "UNSPECIFIED_RECORD_THEME")
    if unspecified_count:
        warnings.append(f"{unspecified_count} 条持仓记录未找到逐股票 theme，positions.csv 使用 UNSPECIFIED_RECORD_THEME。")
    sector_theme_count = sum(1 for row in rows if row["theme"].startswith("行业: "))
    if sector_theme_count:
        warnings.append(f"{sector_theme_count} 条持仓记录缺少显式 theme，已使用 stock_basic industry 生成 '行业: <sector>' 回退标签。")
    return rows, warnings


def build_trade_rows(runs: list[RecordRun], sector_map: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for run in runs:
        theme_map = build_theme_map(run)
        for row in read_csv_rows(run.path / "manual_switch_and_take_profit_orders.csv"):
            status = str(row.get("status") or "").strip().lower()
            if status and status != "filled":
                continue
            symbol = str(row.get("symbol") or "").strip()
            action = str(row.get("action") or "").strip().lower()
            timestamp = str(row.get("timestamp") or run.record_time or run.date).strip()
            if not symbol or action not in {"buy", "sell"}:
                continue
            key = (timestamp, action, symbol, str(row.get("shares") or ""))
            if key in seen:
                continue
            seen.add(key)
            price = parse_float(row.get("execution_price"))
            quantity = parse_float(row.get("shares"))
            amount = parse_float(row.get("trade_value"))
            sector = sector_map.get(symbol, {}).get("sector", "")
            theme = theme_map.get(symbol) or (f"行业: {sector}" if sector else "UNSPECIFIED_RECORD_THEME")
            rows.append(
                {
                    "trade_date": run.date,
                    "ticker": symbol,
                    "name": row.get("name") or "UNKNOWN_NAME",
                    "side": action,
                    "price": f"{price:.4f}" if price is not None else "",
                    "quantity": f"{quantity:.0f}" if quantity is not None else "",
                    "trade_amount": f"{amount:.2f}" if amount is not None else "",
                    "fee": "0",
                    "reason": row.get("reason") or "",
                    "theme": theme,
                }
            )
    return rows


def js_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_generated_js(
    path: Path,
    *,
    generated_at: str,
    source_root: Path,
    latest_record: str,
    record_count: int,
    warnings: list[str],
    nav_csv: str,
    positions_csv: str,
    trades_csv: str,
) -> None:
    payload = (
        "window.DashboardGeneratedRecords = {\n"
        f"  generatedAt: {js_string(generated_at)},\n"
        f"  sourceRoot: {js_string(str(source_root))},\n"
        f"  latestRecord: {js_string(latest_record)},\n"
        f"  recordCount: {record_count},\n"
        f"  warnings: {json.dumps(warnings, ensure_ascii=False, indent=2)},\n"
        "  csv: {\n"
        f"    nav: {js_string(nav_csv)},\n"
        f"    positions: {js_string(positions_csv)},\n"
        f"    trades: {js_string(trades_csv)}\n"
        "  }\n"
        "};\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def export(record_root: Path, dashboard_root: Path, benchmark_source: str = "tushare") -> dict[str, Any]:
    runs, warnings = discover_record_runs(record_root)
    if not runs:
        raise SystemExit(f"No usable records found under {record_root}")
    benchmark_export: BenchmarkExport | None = None
    if benchmark_source in {"auto", "tushare"}:
        benchmark_export, tushare_warnings = load_tushare_benchmark_export(runs)
        warnings.extend(tushare_warnings)
        if benchmark_export is None and benchmark_source == "tushare":
            warnings.append("已按 fail-closed 降级为 strategy_record.market_snapshot.indices benchmark；不得标记为生产级。")
    if benchmark_source == "snapshot":
        warnings.append("benchmark_source=snapshot：按显式参数使用 strategy_record.market_snapshot.indices，非生产级。")
    nav_rows, nav_warnings, nav_fieldnames = build_nav_rows(runs, benchmark_export)
    benchmark_summary = benchmark_source_summary(nav_rows, nav_fieldnames, benchmark_export)
    sector_map, sector_warnings = load_sector_map(DEFAULT_STOCK_BASIC_ROOT)
    positions_rows, position_warnings = build_positions_rows(runs, sector_map)
    trade_rows = build_trade_rows(runs, sector_map)
    warnings.extend(nav_warnings)
    warnings.extend(sector_warnings)
    warnings.extend(position_warnings)
    if not benchmark_summary["production_grade"]:
        warnings.append(
            "benchmark_source_status="
            f"{benchmark_summary['benchmark_source_status']}；"
            f"source_system={benchmark_summary['source_system']}；"
            "Dashboard benchmark 仅供临时展示，不是正式投委会口径。"
        )

    generated_dir = dashboard_root / "generated"
    nav_path = generated_dir / "nav_records.csv"
    positions_path = generated_dir / "positions_records.csv"
    trades_path = generated_dir / "trades_records.csv"
    benchmark_path = generated_dir / "benchmark_records.csv"
    write_csv(
        nav_path,
        nav_fieldnames,
        nav_rows,
    )
    write_csv(
        positions_path,
        [
            "date",
            "ticker",
            "name",
            "weight",
            "theme",
            "sector",
            "sub_sector",
            "daily_return",
            "contribution",
            "market_value",
        ],
        positions_rows,
    )
    write_csv(
        trades_path,
        [
            "trade_date",
            "ticker",
            "name",
            "side",
            "price",
            "quantity",
            "trade_amount",
            "fee",
            "reason",
            "theme",
        ],
        trade_rows,
    )
    write_csv(
        benchmark_path,
        ["date", "ts_code", "field", "close", "nav", "source_system"],
        benchmark_export.raw_rows if benchmark_export is not None else [],
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_generated_js(
        dashboard_root / "js" / "generated_records.js",
        generated_at=generated_at,
        source_root=record_root,
        latest_record=runs[-1].run_id,
        record_count=len(runs),
        warnings=warnings,
        nav_csv=csv_text(nav_path),
        positions_csv=csv_text(positions_path),
        trades_csv=csv_text(trades_path),
    )
    summary = {
        "generated_at": generated_at,
        "source_root": str(record_root),
        "dashboard_root": str(dashboard_root),
        "latest_record": runs[-1].run_id,
        "record_count": len(runs),
        "nav_rows": len(nav_rows),
        "positions_rows": len(positions_rows),
        "trade_rows": len(trade_rows),
        "warnings": warnings,
        "benchmark_source": benchmark_summary,
        "files": {
            "nav": str(nav_path),
            "positions": str(positions_path),
            "trades": str(trades_path),
            "benchmark": str(benchmark_path),
            "generated_js": str(dashboard_root / "js" / "generated_records.js"),
        },
    }
    (generated_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--dashboard-root", type=Path, default=DEFAULT_DASHBOARD_ROOT)
    parser.add_argument(
        "--benchmark-source",
        choices=["tushare", "auto", "snapshot"],
        default=os.environ.get("CN_DASHBOARD_BENCHMARK_SOURCE", "tushare"),
        help="Benchmark source for dashboard NAV fields. Defaults to Tushare index_daily.",
    )
    args = parser.parse_args()
    summary = export(args.record_root, args.dashboard_root, args.benchmark_source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
