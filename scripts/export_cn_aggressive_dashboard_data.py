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
DEFAULT_BENCHMARK_SOURCE = "auto"
DEFAULT_INITIAL_BENCHMARK = 1.0
DEFAULT_STOCK_BASIC_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "dag_core_raw" / "table=stock_basic"
DEFAULT_CN_BARS_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "bars"
SNAPSHOT_BENCHMARK_STATUS = "not_production_grade"
SNAPSHOT_BENCHMARK_SOURCE_SYSTEM = "strategy_record.market_snapshot.indices"
SNAPSHOT_GAP_FILL_COVERAGE = "strategy_record_snapshot_gap_fill"
INDUSTRY_EW_NAV_FIELD = "industry_ew_nav"
INDUSTRY_EW_SOURCE_SYSTEM = "local_parquet.industry_equal_weight"
INDUSTRY_EW_TS_CODE = "LOCAL_INDUSTRY_EW"
SUPPLEMENTAL_BENCHMARK_FIELDS = {INDUSTRY_EW_NAV_FIELD}
DEFAULT_TECH_MANUFACTURING_INDUSTRIES = (
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
LOCAL_BENCHMARK_REQUIRED_COLUMNS = {"date", "ts_code", "close", "source_system"}
LOCAL_BENCHMARK_COVERAGE_VALUES = {"exact_close", "previous_trading_day_ffill"}
LOCAL_BENCHMARK_FORBIDDEN_SOURCE_TOKENS = (
    "sample",
    "mock",
    "demo",
    SNAPSHOT_BENCHMARK_SOURCE_SYSTEM,
)
BENCHMARK_FIELD_ORDER = [
    "benchmark_main_nav",
    "benchmark_nav",
    "csi300_nav",
    "csi500_nav",
    "csi1000_nav",
    "star50_nav",
    "chinext_nav",
    INDUSTRY_EW_NAV_FIELD,
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
    coverage_by_date: dict[str, dict[str, str]]
    value_date_by_date: dict[str, dict[str, str]]
    snapshot_gap_fill_by_date: dict[str, list[str]] | None = None
    calendar_source_system: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TradeCalendarDay:
    is_open: bool
    pretrade_date: str


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


def load_tushare_trade_calendar(
    pro: Any,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, TradeCalendarDay], list[str]]:
    warnings: list[str] = []
    try:
        frame = pro.trade_cal(exchange="", start_date=start_date, end_date=end_date)
    except TypeError:
        try:
            frame = pro.trade_cal(start_date=start_date, end_date=end_date)
        except Exception as exc:  # pragma: no cover - live provider defensive guard.
            return {}, [f"Tushare trade_cal 调用失败：{exc}"]
    except Exception as exc:  # pragma: no cover - live provider defensive guard.
        return {}, [f"Tushare trade_cal 调用失败：{exc}"]
    if frame is None or getattr(frame, "empty", True):
        return {}, ["Tushare trade_cal 未返回数据，非交易日 benchmark 不做前向填充。"]
    if "cal_date" not in frame.columns or "is_open" not in frame.columns:
        return {}, ["Tushare trade_cal 缺少 cal_date 或 is_open 字段，非交易日 benchmark 不做前向填充。"]
    calendar: dict[str, TradeCalendarDay] = {}
    for _, row in frame.iterrows():
        iso_date = tushare_to_iso_date(row.get("cal_date"))
        if not iso_date:
            continue
        is_open = str(row.get("is_open") or "").strip() in {"1", "1.0", "True", "true"}
        pretrade_date = tushare_to_iso_date(row.get("pretrade_date"))
        calendar[iso_date] = TradeCalendarDay(is_open=is_open, pretrade_date=pretrade_date)
    if not calendar:
        warnings.append("Tushare trade_cal 未返回有效交易日历，非交易日 benchmark 不做前向填充。")
    return calendar, warnings


def build_tushare_benchmark_export(
    runs: list[RecordRun],
    pro: Any,
    *,
    snapshot_gap_fill: bool = False,
) -> tuple[BenchmarkExport | None, list[str]]:
    warnings: list[str] = []
    if not runs:
        return None, ["没有可用策略记录，无法拉取 Tushare benchmark。"]
    start_date = iso_to_tushare_date(runs[0].date)
    end_date = iso_to_tushare_date(runs[-1].date)
    trade_calendar, calendar_warnings = load_tushare_trade_calendar(pro, start_date, end_date)
    warnings.extend(calendar_warnings)
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
                    "value_date": iso_date,
                    "coverage": "exact_close",
                }
            )

    if not closes_by_field:
        return None, warnings or ["Tushare 未返回任何可用 benchmark 指数历史。"]

    first_closes = {
        field: close_by_date[min(close_by_date)]
        for field, close_by_date in closes_by_field.items()
        if close_by_date
    }
    snapshot_first_values = {
        field: (
            first_closes.get(field)
            or next((run.benchmark_values[field] for run in runs if run.benchmark_values.get(field)), None)
        )
        for field in closes_by_field
    }
    values_by_date: dict[str, dict[str, float]] = {}
    coverage_by_date: dict[str, dict[str, str]] = {}
    value_date_by_date: dict[str, dict[str, str]] = {}
    snapshot_gap_fill_by_date: dict[str, list[str]] = {}
    for run in runs:
        normalized: dict[str, float] = {}
        coverage: dict[str, str] = {}
        value_dates: dict[str, str] = {}
        for field, close_by_date in closes_by_field.items():
            close_value = close_by_date.get(run.date)
            value_date = run.date
            coverage_status = "exact_close"
            if close_value is None:
                calendar_day = trade_calendar.get(run.date)
                if calendar_day and not calendar_day.is_open and calendar_day.pretrade_date:
                    close_value = close_by_date.get(calendar_day.pretrade_date)
                    if close_value is not None:
                        value_date = calendar_day.pretrade_date
                        coverage_status = "previous_trading_day_ffill"
            first_close = first_closes.get(field)
            if close_value and first_close:
                normalized[field] = close_value / first_close
                coverage[field] = coverage_status
                value_dates[field] = value_date
                if coverage_status == "previous_trading_day_ffill":
                    raw_rows.append(
                        {
                            "date": run.date,
                            "ts_code": next(
                                ts_code
                                for ts_code, benchmark_field in TUSHARE_INDEX_BENCHMARKS.items()
                                if benchmark_field == field
                            ),
                            "field": field,
                            "close": f"{close_value:.6f}",
                            "nav": f"{normalized[field]:.8f}",
                            "source_system": TUSHARE_BENCHMARK_SOURCE_SYSTEM,
                            "value_date": value_date,
                            "coverage": coverage_status,
                        }
                    )
                continue
            if snapshot_gap_fill:
                snapshot_value = run.benchmark_values.get(field)
                snapshot_first = snapshot_first_values.get(field)
                if snapshot_value and snapshot_first:
                    normalized[field] = snapshot_value / snapshot_first
                    coverage[field] = SNAPSHOT_GAP_FILL_COVERAGE
                    value_dates[field] = run.date
                    snapshot_gap_fill_by_date.setdefault(run.date, []).append(field)
                    raw_rows.append(
                        {
                            "date": run.date,
                            "ts_code": next(
                                ts_code
                                for ts_code, benchmark_field in TUSHARE_INDEX_BENCHMARKS.items()
                                if benchmark_field == field
                            ),
                            "field": field,
                            "close": f"{snapshot_value:.6f}",
                            "nav": f"{normalized[field]:.8f}",
                            "source_system": SNAPSHOT_BENCHMARK_SOURCE_SYSTEM,
                            "value_date": run.date,
                            "coverage": SNAPSHOT_GAP_FILL_COVERAGE,
                        }
                    )
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
            component_coverages = [
                coverage[field] for field, _weight in main_components if field in normalized and field in coverage
            ]
            component_value_dates = [
                value_dates[field] for field, _weight in main_components if field in normalized and field in value_dates
            ]
            coverage["benchmark_main_nav"] = (
                SNAPSHOT_GAP_FILL_COVERAGE
                if SNAPSHOT_GAP_FILL_COVERAGE in component_coverages
                else (
                    "previous_trading_day_ffill"
                    if "previous_trading_day_ffill" in component_coverages
                    else "exact_close"
                )
            )
            value_dates["benchmark_main_nav"] = min(component_value_dates) if component_value_dates else run.date
        if "csi300_nav" in normalized:
            normalized["benchmark_nav"] = normalized["csi300_nav"]
            coverage["benchmark_nav"] = coverage.get("csi300_nav", "exact_close")
            value_dates["benchmark_nav"] = value_dates.get("csi300_nav", run.date)
        if normalized:
            values_by_date[run.date] = normalized
            coverage_by_date[run.date] = coverage
            value_date_by_date[run.date] = value_dates

    return BenchmarkExport(
        values_by_date=values_by_date,
        raw_rows=raw_rows,
        source_system=TUSHARE_BENCHMARK_SOURCE_SYSTEM,
        normalization="tushare_index_daily_close_divided_by_first_valid_close_with_trade_cal_previous_trading_day_ffill",
        status_hint="production_source",
        notes=[
            "benchmark 来自 Tushare index_daily 连续指数 close，并按 close/first_valid_close 归一化。",
            "非交易日 strategy record 使用 Tushare trade_cal 的 pretrade_date 做 previous_trading_day_ffill，并在 coverage 中显式标记。",
            "若交易日或 latest strategy record 日期尚无 Tushare index_daily close，则该日期 benchmark 留空并降级为 partial。",
        ],
        coverage_by_date=coverage_by_date,
        value_date_by_date=value_date_by_date,
        snapshot_gap_fill_by_date=snapshot_gap_fill_by_date,
        calendar_source_system="tushare.trade_cal",
    ), warnings


def _is_forbidden_local_benchmark_source(source_system: str) -> bool:
    text = source_system.strip().lower()
    return any(token.lower() in text for token in LOCAL_BENCHMARK_FORBIDDEN_SOURCE_TOKENS)


def _with_composite_benchmark_fields(
    normalized: dict[str, float],
    coverage: dict[str, str],
    value_dates: dict[str, str],
    *,
    run_date: str,
) -> None:
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
        component_coverages = [
            coverage[field] for field, _weight in main_components if field in normalized and field in coverage
        ]
        component_value_dates = [
            value_dates[field] for field, _weight in main_components if field in normalized and field in value_dates
        ]
        coverage["benchmark_main_nav"] = (
            SNAPSHOT_GAP_FILL_COVERAGE
            if SNAPSHOT_GAP_FILL_COVERAGE in component_coverages
            else (
                "previous_trading_day_ffill"
                if "previous_trading_day_ffill" in component_coverages
                else "exact_close"
            )
        )
        value_dates["benchmark_main_nav"] = min(component_value_dates) if component_value_dates else run_date
    if "csi300_nav" in normalized:
        normalized["benchmark_nav"] = normalized["csi300_nav"]
        coverage["benchmark_nav"] = coverage.get("csi300_nav", "exact_close")
        value_dates["benchmark_nav"] = value_dates.get("csi300_nav", run_date)


def load_local_benchmark_export(
    runs: list[RecordRun],
    benchmark_file: Path,
) -> tuple[BenchmarkExport | None, list[str]]:
    warnings: list[str] = []
    if not runs:
        return None, ["没有可用策略记录，无法读取本地 benchmark。"]
    if not benchmark_file.exists():
        return None, [f"未找到本地 benchmark 文件：{benchmark_file}"]

    rows = read_csv_rows(benchmark_file)
    if not rows:
        return None, [f"本地 benchmark 文件为空：{benchmark_file}"]
    missing_columns = sorted(LOCAL_BENCHMARK_REQUIRED_COLUMNS - set(rows[0]))
    if missing_columns:
        return None, [f"本地 benchmark 文件缺少字段：{', '.join(missing_columns)}"]

    start_date = runs[0].date
    end_date = runs[-1].date
    required_fields = set(TUSHARE_INDEX_BENCHMARKS.values())
    parsed_rows: dict[tuple[str, str], dict[str, Any]] = {}
    source_systems: set[str] = set()
    forbidden_sources: set[str] = set()
    invalid_row_count = 0
    ignored_code_count = 0
    for row_number, row in enumerate(rows, start=2):
        iso_date = tushare_to_iso_date(row.get("date"))
        if not iso_date or iso_date < start_date or iso_date > end_date:
            continue
        ts_code = str(row.get("ts_code") or "").strip()
        field = TUSHARE_INDEX_BENCHMARKS.get(ts_code)
        if not field:
            ignored_code_count += 1
            continue
        close_value = parse_float(row.get("close"))
        source_system = str(row.get("source_system") or "").strip()
        coverage = str(row.get("coverage") or "exact_close").strip()
        value_date = tushare_to_iso_date(row.get("value_date") or iso_date)
        if close_value is None or close_value <= 0 or not source_system:
            invalid_row_count += 1
            continue
        if _is_forbidden_local_benchmark_source(source_system):
            forbidden_sources.add(source_system)
            continue
        if coverage not in LOCAL_BENCHMARK_COVERAGE_VALUES:
            warnings.append(f"本地 benchmark 第 {row_number} 行 coverage={coverage} 无效，已跳过。")
            invalid_row_count += 1
            continue
        parsed_rows[(iso_date, field)] = {
            "date": iso_date,
            "ts_code": ts_code,
            "field": field,
            "close": close_value,
            "source_system": source_system,
            "value_date": value_date,
            "coverage": coverage,
        }
        source_systems.add(source_system)

    if forbidden_sources:
        sources = ", ".join(sorted(forbidden_sources))
        return None, [f"本地 benchmark source_system 包含 sample/mock/演示来源：{sources}，不得标记为生产级。"]
    if invalid_row_count:
        warnings.append(f"本地 benchmark 有 {invalid_row_count} 行缺少有效 date/ts_code/close/source_system/coverage，已跳过。")
    if ignored_code_count:
        warnings.append(f"本地 benchmark 有 {ignored_code_count} 行非 Dashboard benchmark 指数代码，已忽略。")

    fields_present = {field for _date, field in parsed_rows}
    missing_fields = sorted(required_fields - fields_present)
    if missing_fields:
        return None, warnings + [f"本地 benchmark 缺少必需指数字段：{', '.join(missing_fields)}"]
    if not parsed_rows:
        return None, warnings + ["本地 benchmark 未提供可用的真实指数 close。"]

    first_closes: dict[str, float] = {}
    for field in required_fields:
        field_dates = sorted(date for date, candidate_field in parsed_rows if candidate_field == field)
        if field_dates:
            first_closes[field] = float(parsed_rows[(field_dates[0], field)]["close"])

    raw_rows: list[dict[str, Any]] = []
    for (iso_date, field), row in sorted(parsed_rows.items()):
        first_close = first_closes.get(field)
        if not first_close:
            continue
        nav_value = row["close"] / first_close
        raw_rows.append(
            {
                "date": iso_date,
                "ts_code": row["ts_code"],
                "field": field,
                "close": f"{row['close']:.6f}",
                "nav": f"{nav_value:.8f}",
                "source_system": row["source_system"],
                "value_date": row["value_date"],
                "coverage": row["coverage"],
            }
        )

    values_by_date: dict[str, dict[str, float]] = {}
    coverage_by_date: dict[str, dict[str, str]] = {}
    value_date_by_date: dict[str, dict[str, str]] = {}
    for run in runs:
        normalized: dict[str, float] = {}
        coverage: dict[str, str] = {}
        value_dates: dict[str, str] = {}
        for field in required_fields:
            row = parsed_rows.get((run.date, field))
            first_close = first_closes.get(field)
            if not row or not first_close:
                continue
            normalized[field] = row["close"] / first_close
            coverage[field] = row["coverage"]
            value_dates[field] = row["value_date"]
        _with_composite_benchmark_fields(normalized, coverage, value_dates, run_date=run.date)
        if normalized:
            values_by_date[run.date] = normalized
            coverage_by_date[run.date] = coverage
            value_date_by_date[run.date] = value_dates

    return BenchmarkExport(
        values_by_date=values_by_date,
        raw_rows=raw_rows,
        source_system="+".join(sorted(source_systems)),
        normalization="local_index_close_divided_by_first_valid_close",
        status_hint="production_source",
        notes=[
            f"benchmark 来自本地真实指数 close 文件：{benchmark_file}",
            "本地 benchmark close 按 close/first_valid_close 归一化；sample/mock/strategy snapshot 来源会被拒绝。",
            "非交易日记录必须在本地文件中显式标记 coverage=previous_trading_day_ffill 和 value_date。",
        ],
        coverage_by_date=coverage_by_date,
        value_date_by_date=value_date_by_date,
    ), warnings


def load_tushare_benchmark_export(
    runs: list[RecordRun],
    *,
    snapshot_gap_fill: bool = False,
) -> tuple[BenchmarkExport | None, list[str]]:
    try:
        import tushare as ts  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local environment.
        return None, [f"未安装或无法导入 tushare，无法拉取生产 benchmark：{exc}"]

    from quant_investor.config import Config
    from quant_investor.credential_utils import create_tushare_pro

    pro = create_tushare_pro(ts, Config.TUSHARE_TOKEN, Config.TUSHARE_URL)
    if pro is None:
        return None, ["TUSHARE_TOKEN 未设置，无法拉取生产 benchmark。"]
    return build_tushare_benchmark_export(runs, pro, snapshot_gap_fill=snapshot_gap_fill)


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


def validate_manual_execution_baseline(run_dir: Path) -> tuple[bool, str]:
    ledger_path = run_dir / "ledger_after_manual_switch.csv"
    if not ledger_path.exists():
        return False, "缺少 ledger_after_manual_switch.csv"
    if not read_csv_rows(ledger_path):
        return False, "ledger_after_manual_switch.csv 无可用行"
    manifest_path = run_dir / "manual_execution_manifest.json"
    if not manifest_path.exists():
        return False, "缺少 manual_execution_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "manual_execution_manifest.json 不是有效 JSON"
    if not isinstance(manifest, dict) or not manifest:
        return False, "manual_execution_manifest.json 为空或格式无效"
    return True, ""


def discover_record_runs(record_root: Path) -> tuple[list[RecordRun], list[str]]:
    warnings: list[str] = []
    runs: list[RecordRun] = []
    for run_dir in sorted(record_root.iterdir() if record_root.exists() else []):
        if not run_dir.is_dir() or not run_dir.name[:8].isdigit():
            continue
        baseline_ok, baseline_reason = validate_manual_execution_baseline(run_dir)
        if not baseline_ok:
            warnings.append(
                f"{run_dir.name}: {baseline_reason}，已跳过；ledger.csv 已停用且不得作为 Dashboard 回退。"
            )
            continue
        pnl_rows = read_csv_rows(run_dir / "pnl_summary.csv")
        if not pnl_rows:
            warnings.append(f"{run_dir.name}: 缺少 pnl_summary.csv，已跳过。")
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


def _load_industry_member_symbols(
    stock_basic_root: Path,
    industries: tuple[str, ...],
) -> tuple[set[str], list[str]]:
    warnings: list[str] = []
    if not stock_basic_root.exists():
        return set(), [f"未找到 stock_basic 行业映射目录：{stock_basic_root}，无法构建 industry_ew_nav。"]
    try:
        with suppress_native_stderr():
            import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return set(), ["当前 Python 环境未安装 pandas/pyarrow，无法构建 industry_ew_nav；可用 ./.venv/bin/python 运行。"]
    try:
        with suppress_native_stderr():
            frame = pd.read_parquet(stock_basic_root, columns=["ts_code", "industry"])
    except Exception as exc:
        return set(), [f"读取 stock_basic 行业映射失败：{exc}，无法构建 industry_ew_nav。"]
    if frame.empty:
        return set(), [f"stock_basic 行业映射为空：{stock_basic_root}，无法构建 industry_ew_nav。"]
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip()
    frame["industry"] = frame["industry"].astype(str).str.strip()
    industry_set = set(industries)
    selected = frame[frame["industry"].isin(industry_set)]
    symbols = {symbol for symbol in selected["ts_code"].tolist() if symbol}
    missing_industries = sorted(industry_set - set(selected["industry"].dropna().unique().tolist()))
    if missing_industries:
        warnings.append(f"industry_ew_nav 行业映射中未找到行业：{', '.join(missing_industries)}。")
    return symbols, warnings


def _read_industry_bar_frame(
    bars_root: Path,
    symbols: set[str],
    start_date: str,
    end_date: str,
) -> tuple[Any | None, list[str]]:
    warnings: list[str] = []
    if not bars_root.exists():
        return None, [f"未找到本地 bars Parquet 目录：{bars_root}，无法构建 industry_ew_nav。"]
    try:
        with suppress_native_stderr():
            import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return None, ["当前 Python 环境未安装 pandas/pyarrow，无法构建 industry_ew_nav；可用 ./.venv/bin/python 运行。"]
    columns = ["ts_code", "trade_date", "close", "adj_close"]
    start_key = iso_to_tushare_date(start_date)
    end_key = iso_to_tushare_date(end_date)
    filters = [
        ("trade_date", ">=", start_key),
        ("trade_date", "<=", end_key),
        ("ts_code", "in", sorted(symbols)),
    ]
    try:
        with suppress_native_stderr():
            frame = pd.read_parquet(bars_root, columns=columns, filters=filters)
    except Exception:
        try:
            with suppress_native_stderr():
                frame = pd.read_parquet(
                    bars_root,
                    columns=columns,
                    filters=[("trade_date", ">=", start_key), ("trade_date", "<=", end_key)],
                )
        except Exception as exc:
            return None, [f"读取本地 bars Parquet 失败：{exc}，无法构建 industry_ew_nav。"]
        warnings.append("bars Parquet 不支持 symbol 下推过滤，已在内存中过滤 industry_ew_nav 成分。")
    if frame.empty:
        return frame, warnings
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip()
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    frame = frame[
        frame["ts_code"].isin(symbols)
        & (frame["trade_date"] >= start_key)
        & (frame["trade_date"] <= end_key)
    ].copy()
    return frame, warnings


def _industry_equal_weight_nav_from_bars(
    bars_frame: Any,
) -> tuple[dict[str, float], dict[str, int]]:
    if bars_frame is None or bars_frame.empty:
        return {}, {}
    import pandas as pd  # type: ignore[import-not-found]

    frame = bars_frame.copy()
    frame["date"] = frame["trade_date"].map(tushare_to_iso_date)
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    if "adj_close" in frame:
        adj_close = pd.to_numeric(frame.get("adj_close"), errors="coerce")
        frame["price"] = adj_close.where(adj_close.notna() & (adj_close > 0), close)
    else:
        frame["price"] = close
    frame = frame.dropna(subset=["date", "ts_code", "price"])
    frame = frame[frame["price"] > 0]
    if frame.empty:
        return {}, {}
    frame = frame.drop_duplicates(["ts_code", "date"], keep="last")
    frame = frame.sort_values(["ts_code", "date"]).reset_index(drop=True)
    frame["daily_return"] = frame.groupby("ts_code", sort=False)["price"].pct_change()
    daily_return = frame.dropna(subset=["daily_return"]).groupby("date")["daily_return"].mean().sort_index()
    daily_member_count = (
        frame.dropna(subset=["daily_return"]).groupby("date")["ts_code"].nunique().sort_index().astype(int).to_dict()
    )
    nav_by_date: dict[str, float] = {}
    current_nav = DEFAULT_INITIAL_BENCHMARK
    for date in sorted(frame["date"].dropna().unique().tolist()):
        if date in daily_return:
            current_nav *= 1.0 + float(daily_return.loc[date])
        nav_by_date[date] = current_nav
    return nav_by_date, {str(date): int(count) for date, count in daily_member_count.items()}


def attach_industry_equal_weight_nav(
    runs: list[RecordRun],
    benchmark_export: BenchmarkExport | None,
    *,
    bars_root: Path = DEFAULT_CN_BARS_ROOT,
    stock_basic_root: Path = DEFAULT_STOCK_BASIC_ROOT,
    industries: tuple[str, ...] = DEFAULT_TECH_MANUFACTURING_INDUSTRIES,
) -> tuple[BenchmarkExport | None, list[str]]:
    if benchmark_export is None:
        return None, []
    if not runs:
        return benchmark_export, ["没有可用策略记录，无法构建 industry_ew_nav。"]
    warnings: list[str] = []
    symbols, symbol_warnings = _load_industry_member_symbols(stock_basic_root, industries)
    warnings.extend(symbol_warnings)
    if not symbols:
        return benchmark_export, warnings + ["industry_ew_nav 成分为空，已跳过。"]
    bars_frame, bar_warnings = _read_industry_bar_frame(bars_root, symbols, runs[0].date, runs[-1].date)
    warnings.extend(bar_warnings)
    nav_by_date, daily_member_count = _industry_equal_weight_nav_from_bars(bars_frame)
    if not nav_by_date:
        return benchmark_export, warnings + ["industry_ew_nav 未从本地 bars 生成任何有效 NAV，已跳过。"]

    values_by_date = {date: dict(values) for date, values in benchmark_export.values_by_date.items()}
    coverage_by_date = {date: dict(values) for date, values in benchmark_export.coverage_by_date.items()}
    value_date_by_date = {date: dict(values) for date, values in benchmark_export.value_date_by_date.items()}
    raw_rows = list(benchmark_export.raw_rows)
    nav_dates = sorted(nav_by_date)
    ffill_rows: list[dict[str, Any]] = []
    for run in runs:
        run_values = values_by_date.setdefault(run.date, {})
        run_coverage = coverage_by_date.setdefault(run.date, {})
        run_value_dates = value_date_by_date.setdefault(run.date, {})
        value_date = run.date if run.date in nav_by_date else ""
        if not value_date:
            earlier_dates = [date for date in nav_dates if date <= run.date]
            if not earlier_dates:
                continue
            value_date = earlier_dates[-1]
        coverage = "exact_close" if value_date == run.date else "previous_trading_day_ffill"
        nav_value = nav_by_date[value_date]
        run_values[INDUSTRY_EW_NAV_FIELD] = nav_value
        run_coverage[INDUSTRY_EW_NAV_FIELD] = coverage
        run_value_dates[INDUSTRY_EW_NAV_FIELD] = value_date
        if coverage == "previous_trading_day_ffill":
            ffill_rows.append(
                {
                    "date": run.date,
                    "ts_code": INDUSTRY_EW_TS_CODE,
                    "field": INDUSTRY_EW_NAV_FIELD,
                    "close": f"{nav_value:.8f}",
                    "nav": f"{nav_value:.8f}",
                    "source_system": INDUSTRY_EW_SOURCE_SYSTEM,
                    "value_date": value_date,
                    "coverage": coverage,
                }
            )
    for date in nav_dates:
        nav_value = nav_by_date[date]
        raw_rows.append(
            {
                "date": date,
                "ts_code": INDUSTRY_EW_TS_CODE,
                "field": INDUSTRY_EW_NAV_FIELD,
                "close": f"{nav_value:.8f}",
                "nav": f"{nav_value:.8f}",
                "source_system": INDUSTRY_EW_SOURCE_SYSTEM,
                "value_date": date,
                "coverage": "exact_close",
            }
        )
    raw_rows.extend(ffill_rows)
    metadata = dict(benchmark_export.metadata or {})
    metadata["industry_equal_weight"] = {
        "field": INDUSTRY_EW_NAV_FIELD,
        "source_system": INDUSTRY_EW_SOURCE_SYSTEM,
        "member_count": len(symbols),
        "industry_list": list(industries),
        "valid_date_count": len(nav_by_date),
        "start_date": min(nav_dates),
        "end_date": max(nav_dates),
        "min_daily_member_count": min(daily_member_count.values()) if daily_member_count else 0,
        "max_daily_member_count": max(daily_member_count.values()) if daily_member_count else 0,
    }
    source_systems = [benchmark_export.source_system]
    if INDUSTRY_EW_SOURCE_SYSTEM not in benchmark_export.source_system.split("+"):
        source_systems.append(INDUSTRY_EW_SOURCE_SYSTEM)
    return BenchmarkExport(
        values_by_date=values_by_date,
        raw_rows=raw_rows,
        source_system="+".join(source_systems),
        normalization=benchmark_export.normalization,
        status_hint=benchmark_export.status_hint,
        notes=[
            *benchmark_export.notes,
            (
                "industry_ew_nav 来自本地 bars Parquet 与 stock_basic.industry，"
                f"对 {len(symbols)} 个策略行业域成员按日收益等权复合。"
            ),
        ],
        coverage_by_date=coverage_by_date,
        value_date_by_date=value_date_by_date,
        snapshot_gap_fill_by_date=benchmark_export.snapshot_gap_fill_by_date,
        calendar_source_system=benchmark_export.calendar_source_system,
        metadata=metadata,
    ), warnings


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
        and field not in SUPPLEMENTAL_BENCHMARK_FIELDS
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
        snapshot_gap_fill_by_date = benchmark_export.snapshot_gap_fill_by_date or {}
        snapshot_gap_fill_dates = sorted(snapshot_gap_fill_by_date)
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
        ffill_dates = [
            str(row.get("date") or "")
            for row in nav_rows
            if any(
                benchmark_export.coverage_by_date.get(str(row.get("date") or ""), {}).get(field)
                == "previous_trading_day_ffill"
                for field in actual_snapshot_fields
            )
        ]
        exact_dates = [
            str(row.get("date") or "")
            for row in nav_rows
            if str(row.get("date") or "") not in missing_dates
            and str(row.get("date") or "") not in ffill_dates
            and str(row.get("date") or "") not in snapshot_gap_fill_dates
        ]
        if missing_date_count == 0 and not ffill_dates:
            status = "production_grade"
        elif missing_date_count == 0 and ffill_dates:
            status = "production_source_with_previous_trading_day_ffill"
        elif latest_only_missing:
            status = "production_source_partial_latest_unavailable"
        else:
            status = "production_source_partial_missing_dates"
        if snapshot_gap_fill_dates:
            status = SNAPSHOT_BENCHMARK_STATUS
        notes = list(benchmark_export.notes)
        if ffill_dates:
            notes.append("部分非交易日 benchmark 使用上一交易日真实 close 前向填充；Dashboard 展示连续，但投委会口径需区分 exact 与 ffill。")
        if missing_date_count:
            source_label = (
                "Tushare benchmark"
                if benchmark_export.source_system == TUSHARE_BENCHMARK_SOURCE_SYSTEM
                else "benchmark source"
            )
            notes.append(f"{source_label} 未覆盖所有策略记录日期；缺失日期不按生产级 benchmark 标记。")
        if snapshot_gap_fill_dates:
            notes.append(
                "部分 Tushare 缺口使用 strategy_record.market_snapshot.indices 实时快照补齐，仅用于 Dashboard 连续展示；不是连续真实指数 close。"
            )
        source_system = benchmark_export.source_system
        if snapshot_gap_fill_dates:
            source_system = f"{benchmark_export.source_system}+{SNAPSHOT_BENCHMARK_SOURCE_SYSTEM}"
        summary = {
            "benchmark_fields": benchmark_fields,
            "benchmark_source_status": status,
            "source_system": source_system,
            "calendar_source_system": benchmark_export.calendar_source_system,
            "production_grade": missing_date_count == 0 and not snapshot_gap_fill_dates,
            "display_continuity_grade": missing_date_count == 0,
            "first_valid_date": min(valid_dates) if valid_dates else "",
            "last_valid_date": max(valid_dates) if valid_dates else "",
            "missing_date_count": missing_date_count,
            "missing_dates": missing_dates,
            "snapshot_gap_fill_count": len(snapshot_gap_fill_dates),
            "snapshot_gap_fill_dates": snapshot_gap_fill_dates,
            "snapshot_gap_fill_by_date": snapshot_gap_fill_by_date,
            "exact_date_count": len(exact_dates),
            "exact_dates": exact_dates,
            "previous_trading_day_ffill_count": len(ffill_dates),
            "previous_trading_day_ffill_dates": ffill_dates,
            "coverage_by_date": benchmark_export.coverage_by_date,
            "value_date_by_date": benchmark_export.value_date_by_date,
            "field_missing_counts": field_missing_counts,
            "normalization": benchmark_export.normalization,
            "notes": notes,
            "raw_row_count": len(benchmark_export.raw_rows),
        }
        for key, value in (benchmark_export.metadata or {}).items():
            summary[key] = value
        return summary
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
            warnings.append(
                f"{run.run_id}: 缺少 ledger_after_manual_switch.csv，positions 已跳过；"
                "ledger.csv 已停用且不得作为 Dashboard 持仓回退。"
            )
            continue
        ledger_rows = read_csv_rows(ledger_path)
        if not ledger_rows:
            warnings.append(f"{run.run_id}: ledger_after_manual_switch.csv 无可用行，positions 已跳过。")
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


def resolve_local_benchmark_file(dashboard_root: Path, benchmark_file: Path | None) -> Path:
    if benchmark_file is not None:
        return benchmark_file
    dashboard_candidate = dashboard_root / "inputs" / "cn_index_benchmark.csv"
    if dashboard_candidate.exists():
        return dashboard_candidate
    default_candidate = DEFAULT_DASHBOARD_ROOT / "inputs" / "cn_index_benchmark.csv"
    if default_candidate.exists():
        return default_candidate
    return dashboard_candidate


def export(
    record_root: Path,
    dashboard_root: Path,
    benchmark_source: str = DEFAULT_BENCHMARK_SOURCE,
    benchmark_gap_fill: str = "snapshot",
    benchmark_file: Path | None = None,
) -> dict[str, Any]:
    runs, warnings = discover_record_runs(record_root)
    if not runs:
        raise SystemExit(f"No usable records found under {record_root}")
    benchmark_export: BenchmarkExport | None = None
    local_benchmark_file = resolve_local_benchmark_file(dashboard_root, benchmark_file)
    if benchmark_source in {"auto", "local"} and (benchmark_source == "local" or local_benchmark_file.exists()):
        benchmark_export, local_warnings = load_local_benchmark_export(runs, local_benchmark_file)
        warnings.extend(local_warnings)
        if benchmark_export is None and benchmark_source == "local":
            warnings.append("已按 fail-closed 降级为 strategy_record.market_snapshot.indices benchmark；不得标记为生产级。")
    if benchmark_source in {"auto", "tushare"}:
        if benchmark_export is None:
            benchmark_export, tushare_warnings = load_tushare_benchmark_export(
                runs,
                snapshot_gap_fill=benchmark_gap_fill == "snapshot",
            )
            warnings.extend(tushare_warnings)
            if benchmark_export is None and benchmark_source == "tushare":
                warnings.append("已按 fail-closed 降级为 strategy_record.market_snapshot.indices benchmark；不得标记为生产级。")
    if benchmark_source == "snapshot":
        warnings.append("benchmark_source=snapshot：按显式参数使用 strategy_record.market_snapshot.indices，非生产级。")
    if benchmark_export is not None:
        benchmark_export, industry_warnings = attach_industry_equal_weight_nav(runs, benchmark_export)
        warnings.extend(industry_warnings)
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
        ["date", "ts_code", "field", "close", "nav", "source_system", "value_date", "coverage"],
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
        choices=["tushare", "auto", "snapshot", "local"],
        default=os.environ.get("CN_DASHBOARD_BENCHMARK_SOURCE", DEFAULT_BENCHMARK_SOURCE),
        help="Benchmark source for dashboard NAV fields. Defaults to local-first auto mode.",
    )
    benchmark_file_env = os.environ.get("CN_DASHBOARD_BENCHMARK_FILE")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path(benchmark_file_env) if benchmark_file_env else None,
        help=(
            "Local real index close CSV for --benchmark-source local/auto. "
            "Required columns: date, ts_code, close, source_system; optional: coverage, value_date."
        ),
    )
    parser.add_argument(
        "--benchmark-gap-fill",
        choices=["snapshot", "none"],
        default=os.environ.get("CN_DASHBOARD_BENCHMARK_GAP_FILL", "snapshot"),
        help=(
            "Fill Tushare benchmark gaps from strategy_record.market_snapshot.indices for display continuity. "
            "Any snapshot fill forces non-production benchmark provenance."
        ),
    )
    args = parser.parse_args()
    summary = export(
        args.record_root,
        args.dashboard_root,
        args.benchmark_source,
        args.benchmark_gap_fill,
        args.benchmark_file,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
