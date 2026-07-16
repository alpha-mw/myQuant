#!/usr/bin/env python3
"""Export CN aggressive strategy records into the static dashboard data bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_DASHBOARD_ROOT = PROJECT_ROOT / "portfolio_dashboard"
DEFAULT_PRIVATE_DASHBOARD_DIRNAME = "private"
DASHBOARD_SCHEMA_VERSION = "dashboard_contract.v3"
DEFAULT_BENCHMARK_SOURCE = "local"
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
EXECUTED_TRADE_STATUSES = {
    "filled",
    "executed",
    "filled_local_manual",
    "filled_local_manual_human_override",
    "filled_local_manual_paper_rebalance",
    "成交",
    "已成交",
}
PARTIAL_EXECUTED_TRADE_STATUSES = {"partial_fill", "partially_filled"}
EXECUTED_TRADE_STATUSES.update(PARTIAL_EXECUTED_TRADE_STATUSES)
TRADE_RECORD_REQUIRED_FIELDS = ["timestamp", "symbol", "side/action", "shares", "price", "trade_value"]
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
    funding_cash_flows: tuple[dict[str, Any], ...] = ()
    manual_manifest_path: Path | None = None
    manual_manifest_sha256: str | None = None
    manual_ledger_path: Path | None = None
    manual_ledger_sha256: str | None = None
    manual_ledger_sha_declared: bool = False
    manual_manifest: dict[str, Any] | None = None
    manual_order_key: tuple[str, str] | None = None


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


def read_ledger_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv_rows(path)
    if path.suffix.lower() != ".parquet":
        return []
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except (ImportError, OSError, ValueError):
        return []
    return [dict(row) for row in frame.to_dict(orient="records")]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    _write_private_text_atomic(path, buffer.getvalue())


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


def parse_percentage_points(value: Any) -> float | None:
    """Convert a field explicitly named ``*_pct`` to a decimal return once.

    Strategy review artifacts store ``today_change_pct=-0.46`` for -0.46%.
    A literal percent sign is already handled by :func:`parse_float`; bare
    values in this field are percentage points and therefore need /100.
    """

    if value is None:
        return None
    text = str(value).strip()
    parsed = parse_float(text)
    if parsed is None:
        return None
    return parsed if text.endswith("%") else parsed / 100.0


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_summary(path: Path | None) -> str | None:
    """Return an audit-useful path without serializing a private absolute root."""

    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        parent = path.parent.name
        return f"<external>/{parent + '/' if parent else ''}{path.name}"


def nullable_float(value: Any) -> float | None:
    parsed = parse_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def explicit_iso_date(value: Any) -> str | None:
    """Normalize only documented date encodings; never search arbitrary text."""

    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        normalized = f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    else:
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
            r"(?:\s*CST|[+-]\d{2}:?\d{2}|Z)?)?",
            text,
        )
        if not match:
            return None
        normalized = match.group(1)
    try:
        datetime.strptime(normalized, "%Y-%m-%d")
    except ValueError:
        return None
    return normalized


def explicit_iso_timestamp(value: Any) -> str | None:
    """Normalize explicit artifact timestamp fields without run-id inference."""

    text = str(value or "").strip()
    if re.fullmatch(r"\d{14}", text):
        normalized = (
            f"{text[:4]}-{text[4:6]}-{text[6:8]}T"
            f"{text[8:10]}:{text[10:12]}:{text[12:14]}+08:00"
        )
    elif re.fullmatch(r"\d{8}_\d{4}", text):
        normalized = (
            f"{text[:4]}-{text[4:6]}-{text[6:8]}T"
            f"{text[9:11]}:{text[11:13]}:00+08:00"
        )
    else:
        match = re.fullmatch(
            r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(?:\s*CST|([+-]\d{2}:?\d{2}|Z))?",
            text,
        )
        if not match:
            return None
        zone = match.group(3) or "+08:00"
        if re.fullmatch(r"[+-]\d{4}", zone):
            zone = f"{zone[:3]}:{zone[3:]}"
        normalized = f"{match.group(1)}T{match.group(2)}{zone}"
    try:
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    return normalized


def load_strict_parquet_trading_calendar(
    bars_root: Path,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], list[str]]:
    """Build the expected open-day mask only from canonical strict Parquet bars."""

    missing = {
        "status": "missing",
        "source_system": "strict_parquet.cn_bars.trade_date",
        "path_summary": path_summary(bars_root),
        "start_date": start_date,
        "end_date": end_date,
        "expected_open_dates": [],
        "expected_open_date_count": 0,
        "first_open_date": None,
        "last_open_date": None,
        "mask_sha256": None,
    }
    if not bars_root.exists():
        return missing, [f"trading_calendar_missing: strict Parquet bars not found: {bars_root}"]
    try:
        with suppress_native_stderr():
            import pyarrow.dataset as ds  # type: ignore[import-not-found]

            dataset = ds.dataset(str(bars_root), format="parquet", partitioning="hive")
            table = dataset.to_table(columns=["trade_date"])
    except Exception as exc:
        return missing, [f"trading_calendar_missing: cannot read strict Parquet trade_date: {exc}"]
    expected = sorted(
        {
            normalized
            for value in table.column("trade_date").to_pylist()
            if (normalized := explicit_iso_date(value)) is not None
            and start_date <= normalized <= end_date
        }
    )
    if not expected:
        return missing, ["trading_calendar_missing: no canonical trade_date in dashboard range"]
    mask_payload = {
        "source_system": "strict_parquet.cn_bars.trade_date",
        "start_date": start_date,
        "end_date": end_date,
        "expected_open_dates": expected,
    }
    return {
        "status": "available",
        "source_system": "strict_parquet.cn_bars.trade_date",
        "path_summary": path_summary(bars_root),
        "start_date": start_date,
        "end_date": end_date,
        "expected_open_dates": expected,
        "expected_open_date_count": len(expected),
        "first_open_date": expected[0],
        "last_open_date": expected[-1],
        "mask_sha256": canonical_json_sha256(mask_payload),
    }, []


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


def load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
    main_field = next(
        (field for field in ("star50_nav", "csi300_nav", "chinext_nav") if field in normalized),
        "",
    )
    if main_field:
        normalized["benchmark_main_nav"] = normalized[main_field]
        coverage["benchmark_main_nav"] = coverage.get(main_field, "exact_close")
        value_dates["benchmark_main_nav"] = value_dates.get(main_field, run_date)
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
        if (
            not value_date
            or (coverage == "exact_close" and value_date != iso_date)
            or (
                coverage == "previous_trading_day_ffill"
                and value_date >= iso_date
            )
        ):
            warnings.append(
                f"本地 benchmark 第 {row_number} 行 value_date/coverage PIT 语义无效，已跳过。"
            )
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

    exact_keys = {
        (row["date"], row["field"])
        for row in parsed_rows.values()
        if row["coverage"] == "exact_close"
    }
    invalid_ffill_keys = [
        key
        for key, row in parsed_rows.items()
        if row["coverage"] == "previous_trading_day_ffill"
        and (row["value_date"], row["field"]) not in exact_keys
    ]
    for key in invalid_ffill_keys:
        parsed_rows.pop(key, None)
        invalid_row_count += 1
        warnings.append(
            "本地 benchmark previous_trading_day_ffill 缺少同指数 exact_close "
            f"value_date 证据：{key[0]} {key[1]}，已跳过。"
        )

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


def iso_date_from_any(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    digits = "".join(char for char in text if char.isdigit())
    return tushare_to_iso_date(digits[:8]) if len(digits) >= 8 else ""


def load_funding_cash_flows(run_dir: Path, record_date: str) -> tuple[dict[str, Any], ...]:
    manifest_path = run_dir / "manual_execution_manifest.json"
    manual_manifest = load_json_object(manifest_path)
    if manual_manifest is None:
        return ()

    embedded = manual_manifest.get("manual_funding_supplement")
    declared_path = str(manual_manifest.get("manual_funding_supplement_path") or "").strip()
    external: dict[str, Any] = {}
    if declared_path:
        evidence_path = Path(declared_path)
        if not evidence_path.is_absolute():
            evidence_path = run_dir / evidence_path
        external = load_json_object(evidence_path) or {}
        if not evidence_path.exists() or not external:
            raise ValueError(
                f"Invalid funding lineage for {run_dir.name}: unreadable funding evidence: {evidence_path}"
            )
    if embedded is not None and not isinstance(embedded, dict):
        raise ValueError(
            f"Invalid funding lineage for {run_dir.name}: manual_funding_supplement must be an object"
        )
    if embedded and external and embedded != external:
        raise ValueError(
            f"Invalid funding lineage for {run_dir.name}: embedded and external funding evidence differ"
        )

    payloads: list[dict[str, Any]] = []
    supplement = external or (embedded if isinstance(embedded, dict) else {})
    if supplement:
        payloads.append(supplement)
    for key in ("external_funding_cash_flows", "funding_cash_flows"):
        rows = manual_manifest.get(key) or []
        if not isinstance(rows, list):
            raise ValueError(f"Invalid funding lineage for {run_dir.name}: {key} must be a list")
        payloads.extend(row for row in rows if isinstance(row, dict))

    flows: list[dict[str, Any]] = []
    for payload in payloads:
        amount = parse_float(payload.get("amount"))
        flow_date = iso_date_from_any(
            payload.get("effective_date")
            or payload.get("capital_base_effective_from")
            or payload.get("date")
            or record_date
        )
        if amount is None or not math.isfinite(amount) or amount == 0:
            raise ValueError(
                f"Invalid funding lineage for {run_dir.name}: funding amount must be finite and non-zero"
            )
        if not flow_date or flow_date != record_date:
            raise ValueError(
                f"Invalid funding lineage for {run_dir.name}: funding date {flow_date or 'missing'} "
                f"does not match record date {record_date}"
            )
        total_before = parse_float(payload.get("total_value_before"))
        total_after = parse_float(payload.get("total_value_after"))
        if (
            total_before is not None
            and total_after is not None
            and not math.isclose(total_before + amount, total_after, abs_tol=0.01)
        ):
            raise ValueError(
                f"Invalid funding lineage for {run_dir.name}: total_value_before + amount != total_value_after"
            )
        flows.append(
            {
                "date": flow_date,
                "amount": amount,
                "source": str(payload.get("source") or "manual_execution_manifest"),
                "schema_version": str(payload.get("schema_version") or ""),
                "evidence_path": declared_path,
                "capital_base_after": parse_float(payload.get("capital_base_after")),
                "total_value_before": total_before,
                "total_value_after": total_after,
            }
        )
    return tuple(flows)


def deduplicate_funding_cash_flows(flows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str, str]] = set()
    for flow in flows:
        identity = (
            str(flow.get("date") or ""),
            float(flow.get("amount") or 0),
            str(flow.get("source") or ""),
            str(flow.get("evidence_path") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(flow)
    return tuple(unique)


_MANUAL_LEDGER_SHA_FIELDS = (
    "next_ledger_sha256",
    "ledger_after_manual_switch_sha256",
    "ledger_sha256",
)


def _manual_manifest_timestamp(manifest: dict[str, Any]) -> str | None:
    for key in (
        "recorded_at",
        "executed_at",
        "updated_at",
        "quote_snapshot",
        "timestamp_long",
        "record_timestamp",
    ):
        timestamp = explicit_iso_timestamp(manifest.get(key))
        if timestamp is not None:
            return timestamp
    return None


def _declared_manual_ledger_sha256(
    manifest: dict[str, Any],
    ledger_path: Path,
) -> str | None:
    suffix_key = (
        "ledger_after_manual_switch_parquet_sha256"
        if ledger_path.suffix.lower() == ".parquet"
        else "ledger_after_manual_switch_csv_sha256"
    )
    for key in (suffix_key, *_MANUAL_LEDGER_SHA_FIELDS):
        value = str(manifest.get(key) or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value
    return None


def _path_within_root(path: Path, record_root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(record_root.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def _validate_manual_baseline(
    *,
    record_root: Path,
    run_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[Path | None, str | None, tuple[str, str] | None, str | None]:
    try:
        from quant_investor.monitoring.cn_aggressive_portfolio_tracker import (
            _manual_manifest_is_valid_baseline,
            _manual_manifest_order_key,
            _resolve_manual_ledger_path,
        )
    except ImportError:
        return None, None, None, "canonical_manual_baseline_helpers_unavailable"
    status = str(manifest.get("status") or "").strip()
    execution_status = str(manifest.get("execution_status") or "").strip()
    if (
        not status
        or status.lower() == "ok"
        or not _manual_manifest_is_valid_baseline(manifest)
        or (execution_status and execution_status != status)
    ):
        return None, None, None, "manual_manifest_status_invalid"
    if _manual_manifest_timestamp(manifest) is None:
        return None, None, None, "manual_manifest_time_missing_or_invalid"
    next_ledger = str(manifest.get("next_ledger_path") or "").strip()
    if not next_ledger:
        return None, None, None, "manual_manifest_next_ledger_path_missing"
    declared_path = Path(next_ledger)
    if not declared_path.is_absolute():
        declared_path = manifest_path.parent / declared_path
    if (
        declared_path.is_symlink()
        or declared_path.stem != "ledger_after_manual_switch"
        or declared_path.suffix.lower() not in {".csv", ".parquet"}
        or not _path_within_root(declared_path, record_root)
    ):
        return None, None, None, "manual_manifest_next_ledger_path_unsafe"
    ledger_path = _resolve_manual_ledger_path(manifest_path, manifest)
    if (
        ledger_path is None
        or ledger_path.is_symlink()
        or ledger_path.stem != "ledger_after_manual_switch"
        or ledger_path.suffix.lower() not in {".csv", ".parquet"}
        or not _path_within_root(ledger_path, record_root)
    ):
        return None, None, None, "manual_ledger_resolution_invalid"
    ledger_path = ledger_path.resolve(strict=True)
    expected_sha = _declared_manual_ledger_sha256(manifest, ledger_path)
    before_sha = sha256_file(ledger_path)
    if expected_sha is not None and before_sha != expected_sha:
        return None, None, None, "manual_ledger_sha256_mismatch"
    ledger_rows = read_ledger_rows(ledger_path)
    after_sha = sha256_file(ledger_path)
    if before_sha is None or after_sha != before_sha:
        return None, None, None, "manual_ledger_readback_changed"
    if not ledger_rows:
        return None, None, None, "manual_ledger_empty_or_unreadable"
    required_fields = {"symbol", "shares", "avg_cost"}
    missing_fields = required_fields - set(ledger_rows[0])
    if missing_fields:
        return None, None, None, "manual_ledger_schema_invalid"
    declared_count = nullable_float(manifest.get("effective_manual_holding_count"))
    if declared_count is not None and int(declared_count) != len(ledger_rows):
        return None, None, None, "manual_ledger_holding_count_mismatch"
    order_key = _manual_manifest_order_key(run_dir, manifest)
    return ledger_path, before_sha, order_key, None


def discover_record_runs(
    record_root: Path,
    *,
    require_effective_manual: bool = True,
) -> tuple[list[RecordRun], list[str]]:
    warnings: list[str] = []
    runs: list[RecordRun] = []
    funding_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        manual_manifest_path: Path | None = None
        manual_manifest_sha256: str | None = None
        manual_ledger_path: Path | None = None
        manual_ledger_sha256: str | None = None
        manual_ledger_sha_declared = False
        manual_manifest: dict[str, Any] | None = None
        manual_order_key: tuple[str, str] | None = None
        manual_record_time: str | None = None
        if require_effective_manual:
            manual_manifest_path = run_dir / "manual_execution_manifest.json"
            if not manual_manifest_path.is_file():
                warnings.append(f"{run_dir.name}: 缺少可读 manual_execution_manifest.json，已跳过。")
                continue
            if (
                manual_manifest_path.is_symlink()
                or not _path_within_root(manual_manifest_path, record_root)
            ):
                warnings.append(
                    f"{run_dir.name}: manual baseline 无效 (manual_manifest_path_unsafe)，已跳过。"
                )
                continue
            manifest_sha_before = sha256_file(manual_manifest_path)
            manual_manifest = load_json_object(manual_manifest_path)
            if manual_manifest is None:
                warnings.append(f"{run_dir.name}: 缺少可读 manual_execution_manifest.json，已跳过。")
                continue
            manifest_sha_after = sha256_file(manual_manifest_path)
            manifest_readback = load_json_object(manual_manifest_path)
            if (
                manifest_sha_before is None
                or manifest_sha_after != manifest_sha_before
                or manifest_readback != manual_manifest
            ):
                warnings.append(
                    f"{run_dir.name}: manual baseline 无效 (manual_manifest_readback_changed)，已跳过。"
                )
                continue
            manual_manifest_sha256 = manifest_sha_before
            (
                manual_ledger_path,
                manual_ledger_sha256,
                manual_order_key,
                baseline_error,
            ) = _validate_manual_baseline(
                record_root=record_root,
                run_dir=run_dir,
                manifest_path=manual_manifest_path,
                manifest=manual_manifest,
            )
            if baseline_error:
                warnings.append(
                    f"{run_dir.name}: manual baseline 无效 ({baseline_error})，已跳过。"
                )
                continue
            manual_ledger_sha_declared = bool(
                manual_ledger_path
                and _declared_manual_ledger_sha256(
                    manual_manifest,
                    manual_ledger_path,
                )
            )
            if not manual_ledger_sha_declared:
                warnings.append(
                    f"{run_dir.name}: manual_ledger_sha_not_declared；"
                    "已使用 contained next_ledger_path、manifest SHA 与 ledger 双读稳定 SHA 建立 provenance。"
                )
            manual_record_time = _manual_manifest_timestamp(manual_manifest)
            manifest_total = parse_float(manual_manifest.get("total_value_after"))
            if manifest_total is not None and manifest_total > 0:
                total_value_after = manifest_total
        record_date = record_date_from_run(run_dir.name)
        funding_cash_flows = load_funding_cash_flows(run_dir, record_date)
        funding_by_date[record_date].extend(funding_cash_flows)
        effective_record_date = (
            explicit_iso_date(manual_record_time)
            if require_effective_manual
            else record_date
        ) or record_date
        runs.append(
            RecordRun(
                run_id=run_dir.name,
                date=effective_record_date,
                path=run_dir,
                record_time=manual_record_time or row.get("record_time", ""),
                initial_capital=initial_capital,
                total_value_after=total_value_after,
                benchmark_values=benchmarks_from_snapshot(run_dir),
                funding_cash_flows=funding_cash_flows,
                manual_manifest_path=manual_manifest_path,
                manual_manifest_sha256=manual_manifest_sha256,
                manual_ledger_path=manual_ledger_path,
                manual_ledger_sha256=manual_ledger_sha256,
                manual_ledger_sha_declared=manual_ledger_sha_declared,
                manual_manifest=manual_manifest,
                manual_order_key=manual_order_key,
            )
        )
    if require_effective_manual:
        runs.sort(key=lambda run: run.manual_order_key or ("", run.run_id))
    latest_by_date: dict[str, RecordRun] = {}
    for run in runs:
        latest_by_date[run.date] = run
    daily_runs = [
        replace(
            latest_by_date[date],
            funding_cash_flows=deduplicate_funding_cash_flows(funding_by_date.get(date, [])),
        )
        for date in sorted(latest_by_date)
    ]
    if require_effective_manual:
        daily_runs.sort(key=lambda run: run.manual_order_key or ("", run.run_id))
    return daily_runs, warnings


def load_sector_map(stock_basic_root: Path) -> tuple[dict[str, dict[str, str | None]], list[str]]:
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

    parquet_files = sorted(stock_basic_root.rglob("*.parquet"))
    generation_sha256 = (
        canonical_json_sha256(
            {str(path.relative_to(stock_basic_root)): sha256_file(path) for path in parquet_files}
        )
        if parquet_files
        else None
    )
    mapping: dict[str, dict[str, str | None]] = {}
    for row in table.to_pylist():
        symbol = str(row.get("ts_code") or "").strip()
        industry = str(row.get("industry") or "").strip()
        if symbol and industry:
            mapping[symbol] = {
                "industry": industry,
                "industry_source": "strict_parquet.stock_basic.industry",
                "industry_as_of": None,
                "industry_generation_sha256": generation_sha256,
            }
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


def build_cash_flow_adjusted_nav_points(runs: list[RecordRun]) -> list[dict[str, float | None]]:
    points: list[dict[str, float | None]] = []
    unit_count: float | None = None
    unit_nav: float | None = None
    first_unit_nav: float | None = None
    previous_unit_nav: float | None = None
    previous_capital: float | None = None
    for run in runs:
        external_flow = sum(float(flow.get("amount") or 0) for flow in run.funding_cash_flows)
        raw_capital_ratio = run.total_value_after / run.initial_capital
        daily_return: float | None = None
        if unit_count is None:
            if run.funding_cash_flows:
                raise ValueError(
                    f"Invalid NAV lineage for {run.run_id}: first NAV record contains funding evidence "
                    "without a prior unit base"
                )
            unit_count = run.initial_capital
            if unit_count <= 0 or run.total_value_after <= 0:
                raise ValueError(f"Invalid NAV lineage for {run.run_id}: non-positive unit base")
            unit_nav = run.total_value_after / unit_count
            first_unit_nav = unit_nav
        else:
            capital_changed = previous_capital is not None and not math.isclose(
                run.initial_capital,
                previous_capital,
                rel_tol=0,
                abs_tol=0.01,
            )
            if capital_changed and math.isclose(external_flow, 0.0, abs_tol=0.01):
                raise ValueError(
                    f"Invalid NAV lineage for {run.run_id}: initial_capital changed from "
                    f"{previous_capital:.2f} to {run.initial_capital:.2f} without manifest-backed funding evidence"
                )
            for flow in run.funding_cash_flows:
                amount = float(flow.get("amount") or 0)
                total_before = parse_float(flow.get("total_value_before"))
                total_after = parse_float(flow.get("total_value_after"))
                if total_before is None or total_after is None:
                    raise ValueError(
                        f"Invalid NAV lineage for {run.run_id}: funding evidence requires "
                        "total_value_before and total_value_after for time-weighted unitization"
                    )
                if total_before <= 0 or total_after <= 0:
                    raise ValueError(
                        f"Invalid NAV lineage for {run.run_id}: funding pre/post values must be positive"
                    )
                flow_unit_nav = total_before / unit_count
                if flow_unit_nav <= 0:
                    raise ValueError(
                        f"Invalid NAV lineage for {run.run_id}: non-positive NAV at funding time"
                    )
                unit_count += amount / flow_unit_nav
                if unit_count <= 0:
                    raise ValueError(
                        f"Invalid NAV lineage for {run.run_id}: funding would leave non-positive units"
                    )
                if not math.isclose(flow_unit_nav * unit_count, total_after, abs_tol=0.01):
                    raise ValueError(
                        f"Invalid NAV lineage for {run.run_id}: funding unitization does not reconcile "
                        "to total_value_after"
                    )
            if run.total_value_after <= 0:
                raise ValueError(
                    f"Invalid NAV lineage for {run.run_id}: non-positive portfolio value"
                )
            unit_nav = run.total_value_after / unit_count
            if previous_unit_nav is None or previous_unit_nav <= 0:
                raise ValueError(f"Invalid NAV lineage for {run.run_id}: missing prior unit NAV")
            daily_return = unit_nav / previous_unit_nav - 1.0
        points.append(
            {
                "unit_nav": unit_nav,
                "unit_count": unit_count,
                "raw_capital_ratio": raw_capital_ratio,
                "rebased_nav": unit_nav / (first_unit_nav or 1.0),
                "daily_return": daily_return,
                "external_funding_cash_flow": external_flow,
            }
        )
        previous_unit_nav = unit_nav
        previous_capital = run.initial_capital
    return points


def portfolio_nav_source_summary(
    runs: list[RecordRun],
    nav_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    funding_events = [flow for run in runs for flow in run.funding_cash_flows]
    return {
        "status": "cash_flow_adjusted_unit_nav",
        "method": "time_weighted_unitization",
        "formula": (
            "flow_unit_nav = total_value_before_flow / units_before_flow; "
            "units_after_flow = units_before_flow + external_funding_cash_flow / flow_unit_nav; "
            "unit_nav_t = total_value_after_t / units_after_flow"
        ),
        "cash_flow_timing_source": "manifest_pre_post_valuation",
        "historical_return_preserved": True,
        "raw_capital_ratio_field": "portfolio_nav_raw",
        "first_valid_date": runs[0].date if runs else "",
        "last_valid_date": runs[-1].date if runs else "",
        "first_unit_nav": parse_float(nav_rows[0].get("portfolio_nav")) if nav_rows else None,
        "last_unit_nav": parse_float(nav_rows[-1].get("portfolio_nav")) if nav_rows else None,
        "display_total_return": (
            parse_float(nav_rows[-1].get("portfolio_nav_rebased")) - 1.0
            if nav_rows and parse_float(nav_rows[-1].get("portfolio_nav_rebased")) is not None
            else None
        ),
        "capital_base_start": runs[0].initial_capital if runs else None,
        "capital_base_end": runs[-1].initial_capital if runs else None,
        "latest_total_value_after": runs[-1].total_value_after if runs else None,
        "unit_count_start": parse_float(nav_rows[0].get("portfolio_units")) if nav_rows else None,
        "unit_count_end": parse_float(nav_rows[-1].get("portfolio_units")) if nav_rows else None,
        "funding_event_count": len(funding_events),
        "funding_events": funding_events,
    }


def build_nav_rows(
    runs: list[RecordRun],
    benchmark_export: BenchmarkExport | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    nav_points = build_cash_flow_adjusted_nav_points(runs)
    points_by_date = {run.date: point for run, point in zip(runs, nav_points)}

    def portfolio_fields(run: RecordRun) -> dict[str, str]:
        point = points_by_date[run.date]
        daily_return = point["daily_return"]
        return {
            "portfolio_nav": f"{point['unit_nav']:.8f}",
            "portfolio_nav_raw": f"{point['raw_capital_ratio']:.8f}",
            "portfolio_nav_rebased": f"{point['rebased_nav']:.8f}",
            "portfolio_return": f"{daily_return:.8f}" if daily_return is not None else "",
            "portfolio_units": f"{point['unit_count']:.8f}",
            "initial_capital": f"{run.initial_capital:.2f}",
            "total_value_after": f"{run.total_value_after:.2f}",
            "external_funding_cash_flow": f"{point['external_funding_cash_flow']:.2f}",
        }

    portfolio_fieldnames = [
        "portfolio_nav",
        "portfolio_nav_raw",
        "portfolio_nav_rebased",
        "portfolio_return",
        "portfolio_units",
        "initial_capital",
        "total_value_after",
        "external_funding_cash_flow",
    ]
    if benchmark_export is not None and benchmark_export.values_by_date:
        benchmark_values_by_date = {
            date: dict(values) for date, values in benchmark_export.values_by_date.items()
        }
        for run in runs:
            values = benchmark_values_by_date.setdefault(run.date, {})
            coverage = dict(benchmark_export.coverage_by_date.get(run.date, {}))
            value_dates = dict(benchmark_export.value_date_by_date.get(run.date, {}))
            _with_composite_benchmark_fields(values, coverage, value_dates, run_date=run.date)
        available_fields = [
            field
            for field in BENCHMARK_FIELD_ORDER
            if any(field in values for values in benchmark_values_by_date.values())
        ]
        for run in runs:
            row: dict[str, Any] = {
                "date": run.date,
                **portfolio_fields(run),
                "cash_weight": "",
                "gross_exposure": "",
                "net_exposure": "",
            }
            values = benchmark_values_by_date.get(run.date, {})
            for field in available_fields:
                row[field] = f"{values[field]:.8f}" if field in values else ""
            rows.append(row)
        fieldnames = [
            "date",
            *portfolio_fieldnames,
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
        benchmark_main_nav = next(
            (normalized[field] for field in ("star50_nav", "csi300_nav", "chinext_nav") if field in normalized),
            DEFAULT_INITIAL_BENCHMARK,
        )
        legacy_benchmark_nav = normalized.get("csi300_nav", benchmark_main_nav)
        row: dict[str, Any] = {
            "date": run.date,
            **portfolio_fields(run),
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
        *portfolio_fieldnames,
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


def _trade_timestamp_date(timestamp: str, default_date: str) -> str:
    text = str(timestamp or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) >= 8:
        return tushare_to_iso_date(digits[:8])
    return default_date


def _trade_side(action: str) -> str:
    text = str(action or "").strip().lower()
    if "buy" in text:
        return "buy"
    if "sell" in text:
        return "sell"
    return ""


def _is_executed_trade(row: dict[str, str], *, legacy_orders: bool = False) -> bool:
    del legacy_orders
    status = str(row.get("status") or "").strip().lower()
    return bool(status and status in EXECUTED_TRADE_STATUSES)


def _is_partial_fill(row: dict[str, str]) -> bool:
    return str(row.get("status") or "").strip().lower() in (
        PARTIAL_EXECUTED_TRADE_STATUSES
    )


def _first_trade_float(row: dict[str, str], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = parse_float(row.get(field))
        if value is not None:
            return value
    return None


def _trade_price(row: dict[str, str]) -> float | None:
    fill_price = _first_trade_float(
        row,
        ("fill_price", "filled_price", "execution_price", "avg_fill_price"),
    )
    if _is_partial_fill(row):
        return fill_price
    return fill_price if fill_price is not None else parse_float(row.get("price"))


def _trade_quantity(row: dict[str, str]) -> float | None:
    fill_quantity = _first_trade_float(
        row,
        (
            "fill_quantity",
            "filled_quantity",
            "filled_shares",
            "execution_quantity",
            "executed_quantity",
        ),
    )
    if _is_partial_fill(row):
        return fill_quantity
    return fill_quantity if fill_quantity is not None else parse_float(
        row.get("shares") or row.get("quantity")
    )


def _trade_amount(
    row: dict[str, str],
    *,
    price: float | None,
    quantity: float | None,
) -> float | None:
    fill_amount = _first_trade_float(
        row,
        (
            "fill_value",
            "filled_value",
            "execution_value",
            "executed_value",
            "fill_amount",
        ),
    )
    if _is_partial_fill(row):
        if fill_amount is not None:
            return fill_amount
        if price is not None and quantity is not None:
            return price * quantity
        return None
    return fill_amount if fill_amount is not None else parse_float(
        row.get("trade_value") or row.get("trade_amount")
    )


def _trade_row_candidates(run: RecordRun, *, legacy_orders: bool) -> list[dict[str, str]]:
    filename = "orders.csv" if legacy_orders else "manual_switch_and_take_profit_orders.csv"
    return read_csv_rows(run.path / filename)


def _trade_source_name(legacy_orders: bool) -> str:
    return "orders.csv" if legacy_orders else "manual_switch_and_take_profit_orders.csv"


def _validate_executed_trade_record(
    row: dict[str, str],
    *,
    run: RecordRun,
    action: str,
    symbol: str,
    timestamp: str,
    price: float | None,
    quantity: float | None,
    amount: float | None,
    source_name: str,
) -> list[str]:
    missing: list[str] = []
    if not timestamp:
        missing.append("timestamp")
    elif not _trade_timestamp_date(timestamp, ""):
        missing.append("timestamp(parseable)")
    if not symbol:
        missing.append("symbol")
    if action not in {"buy", "sell"}:
        missing.append("side/action")
    if quantity is None or quantity <= 0:
        missing.append("shares")
    if price is None or price <= 0:
        missing.append("price")
    if amount is None or amount <= 0:
        missing.append("trade_value")
    if missing:
        return missing
    expected_amount = (price or 0) * (quantity or 0)
    tolerance = max(1.0, abs(expected_amount) * 0.01)
    if amount is not None and abs(amount - expected_amount) > tolerance:
        return ["trade_value(price*shares mismatch)"]
    return []


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


def _position_contribution_date(
    ledger_row: dict[str, Any],
    review_row: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> tuple[str | None, str | None]:
    quote_snapshot = market_snapshot.get("quote_snapshot")
    if not isinstance(quote_snapshot, dict):
        quote_snapshot = {}
    candidates = (
        ("holdings_review.effective_date", review_row.get("effective_date")),
        ("holdings_review.analysis_trade_date", review_row.get("analysis_trade_date")),
        ("holdings_review.quote_date", review_row.get("quote_date")),
        ("holdings_review.quote_at", review_row.get("quote_at")),
        ("holdings_review.quote_time", review_row.get("quote_time")),
        ("holdings_review.trade_time", review_row.get("trade_time")),
        ("ledger.effective_date", ledger_row.get("effective_date")),
        ("ledger.quote_at", ledger_row.get("quote_at")),
        ("market_snapshot.quote_snapshot.as_of", quote_snapshot.get("as_of")),
        ("market_snapshot.quote_snapshot.quote_at", quote_snapshot.get("quote_at")),
        (
            "market_snapshot.analysis_trade_date",
            market_snapshot.get("analysis_trade_date"),
        ),
    )
    for source, value in candidates:
        effective_date = explicit_iso_date(value)
        if effective_date is not None:
            return effective_date, source
    return None, None


def build_positions_rows(
    runs: list[RecordRun],
    sector_map: dict[str, dict[str, str | None]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run in runs:
        ledger_path = run.manual_ledger_path or (
            run.path / "ledger_after_manual_switch.csv"
        )
        before_sha = sha256_file(ledger_path)
        if run.manual_ledger_sha256 and before_sha != run.manual_ledger_sha256:
            raise RuntimeError(
                f"manual ledger changed after manifest selection: {run.run_id}"
            )
        ledger_rows = read_ledger_rows(ledger_path)
        if run.manual_ledger_sha256 and sha256_file(ledger_path) != before_sha:
            raise RuntimeError(
                f"manual ledger changed during readback: {run.run_id}"
            )
        if not ledger_rows:
            warnings.append(f"{run.run_id}: 未找到可用 ledger_after_manual_switch.csv，positions 已跳过。")
            continue
        holdings_review = {
            row.get("symbol", ""): row
            for row in read_csv_rows(run.path / "holdings_review.csv")
            if row.get("symbol")
        }
        market_snapshot = load_json(run.path / "market_snapshot.json")
        equity_sleeve_value = sum(
            value
            for value in (parse_float(item.get("current_value")) for item in ledger_rows)
            if value is not None and value > 0
        )
        for row in ledger_rows:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            value = parse_float(row.get("current_value"))
            reported_sleeve_weight = parse_float(row.get("market_weight"))
            nav_weight = (
                value / run.total_value_after
                if value is not None and run.total_value_after > 0
                else None
            )
            equity_sleeve_weight = reported_sleeve_weight
            if equity_sleeve_weight is None and value is not None and equity_sleeve_value > 0:
                equity_sleeve_weight = value / equity_sleeve_value
            shares = parse_float(row.get("shares"))
            avg_cost = parse_float(row.get("avg_cost"))
            cost_basis = parse_float(row.get("cost_basis"))
            current_price = parse_float(row.get("current_price"))
            review = holdings_review.get(symbol, {})
            contribution_effective_date, contribution_date_source = (
                _position_contribution_date(row, review, market_snapshot)
            )
            daily_return = parse_percentage_points(review.get("today_change_pct"))
            contribution = (
                nav_weight * daily_return
                if nav_weight is not None and daily_return is not None
                else None
            )
            industry_info = sector_map.get(symbol, {})
            rows.append(
                {
                    "date": run.date,
                    "ticker": symbol,
                    "name": row.get("name") or review.get("name") or "UNKNOWN_NAME",
                    # ``weight`` remains a compatibility alias, but its v2
                    # semantics are explicitly NAV weight, never sleeve weight.
                    "weight": f"{nav_weight:.8f}" if nav_weight is not None else "",
                    "nav_weight": f"{nav_weight:.8f}" if nav_weight is not None else "",
                    "equity_sleeve_weight": (
                        f"{equity_sleeve_weight:.8f}"
                        if equity_sleeve_weight is not None
                        else ""
                    ),
                    "industry": industry_info.get("industry"),
                    "industry_source": industry_info.get("industry_source"),
                    "industry_as_of": industry_info.get("industry_as_of"),
                    "industry_generation_sha256": industry_info.get(
                        "industry_generation_sha256"
                    ),
                    "daily_return": f"{daily_return:.8f}" if daily_return is not None else "",
                    "contribution": f"{contribution:.8f}" if contribution is not None else "",
                    "contribution_effective_date": contribution_effective_date or "",
                    "contribution_date_source": contribution_date_source or "unavailable",
                    "market_value": f"{value:.2f}" if value is not None else "",
                    "quantity": f"{shares:.0f}" if shares is not None else "",
                    "avg_cost": f"{avg_cost:.4f}" if avg_cost is not None else "",
                    "cost_basis": f"{cost_basis:.2f}" if cost_basis is not None else "",
                    "current_price": f"{current_price:.4f}" if current_price is not None else "",
                    "unrealized_pnl": (
                        f"{nullable_float(row.get('unrealized_pnl')):.2f}"
                        if nullable_float(row.get("unrealized_pnl")) is not None
                        else ""
                    ),
                    "recommended_action": review.get("recommended_action") or review.get("action") or "",
                    "stop_loss": review.get("stop_loss") or review.get("stop_loss_price") or "",
                    "take_profit": review.get("take_profit") or review.get("take_profit_price") or "",
                    "quote_at": review.get("quote_at") or review.get("quote_time") or review.get("trade_time") or "",
                    "quote_age_seconds": review.get("quote_age_seconds") or "",
                    "thesis": review.get("thesis") or review.get("reason") or "",
                    "risk_status": review.get("risk_status") or review.get("status") or "",
                }
            )
    missing_industry_count = sum(1 for row in rows if not row.get("industry"))
    if missing_industry_count:
        warnings.append(
            f"{missing_industry_count} 条持仓缺少 strict stock_basic.industry；保留 ticker 且 industry=null。"
        )
    return rows, warnings


def build_industry_rows(position_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "date": row.get("date"),
            "ticker": row.get("ticker"),
            "industry": row.get("industry") or None,
            "industry_source": row.get("industry_source") or None,
            "industry_as_of": row.get("industry_as_of") or None,
            "industry_generation_sha256": row.get("industry_generation_sha256") or None,
            "nav_weight": parse_float(row.get("nav_weight")),
        }
        for row in position_rows
    ]


def build_trade_rows(
    runs: list[RecordRun],
    sector_map: dict[str, dict[str, str | None]],
    *,
    warnings: list[str] | None = None,
    completeness: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    strict_warnings = warnings if warnings is not None else []
    skipped_incomplete = 0
    executed_seen = 0
    for run in runs:
        source_rows = [
            (False, _trade_row_candidates(run, legacy_orders=False)),
            (True, _trade_row_candidates(run, legacy_orders=True)),
        ]
        for legacy_orders, trade_rows in source_rows:
            source_name = _trade_source_name(legacy_orders)
            for row in trade_rows:
                if not _is_executed_trade(row, legacy_orders=legacy_orders):
                    continue
                executed_seen += 1
                symbol = str(row.get("symbol") or "").strip()
                action = _trade_side(row.get("action") or row.get("side") or "")
                price = _trade_price(row)
                quantity = _trade_quantity(row)
                amount = _trade_amount(row, price=price, quantity=quantity)
                timestamp = str(row.get("timestamp") or "").strip()
                missing = _validate_executed_trade_record(
                    row,
                    run=run,
                    action=action,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=price,
                    quantity=quantity,
                    amount=amount,
                    source_name=source_name,
                )
                if missing:
                    skipped_incomplete += 1
                    strict_warnings.append(
                        "trade_record_incomplete: "
                        f"{run.run_id} {source_name} 已执行交易缺少/无效 {','.join(missing)}，已跳过；"
                        f"symbol={symbol or '-'} action={action or '-'}。"
                    )
                    continue
                key = (
                    timestamp,
                    action,
                    symbol,
                    f"{quantity:.8f}" if quantity is not None else "",
                    f"{amount:.2f}" if amount is not None else "",
                )
                if key in seen:
                    continue
                seen.add(key)
                industry = sector_map.get(symbol, {}).get("industry")
                fee = parse_float(
                    row.get("fee")
                    or row.get("commission")
                    or row.get("transaction_fee")
                    or row.get("total_fee")
                )
                fee_source = str(row.get("fee_source") or "").strip()
                if fee is None:
                    fee_source = "unknown"
                elif not fee_source:
                    fee_source = f"{source_name}.fee"
                candidates.append(
                    {
                        "trade_date": _trade_timestamp_date(timestamp, run.date),
                        "recommendation_id": row.get("recommendation_id") or "",
                        "decision_id": row.get("decision_id") or "",
                        "order_id": row.get("order_id") or "",
                        "fill_id": row.get("fill_id") or row.get("execution_id") or "",
                        "ticker": symbol,
                        "name": row.get("name") or "UNKNOWN_NAME",
                        "side": action,
                        "price": f"{price:.4f}" if price is not None else "",
                        "quantity": f"{quantity:.0f}" if quantity is not None else "",
                        "trade_amount": f"{amount:.2f}" if amount is not None else "",
                        "fee": f"{fee:.2f}" if fee is not None else "",
                        "fee_source": fee_source,
                        "slippage": row.get("slippage") or "",
                        "ledger_delta": row.get("ledger_delta") or "",
                        "reason": row.get("reason") or "",
                        "industry": industry,
                    }
                )
    candidates.sort(key=lambda item: (item["trade_date"], item["ticker"], item["side"], item["quantity"]))
    if completeness is not None:
        completeness.update(
            {
                "status": "complete" if skipped_incomplete == 0 else "partial",
                "required_fields": TRADE_RECORD_REQUIRED_FIELDS,
                "executed_source_rows": executed_seen,
                "exported_rows": len(candidates),
                "skipped_incomplete_rows": skipped_incomplete,
            }
        )
    return candidates


def apply_position_exposures(
    nav_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
) -> None:
    """Attach account exposure using NAV weights, never equity-sleeve weights."""

    weights_by_date: dict[str, list[float]] = defaultdict(list)
    for row in position_rows:
        weight = nullable_float(row.get("nav_weight") or row.get("weight"))
        if weight is not None:
            weights_by_date[str(row.get("date") or "")].append(weight)
    for row in nav_rows:
        weights = weights_by_date.get(str(row.get("date") or ""), [])
        if not weights:
            continue
        gross = sum(abs(weight) for weight in weights)
        net = sum(weights)
        row["gross_exposure"] = f"{gross:.8f}"
        row["net_exposure"] = f"{net:.8f}"
        row["cash_weight"] = f"{1.0 - net:.8f}"


def _numeric_record(row: dict[str, Any], numeric_fields: set[str]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in row.items():
        if key in numeric_fields:
            record[key] = nullable_float(value)
        else:
            text = "" if value is None else str(value).strip()
            record[key] = text or None
    return record


def _factor_rows_from_registry(registry_path: Path | None) -> list[dict[str, Any]]:
    if registry_path is None or not registry_path.is_file():
        return []
    payload = load_json(registry_path)
    records = payload.get("factors") or payload.get("records") or []
    if not isinstance(records, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        health = metadata.get("health_monitor") if isinstance(metadata.get("health_monitor"), dict) else {}
        rows.append(
            {
                "factor_id": item.get("factor_id") or item.get("name") or item.get("id"),
                "slot": item.get("slot"),
                "family": item.get("family") or item.get("category"),
                "status": item.get("status") or item.get("state"),
                "weight": nullable_float(item.get("weight")),
                "health_window": health.get("last_evaluation_id") or health.get("latest_reviewed_at"),
                "health_status": health.get("status"),
                "challenger": item.get("challenger"),
                "last_transition": item.get("last_transition"),
            }
        )
    return rows


def _factor_canonical_producer_control() -> dict[str, Any]:
    try:
        from quant_investor.factors.governance_protocol_v3 import (
            canonical_replay_producer_control,
        )

        control = canonical_replay_producer_control()
    except (ImportError, TypeError, ValueError):
        control = {}
    return {
        "producer_available": control.get("producer_available") is True,
        "production_apply_eligible": control.get("production_apply_eligible")
        is True,
        "blocker": str(
            control.get("blocker")
            or "canonical_full_chain_replay_producer_unavailable"
        ),
    }


def _load_factor_protocol_v3_readback(
    latest_run: RecordRun,
    *,
    registry_sha: str | None,
    protocol_file: Path | None = None,
    expected_sha256: str = "",
) -> tuple[dict[str, Any], Path | None]:
    try:
        from quant_investor.factors.governance_protocol_v3 import (
            PROTOCOL_HASH,
            PROTOCOL_SCHEMA_VERSION,
        )
    except ImportError:
        PROTOCOL_HASH = ""
        PROTOCOL_SCHEMA_VERSION = "factor-governance-protocol.v3"
    producer_control = _factor_canonical_producer_control()
    if protocol_file is not None:
        candidates = [protocol_file]
        actual_sha = sha256_file(protocol_file)
        if not expected_sha256:
            return _missing_factor_protocol(
                PROTOCOL_HASH,
                "factor_protocol_expected_sha256_missing",
            ), None
        if actual_sha != expected_sha256:
            return _missing_factor_protocol(
                PROTOCOL_HASH,
                "factor_protocol_artifact_sha256_mismatch",
            ), None
    else:
        candidates = [
            latest_run.path / "factor_governance_protocol_v3.json",
            latest_run.path / "factor_protocol_v3.json",
            latest_run.path / "factor_governance_readback.v3.json",
        ]
    for path in candidates:
        payload = load_json(path)
        if not payload:
            continue
        raw = payload.get("factor_protocol")
        if not isinstance(raw, dict):
            raw = payload
        raw = dict(raw)
        reported_blockers = [
            str(value) for value in list(raw.get("blockers") or [])
        ]
        validation_blockers: list[str] = []
        schema_version = str(raw.get("schema_version") or "")
        actual_hash = str(raw.get("protocol_hash") or "")
        if schema_version != PROTOCOL_SCHEMA_VERSION:
            validation_blockers.append("factor_protocol_schema_mismatch")
        if not PROTOCOL_HASH or actual_hash != PROTOCOL_HASH:
            validation_blockers.append("factor_protocol_hash_mismatch")
        if str(raw.get("protocol_version") or "") != "v3":
            validation_blockers.append("factor_protocol_version_mismatch")
        after_registry_sha = str(raw.get("after_registry_sha256") or "")
        if after_registry_sha and registry_sha and after_registry_sha != registry_sha:
            validation_blockers.append("factor_protocol_registry_readback_mismatch")
        transition = raw.get("transition") if isinstance(raw.get("transition"), dict) else {}
        rollback = raw.get("rollback") if isinstance(raw.get("rollback"), dict) else {}
        evidence_hash = str(
            raw.get("evidence_hash")
            or transition.get("evidence_hash")
            or ""
        )
        status = str(raw.get("status") or "blocked")
        before_registry_sha = str(raw.get("before_registry_sha256") or "")
        is_sha = lambda value: bool(
            len(str(value or "")) == 64
            and all(
                character in "0123456789abcdef"
                for character in str(value or "")
            )
        )
        if status in {"report_only", "report_only_ready"}:
            if raw.get("apply_requested") is not False:
                validation_blockers.append("factor_report_only_apply_flag_invalid")
            if (
                not is_sha(before_registry_sha)
                or not is_sha(after_registry_sha)
                or before_registry_sha != after_registry_sha
                or (registry_sha and after_registry_sha != registry_sha)
            ):
                validation_blockers.append("factor_report_only_registry_sha_invalid")
        elif status == "applied":
            if (
                not producer_control["producer_available"]
                or not producer_control["production_apply_eligible"]
            ):
                validation_blockers.append(producer_control["blocker"])
            for field in (
                "evidence_hash",
                "transition_hash",
                "mutation_plan_hash",
            ):
                if not is_sha(raw.get(field)):
                    validation_blockers.append(f"factor_applied_{field}_missing")
            if not str(raw.get("transition_id") or ""):
                validation_blockers.append("factor_applied_transition_id_missing")
            if not is_sha(before_registry_sha) or not is_sha(after_registry_sha):
                validation_blockers.append("factor_applied_registry_sha_missing")
            elif registry_sha and after_registry_sha != registry_sha:
                validation_blockers.append(
                    "factor_applied_registry_readback_mismatch"
                )
            mutation_manifest = raw.get("registry_mutation_manifest")
            if (
                not isinstance(mutation_manifest, dict)
                or mutation_manifest.get("applied") is not True
                or mutation_manifest.get("after_registry_sha256")
                != after_registry_sha
            ):
                validation_blockers.append(
                    "factor_applied_mutation_manifest_invalid"
                )
            wal_path_text = str(raw.get("inverse_wal_path") or "")
            wal_path = Path(wal_path_text) if wal_path_text else None
            if (
                wal_path is None
                or not wal_path.is_file()
                or wal_path.stat().st_mode & 0o777 != 0o600
            ):
                validation_blockers.append("factor_applied_inverse_wal_unverified")
        elif status not in {"blocked", "blocked_readback"}:
            validation_blockers.append("factor_protocol_status_invalid")
        blockers = list(dict.fromkeys(reported_blockers + validation_blockers))
        readback_verified = not validation_blockers and bool(sha256_file(path))
        normalized = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "protocol_version": str(raw.get("protocol_version") or "v3"),
            "expected_protocol_hash": PROTOCOL_HASH or None,
            "protocol_hash": actual_hash or None,
            "protocol_hash_match": bool(PROTOCOL_HASH and actual_hash == PROTOCOL_HASH),
            "status": status if readback_verified else "blocked",
            "blockers": list(dict.fromkeys(blockers)),
            "evidence_hash": evidence_hash or None,
            "evidence_status": (
                "verified"
                if len(evidence_hash) == 64
                else "not_applicable"
                if status == "report_only"
                else "missing"
            ),
            "transition_id": str(
                raw.get("transition_id") or transition.get("transition_id") or ""
            )
            or None,
            "transition_hash": str(
                raw.get("transition_hash") or transition.get("transition_hash") or ""
            )
            or None,
            "transition_applied": bool(status == "applied" and readback_verified),
            "rollback_status": str(
                raw.get("rollback_status")
                or rollback.get("status")
                or ("available" if raw.get("inverse_wal_path") else "not_available")
            ),
            "before_registry_sha256": str(
                before_registry_sha
            )
            or None,
            "after_registry_sha256": after_registry_sha or None,
            "readback_verified": readback_verified,
            "artifact_sha256": sha256_file(path),
            "canonical_producer_available": producer_control[
                "producer_available"
            ],
            "canonical_production_apply_eligible": producer_control[
                "production_apply_eligible"
            ],
            "canonical_producer_blocker": producer_control["blocker"],
        }
        return normalized, path
    return _missing_factor_protocol(
        PROTOCOL_HASH,
        "factor_protocol_v3_missing",
    ), None


def _missing_factor_protocol(
    expected_protocol_hash: str,
    blocker: str,
) -> dict[str, Any]:
    producer_control = _factor_canonical_producer_control()
    return {
        "schema_version": "factor-governance-protocol.v3",
        "protocol_version": "v3",
        "expected_protocol_hash": expected_protocol_hash or None,
        "protocol_hash": None,
        "protocol_hash_match": False,
        "status": "blocked",
        "blockers": [blocker],
        "evidence_hash": None,
        "evidence_status": "missing",
        "transition_id": None,
        "transition_hash": None,
        "transition_applied": False,
        "rollback_status": "not_available",
        "before_registry_sha256": None,
        "after_registry_sha256": None,
        "readback_verified": False,
        "artifact_sha256": None,
        "canonical_producer_available": producer_control["producer_available"],
        "canonical_production_apply_eligible": producer_control[
            "production_apply_eligible"
        ],
        "canonical_producer_blocker": producer_control["blocker"],
    }


def build_attribution_reconciliation(
    nav_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    *,
    trading_calendar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_calendar_supplied = trading_calendar is not None
    calendar_status = str((trading_calendar or {}).get("status") or "missing")
    allowed_open_dates = {
        str(value)
        for value in list((trading_calendar or {}).get("expected_open_dates") or [])
        if explicit_iso_date(value) is not None
    }
    all_nav_return_dates = {
        str(row.get("date") or "")
        for row in nav_rows
        if nullable_float(row.get("portfolio_return")) is not None
    }
    excluded_nav_return_dates = sorted(
        date
        for date in all_nav_return_dates
        if strict_calendar_supplied
        and (calendar_status != "available" or date not in allowed_open_dates)
    )
    contributions_by_date: dict[str, float] = defaultdict(float)
    observations_by_date: dict[str, int] = defaultdict(int)
    sleeve_weights_by_date: dict[str, list[float]] = defaultdict(list)
    sleeve_missing_by_date: dict[str, int] = defaultdict(int)
    total_positions_by_date: dict[str, int] = defaultdict(int)
    invalid_position_lineage_by_date: dict[str, int] = defaultdict(int)
    missing_effective_date_count = 0
    excluded_position_effective_dates: set[str] = set()
    for row in position_rows:
        date = explicit_iso_date(row.get("contribution_effective_date")) or ""
        if not date:
            missing_effective_date_count += 1
        elif strict_calendar_supplied and (
            calendar_status != "available" or date not in allowed_open_dates
        ):
            excluded_position_effective_dates.add(date)
        total_positions_by_date[date] += 1
        nav_weight = nullable_float(row.get("nav_weight"))
        daily_return = nullable_float(row.get("daily_return"))
        contribution = nullable_float(row.get("contribution"))
        if (
            not date
            or not str(row.get("ticker") or "").strip()
            or str(row.get("contribution_date_source") or "").strip()
            in {"", "unavailable"}
            or nav_weight is None
            or daily_return is None
            or contribution is None
            or not math.isclose(
                contribution,
                nav_weight * daily_return,
                abs_tol=1e-8,
            )
        ):
            invalid_position_lineage_by_date[date] += 1
        sleeve_weight = nullable_float(row.get("equity_sleeve_weight"))
        if sleeve_weight is None:
            sleeve_missing_by_date[date] += 1
        else:
            sleeve_weights_by_date[date].append(sleeve_weight)
        if contribution is None:
            continue
        contributions_by_date[date] += contribution
        observations_by_date[date] += 1
    daily: list[dict[str, Any]] = []
    valid_nav_rows = [
        row
        for row in nav_rows
        if nullable_float(row.get("portfolio_return")) is not None
        and (
            not strict_calendar_supplied
            or (
                calendar_status == "available"
                and str(row.get("date") or "") in allowed_open_dates
            )
        )
    ]
    for row in valid_nav_rows:
        date = str(row.get("date") or "")
        portfolio_return = nullable_float(row.get("portfolio_return"))
        contribution = contributions_by_date.get(date) if observations_by_date.get(date) else None
        explicit_residual = nullable_float(row.get("explicit_cash_fee_residual"))
        if explicit_residual is None:
            explicit_components = [
                value
                for value in (
                    nullable_float(row.get("cash_return_contribution")),
                    nullable_float(row.get("fee_return_contribution")),
                )
                if value is not None
            ]
            explicit_residual = sum(explicit_components) if explicit_components else None
        sleeve_weights = sleeve_weights_by_date.get(date, [])
        sleeve_sum = sum(sleeve_weights) if sleeve_weights else None
        position_snapshot_complete = bool(
            contribution is not None
            and observations_by_date.get(date, 0) > 0
            and observations_by_date.get(date, 0)
            == total_positions_by_date.get(date, 0)
            and invalid_position_lineage_by_date.get(date, 0) == 0
            and sleeve_missing_by_date.get(date, 0) == 0
            and sleeve_sum is not None
            and math.isclose(sleeve_sum, 1.0, abs_tol=0.001)
        )
        # Cash/fee residuals explain only a complete position-contribution
        # snapshot. They must never masquerade as position coverage.
        covered = position_snapshot_complete
        residual = (
            portfolio_return - (contribution or 0.0) - (explicit_residual or 0.0)
            if portfolio_return is not None and covered
            else None
        )
        daily.append(
            {
                "date": date,
                "portfolio_return": portfolio_return,
                "position_contribution": contribution,
                "explicit_cash_fee_residual": explicit_residual,
                "covered": covered,
                "unexplained_residual": residual,
                "within_1bp": abs(residual) <= 0.0001 if residual is not None else None,
                "position_observation_count": observations_by_date.get(date, 0),
                "total_position_count": total_positions_by_date.get(date, 0),
                "position_snapshot_complete": position_snapshot_complete,
                "equity_sleeve_weight_sum": sleeve_sum,
            }
        )
    covered_rows = [row for row in daily if row["covered"]]
    valid_count = len(valid_nav_rows)
    coverage_ratio = len(covered_rows) / valid_count if valid_count else 0.0
    all_within_tolerance = bool(covered_rows) and all(
        row["within_1bp"] for row in covered_rows
    )
    reconciled = valid_count > 0 and len(covered_rows) == valid_count and all_within_tolerance
    valid_nav_dates = {str(row.get("date") or "") for row in valid_nav_rows}
    position_dates_without_nav_return = sorted(
        date
        for date in total_positions_by_date
        if date and date in allowed_open_dates and date not in valid_nav_dates
    )
    blockers: list[str] = []
    if strict_calendar_supplied and calendar_status != "available":
        blockers.append("attribution_formal_trading_calendar_missing")
    if excluded_nav_return_dates or excluded_position_effective_dates:
        blockers.append("attribution_non_open_dates_excluded")
    if missing_effective_date_count:
        blockers.append("attribution_position_effective_date_missing")
    if position_dates_without_nav_return:
        blockers.append("attribution_position_date_without_nav_return")
    return {
        "tolerance": 0.0001,
        "daily": daily,
        "valid_nav_return_days": valid_count,
        "covered_days": len(covered_rows),
        "coverage_ratio": coverage_ratio,
        "reconciled_days": sum(1 for row in covered_rows if row["within_1bp"]),
        "status": "reconciled" if reconciled else "partial",
        "coverage_basis": "strict_parquet_trade_date_mask",
        "calendar_status": calendar_status,
        "blockers": blockers,
        "diagnostics": {
            "excluded_nav_return_dates": excluded_nav_return_dates,
            "excluded_position_effective_dates": sorted(
                excluded_position_effective_dates
            ),
            "positions_missing_effective_date_count": missing_effective_date_count,
            "position_dates_without_nav_return": position_dates_without_nav_return,
        },
    }


def _latest_explicit_benchmark_value_dates(
    benchmark_summary: dict[str, Any],
) -> dict[str, str]:
    by_field: dict[str, list[str]] = defaultdict(list)
    by_record = benchmark_summary.get("value_date_by_date") or {}
    if not isinstance(by_record, dict):
        return {}
    for values in by_record.values():
        if not isinstance(values, dict):
            continue
        for field, value in values.items():
            normalized = explicit_iso_date(value)
            if normalized:
                by_field[str(field)].append(normalized)
    return {
        field: max(dates)
        for field, dates in sorted(by_field.items())
        if dates
    }


def build_canonical_as_of_matrix(
    latest_run: RecordRun,
    benchmark_summary: dict[str, Any],
    *,
    factor_registry_sha: str | None,
) -> dict[str, Any]:
    snapshot = load_json(latest_run.path / "market_snapshot.json")
    strategy_record_at = explicit_iso_timestamp(latest_run.record_time)
    strategy_record_date = explicit_iso_date(strategy_record_at)
    analysis_trading_date = explicit_iso_date(snapshot.get("analysis_trade_date"))
    quote_field = snapshot.get("quote_snapshot")
    if isinstance(quote_field, dict):
        quote_value = (
            quote_field.get("as_of")
            or quote_field.get("quote_at")
            or quote_field.get("timestamp")
        )
    else:
        quote_value = quote_field
    quote_at = explicit_iso_timestamp(quote_value)
    return {
        "strategy_record_date": strategy_record_date,
        "strategy_record_at": strategy_record_at,
        "analysis_trading_date": analysis_trading_date,
        "quote_at": quote_at,
        "benchmark_value_dates": _latest_explicit_benchmark_value_dates(benchmark_summary),
        "factor_registry_sha": factor_registry_sha,
    }


def build_dashboard_contract_v3(
    *,
    run_id: str,
    generated_at: str,
    record_root: Path,
    latest_run: RecordRun,
    nav_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    benchmark_summary: dict[str, Any],
    ledger_path: Path,
    manifest_path: Path,
    warnings: list[str],
    registry_path: Path | None = None,
    trading_calendar: dict[str, Any] | None = None,
    factor_protocol_file: Path | None = None,
    expected_factor_sha256: str = "",
    manual_ledger_sha_declared: bool | None = None,
) -> dict[str, Any]:
    """Build the single, versioned snapshot consumed by the static Dashboard."""

    nav_numeric = {
        "portfolio_nav",
        "portfolio_nav_raw",
        "portfolio_nav_rebased",
        "portfolio_return",
        "portfolio_units",
        "initial_capital",
        "total_value_after",
        "external_funding_cash_flow",
        "cash_weight",
        "gross_exposure",
        "net_exposure",
        "cash_return_contribution",
        "fee_return_contribution",
        "explicit_cash_fee_residual",
    }
    nav_numeric.update(
        key
        for row in nav_rows
        for key in row
        if str(key).endswith("_nav")
    )
    position_numeric = {
        "weight",
        "nav_weight",
        "equity_sleeve_weight",
        "daily_return",
        "contribution",
        "market_value",
        "quantity",
        "avg_cost",
        "cost_basis",
        "current_price",
        "unrealized_pnl",
        "stop_loss",
        "take_profit",
        "quote_age_seconds",
    }
    trade_numeric = {
        "price",
        "quantity",
        "trade_amount",
        "fee",
        "slippage",
        "ledger_delta",
    }
    factor_registry_path = registry_path or PROJECT_ROOT / "quant_investor" / "factor_registry" / "mined_factors.json"
    factor_sha = sha256_file(factor_registry_path)
    factor_protocol, factor_protocol_path = _load_factor_protocol_v3_readback(
        latest_run,
        registry_sha=factor_sha,
        protocol_file=factor_protocol_file,
        expected_sha256=expected_factor_sha256,
    )
    manifest = load_json(latest_run.path / "manifest.json")
    readiness_reference = manifest.get("v15_run_readiness")
    if not isinstance(readiness_reference, dict):
        readiness_reference = {}
    readiness_path = latest_run.path / str(
        readiness_reference.get("path") or "v15_run_readiness.json"
    )
    readiness_sha = sha256_file(readiness_path)
    readiness_valid = bool(
        readiness_reference.get("schema_version") == "v15_run_readiness.v1"
        and readiness_reference.get("path") == "v15_run_readiness.json"
        and readiness_sha == readiness_reference.get("sha256")
    )
    industry_rows = build_industry_rows(position_rows)
    calendar = dict(trading_calendar or {})
    if not calendar:
        calendar = {
            "status": "missing",
            "source_system": "strict_parquet.cn_bars.trade_date",
            "path_summary": path_summary(DEFAULT_CN_BARS_ROOT),
            "expected_open_dates": [],
            "expected_open_date_count": 0,
            "first_open_date": None,
            "last_open_date": None,
            "mask_sha256": None,
        }
    reconciliation = build_attribution_reconciliation(
        nav_rows,
        position_rows,
        trading_calendar=calendar,
    )
    as_of_matrix = build_canonical_as_of_matrix(
        latest_run,
        benchmark_summary,
        factor_registry_sha=factor_sha,
    )
    blockers: list[str] = []
    if not nav_rows:
        blockers.append("nav_missing")
    if not position_rows:
        blockers.append("positions_missing")
    if reconciliation["status"] != "reconciled":
        blockers.append("attribution_reconciliation_partial")
    blockers.extend(
        str(value) for value in list(reconciliation.get("blockers") or [])
    )
    if manual_ledger_sha_declared is False:
        blockers.append("manual_ledger_sha_not_declared")
    if calendar.get("status") != "available":
        blockers.append("formal_trading_calendar_missing")
    if not readiness_valid:
        blockers.append("v15_run_readiness_missing_or_hash_mismatch")
    if factor_sha is None:
        blockers.append("factor_registry_missing")
    if factor_protocol.get("status") == "blocked":
        blockers.extend(
            str(value)
            for value in list(factor_protocol.get("blockers") or [])
        )
    if any(str(row.get("fee_source") or "") == "unknown" for row in trade_rows):
        blockers.append("trade_fee_unknown")
    if not benchmark_summary.get("production_grade"):
        blockers.append("benchmark_not_production_grade")
    for field, blocker in (
        ("strategy_record_date", "as_of_strategy_record_missing"),
        ("analysis_trading_date", "as_of_analysis_trade_date_missing"),
        ("quote_at", "as_of_quote_missing"),
    ):
        if as_of_matrix.get(field) is None:
            blockers.append(blocker)
    analysis_date = explicit_iso_date(as_of_matrix.get("analysis_trading_date"))
    if analysis_date:
        quote_date = explicit_iso_date(as_of_matrix.get("quote_at"))
        if quote_date and quote_date > analysis_date:
            blockers.append("as_of_quote_after_analysis_date")
        if any(
            explicit_iso_date(value) is not None
            and explicit_iso_date(value) > analysis_date
            for value in dict(
                as_of_matrix.get("benchmark_value_dates") or {}
            ).values()
        ):
            blockers.append("as_of_benchmark_after_analysis_date")
    if not as_of_matrix.get("benchmark_value_dates"):
        blockers.append("as_of_benchmark_value_dates_missing")
    nav_return_provenance = {
        "source_field": "pnl_summary.total_value_after",
        "return_method": "time_weighted_unitization",
        "gross_or_net": "unknown",
        "trade_fee_inclusion": "unknown",
        "secondary_fee_adjustment_allowed": False,
    }
    blockers.append("nav_fee_provenance_unknown")
    status = "blocked" if not nav_rows or not position_rows else ("partial" if blockers or warnings else "fresh")
    metric_policy = {
        "returns_unit": "decimal",
        "contribution_formula": "nav_weight * daily_return",
        "annualization_min_open_day_coverage": 0.95,
        "annualization_min_valid_daily_returns": 60,
        "annualization_insufficient_status": "insufficient_daily_history",
        "rolling_window_min_open_day_coverage": 0.95,
        "trading_calendar_required": True,
        "excess_curve": "relative_wealth_ratio",
        "monthly_return": "previous_month_end_anchor",
        "unknown_numeric": None,
    }
    schema_path = DEFAULT_DASHBOARD_ROOT / "schema" / "dashboard_contract.v3.schema.json"
    protocol_payload = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "metric_policy": metric_policy,
        "required_tables": ["nav", "positions", "trades", "industries", "factors"],
    }
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "schema_sha256": sha256_file(schema_path),
        "protocol_hash": canonical_json_sha256(protocol_payload),
        "run_id": run_id,
        "generated_at": generated_at,
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "v15_run_readiness": dict(readiness_reference),
        "as_of_matrix": as_of_matrix,
        "trading_calendar": calendar,
        "nav_return_provenance": nav_return_provenance,
        "sources": {
            "strategy_records": {
                "path_summary": path_summary(record_root),
                "latest_record": latest_run.run_id,
                "sha256": sha256_file(latest_run.path / "pnl_summary.csv"),
            },
            "ledger": {
                "path_summary": path_summary(ledger_path),
                "sha256": sha256_file(ledger_path),
            },
            "manual_manifest": {
                "path_summary": path_summary(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "v15_run_readiness": {
                "path_summary": path_summary(readiness_path),
                "sha256": readiness_sha,
                "status": "verified" if readiness_valid else "blocked",
            },
            "factor_registry": {
                "path_summary": path_summary(factor_registry_path),
                "sha256": factor_sha,
                "status": "registry_source" if factor_sha else "missing",
            },
            "factor_protocol": {
                "path_summary": path_summary(factor_protocol_path),
                "sha256": sha256_file(factor_protocol_path),
                "status": str(factor_protocol.get("status") or "blocked"),
            },
        },
        "nav": [_numeric_record(row, nav_numeric) for row in nav_rows],
        "positions": [_numeric_record(row, position_numeric) for row in position_rows],
        "trades": [_numeric_record(row, trade_numeric) for row in trade_rows],
        "industries": industry_rows,
        "factors": _factor_rows_from_registry(factor_registry_path),
        "factor_protocol": factor_protocol,
        "reconciliation": reconciliation,
        "metric_policy": metric_policy,
    }


def write_dashboard_snapshot_v3(
    json_path: Path,
    js_path: Path,
    contract: dict[str, Any],
    *,
    generated_records_js: str,
) -> None:
    _write_private_text_atomic(
        json_path,
        json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _write_private_text_atomic(
        js_path,
        "window.DashboardSnapshotV3 = "
        + json.dumps(contract, ensure_ascii=False, allow_nan=False)
        + ";\n"
        + generated_records_js,
    )


def _write_private_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def dashboard_display_warnings(warnings: list[str]) -> list[str]:
    manual_ledger_skips = 0
    manifest_skips = 0
    incomplete_trade_rows = 0
    visible: list[str] = []
    for warning in warnings:
        text = str(warning)
        if text.endswith(": 缺少可读 ledger_after_manual_switch.csv，已跳过。"):
            manual_ledger_skips += 1
        elif text.endswith(": 缺少可读 manual_execution_manifest.json，已跳过。"):
            manifest_skips += 1
        elif text.startswith("trade_record_incomplete:"):
            incomplete_trade_rows += 1
        elif "条持仓记录缺少strict industry" in text and "回退标签" in text:
            continue
        else:
            visible.append(text)
    if manual_ledger_skips:
        visible.append(
            f"{manual_ledger_skips} 个历史记录因缺少可读 ledger_after_manual_switch.csv 已跳过；详见 export_summary.json。"
        )
    if manifest_skips:
        visible.append(
            f"{manifest_skips} 个历史记录因缺少可读 manual_execution_manifest.json 已跳过；详见 export_summary.json。"
        )
    if incomplete_trade_rows:
        visible.append(
            f"{incomplete_trade_rows} 条已执行交易记录因缺少 timestamp/symbol/action/shares/price/trade_value 被跳过；"
            "详见 export_summary.json 的 trade_record_completeness。"
        )
    return visible


def dashboard_display_infos(warnings: list[str], infos: list[str] | None = None) -> list[str]:
    visible = list(infos or [])
    for warning in warnings:
        text = str(warning)
        if "条持仓记录缺少strict industry" in text and "回退标签" in text:
            visible.append(text)
    return visible


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
    infos: list[str],
    nav_csv: str,
    positions_csv: str,
    trades_csv: str,
    contract: dict[str, Any] | None = None,
) -> None:
    payload = generated_records_js_text(
        generated_at=generated_at,
        source_root=source_root,
        latest_record=latest_record,
        record_count=record_count,
        warnings=warnings,
        infos=infos,
        nav_csv=nav_csv,
        positions_csv=positions_csv,
        trades_csv=trades_csv,
        contract=contract,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def generated_records_js_text(
    *,
    generated_at: str,
    source_root: Path,
    latest_record: str,
    record_count: int,
    warnings: list[str],
    infos: list[str],
    nav_csv: str,
    positions_csv: str,
    trades_csv: str,
    contract: dict[str, Any] | None = None,
) -> str:
    return (
        "window.DashboardGeneratedRecords = {\n"
        f"  generatedAt: {js_string(generated_at)},\n"
        f"  sourceRoot: {js_string(path_summary(source_root) or '')},\n"
        f"  latestRecord: {js_string(latest_record)},\n"
        f"  recordCount: {record_count},\n"
        f"  warnings: {json.dumps(dashboard_display_warnings(warnings), ensure_ascii=False, indent=2)},\n"
        f"  infos: {json.dumps(dashboard_display_infos(warnings, infos), ensure_ascii=False, indent=2)},\n"
        f"  contract: {'window.DashboardSnapshotV3' if contract is not None else 'null'},\n"
        "  csv: {\n"
        f"    nav: {js_string(nav_csv)},\n"
        f"    positions: {js_string(positions_csv)},\n"
        f"    trades: {js_string(trades_csv)}\n"
        "  }\n"
        "};\n"
    )


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


def aggregate_manual_record_warnings(
    record_warnings: list[str],
    effective_run: RecordRun,
) -> list[str]:
    summarized = [
        warning
        for warning in record_warnings
        if "manual baseline 无效" in warning
    ]
    undeclared_count = sum(
        1
        for warning in record_warnings
        if "manual_ledger_sha_not_declared" in warning
    )
    if undeclared_count:
        effective_status = str(
            (effective_run.manual_manifest or {}).get("status") or "unknown"
        )
        readback_verified = bool(
            effective_run.manual_ledger_sha256
            and effective_run.manual_ledger_path
            and sha256_file(effective_run.manual_ledger_path)
            == effective_run.manual_ledger_sha256
        )
        summarized.append(
            "manual_ledger_sha_not_declared: "
            f"count={undeclared_count}；"
            f"effective_manifest_status={effective_status}；"
            "effective_ledger_sha_declared="
            f"{str(effective_run.manual_ledger_sha_declared).lower()}；"
            "effective_computed_sha_readback_verified="
            f"{str(readback_verified).lower()}。"
        )
    return summarized


def export(
    record_root: Path,
    dashboard_root: Path,
    benchmark_source: str = DEFAULT_BENCHMARK_SOURCE,
    benchmark_gap_fill: str = "snapshot",
    benchmark_file: Path | None = None,
    trading_calendar_root: Path | None = None,
    factor_protocol_file: Path | None = None,
    expected_factor_sha256: str = "",
) -> dict[str, Any]:
    nav_runs, warnings = discover_record_runs(record_root, require_effective_manual=False)
    manual_runs, manual_record_warnings = discover_record_runs(
        record_root,
        require_effective_manual=True,
    )
    if not nav_runs:
        raise SystemExit(f"No usable records found under {record_root}")
    if not manual_runs:
        detail = " | ".join(manual_record_warnings[-5:])
        raise SystemExit(
            f"No usable manual ledger records found under {record_root}"
            + (f": {detail}" if detail else "")
        )
    manual_warning_summary = aggregate_manual_record_warnings(
        manual_record_warnings,
        manual_runs[-1],
    )
    warnings.extend(manual_warning_summary)
    benchmark_export: BenchmarkExport | None = None
    local_benchmark_file = resolve_local_benchmark_file(dashboard_root, benchmark_file)
    if benchmark_source in {"auto", "local"} and (benchmark_source == "local" or local_benchmark_file.exists()):
        benchmark_export, local_warnings = load_local_benchmark_export(nav_runs, local_benchmark_file)
        warnings.extend(local_warnings)
        if benchmark_export is None and benchmark_source == "local":
            warnings.append("已按 fail-closed 降级为 strategy_record.market_snapshot.indices benchmark；不得标记为生产级。")
    # Network benchmark access is opt-in only. ``auto`` remains a legacy
    # local/snapshot compatibility mode and must not contact Tushare.
    if benchmark_source == "tushare":
        if benchmark_export is None:
            benchmark_export, tushare_warnings = load_tushare_benchmark_export(
                nav_runs,
                snapshot_gap_fill=benchmark_gap_fill == "snapshot",
            )
            warnings.extend(tushare_warnings)
            if benchmark_export is None and benchmark_source == "tushare":
                warnings.append("已按 fail-closed 降级为 strategy_record.market_snapshot.indices benchmark；不得标记为生产级。")
    if benchmark_source == "snapshot":
        warnings.append("benchmark_source=snapshot：按显式参数使用 strategy_record.market_snapshot.indices，非生产级。")
    if benchmark_export is not None:
        benchmark_export, industry_warnings = attach_industry_equal_weight_nav(nav_runs, benchmark_export)
        warnings.extend(industry_warnings)
    nav_rows, nav_warnings, nav_fieldnames = build_nav_rows(nav_runs, benchmark_export)
    calendar_root = trading_calendar_root or DEFAULT_CN_BARS_ROOT
    trading_calendar, calendar_warnings = load_strict_parquet_trading_calendar(
        calendar_root,
        nav_runs[0].date,
        nav_runs[-1].date,
    )
    warnings.extend(calendar_warnings)
    nav_source_summary = portfolio_nav_source_summary(nav_runs, nav_rows)
    benchmark_summary = benchmark_source_summary(nav_rows, nav_fieldnames, benchmark_export)
    sector_map, sector_warnings = load_sector_map(DEFAULT_STOCK_BASIC_ROOT)
    positions_rows, position_warnings = build_positions_rows(
        manual_runs,
        sector_map,
    )
    apply_position_exposures(nav_rows, positions_rows)
    trade_warnings: list[str] = []
    trade_record_completeness: dict[str, Any] = {}
    trade_rows = build_trade_rows(
        manual_runs,
        sector_map,
        warnings=trade_warnings,
        completeness=trade_record_completeness,
    )
    warnings.extend(nav_warnings)
    warnings.extend(sector_warnings)
    warnings.extend(position_warnings)
    warnings.extend(trade_warnings)
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
            "nav_weight",
            "equity_sleeve_weight",
            "industry",
            "industry_source",
            "industry_as_of",
            "industry_generation_sha256",
            "daily_return",
            "contribution",
            "contribution_effective_date",
            "contribution_date_source",
            "market_value",
            "quantity",
            "avg_cost",
            "cost_basis",
            "current_price",
            "unrealized_pnl",
            "recommended_action",
            "stop_loss",
            "take_profit",
            "quote_at",
            "quote_age_seconds",
            "thesis",
            "risk_status",
        ],
        positions_rows,
    )
    write_csv(
        trades_path,
        [
            "trade_date",
            "recommendation_id",
            "decision_id",
            "order_id",
            "fill_id",
            "ticker",
            "name",
            "side",
            "price",
            "quantity",
            "trade_amount",
            "fee",
            "fee_source",
            "slippage",
            "ledger_delta",
            "reason",
            "industry",
        ],
        trade_rows,
    )
    write_csv(
        benchmark_path,
        ["date", "ts_code", "field", "close", "nav", "source_system", "value_date", "coverage"],
        benchmark_export.raw_rows if benchmark_export is not None else [],
    )
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    dashboard_infos = [
        f"NAV 使用 {len(nav_runs)} 个正式 pnl_summary 记录；positions/trades 使用 "
        f"{len(manual_runs)} 个有效手工持仓基线记录。"
    ]
    if nav_source_summary["funding_events"]:
        funding_text = "，".join(
            f"{event['date']} 外部资金流 {float(event['amount']):,.2f}"
            for event in nav_source_summary["funding_events"]
        )
        dashboard_infos.append(
            "portfolio_nav 使用现金流调整后的单位净值；"
            f"{funding_text} 已从收益计算中剔除，账户总资产与资金基准保留在 NAV 审计字段。"
        )
    effective_manual_run = manual_runs[-1]
    effective_ledger_path = effective_manual_run.manual_ledger_path or (
        effective_manual_run.path / "ledger_after_manual_switch.csv"
    )
    effective_manifest_path = effective_manual_run.manual_manifest_path or (
        effective_manual_run.path / "manual_execution_manifest.json"
    )
    if (
        effective_manual_run.manual_manifest_sha256
        and (
            sha256_file(effective_manifest_path)
            != effective_manual_run.manual_manifest_sha256
            or load_json_object(effective_manifest_path)
            != effective_manual_run.manual_manifest
        )
    ):
        raise RuntimeError("manual manifest changed after baseline selection")
    contract = build_dashboard_contract_v3(
        run_id=f"dashboard_{manual_runs[-1].run_id}",
        generated_at=generated_at,
        record_root=record_root,
        latest_run=manual_runs[-1],
        nav_rows=nav_rows,
        position_rows=positions_rows,
        trade_rows=trade_rows,
        benchmark_summary=benchmark_summary,
        ledger_path=effective_ledger_path,
        manifest_path=effective_manifest_path,
        warnings=warnings,
        trading_calendar=trading_calendar,
        factor_protocol_file=factor_protocol_file,
        expected_factor_sha256=expected_factor_sha256,
        manual_ledger_sha_declared=(
            effective_manual_run.manual_ledger_sha_declared
        ),
    )
    generated_text = generated_records_js_text(
        generated_at=generated_at,
        source_root=record_root,
        latest_record=manual_runs[-1].run_id,
        record_count=len(nav_runs),
        warnings=warnings,
        infos=dashboard_infos,
        nav_csv=csv_text(nav_path),
        positions_csv=csv_text(positions_path),
        trades_csv=csv_text(trades_path),
        contract=contract,
    )
    private_dir = dashboard_root / DEFAULT_PRIVATE_DASHBOARD_DIRNAME
    private_json_path = private_dir / "dashboard_snapshot.v3.json"
    private_js_path = private_dir / "dashboard_snapshot.v3.js"
    write_dashboard_snapshot_v3(
        private_json_path,
        private_js_path,
        contract,
        generated_records_js=generated_text,
    )
    summary = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "schema_sha256": contract["schema_sha256"],
        "protocol_hash": contract["protocol_hash"],
        "run_id": contract["run_id"],
        "status": contract["status"],
        "blockers": contract["blockers"],
        "as_of_matrix": contract["as_of_matrix"],
        "trading_calendar": contract["trading_calendar"],
        "nav_return_provenance": contract["nav_return_provenance"],
        "reconciliation": contract["reconciliation"],
        "v15_run_readiness": contract["v15_run_readiness"],
        "factor_protocol": contract["factor_protocol"],
        "generated_at": generated_at,
        "source_root": path_summary(record_root),
        "dashboard_root": path_summary(dashboard_root),
        "latest_record": manual_runs[-1].run_id,
        "latest_nav_record": nav_runs[-1].run_id,
        "record_count": len(nav_runs),
        "nav_record_count": len(nav_runs),
        "manual_record_count": len(manual_runs),
        "nav_rows": len(nav_rows),
        "positions_rows": len(positions_rows),
        "trade_rows": len(trade_rows),
        "portfolio_nav_source": nav_source_summary,
        "protocol_artifacts": {
            "factor": {
                "path_summary": path_summary(factor_protocol_file),
                "sha256": sha256_file(factor_protocol_file),
            },
        },
        "trade_record_completeness": trade_record_completeness,
        "effective_manual_ledger_status": {
            "status": "valid",
            "record_id": effective_manual_run.run_id,
            "manifest_status": str(
                (effective_manual_run.manual_manifest or {}).get("status") or ""
            ),
            "manifest_recorded_at": effective_manual_run.record_time,
            "manifest_order_key": list(
                effective_manual_run.manual_order_key or ()
            ),
            "ledger_path": path_summary(effective_ledger_path),
            "ledger_sha256": effective_manual_run.manual_ledger_sha256,
            "ledger_sha_declared": (
                effective_manual_run.manual_ledger_sha_declared
            ),
            "ledger_readback_verified": bool(
                effective_manual_run.manual_ledger_sha256
                and sha256_file(effective_ledger_path)
                == effective_manual_run.manual_ledger_sha256
            ),
            "manifest_path": path_summary(effective_manifest_path),
            "manifest_sha256": effective_manual_run.manual_manifest_sha256,
            "manifest_readback_verified": bool(
                effective_manual_run.manual_manifest_sha256
                and sha256_file(effective_manifest_path)
                == effective_manual_run.manual_manifest_sha256
                and load_json_object(effective_manifest_path)
                == effective_manual_run.manual_manifest
            ),
            "legacy_ledger_fallback_used": False,
        },
        "manual_record_warnings": manual_warning_summary,
        "warnings": warnings,
        "benchmark_source": benchmark_summary,
        "files": {
            "nav": path_summary(nav_path),
            "positions": path_summary(positions_path),
            "trades": path_summary(trades_path),
            "benchmark": path_summary(benchmark_path),
            "snapshot_json": path_summary(private_json_path),
            "generated_js": path_summary(private_js_path),
        },
    }
    _write_private_text_atomic(
        generated_dir / "export_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
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
    parser.add_argument(
        "--trading-calendar-root",
        type=Path,
        default=None,
        help="Strict Parquet CN bars root used only to build the formal trade_date mask.",
    )
    parser.add_argument(
        "--factor-protocol-file",
        type=Path,
        default=None,
        help="Explicit Factor Governance Protocol v3 readback JSON artifact.",
    )
    parser.add_argument(
        "--expected-factor-sha256",
        default="",
        help="Required byte SHA-256 when --factor-protocol-file is provided.",
    )
    args = parser.parse_args()
    summary = export(
        args.record_root,
        args.dashboard_root,
        benchmark_source=args.benchmark_source,
        benchmark_gap_fill=args.benchmark_gap_fill,
        benchmark_file=args.benchmark_file,
        trading_calendar_root=args.trading_calendar_root,
        factor_protocol_file=args.factor_protocol_file,
        expected_factor_sha256=args.expected_factor_sha256,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
