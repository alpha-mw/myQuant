#!/usr/bin/env python3
"""Merge verified CN dashboard benchmark fill rows into the local input CSV."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_FILE = PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_index_benchmark.csv"
DEFAULT_FILL_FILE = PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_index_benchmark_missing_rows.csv"
OUTPUT_COLUMNS = ["date", "ts_code", "close", "source_system", "coverage", "value_date"]
VALID_COVERAGE = {"exact_close", "previous_trading_day_ffill"}
SUPPORTED_CODES = {
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000688.SH",
    "399006.SZ",
}
FORBIDDEN_SOURCE_TOKENS = (
    "sample",
    "mock",
    "demo",
    "strategy_record.market_snapshot.indices",
)


class BenchmarkFillError(ValueError):
    """Raised when fill rows are unsafe to merge."""


@dataclass(frozen=True)
class MergeResult:
    benchmark_file: str
    fill_file: str
    output_file: str
    write: bool
    existing_rows: int
    valid_fill_rows: int
    pending_fill_rows: int
    replaced_rows: int
    output_rows: int
    warnings: list[str]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def normalize_date(value: str | None) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    return ""


def parse_positive_close(value: str | None) -> float | None:
    try:
        close = float(str(value or "").strip())
    except ValueError:
        return None
    if close <= 0:
        return None
    return close


def is_forbidden_source(source_system: str) -> bool:
    normalized = source_system.strip().lower()
    return any(token.lower() in normalized for token in FORBIDDEN_SOURCE_TOKENS)


def normalize_existing_row(row: dict[str, str], row_number: int) -> dict[str, str]:
    date = normalize_date(row.get("date"))
    ts_code = str(row.get("ts_code") or "").strip()
    close = parse_positive_close(row.get("close"))
    source_system = str(row.get("source_system") or "").strip()
    coverage = str(row.get("coverage") or "exact_close").strip()
    value_date = normalize_date(row.get("value_date") or date)
    if not date or ts_code not in SUPPORTED_CODES or close is None or not source_system:
        raise BenchmarkFillError(f"benchmark row {row_number} is invalid")
    if is_forbidden_source(source_system):
        raise BenchmarkFillError(f"benchmark row {row_number} has forbidden source_system={source_system}")
    if coverage not in VALID_COVERAGE:
        raise BenchmarkFillError(f"benchmark row {row_number} has invalid coverage={coverage}")
    if not value_date:
        raise BenchmarkFillError(f"benchmark row {row_number} has invalid value_date")
    return {
        "date": date,
        "ts_code": ts_code,
        "close": f"{close:.6f}",
        "source_system": source_system,
        "coverage": coverage,
        "value_date": value_date,
    }


def normalize_fill_row(row: dict[str, str], row_number: int) -> tuple[dict[str, str] | None, str | None]:
    date = normalize_date(row.get("date"))
    ts_code = str(row.get("ts_code") or "").strip()
    close = parse_positive_close(row.get("close"))
    source_system = str(row.get("source_system") or "").strip()
    coverage = str(row.get("coverage") or "exact_close").strip()
    value_date = normalize_date(row.get("value_date") or date)
    if not date or ts_code not in SUPPORTED_CODES:
        raise BenchmarkFillError(f"fill row {row_number} has invalid date or ts_code")
    if close is None and not source_system:
        return None, "pending_close_and_source"
    if close is None or not source_system:
        raise BenchmarkFillError(f"fill row {row_number} must provide both positive close and source_system")
    if is_forbidden_source(source_system):
        raise BenchmarkFillError(f"fill row {row_number} has forbidden source_system={source_system}")
    if coverage not in VALID_COVERAGE:
        raise BenchmarkFillError(f"fill row {row_number} has invalid coverage={coverage}")
    if not value_date:
        raise BenchmarkFillError(f"fill row {row_number} has invalid value_date")
    return {
        "date": date,
        "ts_code": ts_code,
        "close": f"{close:.6f}",
        "source_system": source_system,
        "coverage": coverage,
        "value_date": value_date,
    }, None


def merge_benchmark_fills(
    benchmark_file: Path = DEFAULT_BENCHMARK_FILE,
    fill_file: Path = DEFAULT_FILL_FILE,
    output_file: Path | None = None,
    *,
    write: bool = False,
) -> MergeResult:
    if not benchmark_file.exists():
        raise BenchmarkFillError(f"benchmark file not found: {benchmark_file}")
    if not fill_file.exists():
        raise BenchmarkFillError(f"fill file not found: {fill_file}")
    output = output_file or benchmark_file

    existing_rows = [
        normalize_existing_row(row, row_number)
        for row_number, row in enumerate(read_csv_rows(benchmark_file), start=2)
    ]
    row_map = {(row["date"], row["ts_code"]): row for row in existing_rows}

    valid_fill_rows = 0
    pending_fill_rows = 0
    replaced_rows = 0
    warnings: list[str] = []
    for row_number, row in enumerate(read_csv_rows(fill_file), start=2):
        normalized, pending_reason = normalize_fill_row(row, row_number)
        if pending_reason:
            pending_fill_rows += 1
            continue
        assert normalized is not None
        key = (normalized["date"], normalized["ts_code"])
        if key in row_map:
            replaced_rows += 1
        row_map[key] = normalized
        valid_fill_rows += 1

    if pending_fill_rows:
        warnings.append(f"{pending_fill_rows} fill rows still lack verified close/source_system.")
    if valid_fill_rows == 0:
        warnings.append("No verified fill rows were available to merge.")

    output_rows = [
        row_map[key]
        for key in sorted(row_map, key=lambda item: (item[0], item[1]))
    ]
    if write and valid_fill_rows:
        write_csv_rows(output, output_rows)

    return MergeResult(
        benchmark_file=str(benchmark_file),
        fill_file=str(fill_file),
        output_file=str(output),
        write=write and valid_fill_rows > 0,
        existing_rows=len(existing_rows),
        valid_fill_rows=valid_fill_rows,
        pending_fill_rows=pending_fill_rows,
        replaced_rows=replaced_rows,
        output_rows=len(output_rows),
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--fill-file", type=Path, default=DEFAULT_FILL_FILE)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="Write merged rows. Default is dry-run.")
    args = parser.parse_args()

    try:
        result = merge_benchmark_fills(
            benchmark_file=args.benchmark_file,
            fill_file=args.fill_file,
            output_file=args.output_file,
            write=args.write,
        )
    except BenchmarkFillError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
