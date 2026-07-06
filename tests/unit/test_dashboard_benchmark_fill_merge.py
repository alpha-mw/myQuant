from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_merger():
    spec = importlib.util.spec_from_file_location(
        "dashboard_benchmark_fill_merger",
        ROOT / "scripts" / "merge_cn_dashboard_benchmark_fills.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_merge_benchmark_fills_writes_verified_rows_and_leaves_pending(tmp_path):
    merger = _load_merger()
    benchmark_file = tmp_path / "cn_index_benchmark.csv"
    fill_file = tmp_path / "cn_index_benchmark_missing_rows.csv"
    _write_csv(
        benchmark_file,
        [
            {
                "date": "2026-06-24",
                "ts_code": "000300.SH",
                "close": "4800",
                "source_system": "tushare.index_daily",
                "coverage": "exact_close",
                "value_date": "2026-06-24",
            }
        ],
        merger.OUTPUT_COLUMNS,
    )
    _write_csv(
        fill_file,
        [
            {
                "date": "2026-06-25",
                "ts_code": "000300.SH",
                "close": "4900",
                "source_system": "Wind",
                "coverage": "exact_close",
                "value_date": "2026-06-25",
                "required_field": "csi300_nav",
                "reason": "verified close",
            },
            {
                "date": "2026-06-25",
                "ts_code": "000688.SH",
                "close": "",
                "source_system": "",
                "coverage": "exact_close",
                "value_date": "2026-06-25",
                "required_field": "star50_nav",
                "reason": "pending",
            },
        ],
        merger.OUTPUT_COLUMNS + ["required_field", "reason"],
    )

    result = merger.merge_benchmark_fills(benchmark_file, fill_file, write=True)

    rows = _read_csv(benchmark_file)
    assert result.valid_fill_rows == 1
    assert result.pending_fill_rows == 1
    assert result.output_rows == 2
    assert rows[-1] == {
        "date": "2026-06-25",
        "ts_code": "000300.SH",
        "close": "4900.000000",
        "source_system": "Wind",
        "coverage": "exact_close",
        "value_date": "2026-06-25",
    }


def test_merge_benchmark_fills_replaces_existing_key(tmp_path):
    merger = _load_merger()
    benchmark_file = tmp_path / "cn_index_benchmark.csv"
    fill_file = tmp_path / "cn_index_benchmark_missing_rows.csv"
    _write_csv(
        benchmark_file,
        [
            {
                "date": "2026-06-25",
                "ts_code": "000300.SH",
                "close": "4800",
                "source_system": "tushare.index_daily",
                "coverage": "exact_close",
                "value_date": "2026-06-25",
            }
        ],
        merger.OUTPUT_COLUMNS,
    )
    _write_csv(
        fill_file,
        [
            {
                "date": "2026-06-25",
                "ts_code": "000300.SH",
                "close": "4900",
                "source_system": "Choice",
                "coverage": "exact_close",
                "value_date": "2026-06-25",
            }
        ],
        merger.OUTPUT_COLUMNS,
    )

    result = merger.merge_benchmark_fills(benchmark_file, fill_file, write=True)

    rows = _read_csv(benchmark_file)
    assert result.replaced_rows == 1
    assert result.output_rows == 1
    assert rows[0]["close"] == "4900.000000"
    assert rows[0]["source_system"] == "Choice"


def test_merge_benchmark_fills_rejects_snapshot_or_sample_source(tmp_path):
    merger = _load_merger()
    benchmark_file = tmp_path / "cn_index_benchmark.csv"
    fill_file = tmp_path / "cn_index_benchmark_missing_rows.csv"
    _write_csv(
        benchmark_file,
        [
            {
                "date": "2026-06-24",
                "ts_code": "000300.SH",
                "close": "4800",
                "source_system": "tushare.index_daily",
                "coverage": "exact_close",
                "value_date": "2026-06-24",
            }
        ],
        merger.OUTPUT_COLUMNS,
    )
    _write_csv(
        fill_file,
        [
            {
                "date": "2026-06-25",
                "ts_code": "000300.SH",
                "close": "4900",
                "source_system": "strategy_record.market_snapshot.indices",
                "coverage": "exact_close",
                "value_date": "2026-06-25",
            }
        ],
        merger.OUTPUT_COLUMNS,
    )

    with pytest.raises(merger.BenchmarkFillError, match="forbidden source_system"):
        merger.merge_benchmark_fills(benchmark_file, fill_file, write=True)

    assert len(_read_csv(benchmark_file)) == 1
