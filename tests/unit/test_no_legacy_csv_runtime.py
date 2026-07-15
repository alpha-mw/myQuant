from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = [
    ROOT / "quant_investor",
    ROOT / "daily_runner.py",
    ROOT / "daily_config.py",
    ROOT / "scripts",
]

ALLOWLIST = {
    # Formal-review and audit readers consume strategy-record CSV artifacts,
    # never canonical market bars.
    "quant_investor/monitoring/cn_aggressive_daily_review.py",
    "scripts/build_holdings_fundamental_sheet.py",
    "scripts/print_pipeline_state.py",
    "scripts/run_track_record_audit.py",
    # Dashboard export reads strategy-record CSV artifacts, not runtime market data.
    "scripts/export_cn_aggressive_dashboard_data.py",
    "scripts/backfill_cn_dashboard_benchmark.py",
    "scripts/check_cn_dashboard_export.py",
    "scripts/merge_cn_dashboard_benchmark_fills.py",
    # Offline migration and calibration tools accept explicit CSV inputs.
    "quant_investor/themes/membership_migration.py",
    "scripts/run_theme_threshold_sweep.py",
    # Hash-bound offline retirement replay reads frozen evidence, never market runtime data.
    "scripts/run_v14_retirement_replay_gate.py",
    "scripts/migrate_legacy_csv_state_to_parquet.py",
    "scripts/run_us_aggressive_analysis.py",
}

FORBIDDEN_SNIPPETS = [
    "pd.read_csv(",
    "pandas.read_csv(",
    "read_csv(",
    "SharedCSVReader",
    "SharedCSVReadResult",
    "CSVStore",
    "USLocalCSVDataSource",
    "csv_reader",
    "csv_store",
    "shared_csv_reader",
    "csv.DictReader",
    "csv.reader",
]


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def test_production_runtime_has_no_legacy_csv_read_ports() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        rel_path = path.relative_to(ROOT).as_posix()
        if rel_path in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                violations.append(f"{rel_path}: {snippet}")

    assert violations == []
