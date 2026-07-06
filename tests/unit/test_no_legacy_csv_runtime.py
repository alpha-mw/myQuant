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
    # Dashboard export reads strategy-record CSV artifacts, not runtime market data.
    "scripts/export_cn_aggressive_dashboard_data.py",
    "scripts/migrate_legacy_csv_state_to_parquet.py",
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
